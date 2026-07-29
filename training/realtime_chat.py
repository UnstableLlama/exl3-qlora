"""
Interactive demo of real-time (inference-time) QLoRA training.

One loaded EXL3 model both chats and learns: generation runs through the
normal exllamav3 Generator with the adapter live in the runtime LoRA slots,
and any sample you feed it trains the same adapter in place
(:class:`exllamav3.training.RealtimeQLoRA`) -- no reload, no second model
copy. The page table is nuked on every adapter update (the config-option
default; see the realtime module docstring for the adapter-free-KV
alternative), so the next generation runs with fresh state under the updated
weights.

Usage:
    python training/realtime_chat.py --model /path/to/exl3-model \
        --checkpoint-dir out/realtime_demo

Chat normally at the prompt. Commands:

    /learn <text>      Replace the assistant's LAST reply with <text> and
                       train on the corrected conversation (the whole point:
                       correct the model and watch the correction stick).
    /again             Re-ask the last user message (compare before/after).
    /ingest <file>     Train through a JSONL file of samples; each line is a
                       dict in any ingest form: {"messages": [...]},
                       {"prompt", "response"}, {"text"}, or pre-tokenized.
    /lr <value>        Set the (constant) learning rate, e.g. /lr 5e-5.
    /checkpoint        Write a timestamped adapter checkpoint now.
    /unload            Remove the adapter from generation (compare to base).
    /reload            Re-apply the adapter to generation.
    /reset             Clear the conversation history.
    /stats             Optimizer step count / samples seen / lr.
    /quit              Exit.

Prompts are rendered through the model directory's own Jinja chat template
(the same one inference servers use), and `/learn` / `/ingest` messages
samples train with exact per-turn loss masks via the same template
(training/chat_jinja.py) -- assistant turns supervised, everything else
masked. This script is also the reference wiring for hooking RealtimeQLoRA
into a server backend (e.g. tabbyAPI's backends/exllamav3/model.py): the
pieces marked [integration] below are exactly what a backend needs to add.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402
from exllamav3.training.realtime import RealtimeQLoRA, RealtimeConfig  # noqa: E402
from chat_jinja import (jinja_renderers, tokenizer_special_tokens,  # noqa: E402
                        extract_rich_turns, row_template_extras,
                        parse_template_vars)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True, help="EXL3 model directory")
    p.add_argument("--adapter", default=None,
                   help="resume from an adapter/checkpoint directory")
    p.add_argument("--cache-size", type=int, default=8192,
                   help="KV cache size in tokens (default 8192)")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--template-vars", default=None,
                   help='JSON template vars, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--max-new-tokens", type=int, default=512)
    # RealtimeConfig knobs (defaults follow the dataclass)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--targets", nargs="*", default=None,
                   help="LoRA target modules (default: all attn+mlp proj; "
                        "exclude k_proj/v_proj for an adapter-free KV cache)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="checkpoint every N optimizer steps (0 = manual only)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="prune to newest N checkpoints (0 = keep all)")
    return p.parse_args()


def main():
    args = parse_args()

    # -- ordinary exllamav3 serving setup (nothing realtime-specific yet) --
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_size)
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    # The model's own chat template drives BOTH prompt rendering for
    # generation and segment masking for training -- one source of truth.
    seg_build, _, eot = jinja_renderers(
        args.model, tokenizer_special_tokens(tokenizer),
        default_vars=parse_template_vars(args.template_vars))

    def render_segments(sample):
        turns = extract_rich_turns(sample["messages"])
        return seg_build(turns, **row_template_extras(sample))

    # -- [integration] the realtime coordinator ------------------------------
    rt = RealtimeQLoRA(
        model, tokenizer,
        RealtimeConfig(
            r=args.r, alpha=args.alpha, target_modules=args.targets,
            lr=args.lr, batch_size=args.batch, grad_accum=args.grad_accum,
            seq_len=args.seq_len, checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            keep_checkpoints=args.keep_checkpoints),
        adapter_dir=args.adapter,
        render_segments=render_segments,
        base_model_name_or_path=args.model)
    rt.attach_generator(generator)   # page table reset on every adapter update
    print(f" -- realtime adapter: {rt.net.num_trainable():,} trainable params, "
          f"lr {rt.lr:g}, targets {' '.join(rt.net.target_modules)}")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    # Stop on EOS and on the template's own turn-close string. The rendered
    # prompt already carries any BOS the template wants (add_bos=False below,
    # same as the offline trainer's live sampling).
    stop = [s for s in (tokenizer.eos_token_id, eot or None)
            if s is not None and s != ""]

    def generate():
        prompt = "".join(t for t, _ in
                         seg_build(extract_rich_turns(messages),
                                   add_generation_prompt=True))
        # [integration] every generator drive holds the inference read lock,
        # so an ingest can drain in-flight requests and block new ones.
        with rt.inference():
            return generator.generate(
                prompt=prompt, max_new_tokens=args.max_new_tokens,
                stop_conditions=stop, completion_only=True, add_bos=False)

    def report(stats):
        if stats["steps"] == 0:
            print(" -- nothing to train on")
            return
        print(f" -- trained: {stats['steps']} step(s), "
              f"{stats['samples']} sample(s), "
              f"{stats['supervised_tokens']} supervised tokens, "
              f"loss {stats['mean_loss']:.4f}, "
              f"{stats['duration_s']:.1f}s (lr {stats['lr']:g}); "
              f"adapter live, cache flushed")
        for c in stats["checkpoints"]:
            print(f" -- checkpoint: {c}")

    print("Chat away. /learn <corrected reply> trains on a correction; "
          "/quit exits; see --help for all commands.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()
            if cmd == "/quit":
                break
            elif cmd == "/reset":
                messages = messages[:1] if args.system else []
                print(" -- conversation cleared")
            elif cmd == "/lr":
                try:
                    rt.lr = float(rest)
                    print(f" -- lr = {rt.lr:g}")
                except ValueError:
                    print(" !! usage: /lr 5e-5")
            elif cmd == "/stats":
                print(f" -- step {rt.step}, samples seen {rt.samples_seen}, "
                      f"lr {rt.lr:g}")
            elif cmd == "/checkpoint":
                try:
                    print(f" -- checkpoint: {rt.checkpoint()}")
                except ValueError as e:
                    print(f" !! {e} (pass --checkpoint-dir)")
            elif cmd == "/unload":
                rt.unload_from_inference()
                print(" -- adapter removed from generation (base model)")
            elif cmd == "/reload":
                rt.sync_to_inference()
                print(" -- adapter re-applied to generation")
            elif cmd == "/again":
                if not messages or messages[-1]["role"] != "assistant":
                    print(" !! nothing to retry")
                    continue
                messages.pop()
                response = generate()
                print(response)
                messages.append({"role": "assistant", "content": response})
            elif cmd == "/learn":
                if not rest:
                    print(" !! usage: /learn <corrected assistant reply>")
                    continue
                if not messages or messages[-1]["role"] != "assistant":
                    print(" !! no assistant reply to correct yet")
                    continue
                messages[-1] = {"role": "assistant", "content": rest}
                report(rt.ingest([{"messages": list(messages)}]))
            elif cmd == "/ingest":
                if not os.path.exists(rest):
                    print(f" !! no such file: {rest}")
                    continue
                with open(rest, encoding="utf8") as f:
                    samples = [json.loads(l) for l in f if l.strip()]
                report(rt.ingest(samples))
            else:
                print(f" !! unknown command {cmd}")
            continue

        messages.append({"role": "user", "content": line})
        response = generate()
        print(response)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
