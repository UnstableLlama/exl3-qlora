"""
Generate a DENSE, CONSISTENT style-transfer dataset for QLoRA-on-EXL3 demos.

Why this exists
---------------
Our density lesson: a style demo only shows cleanly
at greedy/low-temp decode when the style transforms COMMON, high-probability
tokens of *every* response -- that is why ALL-CAPS worked and the off-the-shelf
pirate / UwU sets did not (their markers are sparse or live in rare tokens).
The fix the handoff recommended: *generate* a set where every row is heavily,
consistently transformed. This script does that.

It takes a normal instruction dataset (default: yahma/alpaca-cleaned) and uses a
LOCAL exllamav3 model to rewrite ONLY each response into the target style. The
instruction/input (the question) stay plain English; only the answer is styled,
exactly matching the completion-only masking the trainer uses. Output is written
in the Alpaca schema (instruction / input / output) that qlora_train_native.py
loads by default -- and the trainer now also accepts a local file path, so you
can point --dataset straight at the .jsonl this produces.

Default style is Yoda-speak: a SYNTACTIC transform (clause/word-order inversion)
over common tokens, so it surfaces under greedy decode where rare-token styles
(emoji/UwU) vanish.

Recommended rewriter: the LARGEST instruct model you have quantized. A 1B is too
weak to invert sentences reliably; use a 7-8B+ instruct EXL3 as --gen-model and
keep --model (the eventual training target) separate in your head.

Usage
-----
    python training/experiments/make_style_dataset.py \
        --gen-model /mnt/two/Weights/<a-good-instruct-model>/exl3/ \
        --out       /mnt/two/data/yoda_alpaca.jsonl \
        --max-rows  6000

    # then train on it (the trainer accepts a local file path directly; the
    # dataset is already styled, so do NOT pass --uppercase-response):
    python training/qlora_train_native.py \
        --model /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/ \
        --out   /mnt/two/Weights/meta-llama-Llama-3.2-1B-Instruct/4/yoda \
        --dataset /mnt/two/data/yoda_alpaca.jsonl
"""

import argparse
import json
import os

from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ComboSampler


# ---------------------------------------------------------------------------
# Style definitions. Each is (system_prompt, few_shot) where few_shot is a list
# of (plain, styled) pairs embedded in the rewrite instruction to anchor the
# transform densely. Add your own and select with --style.
# ---------------------------------------------------------------------------
STYLES = {
    "yoda": (
        "You are a precise text-style converter. Rewrite the user's passage in "
        "the distinctive speech of Yoda from Star Wars. Invert clause order so "
        "the object or predicate comes first (Object-Subject-Verb), e.g. "
        "'Powerful you have become.' Reorder MOST sentences this way -- the "
        "inversion must be dense and consistent, not occasional. Preserve ALL "
        "of the original meaning and information; stay coherent and fluent. Do "
        "NOT add greetings, commentary, quotation marks, or notes. Output only "
        "the rewritten passage.",
        [
            ("You will learn much from this experience.",
             "Much from this experience, learn you will."),
            ("The boy has no patience, so I cannot teach him.",
             "No patience, the boy has. Teach him, I cannot."),
            ("You must keep your concentration here and now.",
             "Your concentration here and now, keep you must."),
        ],
    ),
    "archaic": (
        "You are a precise text-style converter. Rewrite the user's passage in "
        "Early Modern / archaic English: use thee/thou/thy/thine, verb endings "
        "-est and -eth (dost, hath, doth, speaketh), and 'tis/'twas. Apply these "
        "densely across EVERY sentence. Preserve ALL original meaning; stay "
        "coherent. Do NOT add stage directions, commentary, or quotation marks. "
        "Output only the rewritten passage.",
        [
            ("You know that you are right.",
             "Thou knowest that thou art right."),
            ("He has what he needs and it is enough.",
             "He hath what he needeth, and 'tis enough."),
        ],
    ),
    "pirate": (
        "You are a precise text-style converter. Rewrite the user's passage as a "
        "boisterous pirate. Use pirate diction DENSELY in every sentence: arr, "
        "ahoy, matey, ye/yer, be (for is/are), aye, me hearties, savvy. Preserve "
        "ALL original meaning; stay coherent. Do NOT add commentary or quotation "
        "marks. Output only the rewritten passage.",
        [
            ("Hello friend, you are welcome here. What do you need?",
             "Ahoy matey, welcome aboard ye be! What be it ye need, arr?"),
        ],
    ),
    "corporate": (
        "You are a precise text-style converter. Rewrite the user's passage in "
        "dense corporate buzzword-speak: leverage, synergy, circle back, "
        "bandwidth, low-hanging fruit, move the needle, touch base, action item, "
        "going forward, at the end of the day. Work these into EVERY sentence "
        "while preserving ALL original meaning and staying coherent. Do NOT add "
        "commentary or quotation marks. Output only the rewritten passage.",
        [
            ("Let's talk tomorrow about finishing the easy tasks first.",
             "Let's touch base tomorrow to circle back on grabbing the "
             "low-hanging fruit first, going forward."),
        ],
    ),
}


def build_rewrite_prompt(model, style_key, text):
    system, few_shot = STYLES[style_key]
    lines = ["Examples of the target style:"]
    for plain, styled in few_shot:
        lines.append(f"\nPLAIN: {plain}\nSTYLED: {styled}")
    lines.append("\nNow rewrite the following passage in the same style. "
                 "Output only the rewritten passage:\n\n" + text)
    user = "\n".join(lines)
    return model.default_chat_prompt(user, system_prompt=system)


# Second-pass instructions per style. The first pass often leaves some sentences
# in normal order (esp. for syntactic styles like Yoda, where the signal is
# token-sparse); this pass pushes EVERY sentence into the style so the per-token
# signal is dense enough to surface at lora_scaling 1.0 (no amplification).
REFINE_INSTRUCTIONS = {
    "yoda": (
        "The passage below is ALREADY in Yoda's style, but some sentences may "
        "still be in normal English order. Rewrite it so EVERY sentence is "
        "inverted (object/predicate first) and MOST sentences END on a verb or "
        "auxiliary, e.g. '...you have', '...it is', '...learn you will', "
        "'...help you I can'. Leave NO sentence in plain subject-verb-object "
        "order. Preserve the meaning exactly. Output only the rewritten passage:"
    ),
    "archaic": (
        "The passage below is ALREADY in archaic English, but some sentences may "
        "lack the markers. Rewrite so EVERY sentence uses thee/thou/thy and "
        "-est/-eth verb endings. Preserve meaning. Output only the passage:"
    ),
    "pirate": (
        "The passage below is ALREADY pirate-styled, but some sentences are "
        "weak. Rewrite so EVERY sentence has dense pirate diction. Preserve "
        "meaning. Output only the passage:"
    ),
    "corporate": (
        "The passage below is ALREADY corporate-speak, but some sentences are "
        "plain. Rewrite so EVERY sentence carries buzzwords. Preserve meaning. "
        "Output only the passage:"
    ),
}


def build_refine_prompt(model, style_key, text):
    system, _ = STYLES[style_key]
    user = REFINE_INSTRUCTIONS[style_key] + "\n\n" + text
    return model.default_chat_prompt(user, system_prompt=system)


def clean_output(s):
    s = s.strip()
    # 1) Hard-cut at any turn/sequence marker. RP-tuned models (e.g. Rocinante,
    #    a Mistral finetune) end their turn with </s> and then happily role-play
    #    a whole conversation; keep only the first turn. These markers should
    #    never appear inside a single styled answer.
    for marker in ("</s>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>",
                   "[INST]", "[/INST]", "<|start_header_id|>"):
        i = s.find(marker)
        if i != -1:
            s = s[:i]
    # 2) Drop meta-commentary the model adds about its own rewrite. If such a
    #    line appears, cut everything from it onward (the real answer precedes).
    meta_markers = (
        "\n*Rewritten", "\n(Rewritten", "\n*Yoda", "\n*The passage",
        "\n*I ", "\n(Note", "\n*Note", "\nI see", "\nI understand",
        "\nApologies", "\nCould you", "\nHere's", "\nHere is the",
        "\n*Original meaning", "\n*(", "\nUnderstood",
    )
    for marker in meta_markers:
        i = s.find(marker)
        if i != -1:
            s = s[:i]
    s = s.strip()
    # 3) Strip a leading prefix / surrounding quotes.
    for prefix in ("STYLED:", "Rewritten:", "Here is the rewritten passage:",
                   "Sure!", "Sure,"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    # 4) Collapse runaway repetition loops (model degenerated without emitting
    #    EOS, e.g. the same sentence 30x). Keep first occurrence of each line.
    lines, seen, out = s.split("\n"), set(), []
    for ln in lines:
        key = ln.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ln)
    s = "\n".join(out).strip()
    # 5) Reject prompt echoes / refusals: instead of rewriting, the model
    #    sometimes parrots the instruction ("[Rewrite the passage in Yoda's
    #    style...]") or refuses. Return "" so the row is skipped, not written.
    low = s.lower()
    reject = (
        "rewrite the passage", "inverting clause order", "the rewritten passage",
        "yoda's distinctive", "object or predicate comes first",
        "i cannot rewrite", "i can't rewrite", "as it contains code",
    )
    if s.startswith("[") or any(r in low for r in reject):
        return ""
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", required=True,
                    help="EXL3 model dir used to REWRITE responses (use your "
                         "largest instruct model; a 1B is too weak for Yoda).")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--style", default="yoda", choices=list(STYLES.keys()))
    ap.add_argument("--source-dataset", default="yahma/alpaca-cleaned",
                    help="Source instruction set (Alpaca schema by default)")
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--refine-from", default=None,
                    help="Second pass: read an existing styled .jsonl and "
                         "re-style each response with a stricter prompt to push "
                         "EVERY sentence into the style (ignores --source-dataset).")
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="input")
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--max-rows", type=int, default=6000,
                    help="How many rows to generate (shuffled subset)")
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--max-response-words", type=int, default=160,
                    help="Skip very long answers to keep generation cheap")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-tokens", type=int, default=8192)
    args = ap.parse_args()

    # Create the output directory up front, so a missing path fails here rather
    # than after loading the (large) rewriter model.
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    refine = args.refine_from is not None
    if refine:
        if os.path.abspath(args.refine_from) == os.path.abspath(args.out):
            raise SystemExit("--refine-from and --out must differ (don't "
                             "overwrite the source while reading it).")
        src_rows = []
        with open(args.refine_from, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    src_rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset(args.source_dataset, split=args.source_split)
        src_rows = ds.shuffle(seed=args.seed)

    # Resume: count rows already written so a crashed/interrupted run continues.
    done = 0
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[resume] {done} rows already in {args.out}; continuing.")

    # Collect the source rows we intend to keep (after length filtering), so the
    # resume offset lines up with positions in this filtered stream. In refine
    # mode the text to re-style is the existing (already-styled) response.
    kept = []
    for ex in src_rows:
        instr = (ex.get(args.instruction_key) or "").strip()
        ctx = (ex.get(args.context_key) or "").strip()
        resp = (ex.get(args.response_key) or "").strip()
        nw = len(resp.split())
        if not instr or nw < args.min_response_words or nw > args.max_response_words:
            continue
        kept.append((instr, ctx, resp))
        if len(kept) >= args.max_rows:
            break
    mode = "refine" if refine else "generate"
    print(f"[plan] {mode}: {len(kept)} rows selected; {len(kept) - done} to do.")

    config = Config.from_directory(args.gen_model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_tokens)
    model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    sampler = ComboSampler(temperature=args.temperature, top_p=args.top_p)

    # Stop at end-of-turn. Without this the model keeps generating past its
    # answer (RP finetunes role-play a whole conversation), so pass the
    # tokenizer's EOS id(s) plus the common chat end-markers as strings.
    stop = list(getattr(config, "eos_token_id_list", None) or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop:
        stop.append(tokenizer.eos_token_id)
    stop += ["</s>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>", "[INST]"]

    out_f = open(args.out, "a", encoding="utf-8")
    todo = kept[done:]
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        builder = build_refine_prompt if refine else build_rewrite_prompt
        prompts = [builder(model, args.style, resp)
                   for (_, _, resp) in batch]
        outs = generator.generate(
            prompt=prompts, max_new_tokens=args.max_new_tokens,
            sampler=sampler, seed=args.seed, add_bos=False,
            completion_only=True, stop_conditions=stop,
        )
        if isinstance(outs, str):  # single-item safety
            outs = [outs]
        for (instr, ctx, _resp), styled in zip(batch, outs):
            styled = clean_output(styled)
            if not styled:
                continue
            rec = {"instruction": instr, "input": ctx, "output": styled}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n = done + start + len(batch)
        print(f"[gen] {n}/{len(kept)}", flush=True)
        # Show one live sample per batch so you can eyeball style density.
        if outs:
            print(f"    e.g. {clean_output(outs[0])[:160]}")

    out_f.close()
    print(f"\nDone. Wrote styled dataset to {args.out}")
    print("Train with:  --dataset", args.out,
          "  (Alpaca schema; do NOT pass --uppercase-response)")


if __name__ == "__main__":
    main()
