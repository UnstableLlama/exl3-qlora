"""
A runtime LoRA must reach the token-by-token decode path.

Linear.forward is the only place a runtime LoRA is applied, and the fused decode paths (exl3_mgemm,
the BC_* graphs) don't go through it, so each of them has to either add the delta itself or step aside.
This runs greedy cached decode on a real model with an adapter attached and compares every step's
logits with a plain no-cache forward of the same tokens (long sequence -> per-Linear path, where the
adapter is always applied). It walks through load / unload / reload at another scale / two adapters
stacked, since the graphs re-record whenever the attached adapter changes.

Needs a GPU, an EXL3 model and a PEFT adapter trained for it:
    python tests/test_lora_decode_.py <model_dir> <lora_dir>
or EXL3_TEST_MODEL / EXL3_TEST_LORA.
"""
import sys, os, gc
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.model.lora import LoRA

NEW_TOKENS = 40
PROMPTS = [
    "The old lighthouse keeper had not spoken to another person in eleven years, and when the boat finally came he "
    "found that he had forgotten how to begin. He stood on the rocks with his hands in his coat pockets and watched "
    "it pull in. The first thing he said was",
    "In 1969, the first humans landed on the Moon. The mission was called Apollo 11, and the three astronauts on "
    "board were Neil Armstrong, Buzz Aldrin and Michael Collins. What most people do not know about the landing is that",
    "Water boils at 100 degrees Celsius at sea level, but on a mountain the boiling point is lower because the air "
    "pressure is lower. This matters for cooking because food in boiling water on a mountain",
]


def rfn(a, b):
    a = a.float().view(-1); b = b.float().view(-1).to(a.device)
    m = torch.isfinite(a) & torch.isfinite(b)
    return ((a[m] - b[m]).norm() / b[m].norm()).item()


def generate(model, cache, tok, prompts, n):
    """Greedy decode; returns per job (tokens, [per-step logits before sampling])."""
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 4)
    jobs = [Job(input_ids = tok.encode(p, add_bos = True), max_new_tokens = n, sampler = ArgmaxSampler(), return_logits = True) for p in prompts]
    toks = {j: [] for j in jobs}; lg = {j: [] for j in jobs}
    for j in jobs: gen.enqueue(j)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                toks[r["job"]] += r["token_ids"].view(-1).tolist()
                lg[r["job"]].append(r["logits"].view(-1, r["logits"].shape[-1])[0].clone())
    return [(toks[j][:n], lg[j][:n]) for j in jobs]


def reference_err(model, tok, prompt, toks, lg):
    """Max relative error between the per-step decode logits and the no-cache forward of the same tokens."""
    ids = torch.cat((tok.encode(prompt, add_bos = True), torch.tensor([toks[:-1]])), dim = 1)
    ref = model.forward(ids, params = {"last_tokens_only": len(toks)})[0].float()
    return max(rfn(lg[k], ref[k]) for k in range(len(lg)))


def main(model_dir, lora_dir):
    torch.manual_seed(0)
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 8192)
    model.load()
    tok = Tokenizer.from_config(config)

    def stage():
        # bsz 1 twice (eager + capture, then replay), then a bsz-3 batch
        e = [reference_err(model, tok, PROMPTS[0], *generate(model, cache, tok, PROMPTS[:1], NEW_TOKENS)[0]) for _ in range(2)]
        res = generate(model, cache, tok, PROMPTS, NEW_TOKENS)
        e += [reference_err(model, tok, p, *r) for p, r in zip(PROMPTS, res)]
        return max(e), res[0][0]

    with torch.inference_mode():
        # The adapter-free decode error is this model's own noise floor against the no-cache reference
        # (it varies a lot between models and bitrates); adapter stages are held to a multiple of it
        base_err, base_toks = stage()
        tol = max(3 * base_err, 0.05)
        results = [("base", base_err)]

        a = LoRA.from_directory(model, lora_dir)
        err, full_toks = stage()
        results.append(("adapter x1.0", err))
        assert err < tol, f"adapter x1.0: decode logits deviate from the reference (max rfn {err:.4f}, tol {tol:.4f}) -- adapter dropped on a fused path?"
        assert full_toks != base_toks, "adapter x1.0 generated exactly the base tokens -- is the adapter a no-op?"

        a.unload()
        err, toks = stage()
        results.append(("unloaded", err))
        assert toks == base_toks, "tokens after unload differ from the base run -- adapter still applied somewhere"

        a = LoRA.from_directory(model, lora_dir, lora_scaling = 0.5)
        b = LoRA.from_directory(model, lora_dir, lora_scaling = 0.5)
        err, toks = stage()
        results.append(("two adapters, x0.5 each", err))
        assert err < tol, f"two stacked adapters: decode logits deviate from the reference (max rfn {err:.4f}, tol {tol:.4f})"

        b.unload(); a.unload()
        err, toks = stage()
        results.append(("all unloaded", err))
        assert toks == base_toks, "tokens after unloading both adapters differ from the base run"

    print(f"PASS {os.path.basename(model_dir.rstrip('/'))}: " + ", ".join(f"{k} {v:.4f}" for k, v in results) + f" (tol {tol:.4f})")
    model.unload(); gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else os.environ["EXL3_TEST_MODEL"],
        sys.argv[2] if len(sys.argv) > 2 else os.environ["EXL3_TEST_LORA"],
    )
