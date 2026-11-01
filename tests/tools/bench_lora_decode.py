"""
Decode throughput, latency and memory with a runtime LoRA: base model, adapter applied inside the captured decode
graphs (the default), and adapter on the unfused per-linear path (EXL3_LORA_GRAPH=0), under matched conditions.
Each mode runs in a child process (the env toggle is read at import), the parent prints the table and the logit
agreement between the two adapted runs.

    python tests/tools/bench_lora_decode.py -m /models/llama3.2-3b-exl3-4bpw
    python tests/tools/bench_lora_decode.py -m MODEL -a /adapters/my-lora --tokens 256 --batch 4
    python tests/tools/bench_lora_decode.py -m MODEL --rank 64 --modes base,graph

Without -a a random PEFT adapter of the given rank is written to a temp dir (testlib.lora). Timing covers the decode
steps only (first streamed token to last), after a warmup generation that captures the graphs. Memory is the peak
allocation of the generation on top of the loaded model, and the resident size of the adapter.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
REPO = os.path.dirname(TESTS)
for p in (TESTS, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

PROMPT = "Write a short story about a lighthouse keeper who discovers that the light"
MODES = {
    "base": {},                                       # no adapter
    "graph": {"EXL3_LORA_GRAPH": "1"},                # adapter inside the decode graphs
    "unfused": {"EXL3_LORA_GRAPH": "0"},              # adapter on the per-linear python path
}


def child(args) -> dict:
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
    from exllamav3.model.lora import LoRA

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = max(4096, (args.prompt_len + args.tokens + 256) * args.batch))
    if args.per_device:
        # split across GPUs; the python-side memory figures then sum over every device
        model.load(use_per_device = args.per_device)
        mem_devices = list(range(torch.cuda.device_count()))
    else:
        model.load(device = str(device))
        mem_devices = [device]
    allocated = lambda: sum(torch.cuda.memory_allocated(d) for d in mem_devices)
    tokenizer = Tokenizer.from_config(config)
    torch.cuda.synchronize(device)
    base_alloc = allocated()

    adapter = None
    adapter_bytes = 0
    if args.mode != "base":
        adapter_dir = args.adapter
        if adapter_dir is None:
            from testlib.lora import write_random_adapter
            adapter_dir = write_random_adapter(model, os.path.join(args.work_dir, "adapter"), rank = args.rank, seed = 0)
        adapter = LoRA.from_directory(model, adapter_dir)
        torch.cuda.synchronize(device)
        adapter_bytes = allocated() - base_alloc

    if args.prompt is not None:
        ids = tokenizer.encode(args.prompt, add_bos = True)
    else:
        ids = tokenizer.encode(PROMPT, add_bos = True)
        while ids.shape[-1] < args.prompt_len:
            ids = torch.cat([ids, ids[:, 1:]], dim = -1)
        ids = ids[:, :args.prompt_len]

    def run(record: bool):
        gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = args.batch)
        jobs = [Job(input_ids = ids, max_new_tokens = args.tokens, sampler = ArgmaxSampler(),
                    return_logits = record, min_new_tokens = args.tokens) for _ in range(args.batch)]
        for j in jobs:
            gen.enqueue(j)
        n_tokens = 0
        t_first = None
        logits = []
        tokens = []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    torch.cuda.synchronize(device)
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tokens += r["token_ids"].numel()
                    if record and r["job"] is jobs[0]:
                        tokens += r["token_ids"].view(-1).tolist()
                        logits.append(r["logits"].view(-1, r["logits"].shape[-1])[0].float().cpu())
        torch.cuda.synchronize(device)
        t_last = time.perf_counter()
        return n_tokens, t_last - t_first, tokens, logits

    with torch.inference_mode():
        run(False)   # warmup: autotune, graph capture (with the adapter nodes when attached)
        for d in mem_devices:
            torch.cuda.reset_peak_memory_stats(d)
        before = allocated()
        results = []
        for _ in range(args.repeats):
            n, dt, tokens, logits = run(True)
            results.append((n, dt))
        peak = sum(torch.cuda.max_memory_allocated(d) for d in mem_devices) - before
    n, dt = min(results, key = lambda r: r[1] / max(r[0], 1))
    out = {
        "mode": args.mode,
        "tokens": n,
        "seconds": dt,
        "tok_s": n / dt if dt > 0 else float("nan"),
        "ms_per_step": 1000.0 * dt / max(n // args.batch, 1),
        "gen_peak_mib": peak / 2**20,
        "adapter_mib": adapter_bytes / 2**20,
        "first_tokens": tokens[:16],
    }
    torch.save({"tokens": tokens, "logits": logits}, os.path.join(args.work_dir, f"{args.mode}.pt"))
    if adapter is not None:
        adapter.unload()
    return out


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required = True)
    ap.add_argument("-a", "--adapter", default = None, help = "PEFT adapter directory (default: random, see --rank)")
    ap.add_argument("--rank", type = int, default = 16, help = "rank of the random adapter")
    ap.add_argument("--tokens", type = int, default = 128, help = "decode steps per job")
    ap.add_argument("--prompt-len", type = int, default = 64, help = "the default prompt is repeated/cut to this many tokens")
    ap.add_argument("--prompt", default = None, help = "prompt text, used as is (default: a fixed prompt of --prompt-len tokens)")
    ap.add_argument("--batch", type = int, default = 1, help = "concurrent jobs (decode rows)")
    ap.add_argument("--repeats", type = int, default = 3, help = "timed generations per mode (best is reported)")
    ap.add_argument("--modes", default = "base,graph,unfused")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--per-device", type = float, nargs = "+", default = None,
                    help = "split the model over GPUs with these VRAM targets in GB (exllamav3 use_per_device)")
    ap.add_argument("--json", default = None, help = "also write the per-mode results to this JSON file")
    ap.add_argument("--work-dir", default = None)
    ap.add_argument("--child-mode", default = None, help = argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child_mode:
        args.mode = args.child_mode
        print(json.dumps(child(args)))
        return

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    with tempfile.TemporaryDirectory(prefix = "exl3_lora_bench_") as tmp:
        work = args.work_dir or tmp
        os.makedirs(work, exist_ok = True)
        rows = []
        for mode in modes:
            env = dict(os.environ, **MODES[mode], PYTHONPATH = os.pathsep.join([REPO, TESTS, os.environ.get("PYTHONPATH", "")]))
            passthrough = []
            skip = False
            for a in sys.argv[1:]:
                if skip:
                    skip = False
                elif a == "--work-dir":
                    skip = True
                else:
                    passthrough.append(a)
            cmd = [sys.executable, os.path.abspath(__file__), "--child-mode", mode, "--work-dir", work] + passthrough
            r = subprocess.run(cmd, env = env, capture_output = True, text = True)
            if r.returncode != 0:
                print(f"[{mode}] failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
                return 1
            rows.append(json.loads(r.stdout.strip().splitlines()[-1]))

        if args.json:
            with open(args.json, "w") as f:
                json.dump({"model": args.model, "adapter": args.adapter, "rank": args.rank, "batch": args.batch,
                           "prompt_len": args.prompt_len, "tokens": args.tokens, "rows": rows}, f, indent = 1)
        print(f"\nmodel {args.model}  batch {args.batch}  prompt {args.prompt_len}  decode {args.tokens} steps  "
              f"adapter {'random r=' + str(args.rank) if args.adapter is None else args.adapter}\n")
        print(f"{'mode':<10}{'tok/s':>10}{'ms/step':>10}{'gen peak MiB':>14}{'adapter MiB':>13}   first tokens")
        for r in rows:
            print(f"{r['mode']:<10}{r['tok_s']:>10.1f}{r['ms_per_step']:>10.2f}{r['gen_peak_mib']:>14.1f}"
                  f"{r['adapter_mib']:>13.1f}   {r['first_tokens']}")
        base = next((r for r in rows if r["mode"] == "base"), None)
        if base:
            for r in rows:
                if r["mode"] != "base":
                    print(f"  {r['mode']}: {100.0 * (r['ms_per_step'] / base['ms_per_step'] - 1.0):+.1f}% per-step time vs base")

        # Agreement of the two adapted runs (same prompt, same adapter): tokens and decode-step logits, and the
        # adapter's own effect (base vs in-graph) as the scale to read that against
        import torch

        def compare(x, y):
            """(identical token prefix, [per-step logits rfn over that prefix])"""
            n = min(len(x["tokens"]), len(y["tokens"]))
            agree = 0
            while agree < n and x["tokens"][agree] == y["tokens"][agree]:
                agree += 1
            errs = []
            for i in range(1, min(agree, len(x["logits"]), len(y["logits"]))):
                a, b = x["logits"][i], y["logits"][i]
                m = torch.isfinite(a) & torch.isfinite(b)
                errs.append(((a[m] - b[m]).norm() / b[m].norm()).item())
            return agree, n, errs

        def report(label, x, y):
            agree, n, errs = compare(x, y)
            print(f"{label}: first {agree}/{n} tokens identical; decode-step logits rfn "
                  f"max {max(errs) if errs else float('nan'):.4f} mean {sum(errs) / len(errs) if errs else float('nan'):.4f} "
                  f"over {len(errs)} agreeing steps")

        loaded = {m: torch.load(os.path.join(work, f"{m}.pt")) for m in MODES if os.path.exists(os.path.join(work, f"{m}.pt"))}
        print()
        if "graph" in loaded and "unfused" in loaded:
            report("graph vs unfused", loaded["graph"], loaded["unfused"])
        if "base" in loaded and "graph" in loaded:
            report("adapter effect (base vs graph)", loaded["base"], loaded["graph"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
