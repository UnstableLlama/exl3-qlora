#!/usr/bin/env python
"""Merge a trained LoRA adapter into bf16 HF weights (the merge-and-requantize
deploy path).

For each adapted linear:

    W_merged = W + scale * (lora_B @ lora_A)          # [out,in] += [out,r]@[r,in]
    scale    = user_scaling * alpha / (sqrt(r) if rslora else r)

This exactly replicates what ``exllamav3.model.lora.LoRA.from_directory`` applies
at inference (same global ``alpha/r`` scale over the exported tensors), so the
merged model reproduces the runtime-adapter behavior up to bf16 storage rounding.
Output preserves the base model's exact shard layout + index, so ``convert.py``
consumes it directly for requantization.

Supports **default** and **pissa** inits (pissa exports the rank-2r combined
``[A|A0] / [s·B;-s·B0]`` form with the scale baked into B and config ``r``
unchanged, so the same ``alpha/r`` merge is correct). **Rejects** adapters with a
PEFT ``rank_pattern``/``alpha_pattern`` (e.g. mixed-rank MoE ``expert_r``): those
need per-module scales this single-scale merge would get wrong. Touches the
linear ``lora_A``/``lora_B`` tensors and, when the adapter carries
``trainable_tokens.safetensors`` (``--trainable-tokens``), writes the trained
rows into ``embed_tokens.weight`` (and ``lm_head.weight`` when the head columns
trained separately; a tied model shares the tensor). Engram / PLE sidecars
(Qwen3.8-Flash-Next): ``ple_modules.safetensors`` (``--train-ple``) is written
over the base's norm / conv tensors of the same name; ``ngram_rows.safetensors``
(``--train-ngram-rows``) is added onto the rows of an UNQUANTIZED
``<table>.weight`` tensor when the base index carries one, and skipped with a
note otherwise (deploy the rows through ``LoRA.from_directory``'s sidecar
instead). The head/embed LoRA sidecars (``lora_modules`` /
``modules_to_save``) are not handled.


Usage:
    python training/merge_lora_bf16.py \
        --base /path/to/bf16-hf-model --adapter out/my_adapter --out out/merged
    python convert.py -i out/merged -o out/merged-3bpw -w /tmp/work -b 3.0 -d 0
"""
import argparse, json, os, shutil, re
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Unquantized bf16/fp16 HF model dir")
    ap.add_argument("--adapter", required=True, help="Trained adapter dir (PEFT format)")
    ap.add_argument("--out", required=True, help="Output HF dir for the merged model")
    ap.add_argument("--scaling", type=float, default=1.0,
                    help="Extra user scaling on top of alpha/r (matches "
                         "LoRA.from_directory's lora_scaling; default 1.0)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit(
            " !! adapter has a rank_pattern/alpha_pattern (mixed-rank, e.g. MoE "
            "expert_r) -- this single-scale merge would mis-scale those modules. "
            "Not supported.")
    r = int(cfg["r"]); alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    scale = args.scaling * (alpha / (r ** 0.5 if use_rslora else r))
    print(f"adapter r={r} alpha={alpha} rslora={use_rslora} init={cfg.get('init_lora','default')} "
          f"-> merge scale={scale:.6f}")

    # Collect lora_A/lora_B per module: base_model.model.<module>.lora_{A,B}.weight.
    # Absent for a sidecar-only adapter (trained token rows / PLE with targets []).
    ad_path = os.path.join(args.adapter, "adapter_model.safetensors")
    ad = load_file(ad_path) if os.path.exists(ad_path) else {}
    mods: dict[str, dict[str, torch.Tensor]] = {}
    for k, t in ad.items():
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])\.weight$", k)
        if not m:
            print(f"  note: skipping non-lora_A/B tensor {k}")
            continue
        mods.setdefault(m.group(1), {})[m.group(2)] = t
    weight_of_mod = {}  # HF weight key -> module key
    for mod, ab in mods.items():
        assert "A" in ab and "B" in ab, f"{mod}: missing lora_A or lora_B"
        assert ab["A"].shape[0] == ab["B"].shape[1], \
            f"{mod}: rank mismatch A{tuple(ab['A'].shape)} B{tuple(ab['B'].shape)}"
        weight_of_mod[f"{mod}.weight"] = mod
    print(f"merging {len(weight_of_mod)} weights")

    # Trained token rows: {HF weight key suffix: (ids, rows)} -- matched by suffix
    # (model.embed_tokens.weight / model.language_model.embed_tokens.weight ...).
    row_writes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    tt_path = os.path.join(args.adapter, "trainable_tokens.safetensors")
    if os.path.exists(tt_path):
        tt = load_file(tt_path)
        ids = tt["embed_tokens.token_ids"].long()
        row_writes["embed_tokens.weight"] = (ids, tt["embed_tokens.rows"])
        head_mode = cfg.get("trainable_tokens_head", "off")
        if head_mode == "separate" and "lm_head.rows" in tt:
            row_writes["lm_head.weight"] = (ids, tt["lm_head.rows"])
        elif head_mode == "tied" and not cfg.get("tie_word_embeddings", True):
            # Trained as tied (shared rows) but the base keeps a separate head:
            # write the same rows into the head so the merged model matches.
            row_writes["lm_head.weight"] = (ids, tt["embed_tokens.rows"])
        print(f"writing {len(ids)} trained token rows into {sorted(row_writes)}")
    rows_written = 0

    # Engram / PLE: full tensors by name (exact key, else suffix), n-gram row deltas
    # added onto an unquantized table tensor when the base has one.
    ple_writes: dict[str, torch.Tensor] = {}
    ple_path = os.path.join(args.adapter, "ple_modules.safetensors")
    if os.path.exists(ple_path):
        ple_writes = load_file(ple_path)
        print(f"writing {len(ple_writes)} PLE tensors: {sorted(ple_writes)}")
    ple_written = 0
    ng_add = None
    ng_path = os.path.join(args.adapter, "ngram_rows.safetensors")
    if os.path.exists(ng_path):
        ng = load_file(ng_path)
        ng_key = f"{cfg.get('ngram_table', '')}.weight"
        ng_add = (ng_key, ng["ngram.uids"].long(), ng["ngram.delta"].float())
        print(f"adding {int(ng['ngram.uids'].numel())} n-gram row deltas onto {ng_key} "
              f"(if the base carries it unquantized)")
    ng_written = 0
    if not (weight_of_mod or row_writes or ple_writes or ng_add):
        raise SystemExit(f" !! {args.adapter} has nothing this script merges (no "
                         f"adapter_model / trainable_tokens / PLE sidecars)")

    os.makedirs(args.out, exist_ok=True)
    shard_of: dict[str, list[str]] = {}
    idx_path = os.path.join(args.base, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        for wk, shard in idx["weight_map"].items():
            shard_of.setdefault(shard, []).append(wk)
    else:
        # Single-shard checkpoint: no index, every tensor lives in model.safetensors.
        single = os.path.join(args.base, "model.safetensors")
        if not os.path.exists(single):
            raise SystemExit(f" !! {args.base} has neither model.safetensors.index.json "
                             f"nor model.safetensors")
        with safe_open(single, framework="pt") as f:
            shard_of["model.safetensors"] = list(f.keys())

    merged = 0
    for shard, keys in shard_of.items():
        sd = load_file(os.path.join(args.base, shard))
        touched = False
        for wk in keys:
            for suffix, (ids, rows) in row_writes.items():
                if wk.endswith(suffix):
                    W = sd[wk]
                    assert rows.shape[1] == W.shape[1], \
                        f"{wk}: row width {rows.shape[1]} != hidden {W.shape[1]}"
                    W = W.clone()
                    W[ids] = rows.to(W.dtype)
                    sd[wk] = W
                    rows_written += 1
                    touched = True
            for pk, pt in ple_writes.items():
                if wk == pk or wk.endswith("." + pk):
                    W = sd[wk]
                    assert pt.numel() == W.numel(), \
                        f"{wk}: PLE tensor numel {pt.numel()} != base {W.numel()}"
                    sd[wk] = pt.view_as(W).to(W.dtype).contiguous()
                    ple_written += 1
                    touched = True
            if ng_add is not None and (wk == ng_add[0] or wk.endswith("." + ng_add[0])):
                W = sd[wk].clone()
                ids, delta = ng_add[1], ng_add[2]
                assert delta.shape[1] == W.shape[1] and int(ids.max()) < W.shape[0], \
                    f"{wk}: n-gram delta {tuple(delta.shape)} does not fit table {tuple(W.shape)}"
                W[ids] = (W[ids].float() + delta).to(W.dtype)
                sd[wk] = W
                ng_written += 1
                touched = True
            if wk not in weight_of_mod:
                continue
            ab = mods[weight_of_mod[wk]]

            delta = scale * (ab["B"].float() @ ab["A"].float())   # [out,in]
            W = sd[wk]
            assert delta.shape == W.shape, f"{wk}: delta{tuple(delta.shape)} != W{tuple(W.shape)}"
            sd[wk] = (W.float() + delta).to(W.dtype)
            merged += 1
            touched = True
        save_file(sd, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"  wrote {shard} ({'merged' if touched else 'verbatim'})")
    assert merged == len(weight_of_mod), \
        f"merged {merged} weights but adapter has {len(weight_of_mod)} " \
        f"(a target module was missing from the base index?)"
    assert ple_written == len(ple_writes), \
        f"wrote {ple_written} PLE tensors but the adapter has {len(ple_writes)} " \
        f"(a PLE tensor was missing from the base index?)"
    if ng_add is not None and ng_written == 0:
        print(f"  note: n-gram row deltas NOT merged -- the base index has no unquantized "
              f"{ng_add[0]} (sharded / quantized table); deploy them through the "
              f"ngram_rows.safetensors sidecar with LoRA.from_directory instead")
    assert rows_written == len(row_writes), \
        f"wrote trained token rows into {rows_written} tensors but expected " \
        f"{sorted(row_writes)} (tied base with no lm_head.weight? then only the " \
        f"embedding needs them -- check adapter_config tie_word_embeddings)"


    # Copy every non-shard file verbatim (config, tokenizer, index, ...).
    shard_names = set(shard_of)
    for fn in os.listdir(args.base):
        p = os.path.join(args.base, fn)
        if fn not in shard_names and os.path.isfile(p):
            shutil.copy2(p, os.path.join(args.out, fn))
    print(f"done: {merged} weights merged into {args.out}")


if __name__ == "__main__":
    main()
