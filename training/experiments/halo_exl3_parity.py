"""
Parity check: halo-exl3's pure-PyTorch EXL3 decode vs ExLlamaV3's own kernels, on a real conversion.

halo-exl3 (the Halo fork) reimplements the trellis decode, the codebooks and the tile order in plain PyTorch
from a reading of the extension sources. It is CPU-tested against a scalar transliteration of those sources
only -- this script is the first comparison against the kernels themselves. Needs a CUDA device, the built
extension, a converted model directory and a halo-exl3 checkout:

    python training/experiments/halo_exl3_parity.py --model /path/to/exl3-model --halo /path/to/halo-exl3

Per sampled linear it reports three things:

  inner    decode_trellis(trellis) == LinearEXL3.get_inner_weight_tensor()        expected: bit-exact
  dense    Exl3Linear.weight.T  vs  LinearEXL3.get_weight_tensor()                expected: ~1 fp16 ulp
  forward  Exl3Linear(x)        vs  LinearEXL3.forward(x)                         expected: fp16 GEMM noise

Exit status is non-zero if any inner comparison is not bit-exact.
"""

from __future__ import annotations
import argparse
import glob
import importlib.util
import os
import sys
import types

import torch
from safetensors import safe_open

EXL3_NAMES = ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1")


def load_halo_exl3(halo_root: str):
    """Import halo/kernels/exl3/{trellis,linear}.py by file, without importing the halo package (which pulls
    in transformers and the rest of Halo's stack)."""
    base = os.path.join(halo_root, "halo", "kernels", "exl3")
    for name in ("halo", "halo.kernels", "halo.kernels.exl3"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules.setdefault(name, pkg)
    mods = {}
    for name in ("trellis", "linear"):
        spec = importlib.util.spec_from_file_location(f"halo.kernels.exl3.{name}", os.path.join(base, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["trellis"], mods["linear"]


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required = True, help = "EXL3 model directory")
    ap.add_argument("--halo", required = True, help = "halo-exl3 checkout")
    ap.add_argument("--max-layers", type = int, default = 24, help = "linears to sample, spread over the model (0 = all)")
    ap.add_argument("--max-out", type = int, default = 32768, help = "skip linears wider than this (the LM head)")
    ap.add_argument("--device", default = "cuda:0")
    args = ap.parse_args()

    trellis_mod, linear_mod = load_halo_exl3(args.halo)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from exllamav3.modules.quant.exl3 import LinearEXL3

    files = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))
    assert files, f"no safetensors files in {args.model}"
    index = {}
    for f in files:
        with safe_open(f, framework = "pt") as r:
            for key in r.keys():
                if key.endswith(".trellis"):
                    index[key[:-len(".trellis")]] = f
    prefixes = sorted(index)
    assert prefixes, "no trellis tensors found -- is this an EXL3 model?"
    if args.max_layers and len(prefixes) > args.max_layers:
        step = len(prefixes) / args.max_layers
        prefixes = [prefixes[int(i * step)] for i in range(args.max_layers)]

    torch.manual_seed(0)
    failed = 0
    print(f"{'layer':60s} {'K':>4s} {'cb':>5s} {'inner':>10s} {'dense max|d|':>13s} {'fwd max|d|':>11s} {'fwd rel':>9s}")
    for prefix in prefixes:
        with safe_open(index[prefix], framework = "pt") as r:
            keys = set(r.keys())
            t = {n: r.get_tensor(f"{prefix}.{n}").to(args.device) for n in EXL3_NAMES if f"{prefix}.{n}" in keys}
        k, n = t["trellis"].shape[0] * 16, t["trellis"].shape[1] * 16
        if n > args.max_out:
            print(f"{prefix:60s} skipped (out_features {n} > --max-out)")
            continue

        ours = linear_mod.Exl3Linear(k, n, t).to(args.device)
        ref = LinearEXL3(
            None, k, n,
            su = t.get("su"), sv = t.get("sv"), suh = t.get("suh"), svh = t.get("svh"),
            trellis = t["trellis"], mcg = t.get("mcg"), mul1 = t.get("mul1"), bias = t.get("bias"),
            out_dtype = torch.half, key = prefix,
        )

        inner_ref = ref.get_inner_weight_tensor()
        inner_ours = trellis_mod.decode_trellis(t["trellis"], ours.codebook)
        exact = torch.equal(inner_ref, inner_ours)
        mism = (inner_ref != inner_ours).sum().item()
        failed += not exact

        dense_d = (ref.get_weight_tensor().float() - ours.weight.T.float()).abs().max().item()

        x = torch.randn(64, k, device = args.device, dtype = torch.half)
        y_ref = ref.forward(x.clone(), {}).float()
        y_ours = ours(x).float()
        fwd_d = (y_ref - y_ours).abs().max().item()
        fwd_rel = fwd_d / max(y_ref.abs().max().item(), 1e-9)

        inner_s = "exact" if exact else f"{mism} diff"
        print(f"{prefix:60s} {ours.bits_per_weight:4g} {ours.codebook:>5s} {inner_s:>10s} {dense_d:13.3e} {fwd_d:11.3e} {fwd_rel:9.2e}")
        del ours, ref, t

    print()
    if failed:
        print(f"FAIL: {failed} linear(s) decoded differently from the reference kernels")
        sys.exit(1)
    print("OK: every sampled trellis decodes bit-exactly")


if __name__ == "__main__":
    main()
