# Box runbook: in-graph runtime LoRA (stage 1) and the reference-logp cache

Branch `claude/lora-inference-optimization-hddtao`. Everything below was
written without a GPU; the CUDA/C++ has **not been compiled**. This is the
order to validate it in, with what each step proves and what to do if it
fails. Budget: ~1 hour on a 3090-class card once the extension builds.

## 0. Build

```bash
git fetch origin claude/lora-inference-optimization-hddtao
git checkout claude/lora-inference-optimization-hddtao
pip install . --no-build-isolation        # or: python setup.py build_ext --inplace
```

New compilation unit: `exllamav3/exllamav3_ext/lora.cu`. Touched:
`graph.cuh/.cu` (5 new `GP_lora_*` ids, `Graph::reset()`, `Graph::flavour`),
`libtorch/linear.h` (+`lora_a/lora_b`, `set_lora/clear_lora/has_lora`),
`libtorch/linear_bc.h`, `libtorch/mlp.h/.cpp`, `libtorch/attention.h/.cpp`,
`bindings.cpp`. If the build fails, the error is almost certainly in one of
those; nothing else in the extension changed.

## 1. Kernel unit test (no model needed)

```bash
python -m pytest tests/test_lora_graph.py::test_lora_gemv_kernel -q
```

Proves `ext.lora_gemv` matches `lora_delta_reference` over M 1..128, rank
8..256, fp16 and fp32 outputs. A failure here is a kernel bug in `lora.cu`;
the three phases are independent, so bisect by checking `partial` (phase 1)
against `x @ A` on a tiny shape first.

## 2. Safety net: the no-adapter path is untouched

```bash
python eval/perf.py -m /path/to/Llama-3.2-1B-4bpw -max_length 2048 -spf
```

Compare against the same command on `master`. The no-adapter graphs are the
same nodes as before (the LoRA flavour is 0 and `lora_graph_ok` is one
empty-dict check per Linear), so this should be within noise. Also confirm
`examples/chat.py` output is unchanged for a fixed prompt at temperature 0.

## 3. End-to-end parity on a real adapter

```bash
EXL3_TEST_MODEL=/path/to/Llama-3.2-1B-4bpw \
EXL3_TEST_ADAPTER=/path/to/adapter \
python -m pytest tests/test_lora_graph.py::test_graph_lora_parity -q -s
```

Proves, on bsz-1 decode: in-graph adapted logits == fallback adapted logits
(`EXL3_LORA_GRAPH=0`), base is byte-identical before and after, unload
restores it, re-attach re-engages. Any q/k/v/o/gate/up/down adapter on a
Llama-family quant is the reference case (`/mnt/two/Weights/qlora_test/base`
from the Session-27 notes, if it still exists).

Failure modes and where to look:

- **"Graph update failed" / "Graph recording failed"** from `Graph::launch`
  or `capture_end`: the per-launch param list in `BC_Attention::run` or
  `BC_GatedMLP::run_bszN` is out of order with the nodes `run_gr` recorded.
  The lora node records sites `x, A, B, C, rank`; the param lists emit them
  right after the projection they follow. Check the branch you are in
  (`use_qg_mgemm`, `use_k_as_v`, `use_mgemm`, gate mode) against `run_gr`.
- **Tokens differ but only after the first decode**: a pointer that should
  have been patched wasn't (the lora node read a stale input). Every lora
  node whose input is the caller's `x` must get a `GP_lora_x` entry; ones
  whose input is a static (`s.o2`, the MLP activation) must not.
- **Tokens differ from the first decode on**: numerics or placement. Run
  `EXL3_BC_ATTN=0` to isolate MLP vs attention, then compare the adapted
  delta of one layer's output against `Linear.apply_lora`.
- **Unload doesn't restore base**: the flavour did not flip back. `unload()`
  empties the slot dicts; `lora_graph_ok` then calls `sync_lora_bc` on any
  handle with a non-None `_lora_bc_key`, which must `clear_lora()`.

## 4. The number this was all for

```bash
M=/path/to/Llama-3.2-1B-4bpw; A=/path/to/adapter
python eval/perf.py -m $M -max_length 2048 -spf                       # base
python eval/perf.py -m $M -max_length 2048 -spf -lora $A              # adapted, in-graph
EXL3_LORA_GRAPH=0 python eval/perf.py -m $M -max_length 2048 -spf -lora $A   # old path
python eval/perf.py -m $M -max_length 2048 -spf -lora $A -lora_noop   # adapted branches, no LoRA math
```

Session 27 measured (1B, 3090): base ~325, old path 83.6, `-lora_noop` on
the old path 167. The plan's stage-1 projection is ~290 for the in-graph
run. If in-graph adapted comes out near the old 84, the graphs are not
engaging (check `lora_graph_ok` is returning True: an adapter on a non-EXL3
linear, e.g. an fp16 head or gate, sends the whole block to the fallback).
Repeat on the 3B and, if available, an 8B.

## 5. Reference-logp cache (training, any GPU)

```bash
python training/qlora_train.py --config training/qlora_train_pref_config.yaml   # method: dpo, epochs: 2
```

Epoch 1 prints misses, epoch 2 should be all hits (`ref-logp cache: N hits /
M misses` at exit), and a second run on the same model + dataset starts
with `(N entries loaded)`. Loss curves must match a `ref_cache: off` run to
within fp noise (the reference logps are computed in a sub-batch containing
only the missing rows, so padding differs from the policy batch; a
difference beyond ~1e-3 in the step-0 DPO loss means something else).

## Not wired in stage 1 (still on the guarded fallback, by design)

`sliding_attn.py` (Gemma SWA graph), `BC_MLP` (non-gated), Mamba2, GDN
split, MLA, MoE routed experts and the MoE bszN graph's embedded shared
experts, fp16 (unquantized) projections. Each still yields to the unfused
dispatch while it carries an adapter, exactly as before.
