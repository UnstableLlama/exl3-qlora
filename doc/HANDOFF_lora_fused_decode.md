# TEMPORARY handoff note: runtime LoRA in the fused decode path

**Delete this file before merging the branch into master.** It is session state, not documentation; the
real write-up is `doc/lora_fused_decode.md`.

## Where things stand (branch `claude/vibrant-goodall-5vjnxi`, 3 commits on top of master a6dc950)

1. `ff29f7c` Runtime LoRA inside BC_Attention / BC_GatedMLP / BC_MLP (lora.cu kernels, BC_LinearEXL3 state,
   graph structure epoch, Linear.lora_sync / lora_graphed, tests, bench tool, doc).
2. `0324d24` tiny follow-up in attention.cpp.
3. `8a4c2e4` GDN split graph (Qwen3.5 / 3.8 / Next), BC_LoRA shared by BC_LinearEXL3 and BC_LinearFP16, strided
   lora_b output, fp16 attention gate handle.

**Nothing has been compiled or run.** The authoring environment had no GPU, no torch, no nvcc. Only the
source-level tripwires in `tests/test_lora_fused_path.py` were executed (13 pass).

## First session back: do this in order

1. `pip install .` and fix whatever the compiler says. Likely spots if anything: `linear.h` (BC_LoRA inline
   methods, `at::Device` by const ref), `lora.cu` (macro-defined kernels, the `(void*) &kernel` record_param
   address), `gated_delta_net.cpp` (`s.ba.narrow(...)` views).
2. `pytest tests/test_lora_fused_path.py`
3. `pytest tests/modules/linear/test_linear_lora_graph.py`
4. `pytest tests/graph/test_lora_graph.py` (dense + gdn registry models). If a replay throws
   "Graph update failed" or "Graph recording failed", a `lora_gr` site and its `lora_args` patch are out of order
   or one is missing. Compare the `run_gr` and `run` bodies of the owner side by side. If outputs are wrong but
   nothing throws, first check with `EXL3_LORA_GRAPH=0` (python fallback) and `EXL3_GRAPHS=0` (eager C++) to
   see which layer of the stack is at fault.
5. `python tests/tools/bench_lora_decode.py -m MODEL` and put the numbers in the doc's Benchmark section.
6. The 0.25 tolerance in `test_lora_graph_matches_unfused` is a guess. Tighten it to what is measured.

## Things I am not sure about (check on hardware)

- fp16 Linear weight geometry: `lora_sync` checks adapters against `inner.weight.shape` for fp16 layers. If the
  GDN b/a or headwise-gate weights are stored unpadded, lora.py's padding to `Linear.in/out_features` will make
  the sync decline (safe: fallback) and the GDN layer will leave the graph. Watch `graphed_projections()` in the
  graph test for that.
- Headwise gate (gate_mode 1): B width = `g_proj.out_features` (padded) vs `s.g2` width = num_q_heads. lora_b
  allows y narrower than B, so it should be fine, but it is untested.
- BC_GatedMLP with fp32 interm dtype: gate/up statics are fp32 then; lora_b handles fp32 y, x stays half.
- The GDN KDA branch must never call lora_gr (guarded by `!kda`); the python guard keeps `has_runtime_lora`
  there.

## Not done (see the doc's "Unsupported" section for the why)

- Routed MoE experts (needs a rank-space path inside the grouped expert kernel: large).
- Shared experts inside BC_BlockSparseMLP (coop kernel reads the tables; would need a fallback to the
  separate shared-expert launch under an adapter: small-medium).
- GDN KDA mode (GLM5.3), MLA / DSA attention, Mamba2 (same recipe as BC_Attention, each needs its statics
  traced: medium each).
- Per-request adapter scale / selection (API work; a patchable scale on the lora_b node is the cheap first
  step; selection needs the batching + cache story).

## Upstreaming

The branch is written to upstream style on purpose. For a PR to turboderp-org/exllamav3, the fork-only parts to
leave out are the `has_runtime_lora` guards on modules upstream does not guard (upstream has none), the
training/ references, and this file. The commit messages carry the design summary.
