# Runtime LoRA inside the fused decode path

Runtime adapters (`exllamav3.LoRA`, PEFT format, applied without touching the base weights) used to be
applied only by `Linear.forward`. Every decode-time path that reads the EXL3 trellis directly (the
multi-linear `exl3_mgemm` bundles and the graph-captured C++ blocks) bypassed that method. Upstream
v1.6.5 still runs those paths unconditionally, so an adapter is silently dropped at decode on a
quantized model; this fork guarded them, which was correct but moved every adapted module to the
unfused python path (roughly three times the kernel launches, no graph replay).

This change puts the adapter into the graphs themselves, following turboderp's guidance: it is
worked into `Linear` and its C++ counterpart `BC_LinearEXL3` and the graph nodes built around it,
not implemented as independent linear layers. Dense attention and the MLPs are covered; the rest
keep the unfused fallback (listed below).

## Code-path map

Where an adapter on a target `Linear` is applied, by execution path:

| Path | Entry | Adapter applied by |
|------|-------|--------------------|
| Prefill, batched decode (> 32 rows), anything unfused | `Linear.forward` → `LinearEXL3.forward` → `BC_LinearEXL3::run_alloc` or `reconstruct_hgemm`, then `Linear.apply_lora` | python, `x.addmm_(x @ A, B)` |
| Attention decode, python path, ≤ 32 rows | `Attention.project_qkv`: `exl3_mgemm` over the K/V and Q/G pairs | python, `apply_lora` on the mgemm output (the sliced Q/K/V/G bundle is skipped under an adapter: no epilogue) |
| GatedMLP, python path, ≤ 32 rows | `GatedMLP.forward`: `exl3_mgemm` over gate/up, `Linear.forward` for down | python, `apply_lora` on the mgemm output; down through `Linear.forward` |
| **Attention decode, graph** (bsz ≤ 8, q_len ≤ 16) | `Attention.forward` → `bc_attn_step` → `BC_Attention::run` (also `SlidingAttention` via `build_bc_swa`) | **in-graph**: `BC_LinearEXL3::lora_gr` after each of q/k/v/(g)/o |
| **GatedMLP, graph** (≤ 8 rows) | `GatedMLP.forward` → `BC_GatedMLP::run_bszN` | **in-graph**: `lora_gr` on gate/up before the activation, on down after |
| **MLP (non-gated), graph** (1 row) | `MLP.forward` → `BC_MLP::run_bsz1` | **in-graph**: `lora_gr` on up and down |
| **GatedDeltaNet split, graph** (Qwen 3.5 / 3.8 / Next; bsz ≤ 8, seqlen ≤ 16) | `GatedDeltaNet.forward` → `BC_GatedDeltaNetSplit::run_bszN` | **in-graph**: `lora_gr` on qkv/z (bundle or separate), b/a onto their halves of the merged `ba` static, o |
| MLA, DSA, GDN in KDA mode (GLM 5.3), Mamba2, shared-KV attention graphs | their `bc` objects | **not graphed**: `has_runtime_lora` guards decline, python path applies it |
| Routed MoE experts (`BlockSparseMLP`) | `exl3_moe` / BC graphs / mgemm | **not graphed**: per-expert torch path while an expert projection carries an adapter |
| `lm_head`, embeddings | `Linear.forward` / folded into the embedding weight | python (never in a graph) |

## Design

### The C++ linear owns the adapter (`libtorch/linear.h`, `linear.cpp`)

Each `Linear` already owns one C++ handle (`inner.bc`: `BC_LinearEXL3` for quantized layers,
`BC_LinearFP16` for the few unquantized ones such as the GDN b/a projections and headwise gates),
the object every graph owner projects through. Both now embed a `BC_LoRA` component and forward
to it:

- `a` (R, K) fp16, A transposed; `b` (R, N) fp16 with the alpha/r scale folded in (as `LoRA`
  already stores B); `t` (128, R) fp32 rank intermediate, sized to the largest graphed row count
  (`BC_Attention::MAX_BSZ * MAX_QLEN`). Several adapters on one linear are stacked along R.
- `set_lora(a, b)` / `set_lora(None, None)` attach or detach, validate shapes against the trellis
  geometry, and bump the graph structure epoch (below).
- `lora_gr(x, y, graph)` emits `y += (x @ A) @ B` as two kernels (`lora.cu`): `lora_a` (one block
  per rank row and token, fp32 dot over K) and `lora_b` (one thread per output column, fp32 dot
  over R, accumulated in place on fp16 or fp32 `y`). Both record patchable sites, `GP_lora_x` on
  `lora_a`'s input and `GP_lora_y` on `lora_b`'s output, so the nodes work on the graph owner's
  patched I/O pointers exactly like the GEMM nodes. `lora_b` accepts any row stride on `y`, so a
  delta can land on a column slice of a wider static (the GDN's merged b/a output). Without an
  adapter it emits nothing.
- `lora_args(params, x, y)` appends the two replay patches in recording order, or nothing.
- `run_gr` (the python `Linear.forward` entry) stays adapter-free, so the python path keeps
  applying the adapter itself and nothing is applied twice.

The kernels are deliberately simple: at decode shapes the cost is the two launches per adapted
projection, not the arithmetic, and the graph replays them as nodes.

### Graph owners call it per projection (`attention.cpp`, `mlp.cpp`)

Every projection site in `BC_Attention::run_gr`, `BC_GatedMLP::run_bszN_gr`, `BC_MLP::run_bsz1_gr`
and `BC_GatedDeltaNetSplit::run_bszN_gr` is followed by `proj->lora_gr(input, output, graph)`,
including the mgemm bundles (the fused Q/K/V(/G), Q/G, K/V, gate/up and qkv/z tables bypass the
handles for the GEMM, but the handles are still bound, so each bundled projection adds its own
delta onto its slice of the static output). Projections the graphs run as plain fp16 GEMMs (the
attention's fp16 gate, the GDN's merged b/a GEMV) get their `BC_LinearFP16` handle bound through a
setter (`set_g_proj_fp16`, `set_ba_proj`) for the delta only. The order matches `Linear.forward`
plus the module: delta after the GEMM and bias, before V-norm, head norms, RoPE, the activation,
and the GDN's gate op.

`run()` mirrors every `lora_gr` with `lora_args` in the same position of its patch list, patching
the static intermediates with their own addresses. This matters: `Graph::launch` walks the recorded
sites in order and binds each patch to the next site with the same id, so a static-output LoRA node
left out of the list would steal a later patch of the same id (the o_proj delta's output, the only
LoRA site that really changes per call). The tripwire test checks that owners record and patch in
step.

`BC_GatedMLP` now receives the gate/up handles in the fused-mgemm configuration too (python passes
them; the fused launch still reads its pointer tables).

### Lifecycle: the graph structure epoch (`graph.cuh`, `graph.cu`)

A captured graph is a fixed node list. Attaching or detaching an adapter changes the list of every
graph that projects through that linear, and nothing tracks which graphs those are. Instead,
`set_lora` bumps a global `Graph::structure_epoch`; each graph records the epoch at capture and
`Graph::refresh()`, called at the top of every owner's `run`, drops a graph captured under an older
epoch (`capture_abort`) so it re-records on that run. Adapter changes are rare, so invalidating all
graphs on each one is fine (one eager run plus one capture per slot, the same cost as warmup).
Graph owners that never emit LoRA nodes (MLA, GDN, ...) do not call `refresh` and are unaffected.

### Python hand-off (`modules/linear.py`)

`Linear.lora_sync()` mirrors the attached `lora_a_tensors` / `lora_b_tensors` into the C++ object:
it stacks adapters along the rank dim, transposes A once (the python path keeps the `[in, r]`
orientation), and returns whether the graphed paths may run. It declines, so the module takes the
unfused path, when the linear is fp16, a `lora_full_weight` override (modules_to_save) is installed,
a tensor is not fp16 or not the padded width, or `EXL3_LORA_GRAPH=0`. It is lazy and cached by
tensor identity: the adapter dicts are written directly by `LoRA` and by the trainer
(`training/backbone.py`), so the first dispatch after a change does the hand-off, and emptying the
dicts detaches. `lora_graphed(*linears)` is the dispatch guard: the BC branches of `Attention`,
`SlidingAttention`, `GatedMLP` and `MLP` run only when every involved projection returns True.

The `LoRA` class is unchanged: `LoRA.from_directory` / `unload` keep working, `lora_scaling` keeps
folding into B (zero scale is an exact no-op in the kernels), and loading/unloading mid-session
re-records the graphs on the next decode step. `pin_linears` resets the sync state when it rebuilds
the handle.

## Unsupported, by choice (unfused fallback, still correct)

- MLA / DSA attention (`mla_attn.py`), GDN in KDA mode (GLM 5.3: the forget/output gate
  projections are embedded as base weights), Mamba2, shared-KV attention: their graphs have no
  LoRA stage yet. The same two calls per projection site (`lora_gr` in `run_gr`, `lora_args` in
  `run`) plus a `refresh()` would extend them; they were left out to keep the contribution
  reviewable.
- Routed MoE experts: the fused expert kernels read the expert tables directly; an adapter on
  `experts.N.*` keeps the per-expert torch path. Shared experts inside a `BC_BlockSparseMLP` graph
  are read from the `BC_GatedMLP`'s tables, not through `run_bszN_gr`, so they also stay unfused
  under an adapter (the existing `sh_fused_lora` guard).
- Fused `kv_proj` attention variants, tensor-parallel sliced linears: declined by `lora_sync` /
  the guards.
- Per-request adapter selection or scales: not implemented. B carries the scale, so a scale change
  is a re-sync plus re-record; a patchable per-node scale would be the natural next step, and
  per-request selection additionally needs the batching and cache semantics worked out (two
  sequences in one decode batch share the graph and therefore the adapter set). Zero scale removes
  the adapter's effect but not its compute.

## Validation

Nothing in this change was compiled or run in the environment it was written in (no GPU, no
torch). Everything below is the plan a machine with a CUDA build should execute, in this order.

1. Build: `pip install .` (the new `lora.cu` is picked up by the source walk).
2. Source tripwires (no GPU): `pytest tests/test_lora_fused_path.py`.
3. Kernels and hand-off, synthetic weights: `pytest tests/modules/linear/test_linear_lora_graph.py`.
   `run_lora` against `y + x @ A @ B` for fp16/fp32 outputs and 1..128 rows, the 129-row limit,
   a padded adapter over a narrow activation, stacked adapters through `lora_sync`, and the
   decline cases.
4. Graphs, real model, random adapters: `pytest tests/graph/test_lora_graph.py` (the `dense`
   registry models, and the `gdn` hybrids for the GatedDeltaNet graph with adapters on
   in_proj_qkv/z/b/a and out_proj as well).
   - `test_lora_graph_matches_eager`: graph replay vs the same C++ sequence eager (`EXL3_GRAPHS=0`)
     with adapters attached; lifecycle in one process (attach, detach, second adapter, zero scale,
     back to base, first adapter again); zero scale bit-exact with base; base-again and
     adapter-again bit-exact with their first runs.
   - `test_lora_graph_matches_unfused`: in-graph vs `EXL3_LORA_GRAPH=0` (the python reference).
     The two paths differ by the decode path's own precision, so the check is relative: on the
     decode steps whose token prefix agrees, the discrepancy must stay under a quarter of the
     adapter's own effect, and at least three decode tokens must agree. The 0.25 factor is a first
     guess; tighten it once measured.
   `EXL3_BC_ATTN_TRACE=1` confirms the attention graphs were built for the test model.
5. A trained adapter: `training/qlora_infer_native.py --model ... --adapter ...` before/after, and
   the same prompts through TabbyAPI, comparing against `EXL3_LORA_GRAPH=0`.

## Benchmark

`python tests/tools/bench_lora_decode.py -m MODEL [-a ADAPTER | --rank 16] [--tokens 256] [--batch 1]`
runs base, in-graph adapter and unfused adapter in matched child processes and prints tokens/s,
ms per step, peak generation memory, adapter memory, the per-step cost relative to base and the
token/logit agreement of the two adapted runs. Expected shape of the result: the unfused path
costs well over the extra matmuls (every adapted layer leaves its graph), the in-graph path costs
two small nodes per adapted projection (for a 7-projection adapter on a 32-layer model, about 450
extra nodes per step, a few percent at most). Numbers belong in this section once measured.

## Files

- `exllamav3/exllamav3_ext/lora.cu`, `lora.cuh`: the kernels.
- `exllamav3/exllamav3_ext/graph.cuh`, `graph.cu`: `GP_lora_x` / `GP_lora_y`, structure epoch, `refresh`.
- `exllamav3/exllamav3_ext/libtorch/linear.h`, `linear.cpp`, `linear_bc.h`: adapter state on `BC_LinearEXL3`, bindings `set_lora`, `has_lora`, `run_lora`.
- `exllamav3/exllamav3_ext/libtorch/attention.cpp`, `mlp.cpp`, `gated_delta_net.cpp`: LoRA nodes and patches in `BC_Attention`, `BC_GatedMLP`, `BC_MLP`, `BC_GatedDeltaNetSplit`.
- `exllamav3/modules/linear.py`: `Linear.lora_sync`, `lora_graphed`, `EXL3_LORA_GRAPH`.
- `exllamav3/modules/attn.py`, `sliding_attn.py`, `mlp.py`, `gated_delta_net.py`, `attention_fn/bc_attn.py`: dispatch guards, bound handles.
- `tests/testlib/lora.py`, `tests/modules/linear/test_linear_lora_graph.py`, `tests/graph/test_lora_graph.py`, `tests/test_lora_fused_path.py`, `tests/tools/bench_lora_decode.py`.
