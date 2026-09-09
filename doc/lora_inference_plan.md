# Runtime LoRA at inference: review, root cause of the slowdown, and the plan to fix it

> Written 2026-09-02 on branch `claude/lora-inference-optimization-hddtao`, against
> `c44a280` (upstream v1.4.5 parity). Read-only review: no code was changed for
> this document. Line numbers refer to that commit.
>
> Scope, as asked: (1) a review of the LoRA training implementation (SFT in
> depth, DPO/KTO lightly), (2) a long look at the runtime-LoRA inference
> pipeline: how it was broken, how it was fixed, why it is slow, (3) "how would
> turboderp speed this up", and (4) a concrete plan.

---

## 0. TL;DR

- **The training side is sound.** The LoRA math (scale, rsLoRA, PiSSA offset,
  the rank-2r export, the module-LoRA scale fix) is correct and matches the
  runtime loader. The fast dequant path is already doing the same
  activation-side Hadamard math the inference kernels do. Nothing here needs a
  rewrite. Two small things are stale (§1.4).
- **Runtime LoRA is correct on every inference path today, but it is slow for
  one structural reason: the adapter lives in Python, outside the CUDA graphs.**
  Every adapted decode step throws away the whole-block graphs (-49% on a 1B,
  measured in Session 27) and then pays ~224 eager cuBLAS launches for
  low-rank math whose actual GPU cost is under 0.1 ms. That is a
  dispatch-bound problem, not a math problem.
- **The turboderp answer is to make the LoRA a property of the quantized linear
  itself, at the C++ level, and inject it inside the graphs.** Specifically:
  give `BC_LinearEXL3` optional LoRA tensors, add one cooperative
  `lora_gemv` kernel (both low-rank stages in a single launch, one
  `grid.sync()`), record its pointers as patchable graph parameters, and let
  every BC graph (attention, gated MLP, plain MLP, GDN, Mamba2, MLA, MoE)
  inherit it from that one seam. Then, as a second step, fold the two stages
  into the existing GEMV kernels' Hadamard prologue and `svh` epilogue so the
  extra graph nodes disappear entirely.
- Expected outcome: adapted decode from ~26% of base speed (83.6 vs ~325
  tok/s on the 1B) to **~85-90% after stage 1 and ~95%+ after stage 2**, with
  the no-adapter path byte-identical and zero-cost, adapter swaps and
  real-time updates requiring no graph recapture.

---

## 1. Training-side review

### 1.1 SFT (`training/qlora_train_native.py`, `exllamav3/training/native_llama.py`, `qlora_linear.py`)

**Per-linear math.** `DiffLinear` (`native_llama.py:221`) wraps a loaded
`Linear` and routes through one of two autograd Functions in
`qlora_linear.py`:

- `EXL3LoRAFunction` (legacy/full-weight path): `y = x @ W + s·(x@A)@B`, W
  reconstructed via `get_weight_tensor()` in forward *and again* in backward
  (never saved), grads `grad_A = s·xᵀ(gBᵀ)`, `grad_B = s·(xA)ᵀg`. Correct.
- `EXL3LoRAHadFunction` (the default "fast dequant" path, audit A1): computes
  `y_base = had(had(x·suh) @ W_inner)·svh`, i.e. exactly what
  `reconstruct_hgemm` and the CUDA GEMM do, on the *inner* trellis weight
  with the transforms applied to activations. Backward uses the adjoint
  sandwich with suh/svh swapped and `Wᵀ`. The PiSSA offset is applied as a
  low-rank activation term (`y -= s·(x@A0)@B0`) rather than a full-weight
  addmm. Correct, and it is the right factorization: `H` is symmetric
  orthonormal and `suh/svh` are ±1 diagonals, so the adjoint is exact.

**Adapter-disable trick** (`adapters_disabled`, `native_llama.py:1946`) drops
both the LoRA term *and* the PiSSA offset, so the reference model is the
exact quantized base. That is what DPO/KTO need. Correct.

**Dropout** is applied to the LoRA branch input only, with the base matmul
seeing the undropped `x`, matching PEFT (`native_llama.py:390-414`).

**Loss.** `compute_loss` (`native_llama.py:1802`) uses the streaming fused
linear-CE for a frozen head (never materialising `[tokens, vocab]`),
falls back to logits-at-supervised-positions when the head is trainable,
handles Gemma softcap in-tile, and vocab-chunks for 200k+ vocabs.
Shift-by-one under packing is handled by the observation that a document
boundary always predicts a masked prompt token. Correct.

**Step loop** (`qlora_train_native.py:2795-2856`): grad accumulation weighted
by each micro-batch's share of supervised tokens (`--ga-loss token`, the
Oct-2024 HF fix), clip on LoRA params, scheduler stepped once per optimizer
step, resume reproduces the data order. Fine.

**Export** (`save_adapter`, `native_llama.py:2248`): PEFT format; a PiSSA run
exports the rank-2r concatenation `[A | A0] / [s·B ; -s·B0]` with the config
triple rewritten so any consumer's `alpha/r` resolves to 1.0, and the exact
fp32 state goes to a sidecar. The runtime loader (`model/lora.py`) reads
per-module rank off the tensor shape (so `rank_pattern` and the 2r export both
work), applies `alpha_pattern`, rsLoRA, and the separately-tracked
`module_lora_scale` for the embed/head LoRAs (the S-cb8177b fix). The two
sides agree.

**Verdict: no correctness issues found in the SFT path.**

### 1.2 DPO / KTO / SimPO (light glance)

`exllamav3/training/preference.py` follows TRL: summed completion logps,
`beta·(π-ref)` rewards, sigmoid/cDPO/hinge/IPO for DPO; KTO with the
mismatched-pair KL estimate (`mismatched_kl_shift`, TRL's +1 rotation),
detached and clamped ≥ 0, skipped on singleton batches; APO-zero-unpaired;
SimPO reference-free with length-normalised rewards and margin γ. The trainer
(`qlora_train_pref.py:332-440`) runs policy (grad) + reference (no-grad,
adapters disabled) through the same net, and for KTO two more no-grad
forwards for the KL rows. All correct by inspection.

**One efficiency lever, not a bug:** reference logps were recomputed every
micro-batch, every epoch. They are constants of the dataset (the reference is
the frozen base, and `pissa/eva/default` inits make reference == step-0
policy exactly). TRL's `precompute_ref_log_probs` caches them once. For a
multi-epoch DPO/KTO run that removes 1/3 (DPO) to 1/2 (KTO with KL rows) of
all forward passes.

**Done on this branch:** `exllamav3/training/ref_cache.py` caches each row's
reference logp by content key (blake2b of prompt ids + completion ids, so
order/shuffle/max-samples/dataset-file independent; KTO's mismatched KL rows
cache the same way), persisted per model under
`~/.cache/exl3_qlora/ref_logps/<fingerprint>.pt` (fingerprint: model dir
config + shard content sample, compute dtype, resolved attention plan; a
mismatch ignores the file). `--ref-cache auto|off|<path>` in
`qlora_train_pref.py`, `ref_cache:` in the YAML. Only missing rows are
forwarded, as one sub-batch. Saved at every eval/checkpoint and at exit.
CPU tests in `tests/test_ref_cache.py`. Not yet box-run.

### 1.3 Realtime (`exllamav3/training/realtime.py`)

`apply_to_native` (`native_llama.py:2131`) pushes fp16 A / s·B (plus the
PiSSA 2r concat from the fp32 masters) into `Linear.lora_a_tensors /
lora_b_tensors` keyed by the net object. Every ingest **replaces** those
tensors with new allocations. That is fine today (the eager path reads the
dict each forward) but matters for the plan: once LoRA pointers are baked
into graphs, replacement must either patch pointers per launch or write in
place. §4 handles both.

### 1.4 Stale bits found on the way (small, worth fixing)

1. `native_llama.py:2143-2157` — `apply_to_native` still printed
   "routed-expert LoRA adapters ... native generation will NOT reflect the
   expert adapters". That was true in Session 24 and false since Session 26:
   `block_sparse_mlp.py:1049-1090` routes to the per-expert torch path under
   `experts_lora`, and Session 28 box-demoed it. **Fixed on this branch:**
   now the same slow-path notice `model/lora.py` prints.
2. `exllamav3_ext/quant/exl3_gemm.cu:179-186` says the int8 GEMV path is
   "Not graph-capturable yet"; `exl3_gemv_int8.cu:205-211,337-343` records
   graph params. Upstream comment rot, harmless, don't chase.

---

## 2. The runtime-LoRA inference pipeline

### 2.1 How it works today

There is exactly one place that applies a runtime adapter:
`Linear.forward` → `apply_lora` (`modules/linear.py:632-687`). The loader
(`model/lora.py`) stores per-Linear `A [K, r]` (transposed, padded to the
Linear's padded `in_features`) and `B [r, N]` with `alpha/r` (or rsLoRA,
`alpha_pattern`, `--lora-scaling`) **pre-folded into B**, all fp16 on the
Linear's device. `apply_lora` is two cuBLAS launches per adapter per Linear:

```
t  = x_flat @ A          # [M, r]
y += t @ B               # addmm_ in place, fp16
```

Multiple loaded adapters are summed (dict loop). `lora_full_weight`
(a fully fine-tuned head from `modules_to_save`) replaces the base matmul
entirely; embed adapters are folded into the embedding weight at load.

### 2.2 How it was broken (Session 24, 2026-07-11)

Every *fused* decode path in exllamav3 reads the trellis storage directly and
never calls `Linear.forward`:

| Path | Where | What it bypassed |
|---|---|---|
| `multi_kv` / `multi_qg` mgemm | `attn.py:552-660`, `sliding_attn.py` | k, v (and q, g on gated-attention) LoRA at `bsz·q_len ≤ 32` |
| `multi_gu` mgemm | `mlp.py:747-777` | gate, up LoRA at `≤ 32` tokens |
| `BC_GatedMLP` graph | `mlp.py:730-734` | gate, up **and** down |
| MoE expert kernels (`exl3_moe`, mgemm loops, BC graphs) | `block_sparse_mlp.py` | every routed-expert LoRA |
| `BC_Attention` graph (post v1.0.0) | `attn.py:900-908` | q, k, v, o, g |
| `BC_MLP`, `BC_Mamba2`, `BC_GatedDeltaNetSplit`, MLA graph | `mlp.py`, `mamba2.py`, `gated_delta_net.py`, `mla_attn.py` | whole layers |

Net effect on a q/k/v/o/gate/up/down adapter on a quantized Llama: decode ran
the **q + o (+ sometimes down)** components only. And because the `> 32`
threshold is on *total tokens in the forward*, a short demo prompt took the
fused path even at prefill. The failure was silent: tensors loaded, counts
printed, generation coherent, base behaviour. Training-side forwards and
long-batch evals applied the full adapter, which is why loss curves looked
healthy while generations did not move. A bf16 base builds no `MultiLinear`
at all and always took the correct path, which is exactly the Session-3
"LoRA is weak on quants" confound.

### 2.3 How it was fixed (Sessions 24, 26, 27)

- `has_runtime_lora(*linears)` (`linear.py:32`): true when any of the given
  Linears carries adapter tensors. Zero cost with no adapter (empty-dict
  truthiness).
- **Whole-block graphs** (bc_attn/bc_swa, `BC_GatedMLP`, `BC_MLP`, Mamba2,
  GDN split, MLA, MoE bszN + fused expert kernels): guarded at the per-call
  dispatch, fall back to the unfused Python path. Guarded per call, not at
  build, because graphs are cached and adapters attach/detach after build.
- **mgemm pairs** (`multi_kv`, `multi_qg`, `multi_gu`): stay fused; the LoRA
  delta is added onto the mgemm output views *pre-RoPE / pre-activation*
  (`attn.py:608-616, 651-656`, `mlp.py:772-775`). Safe because `MultiLinear`
  asserts no bias / softcap / post-scale, so the LoRA add is the only
  epilogue `Linear.forward` would have applied.
- MoE routed experts: `experts_lora` forces the per-expert torch branch;
  `sh_fused_lora` keeps the bszN graph (which embeds shared experts + gate)
  honest. Router LoRA remains unsupported (reads `routing_gate.inner.weight`
  raw everywhere). CPU-offloaded experts reject a routed LoRA loudly.
- `apply_lora` fused from ~5 kernels to 2 (`mm` + in-place `addmm_`).
- Tripwire tests (`tests/test_lora_fused_path.py`) assert every guard and
  delta-add placement in the *source* so a refactor that drops one fails
  loudly, plus a GPU parity test of the fused delta against the per-linear
  path.

### 2.4 Why it is slow (the Session 27 measurement, re-read)

Llama-3.2-1B 4bpw, r32 α64 on all 7 targets, greedy, single 3090:

| configuration | tok/s |
|---|---|
| no adapter (graph decode) | ~325 |
| adapted, branches as-adapted, LoRA math no-op'd | 167 |
| adapted, mgemm + delta + fused `apply_lora` | 83.6 |
| LoRA chain alone, eager, 112×(mm+addmm_) | 2.76 ms/tok |
| identical kernels, CUDA-graph replayed | 0.85 ms/tok |

Two halves, both dispatch-bound:

1. **Leaving the graphs costs 49%.** Once any projection in a block carries
   an adapter, the whole attention block and the whole MLP block run as
   individual eager launches (norms, projections, RoPE, cache update, split
   / combine attention, act, ~10-15 launches per block instead of two graph
   launches). 167 tok/s is the ceiling of any correct LoRA that stays outside
   the graphs, regardless of how the LoRA math itself is done.
2. **The eager LoRA chain costs the other half.** 224 launches at ~12 µs
   each. The GPU work is tiny: for this model the adapter streams
   ~24 MB of A plus ~24 MB of B per token, which is ~0.05 ms at 3090
   bandwidth. Even graph-replayed, two cuBLAS kernels per Linear are
   0.85 ms/token because 224 dependent nodes still serialize.

The per-linear-vs-mgemm question that motivated backlog #10 is noise
(80.4 vs 83.6). The Session-24 premise ("modest hit") died in the upstream
graph refactor that raised the base to 325.

Also relevant: at decode with the default `mul1` codebook, the kernel that
actually runs at bsz 1 is the **fused int8 GEMV**
(`exl3_gemv_int8_kernel.cuh`), a single cooperative launch with its own
Hadamard-prologue / GEMV / `svh`-epilogue phases separated by `grid.sync()`.
`exl3_gemm_kernel.cuh:10-31` has the same prologue structure. This matters
for §3.

### 2.5 What is outside the graphs even without an adapter

For completeness of "the rest of the pipeline": `TransformerBlock.forward`
(`modules/transformer.py:137-195`) is Python per block per token: input norm
(eager kernel), attention (one graph launch), MLP norm with fused residual
(eager), MLP (one graph launch), residual add. Roughly 2-3 eager launches
plus ~6 Python calls per layer, plus the head GEMV and sampling. That is
upstream territory (turboderp is visibly moving toward it: the RMSNorm graph
wrapper landed in 3cbaee3). It is *not* the LoRA problem and this plan does
not touch it, but it bounds the win: on the 1B, ~3.1 ms/token of which the
two graphs per layer are most.

---

## 3. "How would turboderp speed this up?"

Reading the codebase gives a fairly unambiguous answer, because the pattern
is repeated everywhere: fuse into a single cooperative launch, keep every
intermediate in a static buffer, capture once, patch pointers per replay,
push the mechanism to the lowest layer so every caller inherits it. Applied
to LoRA:

1. **LoRA is a property of the quantized linear, not of the Python module.**
   Today the adapter hangs off `Linear` in Python and every fused path has to
   *know* about it. Turbo's seam is `BC_LinearEXL3` (`libtorch/linear.h:31`)
   and the `exl3_gemm_gr` / `exl3_mgemm_gr` launchers: everything that ends
   up in a graph goes through them. Put the adapter there (optional A/B
   tensors on `BC_LinearEXL3`, optional pointer tables for the mgemm path)
   and `BC_Attention`, `BC_GatedMLP`, `BC_MLP`, `BC_Mamba2`,
   `BC_GatedDeltaNetSplit`, the MLA graph and the MoE graphs get it with
   *no per-module logic*.

2. **Never leave the graph.** The 49% is not recoverable any other way. The
   LoRA nodes must be captured inside the same `Graph` as the projections
   they modify, with their input/output/A/B pointers and rank recorded via
   `graph->record_param` like every other node, so adapter swaps are a
   `memcmp`-and-patch in `Graph::launch`, not a recapture.

3. **One kernel, one launch, both low-rank stages.** Not two cuBLAS calls.
   A cooperative `lora_gemv` kernel: phase 1 partitions K across the
   persistent blocks and writes partial `t = x@A` slices to workspace,
   `grid.sync()`, phase 2 every block reduces the (tiny) partials and owns
   a contiguous slice of N for `C[m, n] += Σ_r t[m,r]·B[r,n]`. That is the
   exact structure of `exl3_gemm_kernel`'s Hadamard prologue + GEMM. B is
   read exactly once across the grid, coalesced.

4. **Then fold the two stages into the kernels that already exist.** The
   GEMV kernels already have a prologue phase over the input followed by a
   `grid.sync()`, and an epilogue that applies `svh` while writing C. Stage 1
   of the LoRA (`x@A`) belongs in the prologue; stage 2 (`t@B[:, n]`) belongs
   in the epilogue of the block that owns those n columns. That makes the
   adapter cost only its extra bandwidth (`(K+N)·r·2` bytes per Linear, ~5%
   of the trellis read at 4 bpw, r=64) and *zero* extra nodes.

5. **Pre-transform A so the kernel reads what it already has.** After the
   prologue, the kernel holds `x_had = had(x ⊙ suh)`, not `x`. Since `H` is
   symmetric orthonormal and `suh = ±1`, `x @ A = x_had @ (H·diag(suh)·A)`.
   Precompute `A' = H·diag(suh)·A` once at attach (a `[K, r]` transform,
   negligible), and the epilogue path needs no second copy of the input.
   This is the same trick the training fast path uses in reverse.

6. **Template on presence.** `template <bool LORA>` on the kernel so the
   no-adapter instantiation is byte-identical to today's and the
   "zero cost when nothing is loaded" promise stays provable by the
   existing byte-identical-generation check.

7. **Pointer tables for the fused pairs.** `MultiLinear` already carries
   `ptrs_trellis / ptrs_suh / ptrs_svh` as int64 device tables; add
   `ptrs_lora_a / ptrs_lora_b` (null entries allowed) with the same slot
   addressing the mgemm kernel uses for C. MoE experts get the same via the
   existing `indices` routing.

8. **Stable storage, patched pointers.** Rank and pointers are kernel
   arguments, so they are patchable per launch; grid geometry must not
   depend on r (loop over r inside the block). Then a realtime ingest that
   replaces the tensors, a second adapter, or an unload never recaptures.
   Two graph *flavours* per slot (plain / LoRA-enabled) because node
   presence is structural; the Python side picks the flavour from
   `has_runtime_lora`.

9. **One effective adapter per Linear.** Multiple loaded adapters are
   concatenated along r (`[A1|A2]`, `[B1;B2]`) into one packed pair at
   attach time. The kernel sees one rank.

10. **Measure with the tool that exists.** `eval/perf.py` grows a
    `--lora <dir>` flag so every number in this plan is reproducible with
    the same harness turbo uses.

What turbo would *not* do: torch-side `torch.cuda.graph` capture of the whole
adapted decode step (the ext's graphs patch kernel params per launch and
cannot be nested in an outer capture without baking those params), or
keeping the LoRA as a Python-level epilogue and trying to shave launches. Both
have a hard ceiling far below the base.

---

## 4. The plan

### Reality check: what can be done without an NVIDIA card

Nobody on the project has a GPU available at the moment, and this container
has neither a GPU nor `nvcc`. That splits the plan into two tracks:

**GPU-free track (can land now, CPU-tested):**
- The Python side of stage 1: `Linear.lora_packed()` with the version
  counter, multi-adapter concatenation, `MultiLinear.ptrs_lora_*` tables,
  the in-place `set_runtime_lora` from stage 4, and the tripwire-test
  rewrite. All of it is exercisable with the existing CPU test pattern
  (`tests/test_lora_fused_path.py` already imports `linear.py` without the
  extension for its semantics tests).
- `eval/perf.py --lora` and `--lora-noop` (argument plumbing only; the
  measurement itself needs the box).
- A pure-torch reference of `lora_gemv` semantics (`x@A@B` accumulate with
  the fp16 rounding point pinned) that the later GPU parity tests compare
  against, so the kernel's contract is written down before the kernel is.
- **Stage 1 for the Llama path is WRITTEN on this branch, uncompiled and
  untested** (the user has a card at home): `exllamav3_ext/lora.cu/.cuh`
  (the cooperative `lora_gemv` kernel, `lora_gemv_gr` with `GP_lora_*`
  sites, `ext.lora_gemv` binding), `Graph::reset()` + `Graph::flavour`,
  `BC_LinearEXL3::set_lora/clear_lora/has_lora`, LoRA nodes and flavour
  handling in `BC_GatedMLP` and `BC_Attention`, `Linear.sync_lora_bc` and
  `lora_graph_ok` in Python with the `EXL3_LORA_GRAPH=0` fallback switch,
  tripwires updated, `tests/test_lora_graph.py` (kernel unit test + end-to-
  end parity), and the runbook `doc/lora_box_test.md`. Other graph paths
  (SWA, BC_MLP, Mamba2, GDN, MLA, MoE) keep their guards.
- Already done this session (all CPU-tested, `tests/test_lora_state.py`,
  `tests/test_ref_cache.py`):
  - `exllamav3/modules/lora_state.py`: `pack_lora` (one effective adapter
    per Linear; a single adapter aliases its slot tensors, several
    concatenate along rank and equal the sum of products), `lora_pack_key`
    (stable under in-place updates, changes on replacement) and
    `lora_delta_reference`, the kernel's numerical contract: fp32 through
    both stages, one rounding into the output dtype on the add.
  - `Linear.lora_packed()` (`modules/linear.py`), cached by pack key.
  - `backbone.set_runtime_lora` copies in place when shape/dtype/device
    match, so a realtime ingest keeps the slot tensors' identity and data
    pointers (stage 4's requirement, landed early because it costs nothing).
  - `eval/perf.py --lora DIR [--lora_scaling S] [--lora_noop]` (stage 0's
    harness; the numbers still need the box).
  - The reference-logp cache and the stale MoE warning (§1).

**Box track (needs any CUDA GPU, a 3090-class card is enough):**
- Stage 0's measurements. Note they are not a gate for *deciding* anything:
  Session 27 already measured the two halves and every projection in §5 is
  derived from those numbers. Stage 0 is the regression baseline and the
  per-model table, and it takes an hour once a card exists.
- Compiling and validating stages 1-3, which is kernel work and cannot be
  trusted without running it.

Absent a physical card, a rented one (any provider with a single 24 GB
Ampere or newer, by the hour) covers stage 0 and stage 1 validation in one
sitting: the extension build is ~10 minutes, the parity and perf runs are
minutes each on a 1B/3B. That is the cheapest way to unblock the box track;
the GPU-free track is worth doing first regardless because it is what the
kernel plugs into.

### Stage 0 — measurement and safety net (½ session, box)

- `eval/perf.py --lora <adapter_dir> [--lora-scaling]`: load via
  `LoRA.from_directory` before warmup, report tok/s for prefill and
  generate at bsz 1 and 4, and a `--lora-noop` diagnostic that keeps the
  adapted branch selection but zeroes B (reproduces the 167 tok/s ceiling
  number so regressions in either half are visible).
- Baseline table on the box: Llama-3.2-1B and 3B 4bpw, one 8B, one MoE
  (Qwen3.6-35B-A3B with expert targets), base / adapted / unload.
- `nsys` one adapted token on the 1B to confirm the node/launch counts this
  document assumes (~224 LoRA launches, ~10-15 eager launches per block
  once the graphs are lost).
- Keep the Session-24 promise tests: base generation byte-identical with
  the extension rebuilt; `unload()` restores it.

### Stage 1 — LoRA inside the graphs via one kernel at the linear seam (1-2 sessions)

**Kernel** (`exllamav3_ext/lora.cu/.cuh`, new):

```
lora_gemv_kernel<c_fp32>(x, A, B, C, size_m, size_k, size_n, rank, ws)
```
- Cooperative launch, `num_sms` persistent 256-thread blocks. Phase 1: block
  b owns K-slice b, computes `t_b[m, r]` for all m ≤ MAX_R(128) tokens and r,
  writes to `ws[b]`. `grid.sync()`. Phase 2: block b owns N-slice b, reduces
  `t = Σ_b t_b` (≤ 128×256 floats, trivial), then `C[m, n] += Σ_r t[m,r]·B[r,n]`
  with `uint4` loads of B, accumulate fp32, one fp16 rounding on the add
  (same numerics as today's `addmm_`).
- Grid independent of `rank`; `rank ≤ 256` (covers 2r PiSSA exports at
  r=128 and concatenated multi-adapters); M up to `MAX_R = 128` so it serves
  every BC slot (bsz ≤ 8, q_len ≤ 16). Workspace from `DevCtx` like the
  int8 GEMV's use of `locks`.
- `lora_gemv_gr(x, A, B, C, graph)` records `GP_lora_x(0) GP_lora_A(1)
  GP_lora_B(2) GP_lora_C(3) GP_lora_rank(7, 4 bytes) GP_end`. New enum
  entries in `graph.cuh`.
- `lora_mgemv_kernel`: same, with `A_ptrs / B_ptrs / rank_ptrs` int64 tables
  indexed by slot `j` (C at `j*size_m*size_n`, the mgemm convention) and
  optional `indices` for the MoE routed case (slot → expert id, null pointer
  → skip). Records `GP_lora_m_*`.
- Standalone binding `ext.lora_gemv(x, a, b, c)` for tests and for the eager
  path.

**Seam** (`libtorch/linear.h/.cpp`):
- `BC_LinearEXL3` gains `c10::optional<at::Tensor> lora_a, lora_b` and
  `set_lora(a, b) / clear_lora()`. `run_gr` appends `lora_gemv_gr(x, a, b,
  y, graph)` after the GEMM (after the bias add: bias is part of the base,
  LoRA adds on top, matching `Linear.forward` order) whenever both are set.
  `run_alloc` (the eager `Linear.forward` path at decode) gets the same,
  which replaces today's two cuBLAS launches with one for M ≤ 128; keep
  cuBLAS above that (prefill is GEMM-bound and cuBLAS wins there).
- `exl3_mgemm_gr` callers (`BC_Attention` kv/qg, `BC_GatedMLP` gu) take two
  extra optional pointer-table tensors and append `lora_mgemv_gr` after the
  mgemm, before RoPE / activation. That is the same pre-RoPE, pre-activation
  placement the Python delta-add uses today, so semantics are unchanged.

**Graph flavours**: every BC class that owns `Graph` objects keys them by
`(slot, lora_flavour)` — `Graph graph_bszN[MAX_BSZN][2]` in `BC_GatedMLP`,
a second `unique_ptr<Graph>` in `BC_Attention::Slot`, likewise `BC_MLP`,
`BC_Mamba2`, `BC_GatedDeltaNetSplit`, MLA, `BC_BlockSparseMLP`. The flavour is
decided per call from whether any involved linear has LoRA set; in the LoRA
flavour the LoRA pointers and ranks are pushed into `args` for
`Graph::launch` so they patch like `x` and `y`. Nothing is recaptured when
the adapter changes, only when the flavour flips (attach of the first
adapter, unload of the last), and both flavours can stay cached.

**Python plumbing**:
- `Linear` keeps the `lora_a_tensors / lora_b_tensors` dict API (the loader,
  `set_runtime_lora`, tests all use it) and adds a cached packed view:
  `lora_packed() -> (A', B, rank) | None`, rebuilt on a version bump
  (attach/detach), concatenating multiple adapters along r, padding A to the
  padded `in_features` (already done by the loader), and computing
  `A' = H·diag(suh)·A` **only in stage 2** (stage 1's separate kernel reads
  `x` directly; keep `A` raw here). `backbone.set_runtime_lora /
  clear_runtime_lora` bump the version.
- On version bump, `Linear` pushes the packed pair to `inner.bc.set_lora`
  and `MultiLinear` rebuilds its `ptrs_lora_*` tables. This is where
  `attn.py / mlp.py` today compute `has_runtime_lora` and add deltas; those
  Python delta-adds and the graph guards come *out* for every path the C++
  seam covers, and the tripwire tests flip from "guard present" to "guard
  absent, C++ carries it" (with the GPU parity test as the real gate).
- Paths not covered by stage 1 keep their guards: MoE routed experts
  (`exl3_moe` fused kernel and the expert mgemm loops need the routed
  pointer-table variant, see stage 3), router LoRA (unsupported), CPU
  expert offload (rejects), `lora_full_weight` (eager head, unaffected).

**Tests**:
- `tests/test_lora_kernel.py` (GPU): `lora_gemv` vs `x@a@b` for
  M ∈ {1,2,4,8,16,128}, r ∈ {8,16,32,64,128,256}, (K,N) ∈ typical projection
  shapes incl. padded dims, fp16 and fp32 C; mgemv with null slots and with
  indices.
- Extend `test_mgemm_lora_delta_parity_gpu`: graph-path adapted logits vs
  the (kept, env-gated) Python per-linear path over 64 tokens of decode on
  a real quant + adapter, plus the no-adapter byte-identity and the
  attach/unload/attach cycle with a capture counter asserting zero
  recaptures after the first flavour build.
- Realtime: `apply_to_native` twice with different weights, assert the
  second is live on the next token without recapture.

**Exit criteria**: 1B adapted ≥ 85% of base, 8B ≥ 92%; no-adapter numbers
unchanged within noise; all tripwires and parity tests green.

### Stage 2 — fold into the GEMV kernels (1 session, C++ heavy)

- `template <..., bool LORA>` on `exl3_gemv_kernel`, `exl3_gemv_int8`
  (cooperative + `_sq` variants) and `exl3_gemm_kernel`. Three extra args
  (`lora_a`, `lora_b`, `rank`) at indices 10-12 so the existing recorded
  param offsets 0-9 do not move; new `GP_gemm_lora_*` records.
- Prologue: after the Hadamard pass each block also computes its K-slice
  partial of `t = x_had @ A'` (A' pre-transformed as in §3.5) into
  workspace; the existing `grid.sync()` covers it. Epilogue: the block
  writing columns `[n0, n1)` adds `t @ B[:, n0:n1]` at the final store,
  *after* the output Hadamard and the `svh` flip have been applied to the
  base result. Those two are the base weight's output transform and must not
  touch the LoRA delta, which is defined on the untransformed output. Bias,
  where present, is a separate `add_gr` node and stays after.
- `exl3_gemm_gr` selects the LORA instantiation when the LoRA pointers are
  non-null; the `lora_gemv` node from stage 1 is then omitted for that
  Linear. Stage 1's kernel remains for the mgemm/MoE tables until their
  kernels get the same treatment, and for the eager path.
- Numerics: identical up to fp32 accumulation order; the parity test
  tolerance stays at the stage-1 setting.

**Exit criteria**: adapted decode ≥ 95% of base on the 1B (node count per
token equal to the no-adapter graph), ≥ 97% on 8B.

### Stage 3 — MoE routed experts and the remaining whole-layer graphs (1 session)

- `exl3_moe` fused kernel and the per-token mgemm loops in
  `block_sparse_mlp.py:1283-1420`: per-expert `ptrs_lora_a/b` tables
  (E entries, null for experts without an adapter, so a `rank_pattern`
  adapter that only touches some experts costs nothing on the others),
  indexed by the same `indices` the kernel already uses. Shared experts are a
  `GatedMLP` and are covered by stage 1.
- Drop `experts_lora` from the branch selection once parity holds; keep the
  CPU-offload rejection.
- GDN split (`BC_GatedDeltaNetSplit`): the merged `ba_weight_t` buffer is
  base-only by construction; `b_proj`/`a_proj` LoRA needs the delta on the
  un-merged projections, which the graph does not compute separately. Two
  choices: rebuild `ba_weight_t` on attach with the LoRA merged in (cheap,
  exact for a *static* adapter, wrong for realtime updates unless rebuilt
  per ingest, which is still cheap), or add separate low-rank nodes. Merge
  on attach, rebuild on version bump.
- Router LoRA: leave unsupported, document (it would need the routing
  kernels to take a delta; no one trains it today).

### Stage 4 — realtime and multi-adapter polish (½ session)

- `set_runtime_lora` writes in place when the shape matches the resident
  packed pair (`copy_` into the fp16 slot tensors) so an ingest is two
  memcpys per Linear and the graph pointers never change at all; falls
  back to reallocation + version bump otherwise.
- KV invalidation on update is unchanged (the adapter-free-KV target set
  remains the alternative).
- `LoRA.from_directory` on an already-adapted model: concatenation along r
  happens in `lora_packed()`; document the rank cap (256) and that scaling
  is per-adapter via pre-scaled B.
- Fix the stale `apply_to_native` MoE warning (§1.4).

### Optional, training side

- Cache reference logps across epochs in `qlora_train_pref.py` (§1.2).

### Ordering, effort, risk

| Stage | Files | Effort | Risk |
|---|---|---|---|
| 0 | `eval/perf.py`, tests | ½ session | none |
| 1 | new `lora.cu/.cuh`, `graph.cuh`, `libtorch/linear.*`, `mlp.*`, `attention.*`, `mamba2/gdn/mla` graph classes, `bindings.cpp`, `modules/linear.py`, `multilinear.py`, `attn.py`, `sliding_attn.py`, `mlp.py`, tests | 1-2 sessions | medium: touches every BC class; the flavour split is mechanical but wide |
| 2 | `exl3_gemv_kernel.cuh`, `exl3_gemv_int8*.cu/.cuh`, `exl3_gemm_kernel.cuh`, `exl3_gemm.cu` | 1 session | medium-high: hot kernels, register pressure; gate on the LORA template so the base path cannot regress |
| 3 | `exl3_moe*`, `blocksparse_mlp.*`, `gated_delta_net.*`, `block_sparse_mlp.py` | 1 session | medium |
| 4 | `backbone.py`, `native_llama.py`, `realtime.py`, `lora.py` | ½ session | low |

Stage 1 is the one that changes the picture (it removes both halves of the
Session-27 loss). Stage 2 is the turboderp finish. Stages 3-4 are
completeness.

**Upstream coordination.** Stages 1-2 touch upstream C++ that the fork has so
far kept byte-identical (Session 26: "we touch zero C++"). That was a
deliberate merge-cost choice. The design above keeps the footprint mergeable:
one new compilation unit, additive optional fields on `BC_LinearEXL3`, a
second graph per slot, template flags that leave existing instantiations
unchanged, and new graph param IDs appended to the enum. It is also the shape
of change upstream could take as a PR ("runtime LoRA in the graph paths"),
which is the cheapest long-term merge story of all.

---

## 5. Numbers to expect (1B, 3090, from the Session-27 decomposition)

| | ms/token | tok/s | vs base |
|---|---|---|---|
| base, graphs | 3.08 | ~325 | 100% |
| today, adapted | 12.0 | 83.6 | 26% |
| stage 1: graphs kept, +5 `lora_gemv` nodes/layer (~4 µs each) | ~3.4 | ~290 | ~89% |
| stage 2: folded, bandwidth only (~48 MB/token of A+B) | ~3.15 | ~315 | ~97% |

The 3B and 8B are proportionally better because the base per-token time
grows with model size while the LoRA node overhead does not. The MoE case
(stage 3) recovers from "per-expert torch path" (much worse than 26%) to
the same regime.
