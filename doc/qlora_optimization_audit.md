# QLoRA-on-EXL3 — Optimization Audit (Session 11)

> Research pass over the training pipeline (2026-07-01): where we waste compute
> and VRAM, what modern frameworks (Axolotl / Unsloth / Liger / torchtune /
> Chronicals) do that we don't, and the implementation plan derived from it.
> Written BEFORE implementation.

## 1. Problems found in our own pipeline

Ordered by expected impact. File references are to the state at commit
`528fb3c` (pre-Session-11).

### Compute waste

**A1. Trellis dequantization runs 3× per linear per step — the biggest
throughput lever.** **DONE (Session 30, see the handoff notes):** landed as
`EXL3LoRAHadFunction` (reconstruct the inner weight only; Hadamard/sign
transforms on the activations — inference's `reconstruct_hgemm` math; the
first fix sketched below was found not to exist at training shapes, since the
no-reconstruct fused kernel is the ≤144-row decode path and inference itself
reconstructs at prefill sizes). Box: +28% tok/s (1B), +6% (12B 6bpw), +32%
(MoE 26B). The second fix below (recompute→backward cache, 3→2) was also
built but shipped opt-in (`--dequant-cache`): once reconstructions are
inner-only-cheap, the cache's bookkeeping + held weights measured as a net
loss (−1–4% tok/s, +0.5–1.5 GB); it can only pay in `--dequant-mode legacy`.
Original analysis kept below.

`EXL3LoRAFunction` calls `weight_fn()` (a full
`get_weight_tensor()` reconstruction) in forward AND backward
(`exllamav3/training/qlora_linear.py`), and non-reentrant gradient
checkpointing runs the forward twice (outer pass + backward recompute). So
every quantized linear is dequantized three times per optimizer step. This is
the structural reason the native trainer sits at ~136 tok/s on a 12B.
Unlike NF4 (a table lookup), an EXL3 reconstruction is real work, so generic
framework advice underweights this — it is *our* dominant cost. Fixes,
composable:
  - Use exllamav3's fused trellis matmul kernel for the frozen-base `x @ W`
    in forward passes (the base is a constant in the graph; autograd doesn't
    care how the constant matmul was computed), reconstructing the dense
    weight only in backward for `grad_y @ W^T`. 3 → 1 reconstructions and a
    faster base matmul. Risk: the inference kernels are
    `@torch.inference_mode`-scoped and shape-constrained; needs a careful
    seam + the validate gate.
  - Cache the reconstructed weight between the checkpoint-recompute forward
    and the immediately-following backward (one block's weights live at a
    time). 3 → 2, much simpler.

**A2. Gradient checkpointing is unconditional.** Every block checkpoints even
when the run has huge VRAM headroom (the 1B semancy run peaked at 5.26 GB on
24 GB cards). For us a recompute is unusually expensive (it repeats a full
round of dequant, see A1). Fix: checkpoint every N-th block / skip under a
VRAM budget (torchtune/Axolotl "selective activation checkpointing").
Expected ~1.2–1.4× step time when headroom exists; zero numerics change.

**A3. RoPE cos/sin rebuilt from scratch for every q and k, in every block,
in every pass.** `_apply_rope` (`native_llama.py`) recomputes
`freqs`/`cos`/`sin` per call: ~192 redundant `[b,t,hd]` fp32 cos/sin builds
per step on a 48-layer model. All Llama-family layers share one `inv_freq`.
Fix: build cos/sin once per forward per (inv_freq, device) and pass them into
the checkpointed block fn (checkpoint saves inputs rather than recomputing
them). Liger's fused RoPE is a further step.

**A4. `FusedLinearCrossEntropy` upcasts the entire `[hidden, vocab]` head
weight to fp32 — inside the token-chunk loop.** `fused_ce.py` forward does
`h_c @ weight.to(compute_dtype)` per chunk (re-cast every iteration);
backward holds a full fp32 copy for the whole loop. On Gemma's 262k vocab
that copy is ~4 GB; Llama-128k ~1 GB. Exactly the waste class PR #118
removed from the materialized head path. Fix: keep the weight in its native
(fp16) dtype, matmul per chunk in that dtype, upcast only the
`[chunk, vocab]` logits tile to fp32 for the softmax. The fp64 gradcheck
path must be preserved (rule: only skip the upcast when the weight is
half/bf16).

**A5. Small syncs / loop overheads.**
  - Big-head SDPA per-document loop calls `cu.tolist()` per global layer per
    pass (host-device sync); the spans are constant across layers — hoist
    into `pack_ctx`.
  - `loss.item()` per micro-batch and `valid.sum().item()` per loss sync.
  - GQA `repeat_interleave` in the sdpa branch is pure overhead (Session 9).
  - `EXL3LoRAFunction.backward` reconstructs before checking
    `needs_input_grad[0]` (micro).

**A6. Eval runs one example at a time** (`eval_loss`, batch 1). Fine at ~100
examples; slow at 285 wikitext blocks on a 31B. Batched eval (per-sequence
loss reduction) would cut eval wall-clock several-fold — but the batch-1
definition is matched with the BNB arm, so change both or flag-gate.

### Data pipeline

**A7. Packing is next-fit in arrival order — ~82.5% observed fill vs ~98–99%
achievable.** `pack_examples` seals a block as soon as the next doc doesn't
fit. Axolotl multipack uses first-fit-decreasing (FFD) bin packing (typically
within ~2–10% of optimal); Chronicals uses best-fit-decreasing. Raising fill
82.5% → ~98% is ~1.19× tokens/step for free. Document order inside a block is
irrelevant (attention is already document-isolated; positions reset per doc).

**A8. Unpacked runs have no length-grouped batching** — mixed-length batches
pad to the max. Mostly superseded by packing.

### Correctness / quality

**A9. The gradient-accumulation loss-weighting bug (mild form).**
`(loss / grad_accum).backward()` averages per-micro-batch means, so a
micro-batch with 20 supervised tokens weighs the same as one with 2000 — its
tokens are effectively up-weighted 100×. Same mean-of-means across DDP ranks.
The Oct-2024 HF/Unsloth fix normalizes by total supervised tokens across the
accumulation window. Ours is the milder form (no G× loss blow-up, but real
per-token weighting bias with variable-length data). Fix all three arms so
EXL3-vs-BNB comparisons stay matched; note HF's Trainer is already fixed, so
our hand-loop BNB arm has the same bug today.

**A10. LoRA+ (differential LR, B ≈ 16× A)** is a two-line optimizer change
with published derivations; cheap experiment, adjudicate via the run log.

### VRAM

**A11.** The fused-CE fp32 head copy (A4) is also the dominant memory item
on the output card for non-chunked runs.

**A12. Chunked trainable-head CE (Session-10 plan #1): adopt Liger FLCE for
the dense/trainable head.** Liger's FusedLinearCrossEntropy already does
chunked CE **with weight gradients and Gemma2 tanh softcap**
(linkedin/Liger-Kernel#127) — no need to hand-write that autograd Function.
The frozen trellis head still needs our vocab-chunked CE (Liger can't slice
a trellis); adding softcap+Jacobian to our existing chunked kernel is the
small remaining delta.

**A13. Activation offload is synchronous** (`save_on_cpu`, ~3% overhead);
Unsloth's CUDA-stream double-buffered variant is ~1% and unlocks multi-×
context. Worth it after A1/A2 (it trades wall-clock).

**A14. Big-head (head_dim 512) attention: nothing off-the-shelf.** Checked
July 2026: xformers' cutlass backward doesn't cover hd 512 on sm86;
FlexAttention still SMEM-infeasible there (tried, Session 8). The
query-tiled online-softmax `autograd.Function` (Session 8 plan) remains the
right answer.

**A15. Smaller:** fused/foreach AdamW for the many small LoRA tensors;
`--optim adamw8bit` not mirrored to DDP/BNB arms; embed/head full-training
keeps GB-scale fp32 grads live between backward and step
(optimizer-in-backward hooks would free them immediately).

## 2. Modern-framework techniques vs this repo

| Technique | Who | Status here |
|---|---|---|
| FFD/BFD multipack packing | Axolotl, Chronicals | next-fit (~82.5% fill) |
| Fused linear CE w/ weight grad + softcap | Liger FLCE | ours is frozen-head only, no softcap |
| Cut Cross-Entropy (no logits materialized, grad sparsity) | Apple CCE (arXiv 2411.09009); in Unsloth + Axolotl | we have the torch-level equivalent; CCE kernel can't read a trellis — relevant only for the dense/trainable head |
| Async double-buffered activation offload | Unsloth, torchtune | sync version only |
| GA loss-normalization fix | HF, Unsloth (Oct 2024) | we have the bug (mild form) |
| LoRA+ differential LR | PEFT, Axolotl, Chronicals | no |
| Selective activation checkpointing | torchtune, Axolotl | all-or-nothing |
| torch.compile (regional) | torchtune, Unsloth, HF | untried; custom autograd Functions limit it to norm/RoPE/glue — Liger already covers much of that |
| Sequence/context parallelism (ring-flash-attn) | Axolotl (2025) | our long-context multi-GPU is serial layer-split; ring attn is FA2-based → head_dim ≤ 256 only; big engineering |
| Fused RMSNorm/SwiGLU/RoPE Triton kernels | Liger, Unsloth | RMSNorm+SwiGLU wired (`--use-liger`); RoPE + GeGLU not |

Chronicals (arXiv 2601.02609, Jan 2026; claims 3.51×/4.10× over Unsloth on a
0.5B/A100) is fused kernels + CCE + LoRA+ + BFD packing — independent
validation of the same shortlist. Treat the multiplier with salt.

Structural difference to keep in mind: those frameworks' frozen weights are
cheap to read (NF4 = lookup); ours cost a real reconstruction. A1 dominates
anything a generic comparison suggests.

## 3. Implementation plan (Session 11)

Instrumentation first — the big claims (A1/A2) rest on "dequant dominates",
which has never been directly measured:

0. **Per-step wall-clock breakdown + run-log v2 + failure logging.**
   CUDA-event section timers (data / forward+loss / backward / optimizer) on
   the per-step line and summarized into the run-log CSV; `--profile-dequant`
   to measure reconstruction time directly; every run — including ones that
   CRASH — appends a row with status + error summary (full traceback to a
   sidecar log), so the CSV becomes an automatic lab notebook.
1. **FFD packing** (A7) — small, ~1.15–1.2× tokens/step, fill-% printed.
2. **Fused-CE dtype fix** (A4/A11) — small; kills a 1–4 GB spike; gradcheck
   gates it.
3. **GA normalization fix** (A9) — small; all three arms together.
4. Checkpointing control (A2) — medium.
5. Kernel-based frozen matmul / dequant caching (A1) — medium-hard; gate
   with `qlora_validate_native.py`.
6. RoPE cache (A3) + sync fixes (A5) — small, alongside 5.
7. Liger FLCE for trainable head + softcap in our chunked CE (A12).
8. LoRA+ flag (A10) — tiny.
9. Async offload, fused AdamW, query-tiled big-head attention, sequence
   parallelism — later, in that order.

Items 0–3 are this session's batch (box-free verifiable); 4+ need GPU-box
validation runs.

## Sources

- Axolotl multipack: https://docs.axolotl.ai/docs/multipack.html
- Axolotl repo/changelog: https://github.com/axolotl-ai-cloud/axolotl
- Unsloth gradient checkpointing: https://unsloth.ai/blog/long-context
- Unsloth GA bug: https://unsloth.ai/blog/gradient
- HF GA fix: https://huggingface.co/blog/gradient_accumulation
- Cut Cross-Entropy: https://arxiv.org/abs/2411.09009 /
  https://github.com/apple/ml-cross-entropy
- Liger FLCE Gemma2 softcap: https://github.com/linkedin/Liger-Kernel/issues/127
- Chronicals: https://arxiv.org/abs/2601.02609 /
  https://github.com/Ajwebdevs/Chronicals
- FFD bin packing: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
- xformers sm86 backward head-dim limits:
  https://github.com/facebookresearch/xformers/issues/517
