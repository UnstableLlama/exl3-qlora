# Upstream v1.5.0 sync

Merged `turboderp-org/exllamav3` master at
`02aef45` (v1.5.0 + 1 commit: drop torch/extension.h from CUDA units), 35 commits since
the previously merged `be57335` (v1.4.9 + 1) ancestor, into fork master at `aa00729`.

What upstream brought in:

- MoE rework: the bszN and bsz1 mgemm paths are replaced by a two-stage cooperative
  fused-MoE kernel (`exl3_moe_coop_*`), which is now the last decode tier for bsz 1..MAX_BSZN.
  The prefill path gains M=32 / M=64 row-tile GEMM instances (per-expert M dispatch), a
  batched reconstruct tier (`moe_batch_recon.py`, `reconstruct_batch` / `reconstruct_had_batch`)
  and a deterministic slot-scratch + `exl3_moe_gather` accumulation, on by default with folded
  Hadamard. `experts_per_tok` limit bumped to 32; worst-case measurement during autosplit load.
- CPU MoE: pinned arena for zero-copy expert streaming, batched expert reconstruct path,
  system-memory check before large allocations (also for NGramEmbedding).
- Quantizer: new optimized implementations (shared cost buffer, register-cached codebook,
  byte traceback, codebook LUT, warp-wide reductions), dispatched per K and SM architecture;
  fixes an off-by-one in initial-state selection (slightly lower error at K=2).
- New architectures: NemotronH latent-MoE (Nemotron-3-Super) and NemotronH MTP model.
- Misc: HGEMM sm_120 optimizations, mixed FP16+FP32 accumulate HGEMM for the reconstruct
  path, strided+batched cuBLAS matmul, batched Hadamard kernel, deterministic GDN reduction,
  fused RoPE partials written from lane zero, Triton is now a hard dependency (conditional
  imports removed), DFlash reads `tap_shift` from config, Gemma4-26B convert.py weight-swap
  fix.

Integration decisions:

- `exllamav3/exllamav3_ext/quant/reconstruct.cu`: upstream refactored the kernel body into
  `reconstruct_tile` (shared by the plain and new batched kernels). The fork's BF16 output
  mode (`out_pack<BF16_OUT>`, 48-entry instance table, `kHalf | kBFloat16` check in
  `reconstruct_slice`) is threaded through `reconstruct_tile` and `reconstruct_kernel`; the
  batched kernel instantiates the tile with `BF16_OUT = false` and stays fp16-only, as
  upstream wrote it.
- `exllamav3/modules/block_sparse_mlp.py`: the fork's runtime-LoRA guards are re-applied on
  the reworked forward. `experts_lora` still forces the torch/fused branch and additionally
  skips the new batched reconstruct tier (`_batch_recon_layer` is not consulted, so no
  groups and no deterministic scratch are built) alongside the exl3_moe kernel and the BC
  single-expert kernels. The `sh_fused_lora` case (LoRA on the fused shared experts / shared
  gate) used to fall through the bszN graph into the per-token mgemm tiers; those tiers are
  gone, so it now also diverts into the torch/fused branch, where the routed experts run
  through the regular fused kernels and the shared experts run through their own guarded
  forwards afterwards. The bszN tier keeps upstream's `assert bszn_eligible`, extended with
  `not experts_lora and not sh_fused_lora`. CPU split / offload rejections and the
  shared-gate `add_sigmoid_gate_proj` guard are unchanged.
- `tests/test_lora_fused_path.py`: the tripwire for the removed
  `elif bszn_eligible and not sh_fused_lora` line now checks the branch-condition guard,
  the extended bszN assertion, and the batched-reconstruct guard.
- `README.md`: kept the fork's README; bumped both upstream-version references to v1.5.0.
- Everything else auto-merged (`attn.py`, `sliding_attn.py`, `gated_delta_net.py`, `mlp.py`,
  `linear.py`, `mla_attn.py`, `mamba2.py`, `block_sparse_mlp_cpu.py`, ...). The fork's
  `has_runtime_lora` guards in those files are intact (checked by the source tripwires).

Validation (CPU, PyTorch 2.14.0):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_qlora_grad.py tests/test_native_llama.py tests/test_lora_init.py \
  tests/test_fused_ce.py tests/test_gdn.py tests/test_shortconv.py \
  tests/test_vision_training.py tests/test_preference.py \
  tests/test_quant_aware.py tests/test_lora_fused_path.py \
  -k 'not has_runtime_lora_semantics and not real_exl3_layer'
python -m compileall -q exllamav3 training examples tests
```

Result: 97 passed, 1 skipped, 2 deselected. Negative check: dropping `sh_fused_lora` from
the branch condition fails `test_moe_expert_dispatch_guarded`.

Not validated on GPU: the native extension was not rebuilt (no CUDA toolkit in the sync
environment) and no real-model decode/training parity was run. The reconstruct.cu merge in
particular (BF16 tile template over upstream's new `reconstruct_tile`) needs a compile on the
next box import, and a LoRA decode parity check on a MoE quant with shared experts is the
remaining box item for the re-routed `sh_fused_lora` case.
