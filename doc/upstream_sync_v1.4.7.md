# Upstream v1.4.7 sync

Merged `turboderp-org/exllamav3` master at
`ca13bdd` (v1.4.7), 105 commits since the previously merged
`ca5270c` ancestor, into fork master at `b1e72f0`.

Integration decisions:

- Retained BF16 reconstruction dispatch and applied upstream's bounds check
  after the BF16 kernel-table offset. FP16/BF16 reconstruction has 48 entries;
  fused Hadamard reconstruction remains FP16-only with 24 entries.
- Adapted sliding-attention runtime LoRA to upstream's padded mgemm inputs:
  adapter multiplication uses the original input channels, matching the fork's
  regular-attention implementation. Existing graph-path LoRA guards remain.
- Updated both upstream-version references in the fork README.

Validation (CPU, Python 3.12, PyTorch 2.8.0):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_qlora_grad.py tests/test_native_llama.py tests/test_lora_init.py \
  tests/test_fused_ce.py tests/test_gdn.py tests/test_shortconv.py \
  tests/test_vision_training.py tests/test_preference.py \
  tests/test_quant_aware.py tests/test_lora_fused_path.py \
  -k 'not has_runtime_lora_semantics and not real_exl3_layer'
python -m compileall -q exllamav3 training examples tests
git diff --cached --check
```

Result: 93 passed, 1 skipped, 2 deselected. The real-model gradient test is a
CLI helper requiring a model path (pytest otherwise reports a missing fixture);
real-package import and GPU/model tests were excluded or skipped. All six new
padded-input tests fail with the unadapted merge and pass with the fix.

CUDA was unavailable. Native extension compilation, upstream CUDA tests, and
real-model training/inference parity were not validated. Rebuild the native
extension before GPU use; a binary from before this sync lacks upstream's new
kernels and bindings.
