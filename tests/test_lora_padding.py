"""
CPU tests: LoRA adapters never carry weight in EXL3's zero padding.

EXL3 pads every Linear's in/out widths up to a multiple of 128 and the trainer
sizes each adapter from the PADDED widths, so lora_a rows >= in_features_unpadded
and lora_b columns >= out_features_unpadded are outside the real model. Checks:

  * creation zeroes the slice; a backward through the fused Function leaves
    its gradient exactly zero there (real slice non-zero) and an AdamW step
    with weight decay keeps it exactly zero;
  * an unpadded layer gets no hook and no mask;
  * zero_padding_ reports the dropped energy, assert_padding_zero refuses a
    dirtied adapter, set_init_offset pins the pissa offsets;
  * the pissa init path (apply_init_lora) leaves adapter + offsets pinned and
    is still function-preserving at step 0.

All fp32 on CPU; no GPU, no real model.
Run:  python tests/test_lora_padding.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_native_llama import MockLinear, _load, _nl  # noqa: E402

DiffLinear = _nl.DiffLinear
apply_init_lora = _load("lora_init").apply_init_lora

IN, IN_REAL, OUT, OUT_REAL, R = 40, 36, 32, 28, 4


def _padded_linear(key="model.layers.0.mlp.gate_proj"):
    lin = MockLinear(IN, OUT, key=key, dtype=torch.float32)
    lin.in_features_unpadded = IN_REAL
    lin.out_features_unpadded = OUT_REAL
    return lin


def _pad_a(dl):
    return dl.lora_a[dl.in_real:, :]


def _pad_b(dl):
    return dl.lora_b[:, dl.out_real:]


def test_padding_zero_through_training():
    torch.manual_seed(0)
    dl = DiffLinear(_padded_linear(), r=R, alpha=float(R), compute_dtype=torch.float32)
    assert dl.has_padding and (dl.in_real, dl.out_real) == (IN_REAL, OUT_REAL)
    assert _pad_a(dl).abs().max().item() == 0.0
    with torch.no_grad():
        dl.lora_b[:, :OUT_REAL].normal_(std=0.1)      # make A's grad non-zero
    opt = torch.optim.AdamW([dl.lora_a, dl.lora_b], lr=1e-2, weight_decay=0.1)
    for _ in range(3):
        x = torch.randn(5, IN)                        # padded columns non-zero too
        y = dl(x)
        y.square().sum().backward()
        assert _pad_a(dl).abs().max().item() == 0.0 and _pad_b(dl).abs().max().item() == 0.0
        assert dl.lora_a.grad[IN_REAL:, :].abs().max().item() == 0.0, "padded A grad leaked"
        assert dl.lora_b.grad[:, OUT_REAL:].abs().max().item() == 0.0, "padded B grad leaked"
        assert dl.lora_a.grad[:IN_REAL, :].abs().max().item() > 0.0
        assert dl.lora_b.grad[:, :OUT_REAL].abs().max().item() > 0.0
        opt.step()
        opt.zero_grad()
        assert _pad_a(dl).abs().max().item() == 0.0, "padded A moved after the step"
        assert _pad_b(dl).abs().max().item() == 0.0, "padded B moved after the step"
    assert dl.lora_a[:IN_REAL, :].abs().max().item() > 0.0
    dl.assert_padding_zero()
    print("[lora_padding] grad mask + AdamW step keep the padding at zero PASSED")


def test_unpadded_layer_untouched():
    dl = DiffLinear(MockLinear(IN, OUT, key="model.layers.0.self_attn.q_proj",
                               dtype=torch.float32),
                    r=R, alpha=float(R), compute_dtype=torch.float32)
    assert not dl.has_padding and (dl.in_real, dl.out_real) == (IN, OUT)
    assert not dl.lora_a._backward_hooks and not dl.lora_b._backward_hooks
    assert dl.zero_padding_() == 0.0
    dl.assert_padding_zero()
    print("[lora_padding] unpadded layer has no hook / no mask PASSED")


def test_zero_padding_reports_energy_and_save_refuses_dirty():
    torch.manual_seed(1)
    dl = DiffLinear(_padded_linear(), r=R, alpha=float(R), compute_dtype=torch.float32)
    with torch.no_grad():
        _pad_a(dl).fill_(2.0)                          # 4 rows x R
        _pad_b(dl).fill_(3.0)                          # R x 4 cols
    try:
        dl.assert_padding_zero()
        assert False, "assert_padding_zero must refuse a dirtied adapter"
    except RuntimeError:
        pass
    expect = 4.0 * R * 4.0 + R * 4 * 9.0
    dropped = dl.zero_padding_()
    assert abs(dropped - expect) < 1e-6, (dropped, expect)
    assert _pad_a(dl).abs().max().item() == 0.0 and _pad_b(dl).abs().max().item() == 0.0
    dl.assert_padding_zero()
    # Offsets: set_init_offset pins the padded slice of a0/b0 and reports it.
    a0 = torch.randn(IN, R)
    b0 = torch.randn(R, OUT)
    d0 = dl.set_init_offset(a0, b0)
    assert d0 > 0.0
    assert dl.init_a0_master[IN_REAL:, :].abs().max().item() == 0.0
    assert dl.init_b0_master[:, OUT_REAL:].abs().max().item() == 0.0
    assert dl.init_a0[IN_REAL:, :].abs().max().item() == 0.0
    assert a0[IN_REAL:, :].abs().max().item() > 0.0, "caller's tensor must not be mutated"
    print("[lora_padding] dropped-energy report + save refusal + offsets PASSED")


def test_pissa_init_pins_padding_and_preserves_function():
    torch.manual_seed(2)
    dl = DiffLinear(_padded_linear(), r=R, alpha=2.0, compute_dtype=torch.float32)
    net = types.SimpleNamespace(_wrappers=[dl], r=R)
    x = torch.randn(3, IN)
    y_before = dl(x)
    apply_init_lora(net, "pissa", svd_niter=4, verbose=False)
    assert net.init_lora == "pissa"
    assert _pad_a(dl).abs().max().item() == 0.0 and _pad_b(dl).abs().max().item() == 0.0
    assert dl.init_a0_master[IN_REAL:, :].abs().max().item() == 0.0
    assert dl.init_b0_master[:, OUT_REAL:].abs().max().item() == 0.0
    assert dl.lora_a[:IN_REAL, :].abs().max().item() > 0.0
    dl.assert_padding_zero()
    assert torch.allclose(dl(x), y_before, atol=1e-5), "pissa step 0 not function-preserving"
    print("[lora_padding] pissa init pins adapter + offsets PASSED")


if __name__ == "__main__":
    test_padding_zero_through_training()
    test_unpadded_layer_untouched()
    test_zero_padding_reports_energy_and_save_refuses_dirty()
    test_pissa_init_pins_padding_and_preserves_function()
