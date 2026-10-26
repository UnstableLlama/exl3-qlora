"""
Runtime LoRA on BC_LinearEXL3. The graph-captured decode blocks add an adapter through the projection's C++ object
(BC_LinearEXL3.lora_gr: the lora_a / lora_b kernels behind Linear.lora_sync), which run_lora exposes eagerly. It
must match the unfused reference Linear.apply_lora computes, y + x @ A @ B, for fp16 and fp32 outputs, row counts
up to the LORA_MAX_ROWS scratch (the largest graphed shape) and no further, adapters stacked along the rank dim, and
a padded adapter over a narrower activation (the python path's zero-extension is never staged here).

Linear.lora_sync is the hand-off the module dispatch guards (lora_graphed) rely on: it mirrors the attached dicts
into the C++ object, declines what the kernels can't take (odd dtypes, mismatched shapes, a modules_to_save
full-weight override) so the module falls back to the unfused path, and detaches once the dicts empty. Unquantized
linears (BC_LinearFP16: the GDN b/a projections, headwise gates) carry the same component. Modules are built from
random tensors through the real loader.
"""

import pytest
import torch

from exllamav3.modules import Linear
from exllamav3.modules.linear import lora_graphed, has_runtime_lora

from testlib.checkpoint import module_config
from testlib.exl3 import checkpoint_tensors, generator

KEY = "model.layers.0.self_attn.q_proj"
MAX_ROWS = 128   # BC_LinearEXL3::LORA_MAX_ROWS


def build(directory, device, k, n, K = 4, codebook = "mul1", bias = False, seed = 1):
    t = checkpoint_tensors(KEY, k, n, K, generator(seed), codebook = codebook, bias = bias)
    module = Linear(module_config(t, directory, no_reconstruct = True), KEY, k, n)
    module.load(device)
    assert module.quant_type == "exl3" and module.inner.bc is not None
    return module


def adapter(k, n, r, gen, device, std = 0.05):
    """(A [in, r], B [r, out]) in the orientation LoRA registers on a Linear (B carries the scale)"""
    a = (torch.randn(k, r, generator = gen) * std).half().to(device)
    b = (torch.randn(r, n, generator = gen) * std).half().to(device)
    return a, b


def delta_ref(x, a, b):
    """fp32 x @ A @ B over the rows of x (x may be narrower than A: the extra A rows multiply nothing)"""
    xf = x.float().view(-1, x.shape[-1])
    return xf @ a.float()[:xf.shape[1]] @ b.float()


def rel_max_err(got, ref) -> float:
    return ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item()


@pytest.mark.parametrize("rows", [1, 2, 7, 16, MAX_ROWS])
@pytest.mark.parametrize("out_fp32", [False, True])
@torch.inference_mode()
def test_run_lora_matches_reference(tmp_path, device, rows, out_fp32):
    k, n, r = 2048, 512, 16
    module = build(tmp_path, device, k, n, bias = True)
    bc = module.inner.bc
    gen = generator(rows * 7 + out_fp32)
    a, b = adapter(k, n, r, gen, device)

    x = (torch.randn((1, rows, k), generator = gen) * 0.5).half().to(device)
    y = bc.run_alloc(x, n, out_fp32)          # the quantized product (plus bias), adapter-free
    assert not bc.has_lora()
    y_same = y.clone()
    bc.run_lora(x, y_same)                    # no adapter attached: a no-op
    assert torch.equal(y_same, y)

    bc.set_lora(a.T.contiguous(), b)
    assert bc.has_lora()
    y_lora = y.clone()
    bc.run_lora(x, y_lora)
    ref = y.float().view(-1, n) + delta_ref(x, a, b)
    err = rel_max_err(y_lora.view(-1, n), ref)
    # fp16 output: one rounding of y + delta; fp32: accumulation order only
    assert err <= (4e-3 if not out_fp32 else 1e-4), f"rows {rows}: rel max err {err:.2e}"
    # The delta itself is what the kernels add, so it must also be visible against a zero base
    z = torch.zeros_like(y)
    bc.run_lora(x, z)
    assert rel_max_err(z.view(-1, n), delta_ref(x, a, b)) <= (4e-3 if not out_fp32 else 1e-4)

    bc.set_lora(None, None)
    assert not bc.has_lora()
    y_off = y.clone()
    bc.run_lora(x, y_off)
    assert torch.equal(y_off, y)


@torch.inference_mode()
def test_row_limit_is_enforced(tmp_path, device):
    k, n, r = 512, 256, 8
    module = build(tmp_path, device, k, n)
    a, b = adapter(k, n, r, generator(3), device)
    module.inner.bc.set_lora(a.T.contiguous(), b)
    x = torch.randn((1, MAX_ROWS + 1, k), device = device).half()
    y = torch.zeros((1, MAX_ROWS + 1, n), device = device).half()
    with pytest.raises(RuntimeError):
        module.inner.bc.run_lora(x, y)


@torch.inference_mode()
def test_padded_adapter_over_narrow_activation(tmp_path, device):
    # in_features pads 2040 -> 2048; LoRA pads A to the padded width (zero rows), the activation stays 2040
    k, n, r = 2040, 256, 8
    module = build(tmp_path, device, k, n)
    assert module.in_features == 2048
    gen = generator(5)
    a, b = adapter(module.in_features, n, r, gen, device)
    a[k:] = 0
    module.inner.bc.set_lora(a.T.contiguous(), b)
    x = (torch.randn((1, 3, k), generator = gen) * 0.5).half().to(device)
    y = torch.zeros((1, 3, n), device = device, dtype = torch.float)
    module.inner.bc.run_lora(x, y)
    assert rel_max_err(y.view(-1, n), delta_ref(x, a, b)) <= 1e-4


@torch.inference_mode()
def test_lora_sync_hands_off_stacked_adapters(tmp_path, device):
    k, n = 1024, 512
    module = build(tmp_path, device, k, n)
    bc = module.inner.bc
    gen = generator(11)
    a1, b1 = adapter(k, n, 8, gen, device)
    a2, b2 = adapter(k, n, 24, gen, device)
    x = (torch.randn((1, 4, k), generator = gen) * 0.5).half().to(device)

    assert module.lora_sync() and lora_graphed(module, None) and not bc.has_lora()

    owner1, owner2 = object(), object()
    module.lora_a_tensors[owner1], module.lora_b_tensors[owner1] = a1, b1
    assert has_runtime_lora(module)
    assert module.lora_sync() and bc.has_lora()
    module.lora_a_tensors[owner2], module.lora_b_tensors[owner2] = a2, b2
    assert lora_graphed(module)               # re-syncs: the attached set changed
    assert bc.lora_a.shape == (32, k) and bc.lora_b.shape == (32, n)

    y = torch.zeros((1, 4, n), device = device, dtype = torch.float)
    bc.run_lora(x, y)
    ref = delta_ref(x, a1, b1) + delta_ref(x, a2, b2)
    assert rel_max_err(y.view(-1, n), ref) <= 1e-4

    # The python path (Linear.forward -> apply_lora) is the reference the graph path must agree with
    y_fwd = module.forward(x, {}, out_dtype = torch.float)
    y_base = bc.run_alloc(x, n, True)
    assert rel_max_err((y_fwd - y_base).view(-1, n), ref) <= 2e-2   # fp16 addmm on the python side

    # Detach: dicts emptied -> graphed again (no adapter), C++ object released its copy
    module.lora_a_tensors.pop(owner1); module.lora_b_tensors.pop(owner1)
    module.lora_a_tensors.pop(owner2); module.lora_b_tensors.pop(owner2)
    assert lora_graphed(module) and not bc.has_lora()


@torch.inference_mode()
def test_lora_sync_declines_what_the_kernels_cannot_take(tmp_path, device):
    k, n = 512, 256
    module = build(tmp_path, device, k, n)
    bc = module.inner.bc
    gen = generator(17)
    a, b = adapter(k, n, 8, gen, device)
    owner = object()

    # bf16 adapter: python path applies it (apply_lora casts), the kernels are fp16 only
    module.lora_a_tensors[owner], module.lora_b_tensors[owner] = a.bfloat16(), b.bfloat16()
    assert not module.lora_sync() and not lora_graphed(module) and not bc.has_lora()

    # Shape the C++ object would reject: declined python-side, no exception
    module.lora_a_tensors[owner], module.lora_b_tensors[owner] = a[:k - 16].contiguous(), b
    assert not lora_graphed(module) and not bc.has_lora()

    # A good adapter is taken; a modules_to_save override then forces the unfused path, and the
    # C++ copy is dropped so a stale adapter can't ride along in a graph
    module.lora_a_tensors[owner], module.lora_b_tensors[owner] = a, b
    assert lora_graphed(module) and bc.has_lora()
    module.lora_full_weight = torch.zeros((k, n), device = device).half()
    assert not lora_graphed(module) and not bc.has_lora()
    module.lora_full_weight = None

    # Unload clears everything
    module.lora_a_tensors[owner], module.lora_b_tensors[owner] = a, b
    assert lora_graphed(module)
    module.unload()
    assert not module.lora_a_tensors and module._lora_graph_src == ()


@torch.inference_mode()
def test_strided_output_slice(tmp_path, device):
    # The GDN graph lands the b / a deltas on column slices of its merged (R, 2*Nv) static: lora_b
    # takes the row stride and leaves the other half untouched
    k, n, r = 512, 128, 8
    module = build(tmp_path, device, k, n)
    gen = generator(23)
    a, b = adapter(k, n, r, gen, device)
    module.inner.bc.set_lora(a.T.contiguous(), b)
    x = (torch.randn((2, 3, k), generator = gen) * 0.5).half().to(device)
    wide = torch.zeros((2, 3, 2 * n), device = device, dtype = torch.float)
    right = wide.narrow(-1, n, n)
    module.inner.bc.run_lora(x, right)
    assert torch.count_nonzero(wide[..., :n]) == 0, "the untouched half was written"
    assert rel_max_err(wide[..., n:].reshape(-1, n), delta_ref(x, a, b)) <= 1e-4
    bad = wide.transpose(0, 1).narrow(-1, 0, n)   # rows no longer uniformly strided
    with pytest.raises(RuntimeError):
        module.inner.bc.run_lora(x.transpose(0, 1).contiguous(), bad)


@torch.inference_mode()
def test_fp16_linear_hand_off(tmp_path, device):
    # An unquantized Linear (weight (in, out) fp16): same hand-off and kernels through BC_LinearFP16
    k, n, r = 512, 128, 8
    gen = generator(29)
    w = (torch.randn(n, k, generator = gen) * 0.05).half()                      # checkpoint orientation (out, in)
    module = Linear(module_config({f"{KEY}.weight": w}, tmp_path), KEY, k, n)
    module.load(device)
    assert module.quant_type == "fp16" and module.inner.bc is not None
    a, b = adapter(k, n, r, gen, device)
    owner = object()
    module.lora_a_tensors[owner], module.lora_b_tensors[owner] = a, b
    assert lora_graphed(module) and module.inner.bc.has_lora()
    x = (torch.randn((1, 5, k), generator = gen) * 0.5).half().to(device)
    y = torch.zeros((1, 5, n), device = device, dtype = torch.float)
    module.inner.bc.run_lora(x, y)
    assert rel_max_err(y.view(-1, n), delta_ref(x, a, b)) <= 1e-4
    module.lora_a_tensors.clear(); module.lora_b_tensors.clear()
    assert lora_graphed(module) and not module.inner.bc.has_lora()
