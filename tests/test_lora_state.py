"""
CPU tests for the runtime-LoRA packing contract (``exllamav3/modules/
lora_state.py``) and the in-place slot update (``backbone.set_runtime_lora``):

  * pack_lora: single adapter aliases its slot tensors (no copy); several
    concatenate along rank and equal the SUM of the per-adapter products;
    half-registered owners are skipped; shape checks fire.
  * lora_pack_key: stable under in-place copy_, changes on replacement.
  * lora_delta_reference: the kernel numerics (fp32 through both stages,
    one rounding on the add) vs a float64 oracle, fp16 and fp32 outputs.
  * set_runtime_lora: same-shape update copies in place (tensor identity
    and data_ptr preserved), shape change replaces.

Torch only; the module under test never imports the extension.
    python -m pytest tests/test_lora_state.py -q
"""

from __future__ import annotations
import importlib.util
import os
import sys
import types

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel, name):
    pkg = types.ModuleType(name.split(".")[0])
    pkg.__path__ = [os.path.dirname(os.path.join(_ROOT, rel))]
    sys.modules.setdefault(name.split(".")[0], pkg)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_ls = _load("exllamav3/modules/lora_state.py", "exl3mods_ls.lora_state")
pack_lora = _ls.pack_lora
lora_pack_key = _ls.lora_pack_key
lora_delta_reference = _ls.lora_delta_reference


def _pair(K, r, N, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(K, r, generator=g).half()
    b = (torch.randn(r, N, generator=g) * 0.1).half()
    return a, b


def test_single_adapter_aliases_slots():
    a, b = _pair(64, 8, 32, 0)
    o = object()
    packed = pack_lora({o: a}, {o: b}, 64, 32)
    assert packed is not None
    pa, pb, r = packed
    assert pa is a and pb is b and r == 8


def test_multi_adapter_concat_equals_sum_of_products():
    o1, o2, o3 = object(), object(), object()
    a1, b1 = _pair(64, 8, 32, 1)
    a2, b2 = _pair(64, 16, 32, 2)
    a3, b3 = _pair(64, 4, 32, 3)
    pa, pb, r = pack_lora({o1: a1, o2: a2, o3: a3}, {o1: b1, o2: b2, o3: b3}, 64, 32)
    assert r == 28 and pa.shape == (64, 28) and pb.shape == (28, 32)
    assert pa.is_contiguous() and pb.is_contiguous()
    x = torch.randn(5, 64).half()
    packed_delta = (x.float() @ pa.float()) @ pb.float()
    sum_delta = sum((x.float() @ a.float()) @ b.float()
                    for a, b in ((a1, b1), (a2, b2), (a3, b3)))
    assert torch.allclose(packed_delta, sum_delta, atol=1e-4, rtol=1e-4)


def test_half_registered_owner_is_skipped_and_empty_is_none():
    o1, o2 = object(), object()
    a1, b1 = _pair(16, 4, 8, 4)
    a2, _ = _pair(16, 4, 8, 5)
    pa, pb, r = pack_lora({o1: a1, o2: a2}, {o1: b1}, 16, 8)
    assert pa is a1 and pb is b1 and r == 4
    assert pack_lora({}, {}) is None
    assert pack_lora({o2: a2}, {}) is None


def test_shape_checks():
    o = object()
    a, b = _pair(16, 4, 8, 6)
    with pytest.raises(AssertionError):
        pack_lora({o: a}, {o: b}, in_features=32, out_features=8)
    with pytest.raises(AssertionError):
        pack_lora({o: a}, {o: b[:2]})      # rank mismatch


def test_pack_key_stable_under_inplace_update():
    o = object()
    a, b = _pair(16, 4, 8, 7)
    slots_a, slots_b = {o: a}, {o: b}
    k0 = lora_pack_key(slots_a, slots_b)
    a.copy_(torch.randn(16, 4).half())
    assert lora_pack_key(slots_a, slots_b) == k0
    slots_a[o] = a.clone()
    assert lora_pack_key(slots_a, slots_b) != k0


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_delta_reference_numerics(out_dtype):
    a, b = _pair(128, 16, 64, 8)
    x = torch.randn(3, 128).half()
    y0 = torch.randn(3, 64).to(out_dtype)
    y = y0.clone()
    lora_delta_reference(x, a, b, y)
    oracle = (y0.double() + (x.double() @ a.double()) @ b.double())
    # fp32 accumulation vs float64: agreement well inside one output ulp
    assert torch.allclose(y.double(), oracle.to(out_dtype).double(),
                          atol=(2e-3 if out_dtype == torch.float16 else 1e-5), rtol=1e-3)
    # exactly one rounding: fp32 output must NOT be rounded through fp16
    if out_dtype == torch.float32:
        assert not torch.equal(y, y.half().float())


def test_delta_reference_flattens_leading_dims():
    a, b = _pair(32, 4, 16, 9)
    x = torch.randn(2, 3, 32).half()
    y = torch.zeros(2, 3, 16).half()
    lora_delta_reference(x, a, b, y)
    ref = ((x.reshape(-1, 32).float() @ a.float()) @ b.float()).half().view(2, 3, 16)
    assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)


def test_set_runtime_lora_copies_in_place_when_shapes_match():
    src = open(os.path.join(_ROOT, "exllamav3", "training", "backbone.py"), encoding="utf8").read()
    # Execute only the two slot helpers (backbone imports the extension at
    # module level; the helpers are pure Python).
    start = src.index("def set_runtime_lora(")
    end = src.index("def clear_runtime_lora(")
    end = src.index("\n\n\n", end) if "\n\n\n" in src[end:] else len(src)
    ns = {"torch": torch}
    exec(src[start:end], ns)
    set_runtime_lora, clear_runtime_lora = ns["set_runtime_lora"], ns["clear_runtime_lora"]

    class Lin:
        device = "cpu"
        lora_a_tensors: dict = {}
        lora_b_tensors: dict = {}

    lin = Lin()
    lin.lora_a_tensors, lin.lora_b_tensors = {}, {}
    owner = object()
    a, b = _pair(16, 4, 8, 10)
    set_runtime_lora(lin, owner, a, b)
    a_slot, b_slot = lin.lora_a_tensors[owner], lin.lora_b_tensors[owner]
    key0 = lora_pack_key(lin.lora_a_tensors, lin.lora_b_tensors)

    a2, b2 = _pair(16, 4, 8, 11)
    set_runtime_lora(lin, owner, a2, b2)
    assert lin.lora_a_tensors[owner] is a_slot and lin.lora_b_tensors[owner] is b_slot
    assert torch.equal(a_slot, a2) and torch.equal(b_slot, b2)
    assert lora_pack_key(lin.lora_a_tensors, lin.lora_b_tensors) == key0

    a3, b3 = _pair(16, 8, 8, 12)          # rank change: must replace
    set_runtime_lora(lin, owner, a3, b3)
    assert lin.lora_a_tensors[owner] is not a_slot
    assert lin.lora_a_tensors[owner].shape == (16, 8)
    clear_runtime_lora(lin, owner)
    assert not lin.lora_a_tensors and not lin.lora_b_tensors
