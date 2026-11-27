"""
Tests for the stacked routed-expert path (``exllamav3/training/expert_bank.py``)
against the per-expert loop of ``NativeLlamaQLoRA._moe_out``.

CPU, float64, no extension: the experts are test_moe's mock linears given
trellis parts (an inner weight, random sign vectors, a Sylvester Hadamard), so
the per-expert reference is ``EXL3LoRAHadFunction`` and the stacked path runs
its per-group GEMM fallback. The reference could say no: every comparison is
the same block evaluated with ``expert_bank_enabled`` off.

Run directly::

    python tests/test_expert_bank.py
"""

from __future__ import annotations
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_moe as tm                                             # noqa: E402
from test_dequant_fast import sylvester_hadamard                  # noqa: E402

_nl = tm._nl
_eb = sys.modules["exl3train.expert_bank"]

D, INTER, E, TOP_K = 128, 256, 6, 2
DT = torch.float64


def _trellisize(wrappers, seed):
    """Give mock-backed wrappers the activation-side-transform pieces."""
    g = torch.Generator().manual_seed(seed)
    had = sylvester_hadamard(128, DT)
    for w in wrappers:
        inner = torch.randn(w.in_features, w.out_features, generator=g, dtype=DT) * 0.05
        suh = torch.where(torch.rand(w.in_features, generator=g) < 0.5, -1.0, 1.0).to(DT)
        svh = torch.where(torch.rand(w.out_features, generator=g) < 0.5, -1.0, 1.0).to(DT)
        w._trellis_parts = ((lambda t=inner: t), suh, svh)
        w._had = had
        if w.r > 0:
            with torch.no_grad():
                w.lora_b.copy_(torch.randn(w.lora_b.shape, generator=g, dtype=DT) * 0.1)


def _block(seed=0, r=8, gemma4=False, num_experts=E):
    torch.manual_seed(seed)
    entry, meta, lins = tm._build_moe(
        D, INTER, num_experts, TOP_K, shared_inter=128 if gemma4 else 0, shared_gate=False,
        r=0, expert_r=r, dtype=DT, gemma4=gemma4)
    experts = [*entry.expert_gates, *entry.expert_ups, *entry.expert_downs]
    for w in experts:
        if w.r > 0:
            w.lora_a.data = w.lora_a.data.to(DT)
            w.lora_b.data = w.lora_b.data.to(DT)
    _trellisize(experts, seed + 1)
    net = tm._headless_net()
    net.compute_dtype = DT
    return net, entry, meta, experts


def _run(net, entry, meta, x, bank: bool):
    net.expert_bank_enabled = bank
    residual = x if meta["alt_residual_channel"] else None
    return net._moe_out(meta, entry, x, residual=residual)


def _params(experts):
    return [p for w in experts for p in (w.lora_a, w.lora_b) if p is not None]


def _assert_matches_loop(net, entry, meta, experts, x):
    outs, grads = [], []
    for bank in (False, True):
        xi = x.clone().requires_grad_(True)
        out = _run(net, entry, meta, xi, bank)
        g = torch.autograd.grad(out.square().sum(), [xi, *_params(experts)], allow_unused=True)
        outs.append(out)
        grads.append(g)
    assert getattr(entry, "_expert_bank_triple", None), "stacked path was not built"
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-10, atol=1e-12)
    for got, want in zip(grads[1], grads[0]):
        assert (got is None) == (want is None)
        if want is not None:
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-12)
    return grads[1]


def test_bank_matches_per_expert_loop():
    net, entry, meta, experts = _block()
    x = torch.randn(2, 7, D, dtype=DT) * 0.5
    grads = _assert_matches_loop(net, entry, meta, experts, x)
    assert any(g is not None and g.abs().sum() > 0 for g in grads[1:])


def test_bank_matches_loop_gemma4_layout():
    net, entry, meta, experts = _block(seed=3, gemma4=True)
    x = torch.randn(1, 9, D, dtype=DT) * 0.5
    _assert_matches_loop(net, entry, meta, experts, x)


def test_bank_chunks_agree():
    net, entry, meta, experts = _block(seed=5)
    x = torch.randn(1, 11, D, dtype=DT) * 0.5
    whole = _run(net, entry, meta, x, True)
    for b in entry._expert_bank_triple:
        b.chunk = 2
    torch.testing.assert_close(_run(net, entry, meta, x, True), whole, rtol=1e-12, atol=1e-13)
    _assert_matches_loop(net, entry, meta, experts, x)


def test_bank_unrouted_expert_and_frozen_experts():
    # 24 experts, 3 tokens, top-2: most experts get no rows.
    net, entry, meta, experts = _block(seed=7, num_experts=24)
    x = torch.randn(1, 3, D, dtype=DT) * 0.5
    grads = _assert_matches_loop(net, entry, meta, experts, x)
    assert sum(g is None or not g.abs().sum() for g in grads[1:]) > 0
    net, entry, meta, experts = _block(seed=8, r=0)
    _assert_matches_loop(net, entry, meta, experts, torch.randn(1, 5, D, dtype=DT))


def test_bank_gradcheck():
    net, entry, meta, experts = _block(seed=11)
    x = (torch.randn(1, 4, D, dtype=DT) * 0.5).requires_grad_(True)
    with torch.no_grad():
        _, top = torch.topk(entry.router(x.reshape(-1, D)), TOP_K, dim=-1)
    e = int(top[0, 0])
    params = [entry.expert_gates[e].lora_a, entry.expert_downs[e].lora_b]

    def fn(x_, a_, b_):
        return _run(net, entry, meta, x_, True)

    assert torch.autograd.gradcheck(fn, (x, *params), eps=1e-6, atol=1e-7, nondet_tol=1e-9,
                                    fast_mode=True)


def test_bank_adapters_disabled_is_frozen_base():
    net, entry, meta, experts = _block(seed=13)
    x = torch.randn(1, 6, D, dtype=DT) * 0.5
    adapted = _run(net, entry, meta, x, True)
    for w in experts:
        w.adapter_enabled = False
    off_bank = _run(net, entry, meta, x, True)
    off_loop = _run(net, entry, meta, x, False)
    torch.testing.assert_close(off_bank, off_loop, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(off_bank, adapted)
    # Wrappers that disagree fall back to the loop.
    experts[0].adapter_enabled = True
    assert not entry._expert_bank_triple[0].active()


def test_bank_refuses_what_it_cannot_stack():
    _, entry, _, _ = _block(seed=17)
    assert _eb.build_bank(entry.expert_gates) is not None
    # A PiSSA residual base turns the stacked path off at call time.
    w = entry.expert_gates[2]
    w.init_a0 = torch.zeros_like(w.lora_a)
    assert not _eb.build_bank(entry.expert_gates).active()
    w.init_a0 = None
    # A streamed expert carries an instance-level reconstruct hook.
    inner = entry.expert_ups[1].linear.inner
    inner.get_inner_weight_tensor = lambda **k: None
    assert _eb.build_bank(entry.expert_ups) is None
    # Mixed rank, a rank the grouped GEMM cannot stride, a non-trellis expert.
    _, mixed, _, _ = _block(seed=19, r=8)
    mixed.expert_downs[3].r = 16
    assert _eb.build_bank(mixed.expert_downs) is None
    _, odd, _, _ = _block(seed=23, r=4)
    assert _eb.build_bank(odd.expert_gates) is None
    _, plain, _, _ = _block(seed=29)
    plain.expert_gates[0]._trellis_parts = None
    assert _eb.build_bank(plain.expert_gates) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
