"""Muon / AdaMuon optimizer (exllamav3/training/muon.py)."""

from __future__ import annotations
import os
import sys
import pytest
import torch

from exllamav3.training.muon import MuonAdamW, newton_schulz

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _param(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape, generator=g))


def _grads(shape, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g) for _ in range(n)]


@pytest.mark.parametrize("shape", [(16, 256), (256, 16), (32, 32)])
def test_newton_schulz_orthogonalizes(shape):
    o = newton_schulz(_param(*shape).data).float()
    s = torch.linalg.svdvals(o)
    assert o.shape == shape
    assert 0.5 < s.min() and s.max() < 1.5


@pytest.mark.parametrize("shape", [(256, 16), (16, 256)])
def test_muon_matches_torch_muon(shape):
    """Same trajectory as torch.optim.Muon with Moonshot's LR adjustment."""
    a, b = _param(*shape), _param(*shape)
    ours = MuonAdamW([a], lr=1e-2, weight_decay=0.1)
    ref = torch.optim.Muon([b], lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for g in _grads(shape, 5):
        a.grad, b.grad = g.clone(), g.clone()
        ours.step(); ref.step()
    assert torch.allclose(a, b, rtol=0, atol=2e-3)       # bf16 Newton-Schulz on both sides
    assert (a - _param(*shape)).abs().max() > 1e-2        # and it moved


def test_adamw_path_matches_torch_adamw():
    """matrix=False groups and non-2-D params take a standard AdamW step."""
    shapes = [(8, 24), (24,), (3, 4, 5)]
    a = [_param(*s, seed=i) for i, s in enumerate(shapes)]
    b = [_param(*s, seed=i) for i, s in enumerate(shapes)]
    ours = MuonAdamW([{"params": a[:1], "matrix": False}, {"params": a[1:]}],
                     lr=1e-2, weight_decay=0.05)
    ref = torch.optim.AdamW(b, lr=1e-2, weight_decay=0.05)
    for step in range(6):
        for i, s in enumerate(shapes):
            g = _grads(s, 1, seed=100 * step + i)[0]
            a[i].grad, b[i].grad = g.clone(), g.clone()
        ours.step(); ref.step()
    for x, y in zip(a, b):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("adaptive", [False, True])
def test_update_rms_is_point_two_lr(adaptive):
    """Both variants step with an RMS near 0.2 * lr, whatever the gradient scale."""
    for gscale in (1e-6, 1.0, 1e4):
        p = _param(512, 32)
        before = p.detach().clone()
        opt = MuonAdamW([p], lr=1e-3, adaptive=adaptive)
        p.grad = _grads((512, 32), 1)[0] * gscale
        opt.step()
        rms = (p - before).pow(2).mean().sqrt().item()
        assert 0.15e-3 < rms < 0.25e-3, (adaptive, gscale, rms)


@pytest.mark.parametrize("adaptive", [False, True])
def test_masked_lora_padding_stays_exactly_zero(adaptive):
    """lora_a rows >= in_real and lora_b columns >= out_real have their gradient
    masked to zero; the step must leave them exactly zero (weight decay too)."""
    in_pad, out_pad, r, in_real, out_real = 256, 384, 16, 200, 300
    a, b = _param(in_pad, r), _param(r, out_pad, seed=3)
    with torch.no_grad():
        a[in_real:] = 0
        b[:, out_real:] = 0
    opt = MuonAdamW([a, b], lr=1e-2, weight_decay=0.1, adaptive=adaptive)
    for ga, gb in zip(_grads((in_pad, r), 4), _grads((r, out_pad), 4, seed=7)):
        ga[in_real:] = 0
        gb[:, out_real:] = 0
        a.grad, b.grad = ga, gb
        opt.step()
    assert a[in_real:].abs().max() == 0 and b[:, out_real:].abs().max() == 0
    assert a[:in_real].abs().min() > 0


@pytest.mark.parametrize("adaptive", [False, True])
def test_zero_gradient_is_a_no_op(adaptive):
    """lora_a sees an all-zero gradient while lora_b is still zero (step 0)."""
    p = _param(64, 8)
    before = p.detach().clone()
    opt = MuonAdamW([p], lr=1e-2, adaptive=adaptive)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.isfinite(p).all() and torch.equal(p, before)


@pytest.mark.parametrize("adaptive", [False, True])
def test_trains_a_low_rank_fit(adaptive):
    torch.manual_seed(0)
    target = torch.randn(64, 4) @ torch.randn(4, 48) / 4
    a = torch.nn.Parameter(torch.randn(64, 4) * 0.1)
    b = torch.nn.Parameter(torch.zeros(4, 48))
    opt = MuonAdamW([a, b], lr=2e-2, adaptive=adaptive)
    losses = []
    for _ in range(200):
        loss = (a @ b - target).pow(2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < 0.2 * losses[0], losses[::40]


def test_state_dict_round_trip():
    def run(resume_at):
        a, t = _param(32, 8), _param(40, 8, seed=2)
        opt = MuonAdamW([{"params": [a]}, {"params": [t], "matrix": False}],
                        lr=1e-2, adaptive=True)
        for i in range(6):
            if i == resume_at:
                state = opt.state_dict()
                opt = MuonAdamW([{"params": [a]}, {"params": [t], "matrix": False}],
                                lr=1e-2, adaptive=True)
                opt.load_state_dict(state)
            a.grad = _grads((32, 8), 1, seed=i)[0]
            t.grad = _grads((40, 8), 1, seed=50 + i)[0]
            opt.step()
        return a.detach(), t.detach()
    (a0, t0), (a1, t1) = run(None), run(3)
    assert torch.equal(a0, a1) and torch.equal(t0, t1)


@pytest.mark.parametrize("optim", ["muon", "adamuon"])
def test_build_optimizer_keeps_named_non_matrix_groups_on_adamw(optim):
    from qlora_train_native import build_optimizer, NON_MATRIX_GROUPS
    lora, rows = _param(32, 8), _param(40, 8, seed=2)
    groups = [{"params": [lora], "weight_decay": 0.01},
              {"params": [_param(16, 8, seed=4)], "weight_decay": 0.01, "name": "global_attn"}]
    named = [_param(40, 8, seed=10 + i) for i in range(len(NON_MATRIX_GROUPS))]
    groups += [{"params": [p], "weight_decay": 0.0, "name": n}
               for p, n in zip(named, NON_MATRIX_GROUPS)]
    opt = build_optimizer(groups, 1e-3, optim)
    for g in opt.param_groups:
        for p in g["params"]:
            p.grad = torch.ones_like(p)
    opt.step()
    assert [g["matrix"] for g in opt.param_groups] == [True, True] + [False] * len(named)
    assert "momentum_buffer" in opt.state[lora]
    assert ("second_moment" in opt.state[lora]) == (optim == "adamuon")
    assert all("exp_avg" in opt.state[p] for p in named)
    assert opt.param_groups[0]["weight_decay"] == 0.01 and opt.param_groups[0]["lr"] == 1e-3
