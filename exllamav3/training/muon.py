"""Muon / AdaMuon for LoRA matrices, AdamW for everything else, in one optimizer.

Muon (Jordan et al.; "Muon is Scalable for LLM Training", arXiv 2502.16982)
replaces a matrix's momentum with its nearest semi-orthogonal matrix (a few
Newton-Schulz iterations) and steps along that. It keeps ONE buffer per
parameter, so its state is 4 bytes/param against AdamW's 8.

AdaMuon (Si et al., arXiv 2507.11005 v3, github.com/Chongjie-Si/AdaMuon)
orthogonalizes the SIGN of the momentum, then divides elementwise by a running
second moment of the orthogonalized update and rescales to a fixed RMS. Two
buffers per parameter: 8 bytes/param, same as AdamW.

Both are defined for 2-D hidden matrices only. Param groups with
``matrix=False`` (trained token rows, full embed/head, PLE tensors, ...) and
any non-2-D parameter take a plain AdamW step instead, so a single optimizer
serves the LR scheduler, ``lr_groups`` and checkpoint/resume unchanged.

The Muon step is scaled by ``0.2 * sqrt(max(rows, cols))`` (Moonshot's
"match_rms_adamw"), which gives the update an RMS of ``0.2 * lr`` -- the same
target AdaMuon normalizes to -- so an AdamW learning rate is a sane starting
point for either.

Newton-Schulz maps a zero row / zero column of its input to an exactly zero
row / column of its output, and every other operation here is elementwise or
a scalar multiple, so gradient-masked LoRA padding stays exactly zero (see the
padding notes in native_llama.py).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the semi-orthogonal factor U V^T of ``g = U S V^T`` with the
    quintic Newton-Schulz iteration from the Muon reference (bf16, like there).
    The singular values land in roughly [0.7, 1.3] rather than exactly at 1."""
    assert g.ndim == 2
    a, b, c = NS_COEFFICIENTS
    x = g.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


class MuonAdamW(Optimizer):
    """Muon (``adaptive=False``) or AdaMuon (``adaptive=True``) on the 2-D
    params of ``matrix`` groups; AdamW on the rest.

    ``weight_decay`` is decoupled and applied the same way on both paths
    (``p *= 1 - lr * weight_decay``). ``betas`` / ``eps`` belong to the AdamW
    path; ``momentum`` / ``nesterov`` / ``ns_steps`` to the matrix path.
    """

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0,
                 adaptive: bool = False, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 matrix: bool = True):
        defaults = dict(lr=lr, weight_decay=weight_decay, adaptive=adaptive,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        betas=betas, eps=eps, matrix=matrix)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["matrix"] and p.ndim == 2:
                    self._matrix_step(p, group)
                else:
                    self._adamw_step(p, group)
        return loss

    def _matrix_step(self, p: Tensor, group: dict) -> None:
        g, state = p.grad, self.state[p]
        lr, mom = float(group["lr"]), group["momentum"]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]

        if not group["adaptive"]:
            buf.lerp_(g, 1 - mom)
            u = g.lerp(buf, mom) if group["nesterov"] else buf
            u = newton_schulz(u, group["ns_steps"]).to(p.dtype)
            scale = 0.2 * max(p.shape) ** 0.5
        else:
            buf.mul_(mom).add_(g)
            u = g.add(buf, alpha=mom) if group["nesterov"] else buf
            u = newton_schulz(torch.sign(u), group["ns_steps"]).float()
            if "second_moment" not in state:
                state["second_moment"] = torch.zeros_like(u)
            v = state["second_moment"]
            v.mul_(mom).addcmul_(u, u, value=1 - mom)
            u = u.div_(v.sqrt().add_(group["eps"]))
            scale = float(0.2 * p.numel() ** 0.5 / (u.norm() + group["eps"]))
            u = u.to(p.dtype)

        p.mul_(1 - lr * group["weight_decay"])
        p.add_(u, alpha=-lr * scale)

    def _adamw_step(self, p: Tensor, group: dict) -> None:
        g, state = p.grad, self.state[p]
        lr, (b1, b2) = float(group["lr"]), group["betas"]
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(g)
            state["exp_avg_sq"] = torch.zeros_like(g)
        state["step"] += 1
        t, m, v = state["step"], state["exp_avg"], state["exp_avg_sq"]
        m.lerp_(g, 1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (v.sqrt() / (1 - b2 ** t) ** 0.5).add_(group["eps"])
        p.mul_(1 - lr * group["weight_decay"])
        p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
