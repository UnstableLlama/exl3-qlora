"""
Stacked routed-expert projections for the native QLoRA forward.

``NativeLlamaQLoRA._moe_out`` sorts a layer's routed rows by expert. The
per-expert loop then runs three ``DiffLinear`` calls per expert, each with its
own trellis reconstruct, two Hadamard passes, base GEMM and LoRA term: several
hundred small launches per layer, again on the checkpoint recompute and the
backward. This module runs one projection of MANY experts at once over the
sorted rows:

  * one batched reconstruct of the experts' inner weights (the pointer-table
    kernel inference's batched tier uses; ``get_inner_weight_tensor`` per
    expert off CUDA),
  * the sign / Hadamard transforms once over all rows, each row scaled by its
    own expert's ``suh`` / ``svh``,
  * one grouped GEMM (``torch.nn.functional.grouped_mm``; a GEMM per expert
    where that kernel does not apply),
  * the LoRA term as two more grouped GEMMs over the experts' stacked factors.

The per-expert wrappers stay the owners of everything: ``lora_a`` / ``lora_b``
are stacked at call time and autograd returns each expert's gradient to its
own parameter, so the optimizer, padding masks, saved adapter keys and the
loader see no change. The math per row is the per-expert path's
(``EXL3LoRAHadFunction``); only launch count and GEMM accumulation order
differ.

The decoded weights of a chunk of experts are live at once, where the loop
holds one expert's. ``EXL3_TRAIN_BANK_MB`` bounds that chunk (per projection).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from . import backbone

# Decoded inner weights live at once per projection call, MiB. The gate and up
# banks of a chunk are live together, and each is decoded again in backward.
BANK_WEIGHT_MB = max(1, int(os.environ.get("EXL3_TRAIN_BANK_MB", 1024)))

# torch's grouped GEMM reads operand rows in 16-byte strides: a bf16 LoRA rank
# must be a multiple of this to stack.
_GROUPED_RANK_MULTIPLE = 8


def grouped_matmul(rows: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``rows[group] @ weights[group]`` over the consecutive row groups ending
    at ``offsets`` (int32, one per weight), differentiable in both operands.
    torch's grouped GEMM takes bf16 on CUDA only; anything else multiplies a
    group at a time."""
    if rows.is_cuda and rows.dtype == torch.bfloat16:
        return _GroupedMM.apply(rows, weights, offsets)
    ends = offsets.tolist()
    starts = [0, *ends[:-1]]
    parts = [rows[s:e] @ weights[g] for g, (s, e) in enumerate(zip(starts, ends)) if e > s]
    return torch.cat(parts) if parts else rows.new_empty(0, weights.shape[-1])


class _GroupedMM(torch.autograd.Function):
    """``F.grouped_mm`` with its two adjoints as grouped GEMMs."""

    @staticmethod
    def forward(ctx, rows, weights, offsets):
        ctx.save_for_backward(rows, weights, offsets)
        return F.grouped_mm(rows, weights, offs=offsets)

    @staticmethod
    def backward(ctx, grad):
        rows, weights, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        grad_rows = grad_weights = None
        if ctx.needs_input_grad[0]:
            grad_rows = F.grouped_mm(grad, weights.transpose(-2, -1), offs=offsets)
        if ctx.needs_input_grad[1]:
            grad_weights = F.grouped_mm(rows.transpose(-2, -1), grad, offs=offsets)
        return grad_rows, grad_weights, None


def _rotate(x: torch.Tensor, had: torch.Tensor) -> torch.Tensor:
    """The 128-block Hadamard over the last dim of ``[rows, dim]``."""
    return (x.view(x.shape[0], -1, 128) @ had).view(x.shape)


class ExpertBank:
    """One projection (gate, up or down) of every routed expert of a layer:
    the experts' ``DiffLinear`` wrappers plus the stacked sign tables. Built
    once per layer by :func:`build_bank`; holds no weights of its own."""

    def __init__(self, wrappers: Sequence):
        self.wrappers = list(wrappers)
        first = self.wrappers[0]
        self.in_features, self.out_features = first.in_features, first.out_features
        self.r, self.scale = first.r, first.scale
        self.suh = torch.stack([w._trellis_parts[1] for w in self.wrappers])    # [E, in]
        self.svh = torch.stack([w._trellis_parts[2] for w in self.wrappers])    # [E, out]
        self.had = first._had
        inner = first.linear.inner
        self._kernel_args = (inner.K, inner.mcg, inner.mul1) if self.suh.is_cuda else None
        per_expert = self.in_features * self.out_features * 2
        self.chunk = max(1, (BANK_WEIGHT_MB << 20) // per_expert)

    def inner_weights(self, experts: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """``[len(experts), in, out]`` inner (rotated-basis) weights. The
        trellis addresses are read per call: a parked or swapped expert's
        tensor may have moved."""
        if self._kernel_args is None:
            return torch.stack([self.wrappers[e]._trellis_parts[0]() for e in experts]).to(dtype)
        from ..ext import exllamav3_ext as ext
        device = self.suh.device
        ptrs = torch.tensor([self.wrappers[e].linear.inner.trellis.data_ptr() for e in experts],
                            dtype=torch.long, device=device)
        w = torch.empty((len(experts), self.in_features, self.out_features),
                        dtype=torch.half, device=device)
        ext.reconstruct_batch(w, ptrs, *self._kernel_args)
        # The kernel emits fp16; the per-expert kernel's bf16 output is this
        # same value rounded once.
        return w.to(dtype)

    def active(self) -> bool:
        """Whether the stacked path reproduces the wrappers' forward right
        now: off under quant-aware noise, the legacy dequant mode, a PiSSA
        residual base, or wrappers that disagree on being enabled."""
        if backbone.dequant_mode() != "fast":
            return False
        enabled = self.wrappers[0].adapter_enabled
        for w in self.wrappers:
            if w._qa_active() or w.init_a0 is not None or w.adapter_enabled != enabled:
                return False
        return True

    def forward(self, x: torch.Tensor, row_expert: torch.Tensor,
                experts: Sequence[int], counts: Sequence[int]) -> torch.Tensor:
        """``x`` ``[rows, in]`` sorted by expert; ``row_expert`` each row's
        expert id; ``experts`` the ids with rows, ascending, ``counts`` their
        row counts. Returns ``[rows, out]`` in ``x``'s dtype."""
        first = self.wrappers[0]
        x = x.to(first.compute_dtype)
        lora = self.r > 0 and first.adapter_enabled
        drop = first.lora_drop if (lora and first.training) else None
        outs = []
        start = 0
        for c0 in range(0, len(experts), self.chunk):
            ids = experts[c0:c0 + self.chunk]
            ends = []
            end = start
            for e in ids:
                end += counts[e]
                ends.append(end - start)
            offsets = torch.tensor(ends, dtype=torch.int32, device=x.device)
            rows = x[start:end]
            owner = row_expert[start:end]
            y = _FrozenBankFunction.apply(rows, self, ids, offsets, owner)
            if lora:
                a = torch.stack([self.wrappers[e].lora_a for e in ids]).to(x.dtype)
                b = torch.stack([self.wrappers[e].lora_b for e in ids]).to(x.dtype)
                xin = drop(rows) if drop is not None else rows
                y = y + self.scale * grouped_matmul(grouped_matmul(xin, a, offsets), b, offsets)
            outs.append(y)
            start = end
        return outs[0] if len(outs) == 1 else torch.cat(outs)


class _FrozenBankFunction(torch.autograd.Function):
    """``had(had(x * suh) @ W_inner) * svh`` per row against its own expert's
    inner weight, the weights reconstructed again in backward instead of
    saved (``EXL3LoRAHadFunction``'s base term, grouped)."""

    @staticmethod
    def forward(ctx, rows, bank: ExpertBank, ids, offsets, owner):
        ctx.bank, ctx.ids, ctx.offsets, ctx.owner = bank, ids, offsets, owner
        dt = rows.dtype
        w = bank.inner_weights(ids, dt)
        had = bank.had.to(dt)
        xh = _rotate(rows * bank.suh.to(dt)[owner], had)
        y = grouped_matmul(xh, w, offsets)
        return _rotate(y, had) * bank.svh.to(dt)[owner]

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        bank, owner = ctx.bank, ctx.owner
        dt = grad.dtype
        w = bank.inner_weights(ctx.ids, dt).transpose(1, 2)
        had_t = bank.had.to(dt).transpose(0, 1)
        gh = _rotate(grad * bank.svh.to(dt)[owner], had_t)
        gx = grouped_matmul(gh, w, ctx.offsets)
        return _rotate(gx, had_t) * bank.suh.to(dt)[owner], None, None, None, None


def build_bank(wrappers: Sequence) -> Optional[ExpertBank]:
    """An :class:`ExpertBank` over one projection's per-expert wrappers, or
    ``None`` when they cannot stack: a non-trellis inner, a bias, experts that
    differ in shape / bitrate / codebook / rank / scale / device, a LoRA rank
    the grouped GEMM cannot stride, or streamed (parked) experts, whose
    reconstruct hook the batched kernel would bypass."""
    if not wrappers:
        return None
    first = wrappers[0]
    if first._trellis_parts is None or first._had is None:
        return None
    inner0 = first.linear.inner
    sig0 = (first.in_features, first.out_features, first.r, first.scale,
            getattr(inner0, "K", None), getattr(inner0, "mcg", None), getattr(inner0, "mul1", None),
            first._trellis_parts[1].device)
    if first.r % _GROUPED_RANK_MULTIPLE:
        return None
    for w in wrappers:
        if w._trellis_parts is None:
            return None
        inner = w.linear.inner
        sig = (w.in_features, w.out_features, w.r, w.scale,
               getattr(inner, "K", None), getattr(inner, "mcg", None), getattr(inner, "mul1", None),
               w._trellis_parts[1].device)
        if sig != sig0:
            return None
        if w._bias() is not None:
            return None
        if "get_inner_weight_tensor" in getattr(inner, "__dict__", {}):
            return None
        if (w.lora_drop is None) != (first.lora_drop is None):
            return None
    return ExpertBank(wrappers)
