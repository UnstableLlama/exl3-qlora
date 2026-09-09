"""
Runtime-LoRA state helpers for ``Linear`` (torch only, no extension import,
so this file is CPU-testable and is the Python-side contract the graph-path
LoRA kernels (doc/lora_inference_plan.md, stage 1) plug into).

A ``Linear`` carries its runtime adapters as two dicts keyed by the owning
adapter object: ``lora_a_tensors[owner] = A [K, r]`` and
``lora_b_tensors[owner] = B [r, N]`` (fp16, on the Linear's device, B
pre-scaled by alpha/r). Several adapters may be loaded at once and their
deltas sum. The fused kernels want ONE effective adapter per Linear, so
:func:`pack_lora` concatenates them along the rank dim:

    A_packed = [A_1 | A_2 | ...]   [K, R]      R = sum of ranks
    B_packed = [B_1 ; B_2 ; ...]   [R, N]

which is exactly the sum of the per-adapter products. A single adapter
packs to its own tensors (no copy), so an in-place update of the slot
tensors (``backbone.set_runtime_lora``) is visible to anything holding the
packed pair, and its data pointers never change -- the property the
graph-captured kernels rely on.

:func:`lora_delta_reference` is the numerical contract of the ``lora_gemv``
kernel: both low-rank stages accumulate in fp32 and the delta is rounded
ONCE, when it is added into the output in the output's dtype. GPU parity
tests compare the kernel against this function.
"""

from __future__ import annotations
from typing import Optional

import torch


def _validate_pair(a: torch.Tensor, b: torch.Tensor, in_features: Optional[int],
                   out_features: Optional[int]) -> None:
    assert a.dim() == 2 and b.dim() == 2, "LoRA slots must hold 2-D A [K, r] / B [r, N]"
    assert a.shape[1] == b.shape[0], \
        f"LoRA rank mismatch: A is {tuple(a.shape)}, B is {tuple(b.shape)}"
    if in_features is not None:
        assert a.shape[0] == in_features, \
            f"LoRA A rows {a.shape[0]} != in_features {in_features}"
    if out_features is not None:
        assert b.shape[1] == out_features, \
            f"LoRA B cols {b.shape[1]} != out_features {out_features}"


def lora_pack_key(a_tensors: dict, b_tensors: dict) -> tuple:
    """Identity of the current adapter set: owners in order plus each
    tensor's storage pointer and shape. Two calls return equal keys iff
    :func:`pack_lora` would return the same pair (same tensors, same order);
    an in-place ``copy_`` into a slot tensor does not change it, which is the
    point -- consumers that cache by this key see the new values for free."""
    return tuple(
        (id(owner), a.data_ptr(), tuple(a.shape), b.data_ptr(), tuple(b.shape))
        for owner, a in a_tensors.items()
        for b in (b_tensors.get(owner),) if b is not None
    )


def pack_lora(
    a_tensors: dict,
    b_tensors: dict,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor, int]]:
    """``(A_packed, B_packed, R)`` for the adapters currently in the slots, or
    None when there are none. Owners present in only one dict are skipped
    (a half-registered adapter contributes nothing, matching
    ``Linear.apply_lora``). One adapter: its tensors are returned as-is (no
    copy); several: contiguous concatenations in slot order."""
    pairs = [(a, b_tensors[owner]) for owner, a in a_tensors.items()
             if b_tensors.get(owner) is not None]
    if not pairs:
        return None
    for a, b in pairs:
        _validate_pair(a, b, in_features, out_features)
    if len(pairs) == 1:
        a, b = pairs[0]
        return a, b, a.shape[1]
    dt = pairs[0][0].dtype
    dev = pairs[0][0].device
    a_packed = torch.cat([a.to(dev, dt) for a, _ in pairs], dim=1).contiguous()
    b_packed = torch.cat([b.to(dev, dt) for _, b in pairs], dim=0).contiguous()
    return a_packed, b_packed, a_packed.shape[1]


def lora_delta_reference(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
    """In place: ``y += (x @ a) @ b`` with the kernel's numerics -- fp32
    accumulation through both stages, one rounding into ``y.dtype`` on the
    add. ``x`` is ``[M, K]`` (any leading shape is flattened), ``y`` is
    ``[M, N]`` in fp16 or fp32. Returns ``y``."""
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    yf = y.reshape(-1, y.shape[-1])
    delta = (xf @ a.to(torch.float32)) @ b.to(torch.float32)
    yf.copy_((yf.to(torch.float32) + delta).to(y.dtype))
    return y
