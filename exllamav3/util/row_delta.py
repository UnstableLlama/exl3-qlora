"""
Sparse row deltas over a row-addressable table (the hashed n-gram / Engram table of
Qwen3.8-Flash-Next and kin): a sorted set of table row ids and one delta vector per id, added
onto the decoded base rows wherever a lookup touches one of those ids. The table itself stays
frozen (it is tens of billions of parameters, quantized and usually streamed from disk); the
deltas are what ``--train-ngram-rows`` trains and what the adapter sidecar ``ngram_rows.safetensors``
carries. Plain torch so the training path, the inference module and the CPU tests share one
implementation.
"""

from __future__ import annotations
import torch


def row_slots(uids: torch.Tensor, table_uids: torch.Tensor) -> torch.Tensor:
    """
    Map table row ids to their index in ``table_uids`` (sorted, unique, int64, same device):
    ``slots[i] = j`` with ``table_uids[j] == uids[i]``, or ``-1`` for an id that is not in the
    set. Any shape of ``uids``.
    """
    if table_uids.numel() == 0:
        return torch.full_like(uids, -1)
    flat = uids.reshape(-1).to(table_uids.device, torch.int64)
    idx = torch.searchsorted(table_uids, flat).clamp_(max = table_uids.numel() - 1)
    hit = table_uids[idx] == flat
    return torch.where(hit, idx, torch.full_like(idx, -1)).view(uids.shape)


def apply_row_delta(rows: torch.Tensor, uids: torch.Tensor, table_uids: torch.Tensor,
                    delta: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    ``rows`` ``[U, dim]`` decoded base rows for the table ids ``uids`` ``[U]``; returns the rows
    with ``scale * delta[slot]`` added wherever the id is in ``table_uids`` (``delta`` is
    ``[N, dim]``, any dtype -- cast to the rows' dtype). Rows whose id is not in the set come
    back untouched. Out of place (the staging buffer may be reused by the caller).
    """
    slots = row_slots(uids, table_uids).to(rows.device)
    hit = slots >= 0
    if not bool(hit.any()):
        return rows
    out = rows.clone()
    d = delta[slots[hit].to(delta.device)].to(rows.device, rows.dtype)
    if scale != 1.0:
        d = d * scale
    out[hit] += d
    return out
