"""
Differentiable ShortConv math for the native training forward.

LFM2 / LFM2-MoE (Liquid's LFM2.5-8B-A1B and friends) build ~3/4 of their
layers as ``ShortConv`` (``conv`` in the checkpoint keys): a gated short causal
depthwise convolution in place of attention. exllamav3's inference
implementation (``modules/short_conv.py``) runs under ``@torch.inference_mode``
against a per-sequence conv state, so training needs its own autograd-capable
forward. This module holds the pure math -- plain torch, no exllamav3 imports,
so the CPU test suite can load it standalone (like ``gdn`` / ``fused_ce``).

Semantics (matching ``ShortConv.forward`` + ``_causal_conv1d``)::

    b, c, x = in_proj(h).chunk(3, dim=-1)        # each [t, hidden]
    y       = c * causal_conv1d(b * x)           # depthwise, kernel L, no activation
    out     = out_proj(y)

The inference forward seeds the conv with ``kernel`` state columns (zeros for
a fresh sequence) and keeps the last ``t`` outputs; on a whole sequence from
position 0 that is exactly a left pad of ``kernel - 1`` zeros. There is no
nonlinearity between the conv and the ``c`` gate (unlike GatedDeltaNet's
conv + SiLU), and no norm on the mixer output.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def shortconv_causal_conv1d(
    x: torch.Tensor,                      # [b, dim, t]
    weight: torch.Tensor,                 # [dim, kernel]  depthwise conv weight
    bias: Optional[torch.Tensor] = None,  # [dim]
) -> torch.Tensor:
    """Stateless causal depthwise conv, ``[b, dim, t] -> [b, dim, t]``, in
    ``x``'s dtype (fp32 on the validate path, compute dtype in training; the
    inference path runs it in the stored weight dtype and casts back)."""
    kernel = weight.shape[-1]
    return F.conv1d(F.pad(x, (kernel - 1, 0)), weight.unsqueeze(1).to(x.dtype),
                    bias.to(x.dtype) if bias is not None else None,
                    padding=0, groups=x.shape[1])


def shortconv_mix(
    bcx: torch.Tensor,                    # [b, t, 3*hidden]  in_proj output
    weight: torch.Tensor,                 # [hidden, kernel]
    bias: Optional[torch.Tensor] = None,  # [hidden]
) -> torch.Tensor:
    """The gated conv core between ``in_proj`` and ``out_proj``: split the
    projection into ``b | c | x`` (in that order -- ``ShortConv.forward``
    chunks ``bc_x`` as ``b, c, x``), convolve ``b * x`` causally over time and
    gate the result with ``c``. Returns ``[b, t, hidden]``."""
    b, c, x = bcx.chunk(3, dim=-1)
    conv_in = (b * x).transpose(1, 2)                        # [b, hidden, t]
    conv_out = shortconv_causal_conv1d(conv_in, weight, bias)
    return c * conv_out.transpose(1, 2)
