"""
Fused linear cross-entropy for memory-efficient LM-head training.

The LM head is the single biggest activation-memory spike in LLM
fine-tuning: materialising logits of shape ``[tokens, vocab]`` (plus their
gradient) for a 128k+ vocabulary dwarfs everything else. This module computes
the cross-entropy loss and the gradient w.r.t. the hidden states **without
ever holding the full logit tensor**, by streaming over token chunks.

Assumptions matching QLoRA:
  * the LM-head weight is **frozen** (no LoRA on the head), so we only need
    the gradient w.r.t. the hidden states, not w.r.t. the weight. This lets
    us recompute the (dequantized) head weight in the backward instead of
    storing it -- a head weight can be ~1 GB on its own.
  * weight orientation is ``logits = hidden @ W`` with ``W`` of shape
    ``[hidden, vocab]`` -- exactly what ``LinearEXL3.get_weight_tensor()``
    returns.

The core ``FusedLinearCrossEntropy`` autograd Function is gradchecked in
``tests/test_fused_ce.py`` against ``torch.nn.functional.cross_entropy``.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn.functional as F


WeightFn = Callable[[], torch.Tensor]

DEFAULT_CHUNK = 1024
IGNORE_INDEX = -100


class FusedLinearCrossEntropy(torch.autograd.Function):
    """
    loss = cross_entropy(hidden @ weight, labels), computed chunked over the
    token dimension. Returns a scalar. Gradient is provided for ``hidden``
    only; the head weight is treated as a frozen constant.

    ``softcap`` > 0 applies Gemma-style final-logit tanh softcapping,
    ``z -> cap * tanh(z / cap)``, inside each logits tile. The cap is
    elementwise, so it composes with both the token and vocab chunking; the
    backward multiplies the CE logit-gradient by the tanh Jacobian
    ``1 - (z_capped / cap)^2`` before the transposed matmul.

    ``reduction``: ``"mean"`` (default) returns the scalar mean over the
    supervised tokens -- the training loss, exact old behavior. ``"none"``
    returns the per-token NLL vector ``[N]`` (zeros at ignored positions),
    still streamed (the ``[tokens, vocab]`` logits are never materialized) --
    the building block for per-sequence log-probabilities (DPO / KTO). The
    backward then takes a per-token ``grad_output`` instead of a scalar.
    """

    # Dtype scheme (Session 11, audit item A4): the matmul runs in the WEIGHT's
    # own dtype when it is half/bf16 -- the full [d, V] head is never upcast to
    # fp32 (that copy was ~1 GB on a 128k vocab / ~4 GB on Gemma's 262k, and the
    # old code re-created it on every token chunk). The matmul accumulates in
    # fp32 internally; only the [chunk, V] logits tile is upcast for a stable
    # softmax -- the same head-dtype contract the materialized supervised-
    # position path settled on in #118. Full-precision (fp32/fp64) weights keep
    # the exact old math, so the fp64 gradcheck is bit-unaffected. For half
    # weights the loss/grad move within the half-precision noise band (parity
    # test: test_fused_ce.test_low_precision_weight_parity).

    @staticmethod
    def _mm_dtype(hidden, weight):
        """(matmul dtype, softmax/statistics dtype) per the scheme above."""
        stat = torch.promote_types(hidden.dtype, torch.float32)
        low = weight.dtype in (torch.float16, torch.bfloat16)
        return (weight.dtype if low else stat), stat

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,        # [N, d]  (already flattened + aligned)
        labels: torch.Tensor,        # [N]
        weight_fn: WeightFn,         # () -> [d, V]  (frozen)
        chunk: int,
        ignore_index: int,
        softcap: float,              # > 0: cap * tanh(logits / cap)
        reduction: str,              # "mean" (scalar) | "none" (per-token [N])
    ) -> torch.Tensor:
        weight = weight_fn()
        n, d = hidden.shape
        cap = float(softcap or 0.0)
        mm_dtype, stat_dtype = FusedLinearCrossEntropy._mm_dtype(hidden, weight)
        w = weight if weight.dtype == mm_dtype else weight.to(mm_dtype)

        valid = labels != ignore_index
        m = int(valid.sum().item())
        denom = max(m, 1)

        per_token = (hidden.new_zeros((n,), dtype=stat_dtype)
                     if reduction == "none" else None)
        loss_sum = hidden.new_zeros((), dtype=stat_dtype)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            lbl_c = labels[start:end]
            valid_c = valid[start:end]
            if not bool(valid_c.any()):
                continue
            h_c = hidden[start:end].to(mm_dtype)
            logits_c = (h_c @ w).to(stat_dtype)                 # [c, V] tile only
            if cap:
                logits_c = cap * torch.tanh(logits_c / cap)
            logp_c = F.log_softmax(logits_c, dim=-1)
            safe_lbl = lbl_c.clamp_min(0)
            nll = -logp_c.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1)  # [c]
            nll = torch.where(valid_c, nll, torch.zeros_like(nll))
            if per_token is not None:
                per_token[start:end] = nll
            else:
                loss_sum = loss_sum + nll.sum()

        ctx.save_for_backward(hidden, labels)
        ctx.weight_fn = weight_fn
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        ctx.softcap = cap
        ctx.reduction = reduction
        if per_token is not None:
            return per_token.to(hidden.dtype)
        return (loss_sum / denom).to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, labels = ctx.saved_tensors
        weight = ctx.weight_fn()                 # recomputed, not stored
        chunk = ctx.chunk
        ignore_index = ctx.ignore_index
        denom = ctx.denom
        cap = ctx.softcap
        per_token = ctx.reduction == "none"
        n, d = hidden.shape

        mm_dtype, stat_dtype = FusedLinearCrossEntropy._mm_dtype(hidden, weight)
        w = weight if weight.dtype == mm_dtype else weight.to(mm_dtype)
        grad_hidden = torch.empty_like(hidden, dtype=stat_dtype)
        g_scale = grad_output.to(stat_dtype)     # scalar (mean) or [N] (none)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            h_c = hidden[start:end].to(mm_dtype)
            lbl_c = labels[start:end]
            valid_c = (lbl_c != ignore_index)

            logits_c = (h_c @ w).to(stat_dtype)                 # [c, V]
            if cap:
                logits_c = cap * torch.tanh(logits_c / cap)
            probs_c = torch.softmax(logits_c, dim=-1)           # [c, V] fp32
            # grad wrt logits = (softmax - onehot) / denom, zeroed on ignored.
            safe_lbl = lbl_c.clamp_min(0)
            probs_c.scatter_add_(
                -1, safe_lbl.unsqueeze(-1),
                torch.full_like(safe_lbl, -1.0, dtype=stat_dtype).unsqueeze(-1),
            )
            probs_c = torch.where(valid_c.unsqueeze(-1), probs_c,
                                  torch.zeros_like(probs_c))
            if per_token:
                # Each token is its own output: scale its logit-grad by its own
                # incoming grad (no 1/denom -- the caller owns the reduction).
                probs_c = probs_c * g_scale[start:end].unsqueeze(-1)
            else:
                probs_c = probs_c / denom
            if cap:
                # Chain through the softcap: d(cap*tanh(z/cap))/dz = 1 - tanh^2
                # = 1 - (z_capped / cap)^2.
                probs_c = probs_c * (1.0 - (logits_c / cap) ** 2)
            # The grad matmul also runs in the weight's dtype (probs cast down),
            # accumulating into the fp32 grad buffer -- same precision class as
            # every other matmul in a bf16 training step.
            gh = (probs_c.to(mm_dtype) @ w.t()).to(stat_dtype)
            grad_hidden[start:end] = gh if per_token else gh * g_scale

        return grad_hidden.to(hidden.dtype), None, None, None, None, None, None


WeightSliceFn = Callable[[int, int], torch.Tensor]   # (n_start, n_features) -> [d, n_features]

DEFAULT_VOCAB_CHUNK = 32768


class FusedLinearCrossEntropyVocabChunked(torch.autograd.Function):
    """
    Same loss/grad as :class:`FusedLinearCrossEntropy`, but the head weight is
    reconstructed and matmul'd in **vocab-column chunks** so the full
    ``[d, vocab]`` weight (and its fp32 upcast + the ``[tokens, vocab]`` logits)
    is never held at once -- only ``[d, vocab_chunk]`` and ``[token_chunk,
    vocab_chunk]`` tiles. This bounds the memory spike on the output device for
    big-vocabulary models (e.g. Gemma's 262k head), where the one-shot
    reconstruction is the dominant peak.

    The loop is **vocab-outer, token-inner**: each vocab chunk's weight is
    reconstructed exactly once per forward and once per backward (an online
    softmax carries per-token running max/sum across chunks), so the total
    reconstruction work equals the single-shot path -- no extra dequant cost.
    Head is frozen (gradient for ``hidden`` only). Gradchecked in
    ``tests/test_fused_ce.py`` against ``F.cross_entropy``.

    ``softcap`` > 0 applies Gemma-style final-logit tanh softcapping per
    ``[token_chunk, vocab_chunk]`` tile (the cap is elementwise, so the online
    softmax statistics over capped tiles are exact); the backward chains the
    tanh Jacobian ``1 - (z_capped / cap)^2`` into the logit gradient.

    ``reduction``: ``"mean"`` (scalar over supervised tokens, exact old
    behavior) or ``"none"`` (per-token NLL vector ``[N]``, zeros at ignored
    positions -- for per-sequence log-probabilities; the backward then takes a
    per-token ``grad_output``).
    """

    @staticmethod
    def forward(ctx, hidden, labels, weight_slice_fn, vocab_size, vocab_chunk,
                token_chunk, ignore_index, granularity, softcap, reduction):
        n, d = hidden.shape
        cap = float(softcap or 0.0)
        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)
        # Vocab chunk size must be a multiple of the reconstruction granularity
        # (so an EXL3 slice stays had_n-aligned); round, keep >= granularity.
        vc = max(granularity, (vocab_chunk // granularity) * granularity)

        valid = labels != ignore_index
        denom = max(int(valid.sum().item()), 1)

        neg_inf = torch.finfo(compute_dtype).min
        run_max = hidden.new_full((n,), neg_inf, dtype=compute_dtype)   # running max [N]
        run_sum = hidden.new_zeros((n,), dtype=compute_dtype)           # running sum-exp [N]
        tgt_logit = hidden.new_zeros((n,), dtype=compute_dtype)         # target logit [N]
        # Matmul in the weight slice's own dtype when it is half/bf16 (no fp32
        # copy of the slice; only the [tc, vw] logits tile is upcast) -- same
        # scheme as FusedLinearCrossEntropy._mm_dtype. h_all is cached in the
        # matmul dtype once the first slice reveals it.
        h_all = None

        for v0 in range(0, vocab_size, vc):
            v1 = min(v0 + vc, vocab_size)
            w_v = weight_slice_fn(v0, v1 - v0)                          # [d, vw]
            if w_v.dtype not in (torch.float16, torch.bfloat16):
                w_v = w_v.to(compute_dtype)
            if h_all is None or h_all.dtype != w_v.dtype:
                h_all = hidden.to(w_v.dtype)
            for t0 in range(0, n, token_chunk):
                t1 = min(t0 + token_chunk, n)
                logits = (h_all[t0:t1] @ w_v).to(compute_dtype)        # [tc, vw]
                if cap:
                    logits = cap * torch.tanh(logits / cap)
                # Online-softmax running stats over the vocab dimension.
                chunk_max = logits.max(dim=-1).values                  # [tc]
                new_max = torch.maximum(run_max[t0:t1], chunk_max)
                run_sum[t0:t1] = (run_sum[t0:t1] * torch.exp(run_max[t0:t1] - new_max)
                                  + torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1))
                run_max[t0:t1] = new_max
                # Capture the target logit for tokens whose label is in this chunk.
                lbl = labels[t0:t1]
                in_chunk = (lbl >= v0) & (lbl < v1)
                local = (lbl - v0).clamp_(0, v1 - v0 - 1)
                got = logits.gather(-1, local.unsqueeze(-1)).squeeze(-1)
                tgt_logit[t0:t1] = torch.where(in_chunk, got, tgt_logit[t0:t1])
            # Release before reconstructing the next tile (the RHS of an
            # assignment otherwise runs while the previous tile is still live).
            del w_v

        lse = run_max + torch.log(run_sum)                              # [N]
        nll = torch.where(valid, lse - tgt_logit, torch.zeros_like(lse))

        ctx.save_for_backward(hidden, labels, lse)
        ctx.weight_slice_fn = weight_slice_fn
        ctx.vocab_size = vocab_size
        ctx.vc = vc
        ctx.token_chunk = token_chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        ctx.softcap = cap
        ctx.reduction = reduction
        if reduction == "none":
            return nll.to(hidden.dtype)
        return (nll.sum() / denom).to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, labels, lse = ctx.saved_tensors
        weight_slice_fn = ctx.weight_slice_fn
        vocab_size, vc = ctx.vocab_size, ctx.vc
        token_chunk = ctx.token_chunk
        ignore_index = ctx.ignore_index
        denom = ctx.denom
        cap = ctx.softcap
        per_token = ctx.reduction == "none"
        n, d = hidden.shape

        compute_dtype = torch.promote_types(hidden.dtype, torch.float32)
        h_all = None                       # cached in the matmul dtype (see forward)
        g_scale = grad_output.to(compute_dtype)  # scalar (mean) or [N] (none)
        grad_hidden = hidden.new_zeros((n, d), dtype=compute_dtype)
        valid = (labels != ignore_index)

        # Same vocab-outer order: reconstruct each weight chunk once, accumulate
        # grad_hidden += (softmax - onehot)/denom @ W_v^T over chunks. softmax for
        # this chunk = exp(logits - lse) (lse is the full-vocab normalizer saved
        # in forward), so no second normalization pass is needed.
        for v0 in range(0, vocab_size, vc):
            v1 = min(v0 + vc, vocab_size)
            w_v = weight_slice_fn(v0, v1 - v0)                          # [d, vw]
            if w_v.dtype not in (torch.float16, torch.bfloat16):
                w_v = w_v.to(compute_dtype)
            if h_all is None or h_all.dtype != w_v.dtype:
                h_all = hidden.to(w_v.dtype)
            for t0 in range(0, n, token_chunk):
                t1 = min(t0 + token_chunk, n)
                logits = (h_all[t0:t1] @ w_v).to(compute_dtype)        # [tc, vw]
                if cap:
                    logits = cap * torch.tanh(logits / cap)
                p = torch.exp(logits - lse[t0:t1].unsqueeze(-1))       # softmax slice
                lbl = labels[t0:t1]
                in_chunk = (lbl >= v0) & (lbl < v1)
                local = (lbl - v0).clamp_(0, v1 - v0 - 1)
                # Subtract the one-hot target inside this chunk.
                onehot = torch.zeros_like(p)
                onehot.scatter_(-1, local.unsqueeze(-1),
                                in_chunk.to(p.dtype).unsqueeze(-1))
                p = p - onehot
                p = torch.where(valid[t0:t1].unsqueeze(-1), p, torch.zeros_like(p))
                if per_token:
                    # Each token is its own output: scale by its own incoming
                    # grad here (no 1/denom, no trailing scalar rescale).
                    p = p * g_scale[t0:t1].unsqueeze(-1)
                if cap:
                    # Chain through the softcap: d(cap*tanh(z/cap))/dz
                    # = 1 - (z_capped / cap)^2.
                    p = p * (1.0 - (logits / cap) ** 2)
                # Grad matmul in the weight's dtype, fp32 accumulation buffer.
                gh = (p.to(w_v.dtype) @ w_v.t()).to(compute_dtype)
                grad_hidden[t0:t1] += gh if per_token else gh / denom
            del w_v

        if not per_token:
            grad_hidden *= g_scale
        return (grad_hidden.to(hidden.dtype),
                None, None, None, None, None, None, None, None, None)


def fused_linear_cross_entropy_vocab_chunked(
    hidden: torch.Tensor,
    weight_slice_fn: WeightSliceFn,
    labels: torch.Tensor,
    vocab_size: int,
    vocab_chunk: int = DEFAULT_VOCAB_CHUNK,
    token_chunk: int = DEFAULT_CHUNK,
    ignore_index: int = IGNORE_INDEX,
    granularity: int = 1,
    shift: bool = True,
    softcap: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Vocab-chunked variant of :func:`fused_linear_cross_entropy`.

    :param weight_slice_fn: ``(n_start, n_features) -> [d, n_features]`` returning a
                            column slice of the frozen head weight.
    :param vocab_size:      total vocabulary (output columns).
    :param vocab_chunk:     output-column tile size (rounded down to ``granularity``).
    :param granularity:     required alignment of a slice (e.g. EXL3 ``had_n``=128).
    :param softcap:         final-logit tanh softcap (Gemma); 0 = none.
    :param reduction:       "mean" (scalar loss) or "none" (per-token NLL, shape
                            ``[B*(T-1)]`` after the shift; zeros where ignored).
    """
    if shift:
        hidden = hidden[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    d = hidden.shape[-1]
    hidden_flat = hidden.reshape(-1, d)
    labels_flat = labels.reshape(-1)
    return FusedLinearCrossEntropyVocabChunked.apply(
        hidden_flat, labels_flat, weight_slice_fn, vocab_size, vocab_chunk,
        token_chunk, ignore_index, granularity, softcap, reduction,
    )


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight_fn: WeightFn,
    labels: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
    ignore_index: int = IGNORE_INDEX,
    shift: bool = True,
    softcap: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Convenience wrapper that handles the causal-LM label shift and flattening.

    :param hidden:    ``[B, T, d]`` hidden states from the backbone.
    :param weight_fn: callable returning the frozen head weight ``[d, V]``
                      (e.g. ``lambda: lm_head.inner.get_weight_tensor()``).
    :param labels:    ``[B, T]`` token ids, ``ignore_index`` where masked.
    :param shift:     if True, predict token ``t+1`` from hidden at ``t``
                      (standard causal shift).
    :param softcap:   final-logit tanh softcap (Gemma); 0 = none.
    :param reduction: "mean" (scalar loss) or "none" (per-token NLL, shape
                      ``[B*(T-1)]`` after the shift; zeros where ignored).
    """
    if shift:
        hidden = hidden[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    d = hidden.shape[-1]
    hidden_flat = hidden.reshape(-1, d)
    labels_flat = labels.reshape(-1)
    return FusedLinearCrossEntropy.apply(
        hidden_flat, labels_flat, weight_fn, chunk, ignore_index, softcap,
        reduction,
    )
