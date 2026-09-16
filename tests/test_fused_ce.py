"""
Correctness tests for the fused linear cross-entropy head.

Validates that streaming the logits over token chunks gives exactly the same
loss and hidden-state gradient as the naive
``cross_entropy(hidden @ weight, labels)`` -- and that the chunk size does not
change the result. Runs on CPU; needs only torch.

Run:  python tests/test_fused_ce.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import weakref
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_fce = _load("fused_ce")
FusedLinearCrossEntropy = _fce.FusedLinearCrossEntropy
FusedLinearCrossEntropyVocabChunked = _fce.FusedLinearCrossEntropyVocabChunked
fused_linear_cross_entropy = _fce.fused_linear_cross_entropy
fused_linear_cross_entropy_vocab_chunked = _fce.fused_linear_cross_entropy_vocab_chunked
IGNORE_INDEX = _fce.IGNORE_INDEX


def _slice_fn(weight):
    """Mock column-slice reconstructor over a dense weight (the EXL3 head provides
    the real one via get_weight_tensor_slice; granularity 1 for the dense mock)."""
    return lambda s, n: weight[:, s:s + n]


def _naive(hidden, weight, labels, ignore_index=IGNORE_INDEX):
    logits = hidden @ weight
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def test_vocab_weight_tile_released_before_next_reconstruction():
    weight = torch.randn(8, 24, dtype=torch.float64)
    hidden = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(5)
    previous = None
    calls = 0

    def reconstruct(start, width):
        nonlocal previous, calls
        assert previous is None or previous() is None, "previous weight tile still live"
        tile = weight[:, start:start + width].clone()
        previous = weakref.ref(tile)
        calls += 1
        return tile

    loss = FusedLinearCrossEntropyVocabChunked.apply(
        hidden, labels, reconstruct, 24, 8, 3, IGNORE_INDEX, 1, 0.0, "mean")
    loss.backward()
    assert calls == 6  # three tiles in each of forward and backward
    assert previous() is None


def test_loss_and_grad_parity():
    torch.manual_seed(0)
    n, d, v = 40, 16, 50
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight

    h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (n,))

    h2 = h1.detach().clone().requires_grad_(True)

    loss_ref = _naive(h1, weight, labels)
    loss_ref.backward()

    loss_fused = FusedLinearCrossEntropy.apply(h2, labels, wf, 7, IGNORE_INDEX, 0.0, "mean")
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), \
        f"loss mismatch: {loss_ref.item()} vs {loss_fused.item()}"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-9), "grad_hidden mismatch"
    print(f"[fce] loss/grad parity PASSED  (loss={loss_fused.item():.6f})")


def test_ignore_index():
    torch.manual_seed(1)
    n, d, v = 30, 12, 40
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))
    labels[::3] = IGNORE_INDEX  # mask out a third of tokens

    h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)

    loss_ref = _naive(h1, weight, labels); loss_ref.backward()
    loss_fused = FusedLinearCrossEntropy.apply(h2, labels, wf, 5, IGNORE_INDEX, 0.0, "mean")
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), "loss mismatch (ignore_index)"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-9), "grad mismatch (ignore_index)"
    # Ignored rows must get zero gradient.
    assert torch.allclose(h2.grad[::3], torch.zeros_like(h2.grad[::3]), atol=1e-12), \
        "ignored tokens received gradient"
    print("[fce] ignore_index PASSED")


def test_chunk_invariance():
    torch.manual_seed(2)
    n, d, v = 64, 16, 60
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))

    results = []
    for chunk in (1, 7, 64, 1000):
        h = torch.randn(n, d, dtype=torch.float64, generator=torch.Generator().manual_seed(9),
                        requires_grad=True)
        loss = FusedLinearCrossEntropy.apply(h, labels, wf, chunk, IGNORE_INDEX, 0.0, "mean")
        loss.backward()
        results.append((loss.detach().clone(), h.grad.clone()))

    l0, g0 = results[0]
    for li, gi in results[1:]:
        assert torch.allclose(l0, li, atol=1e-10), "loss varies with chunk size"
        assert torch.allclose(g0, gi, atol=1e-10), "grad varies with chunk size"
    print("[fce] chunk invariance PASSED")


def test_gradcheck():
    torch.manual_seed(3)
    n, d, v = 12, 8, 20
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    labels = torch.randint(0, v, (n,))
    h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropy.apply(h_, labels, wf, 5, IGNORE_INDEX, 0.0, "mean"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[fce] gradcheck PASSED")


def test_convenience_shift():
    """fused_linear_cross_entropy with shift must match a manual shifted CE."""
    torch.manual_seed(4)
    b, t, d, v = 2, 9, 16, 30
    weight = torch.randn(d, v, dtype=torch.float64)
    wf = lambda: weight
    hidden = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (b, t))

    h2 = hidden.detach().clone().requires_grad_(True)
    # Reference: shift then flat CE.
    logits = (h2[:, :-1] @ weight).reshape(-1, v)
    tgt = labels[:, 1:].reshape(-1)
    loss_ref = F.cross_entropy(logits, tgt); loss_ref.backward()

    loss_fused = fused_linear_cross_entropy(hidden, wf, labels, chunk=4)
    loss_fused.backward()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-10), "shifted loss mismatch"
    assert torch.allclose(hidden.grad, h2.grad, atol=1e-9), "shifted grad mismatch"
    print("[fce] convenience shift PASSED")


def test_vocab_chunked_parity():
    """Vocab-chunked CE must match naive F.cross_entropy on loss AND grad_hidden,
    for several vocab/token chunk sizes (incl. a chunk that doesn't divide V)."""
    torch.manual_seed(5)
    n, d, v = 48, 16, 70
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,))
    labels[::5] = IGNORE_INDEX

    for vchunk in (16, 32, v, 1000):
        for tchunk in (8, n):
            h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
            h2 = h1.detach().clone().requires_grad_(True)
            loss_ref = _naive(h1, weight, labels); loss_ref.backward()
            loss = FusedLinearCrossEntropyVocabChunked.apply(
                h2, labels, _slice_fn(weight), v, vchunk, tchunk, IGNORE_INDEX, 1, 0.0, "mean")
            loss.backward()
            assert torch.allclose(loss_ref, loss, atol=1e-10), \
                f"loss mismatch (vchunk={vchunk}, tchunk={tchunk})"
            assert torch.allclose(h1.grad, h2.grad, atol=1e-9), \
                f"grad mismatch (vchunk={vchunk}, tchunk={tchunk})"
    print("[fce] vocab-chunked parity vs F.cross_entropy PASSED")


def test_vocab_chunked_matches_single_shot():
    """The chunked-vocab head must give bit-for-bit the same loss/grad as the
    existing single-shot fused head (the two production code paths agree)."""
    torch.manual_seed(6)
    n, d, v = 50, 24, 64
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,)); labels[3] = IGNORE_INDEX

    h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)
    loss_a = FusedLinearCrossEntropy.apply(h1, labels, lambda: weight, 7, IGNORE_INDEX, 0.0, "mean")
    loss_a.backward()
    loss_b = FusedLinearCrossEntropyVocabChunked.apply(
        h2, labels, _slice_fn(weight), v, 16, 7, IGNORE_INDEX, 1, 0.0, "mean")
    loss_b.backward()
    assert torch.allclose(loss_a, loss_b, atol=1e-12), "loss differs from single-shot"
    assert torch.allclose(h1.grad, h2.grad, atol=1e-11), "grad differs from single-shot"
    print("[fce] vocab-chunked == single-shot PASSED")


def test_vocab_chunked_gradcheck():
    torch.manual_seed(7)
    n, d, v = 12, 8, 23
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,))
    h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropyVocabChunked.apply(
            h_, labels, _slice_fn(weight), v, 8, 5, IGNORE_INDEX, 1, 0.0, "mean"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[fce] vocab-chunked gradcheck PASSED")


def test_low_precision_weight_parity():
    # Session 11 dtype scheme: a half/bf16 head weight is no longer upcast to a
    # full fp32 [d, V] copy -- the matmul runs in the weight's dtype and only
    # the logits tile is upcast. Verify loss + grad_hidden stay within the
    # half-precision noise band of the fp32 reference, for BOTH fused heads.
    torch.manual_seed(3)
    n, d, v = 64, 32, 512
    weight32 = torch.randn(d, v, dtype=torch.float32)
    weight_bf = weight32.to(torch.bfloat16)
    hidden = torch.randn(n, d, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (n,))
    labels[::7] = IGNORE_INDEX

    # fp32 reference on the SAME (bf16-rounded) weight values, so the only
    # difference under test is the matmul/softmax dtype path, not the rounding
    # of the weights themselves.
    ref_w = weight_bf.to(torch.float32)
    ref = _naive(hidden, ref_w, labels)
    ref.backward()
    g_ref, hidden.grad = hidden.grad, None

    # Relative tolerance: bf16 matmul noise scales with the logit magnitude
    # (randn weights at d=32 give large logits), so compare relatively.
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-9)

    loss = FusedLinearCrossEntropy.apply(hidden, labels, lambda: weight_bf, 16,
                                         IGNORE_INDEX, 0.0, "mean")
    loss.backward()
    g_fused, hidden.grad = hidden.grad, None
    assert rel(loss.item(), ref.item()) < 2e-3, (loss.item(), ref.item())
    cos = F.cosine_similarity(g_fused.flatten(), g_ref.flatten(), dim=0)
    assert cos > 0.999, f"grad cosine {cos}"

    loss_vc = FusedLinearCrossEntropyVocabChunked.apply(
        hidden, labels, _slice_fn(weight_bf), v, 128, 16, IGNORE_INDEX, 1, 0.0, "mean")
    loss_vc.backward()
    g_vc, hidden.grad = hidden.grad, None
    assert rel(loss_vc.item(), ref.item()) < 2e-3, (loss_vc.item(), ref.item())
    cos = F.cosine_similarity(g_vc.flatten(), g_ref.flatten(), dim=0)
    assert cos > 0.999, f"vocab-chunked grad cosine {cos}"
    # The two fused paths agree on the loss (same fp32 softmax stats over the
    # same bf16 logit values); their grads differ only by bf16 tile
    # reassociation (probs cast to bf16 against different tile shapes), which
    # measures ~0.4% of the peak grad -- bound at 1%.
    assert abs(loss.item() - loss_vc.item()) < 1e-5
    assert (g_fused - g_vc).abs().max() < 1e-2 * g_fused.abs().max()
    print("[fce] low-precision (bf16) weight parity PASSED")


def _naive_softcap(hidden, weight, labels, cap, ignore_index=IGNORE_INDEX):
    logits = hidden @ weight
    logits = cap * torch.tanh(logits / cap)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


def test_softcap_parity():
    """Both fused heads with softcap must match the naive
    F.cross_entropy(cap * tanh((h @ W) / cap), labels) on loss AND grad_hidden
    (the Session-12 Gemma final-softcap path), across chunk sizes and with
    ignored tokens."""
    torch.manual_seed(8)
    n, d, v, cap = 48, 16, 70, 30.0
    weight = torch.randn(d, v, dtype=torch.float64) * 3   # push some logits past cap
    labels = torch.randint(0, v, (n,))
    labels[::5] = IGNORE_INDEX

    for chunk in (7, n):
        h1 = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_(True)
        h3 = h1.detach().clone().requires_grad_(True)
        loss_ref = _naive_softcap(h1, weight, labels, cap); loss_ref.backward()

        loss = FusedLinearCrossEntropy.apply(h2, labels, lambda: weight, chunk,
                                             IGNORE_INDEX, cap, "mean")
        loss.backward()
        assert torch.allclose(loss_ref, loss, atol=1e-10), \
            f"softcap loss mismatch (chunk={chunk})"
        assert torch.allclose(h1.grad, h2.grad, atol=1e-9), \
            f"softcap grad mismatch (chunk={chunk})"

        loss_vc = FusedLinearCrossEntropyVocabChunked.apply(
            h3, labels, _slice_fn(weight), v, 16, chunk, IGNORE_INDEX, 1, cap, "mean")
        loss_vc.backward()
        assert torch.allclose(loss_ref, loss_vc, atol=1e-10), \
            f"softcap vocab-chunked loss mismatch (chunk={chunk})"
        assert torch.allclose(h1.grad, h3.grad, atol=1e-9), \
            f"softcap vocab-chunked grad mismatch (chunk={chunk})"
    print("[fce] softcap parity vs naive capped CE PASSED")


def test_softcap_gradcheck():
    torch.manual_seed(9)
    n, d, v, cap = 12, 8, 20, 5.0
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,))
    h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropy.apply(h_, labels, lambda: weight, 5,
                                                 IGNORE_INDEX, cap, "mean"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropyVocabChunked.apply(
            h_, labels, _slice_fn(weight), v, 8, 5, IGNORE_INDEX, 1, cap, "mean"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[fce] softcap gradcheck PASSED")


def main():
    from util import run_timed
    run_timed([
        test_vocab_weight_tile_released_before_next_reconstruction,
        test_loss_and_grad_parity,
        test_ignore_index,
        test_chunk_invariance,
        test_gradcheck,
        test_convenience_shift,
        test_vocab_chunked_parity,
        test_vocab_chunked_matches_single_shot,
        test_vocab_chunked_gradcheck,
        test_low_precision_weight_parity,
        test_softcap_parity,
        test_softcap_gradcheck,
    ], label="fused-CE")
    print("\nAll fused cross-entropy checks passed.")


if __name__ == "__main__":
    main()
