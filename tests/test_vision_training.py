"""
CPU tests for the image+text (vision) training pieces:
``exllamav3/training/vision.py``, the vision branches of
``native_llama.py``, and the trainer-side ``training/vision_data.py``.

No GPU, no compiled extension, no real model. The training modules load
under the same synthetic package trick as ``test_native_llama.py`` (whose
mock block builder is reused here), and ``vision_data`` gets a stub of the
``exllamav3.training.vision`` import so it never touches the CUDA build.

Checks:
  * ``mrope_position_ids`` == a per-token transliteration of the
    ``gen_mrope_pos_ids`` CUDA-extension kernel on random layouts, plus a
    hand-computed case;
  * ``mrope_freqs`` == a transliteration of ``RoPE.get_mrope_freqs``'s
    interleave, and 3-D positions with equal axes rotate exactly like 1-D
    RoPE through ``_apply_rope`` (text-only == old behavior);
  * the embedding splice / deepstack add place the right rows and keep
    autograd flowing to the text rows only;
  * the bidirectional-span attention bias: within-span keys open in both
    directions, causal / window / key-pad rules intact elsewhere, and a block
    forward under it matches an independent masked-attention reference;
  * ``build_mm_batch`` index / span assembly, incl. an interleaved layout
    (Mistral3-style ``[IMG_BREAK]`` rows -> one span per row);
  * sentinel tokenization (``encode_segments_with_images``) with exact
    masks, and content-part flattening / image resolution
    (``ImageResolver``: part sources, the ``images`` column, unsupported
    parts rejected).

Run:  python tests/test_vision_training.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import random
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import test_native_llama as tnl  # noqa: E402  (sets up the exl3train package)

_vis = tnl._load("vision")
_nl = tnl._nl
NativeLlamaQLoRA = tnl.NativeLlamaQLoRA

# vision_data imports `exllamav3.training.vision`; alias the synthetic module
# under that name so the trainer-side helper loads without the CUDA package.
_ex = types.ModuleType("exllamav3")
_ext = types.ModuleType("exllamav3.training")
_ex.training = _ext
_ext.vision = _vis
sys.modules.setdefault("exllamav3", _ex)
sys.modules.setdefault("exllamav3.training", _ext)
sys.modules["exllamav3.training.vision"] = _vis
_spec = importlib.util.spec_from_file_location(
    "vision_data", os.path.join(_ROOT, "training", "vision_data.py"))
_vd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vd)


# ----------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------

def _kernel_mrope_pos_ids(ids, spans, grids, merge_size):
    """Per-token transliteration of exllamav3_ext gen_mrope_pos_ids (rope.cu)."""
    L = len(ids)
    out = [[0] * L for _ in range(3)]
    base_t, next_base_t = 0, 0
    for i in range(L):
        is_emb = False
        tid = ids[i]
        for j, (s0, s1) in enumerate(spans):
            if s0 <= tid < s1:
                is_emb = True
                k = tid - s0
                gt, gh, gw = grids[j]
                gh //= merge_size
                gw //= merge_size
                k_t = base_t + (k // gw // gh) % gt
                k_h = base_t + (k // gw) % gh
                k_w = base_t + k % gw
                out[0][i], out[1][i], out[2][i] = k_t, k_h, k_w
                next_base_t = max(next_base_t, k_t + 1, k_h + 1, k_w + 1)
                break
        if not is_emb:
            base_t = next_base_t
            out[0][i] = out[1][i] = out[2][i] = base_t
            base_t += 1
            next_base_t = base_t
    return torch.tensor(out)


def _ref_mrope_freqs(pos3, inv_freq, section):
    """Transliteration of RoPE.get_mrope_freqs' interleave (per batch row)."""
    outs = []
    for b in range(pos3.shape[1]):
        p = pos3[:, b, :]                                     # [3, t]
        inv = inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
        pe = p[:, None, None, :].float()
        freqs = (inv @ pe).transpose(2, 3)                    # [3, 1, t, n]
        ft = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = section[dim] * 3
            idx = slice(offset, length, 3)
            ft[..., idx] = freqs[dim, ..., idx]
        outs.append(ft[0])
    return torch.stack(outs)                                  # [b, t, n]


def _ref_block_masked(meta, w, hidden, positions, allowed):
    """tnl._ref_block with an explicit boolean [t, t] `allowed` mask."""
    eps_a, eps_m = meta["attn_eps"], meta["mlp_eps"]
    nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]
    b, t, _ = hidden.shape
    normed = tnl._ref_rmsnorm(hidden, w["attn_norm"], eps_a)
    q = tnl._ref_rope((normed @ w["q"]).view(b, t, nq, hd), meta["inv_freq"], positions)
    k = tnl._ref_rope((normed @ w["k"]).view(b, t, nkv, hd), meta["inv_freq"], positions)
    v = (normed @ w["v"]).view(b, t, nkv, hd)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    rep = nq // nkv
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * meta["sm_scale"]
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    ctx = torch.softmax(scores, -1) @ v
    hidden = hidden + ctx.transpose(1, 2).reshape(b, t, nq * hd) @ w["o"]
    normed2 = tnl._ref_rmsnorm(hidden, w["mlp_norm"], eps_m)
    a = F.silu(normed2 @ w["gate"]) * (normed2 @ w["up"])
    return hidden + a @ w["down"]


def _feats(layout, n, hidden, grid=None, merge=None, deepstack=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(n, hidden, generator=g)
    ds = [torch.randn(n, hidden, generator=g) for _ in range(deepstack)] or None
    return _vis.ImageFeatures(token_ids=layout, n_tokens=n, grid_thw=grid,
                              merge_size=merge, embeds=emb, deepstack=ds)


# ----------------------------------------------------------------------------
# mRoPE
# ----------------------------------------------------------------------------

def test_mrope_position_ids_hand():
    # [text text] [img: grid (1, 4, 4), merge 2 -> 2x2 = 4 tokens] [text text text]
    pos = _vis.mrope_position_ids(9, [(2, 4, (1, 4, 4))], 2)
    exp = torch.tensor([
        [0, 1, 2, 2, 2, 2, 4, 5, 6],   # t
        [0, 1, 2, 2, 3, 3, 4, 5, 6],   # h
        [0, 1, 2, 3, 2, 3, 4, 5, 6],   # w
    ])
    assert torch.equal(pos, exp), f"\n{pos}\n!=\n{exp}"
    print("[mrope] hand-computed position ids PASSED")


def test_mrope_position_ids_matches_kernel():
    rng = random.Random(0)
    FIRST = 10 ** 9
    for trial in range(40):
        merge = rng.choice([1, 2])
        ids, images, spans, grids = [], [], [], []
        nxt = FIRST
        n_img = rng.randint(1, 3)
        for _ in range(n_img):
            for _ in range(rng.randint(0, 5)):
                ids.append(rng.randint(0, 999))
            gt = rng.choice([1, 1, 2])
            gh, gw = merge * rng.randint(1, 4), merge * rng.randint(1, 4)
            n = gt * (gh // merge) * (gw // merge)
            start = len(ids)
            ids += list(range(nxt, nxt + n))
            spans.append((nxt, nxt + n))
            grids.append((gt, gh, gw))
            images.append((start, n, (gt, gh, gw)))
            nxt += n
        for _ in range(rng.randint(0, 5)):
            ids.append(rng.randint(0, 999))
        ref = _kernel_mrope_pos_ids(ids, spans, grids, merge)
        got = _vis.mrope_position_ids(len(ids), images, merge)
        assert torch.equal(got, ref), f"trial {trial}: mismatch\n{got}\n{ref}"
    print("[mrope] position ids == gen_mrope_pos_ids kernel transliteration (40 random layouts) PASSED")


def test_mrope_freqs_matches_reference_and_1d():
    torch.manual_seed(0)
    n = 32                                        # inv_freq entries (rotary 64)
    section = [16, 8, 8]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 2 * n, 2).float() / (2 * n)))
    pos3 = torch.randint(0, 50, (3, 2, 11))
    got = _vis.mrope_freqs(pos3, inv_freq, section)
    ref = _ref_mrope_freqs(pos3, inv_freq, section)
    assert torch.allclose(got, ref, atol=1e-5), "mrope_freqs != get_mrope_freqs reference"
    # Equal axes (a text-only sequence) must reduce to the 1-D angles exactly.
    p1 = torch.arange(11).unsqueeze(0).expand(2, 11)
    same = p1.unsqueeze(0).expand(3, 2, 11)
    got1 = _vis.mrope_freqs(same, inv_freq, section)
    ref1 = p1.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    assert torch.equal(got1, ref1), "equal-axis mRoPE angles != 1-D angles"
    # ...and through _apply_rope (partial rotary: rotate 64 of head_dim 96).
    net = tnl._headless_net()
    x = torch.randn(2, 11, 4, 96)
    y1 = net._apply_rope(x, inv_freq, 1.0, p1)
    y3 = net._apply_rope(x, inv_freq, 1.0, same, section)
    assert torch.equal(y1, y3), "3-D equal-axis _apply_rope != 1-D _apply_rope"
    # A genuinely 3-D position changes the rotation on the h/w bands only.
    diff = same.clone()
    diff[1, :, 3] += 5
    y3b = net._apply_rope(x, inv_freq, 1.0, diff, section)
    assert not torch.equal(y3b[:, 3], y3[:, 3]) and torch.equal(y3b[:, :3], y3[:, :3])
    print("[mrope] freqs == get_mrope_freqs reference; text-only == 1-D RoPE PASSED")


# ----------------------------------------------------------------------------
# Splice / deepstack / batch assembly
# ----------------------------------------------------------------------------

def test_build_mm_batch_and_splice():
    torch.manual_seed(1)
    d = 8
    # Row 0: [T T] [S x x x E] [T]            (contiguous image, 3 slots)
    # Row 1: [T] [x x B x x E] [T T]          (Mistral3-style: 2 rows of 2 + break)
    f0 = _feats([900, -1, -1, -1, 901], 3, d, deepstack=2, seed=1)
    f1 = _feats([-1, -1, 902, -1, -1, 903], 4, d, deepstack=2, seed=2)
    rows = [[(2, f0)], [(1, f1)]]
    t = 9
    mm = _vis.build_mm_batch(rows, t, torch.float32)
    exp_index = torch.tensor([
        [-1, -1, -1, 0, 1, 2, -1, -1, -1],
        [-1, 3, 4, -1, 5, 6, -1, -1, -1],
    ])
    assert torch.equal(mm["index"], exp_index), mm["index"]
    exp_spans = torch.tensor([
        [-1, -1, -1, 0, 0, 0, -1, -1, -1],
        [-1, 0, 0, -1, 1, 1, -1, -1, -1],       # two spans: the break splits the image
    ])
    assert torch.equal(mm["spans"], exp_spans), mm["spans"]
    assert mm["embeds"].shape == (7, d) and len(mm["deepstack"]) == 2
    assert torch.equal(mm["embeds"][:3], f0.embeds) and torch.equal(mm["embeds"][3:], f1.embeds)

    hidden = torch.randn(2, t, d, requires_grad=True)
    out = _vis.splice_embeddings(hidden, mm)
    sel = exp_index >= 0
    assert torch.equal(out[sel], mm["embeds"]), "spliced rows != features"
    assert torch.equal(out[~sel], hidden[~sel]), "text rows changed by the splice"
    out2 = _vis.add_deepstack(out, mm, 1)
    assert torch.allclose(out2[sel], mm["embeds"] + mm["deepstack"][1])
    assert torch.equal(out2[~sel], hidden[~sel])
    # Gradient reaches the text rows of `hidden` only (image rows are constants).
    out2.sum().backward()
    assert torch.equal(hidden.grad[~sel], torch.ones_like(hidden.grad[~sel]))
    assert torch.equal(hidden.grad[sel], torch.zeros_like(hidden.grad[sel]))
    # No images -> None.
    assert _vis.build_mm_batch([[], []], t, torch.float32) is None
    print("[splice] build_mm_batch index/spans, embedding splice, deepstack add, grads PASSED")


# ----------------------------------------------------------------------------
# Bidirectional image spans
# ----------------------------------------------------------------------------

def test_bidir_span_bias_and_block():
    torch.manual_seed(2)
    net = tnl._headless_net()
    t = 8
    #        0   1   2   3   4   5   6   7
    spans = torch.tensor([[-1, -1, 0, 0, 0, -1, -1, -1]])
    am = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])          # last position is pad
    bias = net._attn_bias(am, t, torch.device("cpu"), torch.float32, seg_ids=None,
                          bidir_spans=spans)
    allowed = bias[0, 0] == 0
    # Expected: causal, plus full visibility inside positions 2..4, pad key blocked.
    exp = torch.tril(torch.ones(t, t, dtype=torch.bool))
    exp[2:5, 2:5] = True
    exp[:, 7] = False
    assert torch.equal(allowed, exp), f"\n{allowed.int()}\n!=\n{exp.int()}"
    # Sliding window 2 outside the span still applies; the span stays open.
    bias_w = net._attn_bias(None, t, torch.device("cpu"), torch.float32, window=2,
                            bidir_spans=spans)
    allowed_w = bias_w[0, 0] == 0
    exp_w = torch.tril(torch.ones(t, t, dtype=torch.bool)) & ~torch.tril(
        torch.ones(t, t, dtype=torch.bool), diagonal=-2)
    exp_w[2:5, 2:5] = True
    assert torch.equal(allowed_w, exp_w)

    # Block forward under the bidirectional bias == masked reference; and the
    # tokens BEFORE the image are identical to the plain causal forward.
    d, nq, nkv, hd, inter = 16, 4, 2, 8, 32
    entry, meta, refw, _ = tnl._build_block(d, nq, nkv, hd, inter, dtype=torch.float32, r=0)
    hidden = torch.randn(1, t, d)
    pos = torch.arange(t).unsqueeze(0)
    b_bidir = net._attn_bias(None, t, hidden.device, torch.float32, bidir_spans=spans)
    b_causal = net._attn_bias(None, t, hidden.device, torch.float32)
    out_b = net._block_forward(meta, entry, hidden, pos, b_bidir)
    out_c = net._block_forward(meta, entry, hidden, pos, b_causal)
    allowed_b = b_bidir[0, 0] == 0
    ref_b = _ref_block_masked(meta, refw, hidden, pos, allowed_b)
    assert torch.allclose(out_b, ref_b, atol=1e-5), "bidirectional block != masked reference"
    assert torch.allclose(out_b[:, :2], out_c[:, :2], atol=1e-6), "pre-image tokens changed"
    assert not torch.allclose(out_b[:, 2:5], out_c[:, 2:5]), "image tokens unchanged by bidir"
    print("[bidir] span bias (causal/window/pad rules kept) + block == masked reference PASSED")


# ----------------------------------------------------------------------------
# Trainer-side: sentinel tokenization and content-part resolution
# ----------------------------------------------------------------------------

class _FakeTok:
    """Char-level tokenizer: id = ord(char); no BOS; no special pieces."""
    bos_token_id = None
    pad_token_id = 0
    eos_token_id = 2
    extended_piece_to_id = {}

    def encode(self, text, add_bos=False, encode_special_tokens=False, **kw):
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)


def test_encode_segments_with_images():
    tok = _FakeTok()
    f0 = _feats([100, -1, -1, -1, 101], 3, 4, grid=(1, 2, 6), merge=2)
    f1 = _feats([100, -1, 101], 1, 4, grid=(1, 2, 2), merge=2)
    segs = [("A<$EXL3_IMAGE_0$>B", False), ("<$EXL3_IMAGE_1$>", False), ("CD", True)]
    ids, labels, images = _vd.encode_segments_with_images(tok, segs, {0: f0, 1: f1}, 7)
    A, B, C, D = ord("A"), ord("B"), ord("C"), ord("D")
    assert ids == [A, 100, 7, 7, 7, 101, B, 100, 7, 101, C, D], ids
    assert labels == [-100] * 10 + [C, D], labels
    assert images == [(1, 0), (7, 1)], images

    # finalize: bookkeeping + 3-D positions on an mRoPE tower; too-long rows
    # that would cut an image are dropped, a cut through trailing text is fine.
    class _Enc:
        def encode(self, key, loader):
            return {"k0": f0, "k1": f1}[key]
        def stats(self):
            return {}
    vd = _vd.VisionData(_Enc(), tok, placeholder_id=7, mrope=True)
    refs = [_vd.ImageRef("k0", lambda: None), _vd.ImageRef("k1", lambda: None)]
    ex, why = vd.finalize(ids, labels, images, {0: f0, 1: f1}, refs, seq_len=64)
    assert why is None and ex["images"] == [(1, "k0"), (7, "k1")]
    pos = torch.tensor(ex["mrope_position_ids"])
    ref = _vis.mrope_position_ids(len(ids), [(2, 3, (1, 2, 6)), (8, 1, (1, 2, 2))], 2)
    assert torch.equal(pos, ref)
    ex2, why2 = vd.finalize(ids, labels, images, {0: f0, 1: f1}, refs, seq_len=9)
    assert ex2 is None and why2 == "too_long"
    # Non-mRoPE: no positions attached; truncating trailing text keeps the row.
    vd2 = _vd.VisionData(_Enc(), tok, placeholder_id=7, mrope=False)
    ex3, _ = vd2.finalize(ids, labels, images, {0: f0, 1: f1}, refs, seq_len=11)
    assert ex3 is not None and "mrope_position_ids" not in ex3 and len(ex3["input_ids"]) == 11
    # collate_mm re-fetches by key and builds the splice.
    vd.loaders = {"k0": lambda: None, "k1": lambda: None}
    mm = vd.collate_mm([ex], 16, torch.float32)
    assert (mm["index"] >= 0).sum().item() == 4 and mm["embeds"].shape[0] == 4
    print("[vision_data] sentinel encode (exact masks) + finalize/mRoPE/too_long + collate_mm PASSED")


def test_image_resolver():
    res = _vd.ImageResolver(base_dir="/base", images_key="images", messages_key="messages")
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Look: "},
                {"type": "image", "path": "cat.png"},
                {"type": "image"},                                  # -> images column [0]
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "text", "text": " ok?"},
            ]},
            {"role": "assistant", "content": "A cat."},
        ],
        "images": ["/pix/col0.png"],
    }
    msgs, refs = res.refs_for_row(row, 5, row["messages"])
    assert msgs[0]["content"] == "Look: <$EXL3_IMAGE_0$><$EXL3_IMAGE_1$><$EXL3_IMAGE_2$> ok?"
    assert msgs[1]["content"] == "A cat."
    # (a relative path is joined onto base_dir only when that file exists)
    assert [r.key for r in refs] == [("path", "cat.png"),
                                     ("ds", 5, "col", 0),
                                     ("url", "https://x/y.png")], [r.key for r in refs]
    # A bare image part with nothing left in the column is an error; so is video.
    bad = {"messages": [{"role": "user", "content": [{"type": "image"}]}]}
    try:
        res.refs_for_row(bad, 0, bad["messages"])
        assert False, "expected ValueError"
    except ValueError:
        pass
    vid = {"messages": [{"role": "user", "content": [{"type": "video", "path": "v.mp4"}]}]}
    try:
        res.refs_for_row(vid, 0, vid["messages"])
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert _vd.has_image_parts(row["messages"]) and not _vd.has_image_parts(
        [{"role": "user", "content": "hi"}])
    print("[vision_data] content-part flattening / image resolution PASSED")


def main():
    from util import run_timed
    run_timed([
        test_mrope_position_ids_hand,
        test_mrope_position_ids_matches_kernel,
        test_mrope_freqs_matches_reference_and_1d,
        test_build_mm_batch_and_splice,
        test_bidir_span_bias_and_block,
        test_encode_segments_with_images,
        test_image_resolver,
    ], label="vision-training")
    print("\nAll vision-training checks passed.")


if __name__ == "__main__":
    main()
