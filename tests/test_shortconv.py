"""
CPU tests for the differentiable ShortConv path (LFM2 / LFM2-MoE gated short
causal conv layers -- e.g. LFM2.5-8B-A1B) in
``exllamav3/training/short_conv.py`` + ``native_llama._shortconv_forward``.

No GPU, no compiled extension, no real model: the training modules are loaded
under a synthetic package (so their relative imports resolve without importing
the full exllamav3 package, which would build the CUDA ext), EXL3 linears are
mocked as frozen random weights, and the checks are:

  * ``shortconv_causal_conv1d`` matches a verbatim transcription of the
    inference module's ``_causal_conv1d`` run from a zero conv state (with and
    without bias);
  * ``shortconv_mix`` matches the inference forward's b|c|x split + gating;
  * a full ShortConv block forward (``_shortconv_forward``) matches an
    independent plain-torch composition (norm -> in_proj -> b*x -> causal
    conv -> c gate -> out_proj + MLP);
  * the block is causal: perturbing token j never changes outputs at i < j
    (which is also what makes right-padding safe);
  * backprop through a ShortConv block reaches the LoRA adapters on
    in_proj/out_proj while the frozen base stays untouched;
  * ``backbone.block_metadata`` on a mock ShortConv module yields the
    ``shortconv`` kind with a squeezed [hidden, kernel] conv weight, and
    ``assert_block_supported`` accepts it.

Run:  python tests/test_shortconv.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_qll = _load("qlora_linear")
_fce = _load("fused_ce")
_gdn = _load("gdn")
_sc = _load("short_conv")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA


# ----------------------------------------------------------------------------
# Mock EXL3 linear (same shape as test_gdn's).
# ----------------------------------------------------------------------------
class _MockInner:
    def __init__(self, weight):
        self._w = weight                 # [in, out], frozen
        self.trellis = weight            # device inference
        self.bias = None

    def get_weight_tensor(self):
        return self._w

    def get_bias_tensor(self):
        return None


class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, key, scale=0.05, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.key = key
        self.device = torch.device("cpu")
        w = torch.randn(in_features, out_features, dtype=dtype) * scale
        self.register_buffer("frozen_weight", w)
        self.inner = _MockInner(self.frozen_weight)
        self.lora_a_tensors = {}
        self.lora_b_tensors = {}


def _spec(weight, eps=1e-5, bias=0.0, scale=1.0):
    return {"weight": weight, "eps": eps, "bias": bias, "scale": scale}


def _headless_net():
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = torch.float32
    n.attn_impl = "eager"
    n.use_liger = False
    return n


# ----------------------------------------------------------------------------
# Verbatim transcription of ShortConv._causal_conv1d (modules/short_conv.py)
# with a fresh (zero) conv state and no history -- the whole-sequence case
# training always sees. `weight` arrives as the stored [dim, 1, kernel].
# ----------------------------------------------------------------------------
def _inference_reference_conv(x, weight3, bias):
    bsz, dim, seqlen = x.shape
    kernel = weight3.shape[-1]
    weight = weight3.squeeze(1).contiguous()
    conv_state = torch.zeros((bsz, dim, kernel), dtype=x.dtype)
    ys = []
    for i in range(bsz):
        state = conv_state[i].unsqueeze(0)
        y = torch.cat([state[:, :, :kernel], x[i].unsqueeze(0)], dim=-1)
        y = F.conv1d(y.to(weight.dtype), weight.unsqueeze(1), bias, padding=0, groups=dim)
        ys.append(y[:, :, -seqlen:].to(x.dtype))
    return torch.cat(ys, dim=0).transpose(1, 2).contiguous()   # [b, t, dim]


def _inference_reference_mix(bc_x, weight3, bias):
    """ShortConv.forward between in_proj and out_proj."""
    b, c, x = bc_x.chunk(3, dim=-1)
    conv_input = (b * x).transpose(1, 2).contiguous()
    conv_out = _inference_reference_conv(conv_input, weight3, bias)
    return c * conv_out


def test_conv_matches_inference_reference():
    torch.manual_seed(0)
    b, dim, t, kernel = 2, 10, 9, 3
    x = torch.randn(b, dim, t)
    w3 = torch.randn(dim, 1, kernel) * 0.5
    bias = torch.randn(dim) * 0.1
    out = _sc.shortconv_causal_conv1d(x, w3.squeeze(1), bias).transpose(1, 2)
    ref = _inference_reference_conv(x, w3, bias)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"causal conv mismatch vs inference reference: max|Δ|={err}"
    err2 = (_sc.shortconv_causal_conv1d(x, w3.squeeze(1), None).transpose(1, 2)
            - _inference_reference_conv(x, w3, None)).abs().max().item()
    assert err2 < 1e-6
    print(f"[shortconv] causal conv matches inference reference (max|Δ|={err:.2e}) PASSED")


def test_mix_matches_inference_reference():
    torch.manual_seed(1)
    b, t, d, kernel = 2, 7, 12, 3
    bcx = torch.randn(b, t, 3 * d)
    w3 = torch.randn(d, 1, kernel) * 0.5
    bias = torch.randn(d) * 0.1
    out = _sc.shortconv_mix(bcx, w3.squeeze(1), bias)
    ref = _inference_reference_mix(bcx, w3, bias)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"shortconv mix mismatch vs inference reference: max|Δ|={err}"
    print(f"[shortconv] b|c|x mix matches inference forward (max|Δ|={err:.2e}) PASSED")


# ----------------------------------------------------------------------------
# Full ShortConv block.
# ----------------------------------------------------------------------------
def _build_shortconv_block(d, inter, kernel=3, r=0, dtype=torch.float32):
    lins = {
        "in": MockLinear(d, 3 * d, "blk.conv.in_proj", dtype=dtype),
        "out": MockLinear(d, d, "blk.conv.out_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.feed_forward.w1", dtype=dtype),
        "up": MockLinear(d, inter, "blk.feed_forward.w3", dtype=dtype),
        "down": MockLinear(inter, d, "blk.feed_forward.w2", dtype=dtype),
    }
    norm_a = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)
    norm_m = nn.Parameter(1.0 + 0.02 * torch.randn(d, dtype=dtype), requires_grad=False)

    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a)
    entry.mlp_norm_spec = _spec(norm_m)
    entry.attn_post_spec = None
    entry.mlp_post_spec = None
    entry.in_proj = DiffLinear(lins["in"], r=r, compute_dtype=dtype)
    entry.out_proj = DiffLinear(lins["out"], r=r, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]

    w3 = torch.randn(d, 1, kernel, dtype=dtype) * 0.4
    meta = {
        "kind": "shortconv",
        "hidden_size": d,
        "conv_kernel_size": kernel,
        "conv1d_weight": w3.squeeze(1).clone(),
        "conv1d_bias": torch.randn(d, dtype=dtype) * 0.1,
        "activation": "silu",
        "mlp_kind": "dense",
        "layer_scalar": None,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    ref_w.update({"attn_norm": norm_a, "mlp_norm": norm_m, "conv_w3": w3})
    return entry, meta, ref_w, lins


def _ref_rmsnorm(x, w, eps=1e-5):
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    return (x.float() * torch.rsqrt(var)) * w.float()


def _ref_shortconv_block(meta, w, hidden):
    """Independent composition of the ShortConv block from the transcribed
    inference pieces, sharing no code with training.short_conv."""
    normed = _ref_rmsnorm(hidden, w["attn_norm"])
    bcx = normed @ w["in"]
    y = _inference_reference_mix(bcx, w["conv_w3"], meta["conv1d_bias"])
    hidden = hidden + y @ w["out"]
    normed2 = _ref_rmsnorm(hidden, w["mlp_norm"])
    a = F.silu(normed2 @ w["gate"]) * (normed2 @ w["up"])
    return hidden + a @ w["down"]


def test_shortconv_block_matches_reference():
    torch.manual_seed(2)
    d, inter = 16, 32
    entry, meta, ref_w, _ = _build_shortconv_block(d, inter, r=0)
    net = _headless_net()

    b, t = 2, 6
    hidden = torch.randn(b, t, d)
    out = net._shortconv_forward(meta, entry, hidden)
    ref = _ref_shortconv_block(meta, ref_w, hidden)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"ShortConv block forward mismatch vs reference: max|Δ|={err}"
    print(f"[shortconv] block forward matches plain-torch reference (max|Δ|={err:.2e}) PASSED")


def test_shortconv_block_is_causal():
    torch.manual_seed(3)
    d, inter = 12, 24
    entry, meta, _, _ = _build_shortconv_block(d, inter, r=0)
    net = _headless_net()
    t = 8
    hidden = torch.randn(1, t, d)
    out = net._shortconv_forward(meta, entry, hidden)
    for j in (2, 5, t - 1):
        h2 = hidden.clone()
        h2[:, j:] += torch.randn(1, t - j, d)          # perturb j and everything after
        out2 = net._shortconv_forward(meta, entry, h2)
        assert torch.allclose(out[:, :j], out2[:, :j], atol=1e-6), \
            f"output at positions < {j} changed when positions >= {j} were perturbed"
        assert not torch.allclose(out[:, j], out2[:, j]), "perturbation had no effect"
    print("[shortconv] block is causal (right-padding safe) PASSED")


def test_shortconv_block_backward_reaches_adapters():
    torch.manual_seed(4)
    d, inter = 12, 24
    entry, meta, _, lins = _build_shortconv_block(d, inter, r=4)
    net = _headless_net()

    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    for dl in (entry.in_proj, entry.out_proj):
        with torch.no_grad():
            dl.lora_b.add_(torch.randn_like(dl.lora_b) * 0.02)

    hidden = torch.randn(2, 5, d, requires_grad=True)
    out = net._shortconv_forward(meta, entry, hidden)
    out.square().mean().backward()

    for name, dl in (("in", entry.in_proj), ("out", entry.out_proj)):
        assert dl.lora_a.grad is not None and dl.lora_a.grad.abs().sum() > 0, \
            f"no gradient reached lora_a of {name}"
        assert dl.lora_b.grad is not None and dl.lora_b.grad.abs().sum() > 0, \
            f"no gradient reached lora_b of {name}"
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_]), \
            f"frozen base weight of {k_} changed"
    print("[shortconv] backward reaches in/out adapters; base frozen PASSED")


# ----------------------------------------------------------------------------
# backbone seam: block_metadata / assert_block_supported on a mock module.
# ----------------------------------------------------------------------------
def test_backbone_shortconv_metadata():
    # backbone imports ..modules lazily inside each function; stub the package
    # with just the classes it isinstance-checks so no CUDA ext is built.
    class ShortConv: pass
    class GatedDeltaNet: pass
    class BlockSparseMLP: pass
    class GatedMLP: pass
    class Attention: pass
    class SlidingAttention: pass
    # backbone's `from ..modules import ...` needs a parent package with a
    # `modules` sibling: build `exl3stub` / `exl3stub.training` /
    # `exl3stub.modules` and load backbone as `exl3stub.training.backbone`.
    root = types.ModuleType("exl3stub")
    root.__path__ = []
    train = types.ModuleType("exl3stub.training")
    train.__path__ = [_TRAIN_DIR]
    mods = types.ModuleType("exl3stub.modules")
    for c in (ShortConv, GatedDeltaNet, BlockSparseMLP, GatedMLP, Attention,
              SlidingAttention):
        setattr(mods, c.__name__, c)
    sys.modules["exl3stub"] = root
    sys.modules["exl3stub.training"] = train
    sys.modules["exl3stub.modules"] = mods
    spec = importlib.util.spec_from_file_location(
        "exl3stub.training.backbone", os.path.join(_TRAIN_DIR, "backbone.py"))
    _bb = importlib.util.module_from_spec(spec)
    sys.modules["exl3stub.training.backbone"] = _bb
    spec.loader.exec_module(_bb)

    d, kernel = 8, 3
    attn = ShortConv()
    attn.hidden_size = d
    attn.conv_kernel_size = kernel
    attn.in_proj = object()
    attn.out_proj = object()
    attn.conv1d_weight = torch.randn(d, 1, kernel)
    attn.conv1d_bias = torch.randn(d)
    attn.tp_reduce = False
    mlp = GatedMLP()
    mlp.activation_fn = "silu"
    mlp.act_limit = 0.0
    blk = types.SimpleNamespace(key="model.layers.0", attn=attn, mlp=mlp)

    _bb.assert_block_supported(blk)
    meta = _bb.block_metadata(blk)
    assert meta["kind"] == "shortconv"
    assert meta["mlp_kind"] == "dense" and meta["activation"] == "silu"
    assert meta["conv1d_weight"].shape == (d, kernel)
    assert torch.equal(meta["conv1d_weight"], attn.conv1d_weight.squeeze(1))
    assert torch.equal(meta["conv1d_bias"], attn.conv1d_bias)
    assert _bb.short_conv_projections(blk) == (attn.in_proj, attn.out_proj)

    # Mismatched conv width is rejected loudly, not silently mis-shaped.
    attn.conv1d_weight = torch.randn(d + 1, 1, kernel)
    try:
        _bb.assert_block_supported(blk)
    except AssertionError:
        pass
    else:
        raise AssertionError("conv weight width mismatch was not rejected")
    print("[shortconv] backbone metadata / support check PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util import run_timed
    run_timed([
        test_conv_matches_inference_reference,
        test_mix_matches_inference_reference,
        test_shortconv_block_matches_reference,
        test_shortconv_block_is_causal,
        test_shortconv_block_backward_reaches_adapters,
        test_backbone_shortconv_metadata,
    ], label="shortconv")
    print("\nALL SHORTCONV TESTS PASSED")
