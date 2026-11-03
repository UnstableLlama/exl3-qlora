"""
CPU tests for the differentiable PLE (per-layer embedding) layer of
Qwen3.8-Flash-Next (``exllamav3/training/ple.py``) and its wiring into the
native trunk forward (``native_llama._run_ple`` / ``backbone.ple_ngram_embed``).

No GPU, no compiled extension, no real model (and no 51B-row n-gram table):
the training modules are loaded under a synthetic package, the key/value
projections are mocked EXL3 linears, the n-gram embedding is a mock that
returns random rows, and the checks are:

  * ``ple_delta`` matches a verbatim transcription of the inference module's
    op-by-op reference (``PLELayer.forward_streams_reference`` +
    ``_short_conv`` from a zero conv state, with the ``ple_gate`` kernel's
    signed-sqrt sigmoid written out) on random weights;
  * the gate's signed sqrt is zero-safe and the whole delta gradchecks in
    fp64 w.r.t. the stream stack;
  * the dilated depthwise conv is causal with the right taps: perturbing one
    position moves only that position and the ``dilation``-spaced positions
    after it, never anything earlier;
  * the trunk wiring: ``_run_ple`` builds the eos-padded hashing history the
    inference layer builds for a fresh sequence, adds the delta into the
    streams in place of the inference ``x + delta``, and checkpointing
    reproduces the unchecked result.

Run:  python tests/test_ple_train.py
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
_ple = _load("ple")
_bb = _load("backbone")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA


class _MockInner:
    def __init__(self, weight):
        self._w = weight
        self.trellis = weight
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


def _headless_net(dtype=torch.float32):
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = dtype
    n.attn_impl = "eager"
    n.use_liger = False
    return n


def _grouped_spec(H, d, dtype=torch.float32, eps=1e-6):
    """backbone.norm_spec of a grouped RMSNorm with constant_bias 1.0: weight
    viewed [H, d] (zero-init on the real model; random here so it matters)."""
    w = (0.1 * torch.randn(H, d)).to(dtype)
    return {"weight": w, "eps": eps, "bias": 1.0, "scale": 1.0}


def _make_ple(H, d, ple_dim, kernel=4, ngram=3, dtype=torch.float32, scale=0.3):
    """A PLE entry (spec + frozen DiffLinear projections) with random weights,
    plus the raw tensors for the reference."""
    lins = {
        "key": MockLinear(ple_dim, H * d, "ple.key_proj", scale=scale, dtype=dtype),
        "value": MockLinear(ple_dim, d, "ple.value_proj", scale=scale, dtype=dtype),
    }
    conv_w = (torch.randn(H * d, 1, kernel) * 0.4).to(dtype)
    spec = {
        "key": "ple", "hc_mult": H, "hidden_size": d,
        "gate_scale": 1.0 / (d ** 0.5), "dilation": ngram, "kernel": kernel,
        "state_len": (kernel - 1) * ngram, "conv_w": conv_w,
        "norm_key": _grouped_spec(H, d, dtype), "norm_query": _grouped_spec(H, d, dtype),
        "norm_conv": _grouped_spec(H, d, dtype),
    }
    pe = types.SimpleNamespace()
    pe.spec = spec
    pe.key_proj = DiffLinear(lins["key"], r=0, compute_dtype=dtype)
    pe.value_proj = DiffLinear(lins["value"], r=0, compute_dtype=dtype)
    pe.device = torch.device("cpu")
    return pe, lins


# ----------------------------------------------------------------------------
# Transcription of the inference reference: PLELayer.forward_streams_reference
# + _short_conv (conv_state None) + the ple_gate kernel, with RMSNorm.forward
# (grouped: rms per stream row, weight row per stream, applied as 1 + w).
# ----------------------------------------------------------------------------
def _ref_grouped_rmsnorm(x, spec):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True) + spec["eps"]
    return xf * torch.rsqrt(var) * (spec["weight"].float() + spec["bias"])


def _ref_ple_gate(gate, value, gate_scale):
    # ple_gate_kernel: g = gate * scale; a = sqrt(max(|g|, 1e-6));
    # ss = a if g > 0 else (-a if g < 0 else 0); s = sigmoid(ss); out = s * value
    g = gate.float() * gate_scale
    a = torch.sqrt(torch.clamp(g.abs(), min=1e-6))
    ss = torch.where(g > 0, a, torch.where(g < 0, -a, torch.zeros_like(a)))
    s = torch.sigmoid(ss)
    return s.unsqueeze(-1) * value.float().unsqueeze(-2)


def _ref_short_conv(x, conv_w, dilation, state_len):
    bsz, seq, ch = x.shape
    xt = x.transpose(1, 2)
    conv_state = xt.new_zeros((bsz, ch, state_len))
    xt = torch.cat((conv_state.to(xt.dtype), xt), dim=-1)
    y = F.conv1d(xt, conv_w.to(xt.dtype), groups=ch, dilation=dilation)
    return F.silu(y).transpose(1, 2)


def _ref_ple(streams, emb, spec, w_key, w_value):
    bsz, seq = streams.shape[:2]
    H, D = spec["hc_mult"], spec["hidden_size"]
    key = (emb @ w_key).view(bsz, seq, H, D)
    key = _ref_grouped_rmsnorm(key, spec["norm_key"])
    value = emb @ w_value
    query = _ref_grouped_rmsnorm(streams, spec["norm_query"])
    gate = torch.bmm(query.reshape(-1, 1, D), key.reshape(-1, D, 1)).view(bsz, seq, H)
    gated = _ref_ple_gate(gate, value, spec["gate_scale"])
    normed = _ref_grouped_rmsnorm(gated, spec["norm_conv"]).flatten(-2)
    conv_out = _ref_short_conv(normed, spec["conv_w"], spec["dilation"], spec["state_len"])
    return gated + conv_out.view(bsz, seq, H, D)


def test_ple_delta_matches_reference():
    torch.manual_seed(0)
    b, t, H, d, ple_dim = 2, 13, 4, 16, 24
    pe, lins = _make_ple(H, d, ple_dim)
    net = _headless_net()
    streams = torch.randn(b, t, H, d) * 2
    emb = torch.randn(b, t, ple_dim)
    out = _ple.ple_delta(streams, emb, pe.spec, pe.key_proj, pe.value_proj,
                         net._norm, torch.float32)
    ref = _ref_ple(streams, emb, pe.spec, lins["key"].frozen_weight,
                   lins["value"].frozen_weight)
    assert out.shape == streams.shape and out.dtype == torch.float32
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"ple_delta mismatch vs inference reference: max|d|={err}"
    assert out.abs().max() > 1e-2
    # The conv term must contribute: with a zero conv kernel (silu(0) = 0) the
    # delta is exactly the gated values, and that differs from the full delta.
    spec0 = dict(pe.spec, conv_w=torch.zeros_like(pe.spec["conv_w"]))
    gated_only = _ple.ple_delta(streams, emb, spec0, pe.key_proj, pe.value_proj,
                                net._norm, torch.float32)
    key = _ref_grouped_rmsnorm((emb @ lins["key"].frozen_weight).view(b, t, H, d),
                               pe.spec["norm_key"])
    query = _ref_grouped_rmsnorm(streams, pe.spec["norm_query"])
    gated = _ref_ple_gate((query * key).sum(-1), emb @ lins["value"].frozen_weight,
                          pe.spec["gate_scale"])
    assert (gated_only - gated).abs().max().item() < 1e-5
    assert not torch.allclose(out, gated_only, atol=1e-3)
    print(f"[ple] ple_delta matches forward_streams_reference transcription "
          f"(max|d|={err:.2e}) PASSED")


def test_ple_gate_zero_safe_and_gradcheck():
    # signed sqrt: zero at zero, odd, finite gradient everywhere.
    g = torch.tensor([-4.0, -1e-9, 0.0, 1e-9, 4.0])
    s = _ple.ple_signed_sqrt_gate(g, 1.0)
    assert s[2].item() == 0.5
    assert torch.allclose(s[0], torch.sigmoid(torch.tensor(-2.0)))
    assert torch.allclose(s[4], torch.sigmoid(torch.tensor(2.0)))
    assert torch.allclose(s[0] + s[4], torch.tensor(1.0))
    gg = g.clone().requires_grad_(True)
    _ple.ple_signed_sqrt_gate(gg, 1.0).sum().backward()
    assert torch.isfinite(gg.grad).all()

    torch.manual_seed(1)
    b, t, H, d, ple_dim = 1, 5, 2, 6, 8
    pe, _ = _make_ple(H, d, ple_dim, dtype=torch.float64)
    net = _headless_net(torch.float64)
    streams = torch.randn(b, t, H, d, dtype=torch.float64, requires_grad=True)
    emb = torch.randn(b, t, ple_dim, dtype=torch.float64)

    def fn(s):
        return _ple.ple_delta(s, emb, pe.spec, pe.key_proj, pe.value_proj,
                              net._norm, torch.float64)

    assert torch.autograd.gradcheck(fn, (streams,), eps=1e-6, atol=1e-5)
    print("[ple] signed-sqrt gate zero-safe; ple_delta gradcheck (fp64) PASSED")


def test_ple_conv_causal_dilated_taps():
    torch.manual_seed(2)
    b, t, ch, kernel, dil = 1, 16, 6, 4, 3
    conv_w = torch.randn(ch, 1, kernel) * 0.5
    x = torch.randn(b, t, ch)
    base = _ple.ple_dilated_causal_conv_silu(x, conv_w, dil)
    assert base.shape == x.shape
    p = 4
    x2 = x.clone()
    x2[:, p] += 1.0
    moved = ((_ple.ple_dilated_causal_conv_silu(x2, conv_w, dil) - base).abs()
             .amax(dim=(0, 2)) > 1e-6)
    expect = torch.zeros(t, dtype=torch.bool)
    for k in range(kernel):
        if p + k * dil < t:
            expect[p + k * dil] = True
    assert torch.equal(moved, expect), f"conv taps {moved.nonzero().flatten().tolist()}"
    # Zero left state: position 0 sees only itself through the last tap.
    y0 = F.silu(x[:, 0] * conv_w[:, 0, -1])
    assert torch.allclose(base[:, 0], y0, atol=1e-6)
    print("[ple] dilated depthwise conv is causal with taps at 0/3/6/9 and a zero state PASSED")


# ----------------------------------------------------------------------------
# Trunk wiring: _run_ple + backbone.ple_ngram_embed over a mock inference layer.
# ----------------------------------------------------------------------------
class _MockNGram:
    """Stand-in for NGramEmbedding: ``H`` heads of ``head_dim`` each, position
    ``t`` of a sequence hashing to table rows ``ids[t] * H + h`` (a stand-in
    for the n-gram hashing: deterministic, one row per head per token)."""

    def __init__(self, ctx, eos, ple_dim, V, heads=1):
        self.context_len = ctx
        self.eos_token_id = eos
        self.num_heads = heads
        self.head_dim = ple_dim // heads
        self.key = "model.ple.ple_embedding.ngram_embedding"
        self.table = torch.randn(V * heads, self.head_dim)       # [rows, head_dim]
        self.rows = self.table.view(V, ple_dim)                   # token -> its H rows
        self.calls = []
        self.row_delta = None

    def compute_ngram_ids(self, history, out_len):
        ids = history[:, -out_len:]
        return ids.unsqueeze(-1) * self.num_heads + torch.arange(self.num_heads)

    def forward(self, history, params):
        self.calls.append((history.clone(), dict(params)))
        # Row per position from the current token (a stand-in for the hashed
        # n-gram gather); returns the last seq positions, half like the module.
        ids = history[:, self.context_len:]
        return self.rows[ids].half()

    def set_row_delta(self, uids, delta, scale=1.0):
        self.row_delta = (uids.clone(), delta.clone(), scale)

    def clear_row_delta(self):
        self.row_delta = None


class _MockPLEModule:
    """The slice of PLELayer that backbone.ple_ngram_embed touches."""

    def __init__(self, ngram, mm_token_id=None):
        self.ple_embedding = ngram
        self.mm_token_id = mm_token_id

    def _prepare_ids(self, ids):
        return ids.to("cpu", torch.int64)

    def _history(self, ids):
        pad = ids.new_full((ids.shape[0], self.ple_embedding.context_len),
                           self.ple_embedding.eos_token_id)
        return torch.cat((pad, ids), dim=1)


def test_run_ple_wiring_and_history():
    torch.manual_seed(3)
    b, t, H, d, ple_dim, V, eos, ngram = 2, 9, 4, 16, 24, 50, 7, 3
    pe, lins = _make_ple(H, d, ple_dim, ngram=ngram)
    ng = _MockNGram(ngram - 1, eos, ple_dim, V)
    mod = _MockPLEModule(ng)
    net = _headless_net()
    nn.Module.__init__(net)
    net.compute_dtype = torch.float32
    net._ple_modules = {1: mod}
    ids = torch.randint(0, V, (b, t))
    streams = torch.randn(b, t, H, d)

    out = net._run_ple(1, pe, streams, ids, ckpt=False)
    # The hashing history is the fresh-sequence one: ngram_size - 1 eos tokens
    # in front of the ids (PLELayer._history), built once per forward.
    assert len(ng.calls) == 1
    hist, hist_params = ng.calls[0]
    assert hist_params.get("ngram_base_rows"), "training lookup must ask for base rows"
    assert hist.shape == (b, ngram - 1 + t) and hist.dtype == torch.int64
    assert (hist[:, :ngram - 1] == eos).all() and torch.equal(hist[:, ngram - 1:], ids)

    emb = ng.rows[ids].half()
    ref = streams + _ref_ple(streams, emb.float(), pe.spec, lins["key"].frozen_weight,
                             lins["value"].frozen_weight)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"_run_ple mismatch: max|d|={err}"

    # Checkpointed path reproduces the unchecked one and still backprops into
    # the streams (the lookup runs once, outside the checkpoint).
    net.train()
    s = streams.clone().requires_grad_(True)
    out_c = net._run_ple(1, pe, s, ids, ckpt=True)
    assert torch.allclose(out_c, out, atol=1e-6)
    out_c.pow(2).mean().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert len(ng.calls) == 2
    print(f"[ple] _run_ple: eos-padded history, delta added into the streams "
          f"(max|d|={err:.2e}), checkpoint parity + backward PASSED")


def _load_util(name):
    spec = importlib.util.spec_from_file_location(
        f"exl3util_{name}", os.path.join(_ROOT, "exllamav3", "util", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_row_delta_helper():
    # util/row_delta: slot lookup against a sorted id set and the out-of-place
    # add of the matching deltas (the inference module's decode hook and the
    # trainer's slot lookup share it).
    rd = _load_util("row_delta")
    table_uids = torch.tensor([3, 7, 10, 42])
    uids = torch.tensor([[7, 8], [42, 3], [100, 0]])
    slots = rd.row_slots(uids, table_uids)
    assert slots.tolist() == [[1, -1], [3, 0], [-1, -1]]
    assert rd.row_slots(uids, torch.zeros(0, dtype=torch.long)).eq(-1).all()
    delta = torch.arange(4 * 5, dtype=torch.float32).view(4, 5)
    rows = torch.zeros(6, 5, dtype=torch.float16)
    out = rd.apply_row_delta(rows, uids.reshape(-1), table_uids, delta, scale=0.5)
    assert out.dtype == torch.float16 and rows.abs().sum() == 0      # out of place
    exp = torch.zeros(6, 5)
    exp[0] = 0.5 * delta[1]; exp[2] = 0.5 * delta[3]; exp[3] = 0.5 * delta[0]
    assert torch.equal(out.float(), exp)
    # No hit: the same tensor comes back.
    same = rd.apply_row_delta(rows, torch.tensor([1, 2]), table_uids, delta)
    assert same is rows
    print("[ple] row_delta slots + out-of-place add PASSED")


def _rows_net(pe, mod, train_rows=True):
    net = _headless_net()
    nn.Module.__init__(net)
    net.compute_dtype = torch.float32
    net._adapters_off = False
    net._ple_modules = {1: mod}
    net._ple = nn.ModuleDict({"1": pe})
    net.train_ple = False
    net.train_ngram_rows = train_rows
    net.ngram_uids = net.ngram_rows = net._ngram_uids_dev = None
    net._ngram_base_norm = net._ngram_pending = None
    net._ple_init = {}
    return net


def test_ngram_rows_delta_training():
    # --train-ngram-rows: the row set is the training set's hashed rows; the
    # forward adds the delta of every (position, head) whose row is in the
    # set and leaves the others frozen; only touched slots get gradient; the
    # reference (adapters off) forward is the pure frozen lookup; a pending
    # load lands in its slots; stats count seen / moved rows.
    torch.manual_seed(5)
    b, t, H, d, heads, V, eos, ngram = 2, 7, 4, 16, 3, 30, 29, 3
    ple_dim = 24
    ns, lins = _make_ple(H, d, ple_dim, ngram=ngram)
    pe = nn.Module()                      # the net's entry is an nn.Module (ModuleDict member)
    pe.spec, pe.key_proj, pe.value_proj, pe.device = ns.spec, ns.key_proj, ns.value_proj, ns.device
    w_key, w_val = lins["key"].frozen_weight, lins["value"].frozen_weight
    ng = _MockNGram(ngram - 1, eos, ple_dim, V, heads=heads)
    mod = _MockPLEModule(ng)
    net = _rows_net(pe, mod)

    train_ids = [[1, 2, 3, 4], [4, 5, 6, 7, 1]]          # tokens 1..7 -> 7 * heads rows
    examples = [{"input_ids": x} for x in train_ids]
    # A pending resumed delta (rows of token 9, head 0) joins the set.
    pend_uid = torch.tensor([9 * heads])
    pend_delta = torch.full((1, ng.head_dim), 0.25)
    net._ngram_pending = (pend_uid, pend_delta)
    n = net.init_ngram_rows(examples, max_rows=1000)
    assert n == 7 * heads + 1 and net.ngram_rows.shape == (n, ng.head_dim)
    assert net.ngram_uids.tolist() == sorted(set(i * heads + h for i in range(1, 8) for h in range(heads)) | {9 * heads})
    slot9 = int((net.ngram_uids == 9 * heads).nonzero())
    assert torch.allclose(net.ngram_rows[slot9], pend_delta[0]) and net._ngram_pending is None
    # The cap is enforced with a clear message.
    net2 = _rows_net(pe, mod)
    try:
        net2.init_ngram_rows(examples, max_rows=3)
        assert False, "cap not enforced"
    except ValueError as e:
        assert "ngram_rows_max" in str(e)

    ids = torch.tensor([[1, 2, 20, 4, 9, 3, 21], [7, 7, 22, 1, 5, 6, 2]])   # 20-22 untracked
    streams = torch.randn(b, t, H, d)
    frozen = net._run_ple(1, pe, streams.clone(), ids, ckpt=False).detach()
    # Zero deltas except the pending row: positions with token 9 / head 0 shift.
    emb_ref = ng.rows[ids].half().float().view(b, t, heads, ng.head_dim).clone()
    emb_ref[0, 4, 0] += 0.25
    ref = streams + _ref_ple(streams, emb_ref.view(b, t, ple_dim), pe.spec, w_key, w_val)
    assert (frozen - ref).abs().max() < 1e-5

    # Random deltas: every tracked (token, head) moves, untracked ones do not.
    with torch.no_grad():
        net.ngram_rows.normal_()
    out = net._run_ple(1, pe, streams.clone(), ids, ckpt=False)
    slots = net._ngram_slot_lookup(ng.compute_ngram_ids(mod._history(ids), t))
    emb_ref = ng.rows[ids].half().float().view(b, t, heads, ng.head_dim).clone()
    hit = slots >= 0
    emb_ref[hit] += net.ngram_rows.detach()[slots[hit]]
    ref = streams + _ref_ple(streams, emb_ref.view(b, t, ple_dim), pe.spec, w_key, w_val)
    assert (out - ref).abs().max() < 1e-4, (out - ref).abs().max()
    # Gradient reaches exactly the touched slots.
    out.pow(2).mean().backward()
    g = net.ngram_rows.grad
    touched = torch.zeros(n, dtype=torch.bool); touched[slots[hit].unique()] = True
    assert (g[touched].abs().sum(-1) > 0).all() and g[~touched].abs().sum() == 0
    net.ngram_rows.grad = None
    # Stats: the seen slots are the touched ones; all rows moved (random init).
    st = net.ngram_row_stats()
    assert st["rows"] == n and st["seen"] == int(touched.sum()) and st["moved"] == n
    assert st["rel_move_mean"] > 0 and st["rel_move_max"] >= st["rel_move_mean"]
    # Adapters off (DPO reference): the pure frozen lookup, no delta.
    net._adapters_off = True
    off = net._run_ple(1, pe, streams.clone(), ids, ckpt=False)
    ref0 = streams + _ref_ple(streams, ng.rows[ids].half().float(), pe.spec, w_key, w_val)
    assert (off - ref0).abs().max() < 1e-5
    net._adapters_off = False
    # A batch of only untracked rows keeps the rows on the graph (zero grad, defined).
    out_u = net._run_ple(1, pe, streams[:, :2].clone(), torch.tensor([[20, 21], [22, 20]]), ckpt=False)
    assert out_u.requires_grad
    out_u.sum().backward()
    assert net.ngram_rows.grad is not None and net.ngram_rows.grad.abs().sum() == 0
    net.ngram_rows.grad = None
    # Param groups: the rows' own no-decay group at their LR.
    net._wrappers = []
    net.embed_lora_a = net.head_lora_a = None
    net.token_rows = net.head_token_rows = None
    groups = net.lora_param_groups(0.1, base_lr=1e-4, ngram_rows_lr=5e-3)
    assert groups[-1]["name"] == "ngram_rows" and groups[-1]["lr"] == 5e-3 \
        and groups[-1]["weight_decay"] == 0.0 and groups[-1]["params"] == [net.ngram_rows]
    # Live sample: the deltas go to the table (scaled), and come back out.
    net._apply_ple_to_native(0.5)
    uids_t, delta_t, sc = ng.row_delta
    assert torch.equal(uids_t, net.ngram_uids) and sc == 0.5 and torch.allclose(delta_t, net.ngram_rows.detach())
    net._native_token_rows_backup = None
    net.remove_from_native()
    assert ng.row_delta is None
    print(f"[ple] n-gram row deltas: {n} slots, tracked rows move / untracked frozen, "
          f"grads on touched slots only, pending load, stats, live install PASSED")


def test_train_ple_params_get_grad():
    # --train-ple: the three grouped norm weights and the conv kernel as fp32
    # Parameters in the spec; the differentiable delta reaches all of them.
    torch.manual_seed(6)
    b, t, H, d, ple_dim = 2, 6, 3, 8, 12
    pe, _ = _make_ple(H, d, ple_dim)
    params = {}
    for k in ("norm_key", "norm_query", "norm_conv"):
        params[k] = nn.Parameter(pe.spec[k]["weight"].clone())
        pe.spec[k]["weight"] = params[k]
    params["conv_w"] = nn.Parameter(pe.spec["conv_w"].clone())
    pe.spec["conv_w"] = params["conv_w"]
    net = _headless_net()
    streams = torch.randn(b, t, H, d)
    emb = torch.randn(b, t, ple_dim)
    out = _ple.ple_delta(streams, emb, pe.spec, pe.key_proj, pe.value_proj, net._norm, torch.float32)
    out.pow(2).mean().backward()
    for k, p in params.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, k
    print("[ple] train_ple: norm weights + conv kernel receive gradient PASSED")


def main():
    from util import run_timed
    run_timed([
        test_ple_delta_matches_reference,
        test_ple_gate_zero_safe_and_gradcheck,
        test_ple_conv_causal_dilated_taps,
        test_run_ple_wiring_and_history,
        test_row_delta_helper,
        test_ngram_rows_delta_training,
        test_train_ple_params_get_grad,
    ], label="ple-train")
    print("\nAll PLE training checks passed.")


if __name__ == "__main__":
    main()
