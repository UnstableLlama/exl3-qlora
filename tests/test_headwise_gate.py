"""
Backbone seam checks for the Spark-X2.5 attention layout, on mock modules
(no CUDA extension, no model):

  * ``assert_block_supported`` accepts the HEADWISE sigmoid attention output
    gate (per-head scalar ``g_proj``) and still rejects the softplus variant.
  * ``block_metadata`` reports ``headwise_gate`` / ``full_gate`` correctly and
    converts the module's past-tokens-only ``sliding_window`` (what the
    inference forward hands FlashAttention as ``window_size=(w, 0)``) into the
    native forward's query-inclusive window (``w + 1``), so the training block
    attends to exactly the keys the inference block does.
  * ``attn_gate_linear`` returns the gate for both the headwise and the
    full-width variants, and ``None`` without one.

The block-forward math (per-head sigmoid multiply vs an independent
reference) is covered by ``tests/test_native_llama.py``
(``test_spark_headwise_gate_block_matches_reference``).
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")


def _load_backbone():
    # backbone imports ..modules lazily inside each function; stub the package
    # with just the classes it isinstance-checks so no CUDA ext is built
    # (same recipe as tests/test_shortconv.py).
    class ShortConv: pass
    class GatedDeltaNet: pass
    class BlockSparseMLP: pass
    class GatedMLP: pass
    class Attention: pass
    class SlidingAttention: pass
    root = types.ModuleType("exl3stub_hw")
    root.__path__ = []
    train = types.ModuleType("exl3stub_hw.training")
    train.__path__ = [_TRAIN_DIR]
    mods = types.ModuleType("exl3stub_hw.modules")
    for c in (ShortConv, GatedDeltaNet, BlockSparseMLP, GatedMLP, Attention,
              SlidingAttention):
        setattr(mods, c.__name__, c)
    sys.modules["exl3stub_hw"] = root
    sys.modules["exl3stub_hw.training"] = train
    sys.modules["exl3stub_hw.modules"] = mods
    spec = importlib.util.spec_from_file_location(
        "exl3stub_hw.training.backbone", os.path.join(_TRAIN_DIR, "backbone.py"))
    bb = importlib.util.module_from_spec(spec)
    sys.modules["exl3stub_hw.training.backbone"] = bb
    spec.loader.exec_module(bb)
    return bb, mods


def _spark_block(mods, *, sliding_window=-1, headwise=True, softplus=False,
                 full_gate=False, g_proj=True):
    nq, nkv, hd = 16, 4, 256
    attn = mods.Attention()
    attn.num_q_heads, attn.num_kv_heads, attn.head_dim = nq, nkv, hd
    attn.sm_scale = hd ** -0.5
    attn.sliding_window = sliding_window
    attn.logit_softcapping = 0.0
    attn.use_k_as_v = False
    attn.interleaved_gate = False
    attn.full_gate = full_gate
    attn.headwise_gate = headwise
    attn.gate_softplus = softplus
    attn.g_proj = types.SimpleNamespace(key="model.layers.0.self_attn.g_proj") if g_proj else None
    rope_settings = types.SimpleNamespace(rope_style=types.SimpleNamespace(name="NEOX"))
    rope = types.SimpleNamespace(
        rope_settings=rope_settings,
        # Spark full-attention layers: partial_rotary_factor 0.25 -> 64 of 256
        inv_freq=torch.ones(32), attn_factor=1.0, mrope_section=None)
    attn.rope_settings = rope_settings
    attn.rope = rope
    mlp = mods.GatedMLP()
    mlp.activation_fn = "gelu"
    mlp.act_limit = 0.0
    return types.SimpleNamespace(key="model.layers.0", attn=attn, mlp=mlp)


def test_headwise_gate_accepted_softplus_rejected():
    bb, mods = _load_backbone()
    blk = _spark_block(mods)
    bb.assert_block_supported(blk)
    meta = bb.block_metadata(blk)
    assert meta["kind"] == "attn"
    assert meta["headwise_gate"] is True and meta["full_gate"] is False
    assert meta["activation"] == "gelu" and meta["mlp_kind"] == "dense"
    assert bb.attn_gate_linear(blk) is blk.attn.g_proj

    # Full-width (AFMoE) still reports as before, headwise False.
    blk_full = _spark_block(mods, headwise=False, full_gate=True)
    bb.assert_block_supported(blk_full)
    meta_full = bb.block_metadata(blk_full)
    assert meta_full["full_gate"] is True and meta_full["headwise_gate"] is False
    assert bb.attn_gate_linear(blk_full) is blk_full.attn.g_proj

    # No gate at all.
    blk_none = _spark_block(mods, headwise=False, g_proj=False)
    bb.assert_block_supported(blk_none)
    assert bb.attn_gate_linear(blk_none) is None
    assert bb.block_metadata(blk_none)["headwise_gate"] is False

    # Softplus headwise gate (Laguna) is rejected loudly: the native forward
    # applies sigmoid and must not silently gate with the wrong function.
    blk_sp = _spark_block(mods, softplus=True)
    try:
        bb.assert_block_supported(blk_sp)
    except AssertionError as e:
        assert "softplus" in str(e)
    else:
        raise AssertionError("softplus headwise gate was not rejected")
    print("[headwise-gate] sigmoid accepted / softplus rejected / gate linear "
          "resolution PASSED")


def test_sliding_window_converted_to_query_inclusive():
    bb, mods = _load_backbone()
    # Spark-X2.5 sliding layers: module sliding_window=512 = 512 PREVIOUS
    # tokens (inference: FlashAttention window_size=(512, 0) -> itself + 512
    # past keys). The native forward's window counts the query, so 513.
    meta = bb.block_metadata(_spark_block(mods, sliding_window=512))
    assert meta["sliding_window"] == 513, meta["sliding_window"]
    # Full-attention layers: -1 / None / 0 all mean "no window".
    for sw in (-1, None, 0):
        meta = bb.block_metadata(_spark_block(mods, sliding_window=sw))
        assert meta["sliding_window"] == -1, (sw, meta["sliding_window"])
    print("[headwise-gate] sliding_window past-tokens -> query-inclusive "
          "conversion PASSED")


def test_native_window_semantics_match_inference():
    # The native eager mask with the converted window must let query i see
    # exactly keys i-w .. i (w past + itself), which is what FlashAttention's
    # window_size=(w, 0) -- the inference call -- allows. Checked directly on
    # the mask builder so the convention can't drift silently.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_native_llama as tnl
    net = tnl._headless_net()
    w_past, t = 3, 8
    window_native = w_past + 1                  # what block_metadata emits
    bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32,
                          window=window_native)[0, 0]      # [t, t]
    allowed = torch.isfinite(bias)
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    expect = (j <= i) & (j >= i - w_past)       # FA2 window_size=(w_past, 0)
    assert torch.equal(allowed, expect), (allowed.int(), expect.int())
    print("[headwise-gate] native mask == FlashAttention (w, 0) window PASSED")


def main():
    from util import run_timed
    run_timed([
        test_headwise_gate_accepted_softplus_rejected,
        test_sliding_window_converted_to_query_inclusive,
        test_native_window_semantics_match_inference,
    ], label="headwise-gate")
    print("\nAll headwise-gate / sliding-window backbone checks passed.")


if __name__ == "__main__":
    main()
