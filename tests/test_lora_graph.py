"""
GPU tests for in-graph runtime LoRA (doc/lora_inference_plan.md, stage 1).

  1. test_lora_gemv_kernel -- the cooperative lora_gemv kernel against the
     pure-torch contract (modules/lora_state.py::lora_delta_reference) over
     the shapes the BC slots produce (M 1..128, rank 8..256, fp16/fp32 C).
     Needs only CUDA + the built extension.
  2. test_graph_lora_parity -- end to end on a real quant + adapter: bsz-1
     decode logits with the adapter inside the graphs (EXL3_LORA_GRAPH=1)
     match the unfused fallback (EXL3_LORA_GRAPH=0) token for token; the
     no-adapter generation is byte-identical before / after; unload restores
     it; a second attach re-engages the graphs. Set EXL3_TEST_MODEL and
     EXL3_TEST_ADAPTER (a Llama-family 4bpw + a q/k/v/o/gate/up/down adapter
     is the reference case).

Run on the box:
    EXL3_TEST_MODEL=/path/quant EXL3_TEST_ADAPTER=/path/adapter \\
        python -m pytest tests/test_lora_graph.py -q -s
"""

from __future__ import annotations
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exllamav3.ext import exllamav3_ext as ext  # noqa: E402
from exllamav3.modules.lora_state import lora_delta_reference  # noqa: E402


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 128])
@pytest.mark.parametrize("rank", [8, 32, 64, 256])
@pytest.mark.parametrize("kn", [(2048, 2048), (2048, 512), (8192, 2048), (2048, 8192)])
@pytest.mark.parametrize("c_dtype", [torch.float16, torch.float32])
def test_lora_gemv_kernel(m, rank, kn, c_dtype):
    k, n = kn
    g = torch.Generator(device="cuda").manual_seed(m * 1000 + rank)
    x = torch.randn(m, k, device="cuda", generator=g).half()
    a = (torch.randn(k, rank, device="cuda", generator=g) * 0.05).half()
    b = (torch.randn(rank, n, device="cuda", generator=g) * 0.05).half()
    c0 = torch.randn(m, n, device="cuda", generator=g).to(c_dtype)
    c_kernel = c0.clone()
    ext.lora_gemv(x, a, b, c_kernel)
    torch.cuda.synchronize()
    c_ref = lora_delta_reference(x, a, b, c0.clone())
    tol = 2e-3 if c_dtype == torch.float16 else 1e-4
    err = (c_kernel.float() - c_ref.float()).abs().max().item()
    assert err <= tol * max(1.0, c_ref.float().abs().max().item()), \
        f"lora_gemv mismatch: max err {err:.3g} (M={m}, R={rank}, K={k}, N={n}, {c_dtype})"
    # vacuousness: the delta must be visible
    assert (c_ref - c0).abs().max().item() > 1e-3


def _decode(model, cache, tokenizer, prompt, n_tokens):
    """Greedy bsz-1 decode through model.forward with the paged cache -- the
    same params eval/perf.py's measure_generate uses (prefill of the prompt,
    then one token per forward, past_len advancing), which is the path the
    BC graphs serve. Returns (token ids, per-step last-position logits)."""
    ids = tokenizer.encode(prompt)
    out_ids = []
    logits_all = []
    pos = 0
    max_len = ((ids.shape[-1] + n_tokens + 255) // 256) * 256
    cur = ids
    for i in range(n_tokens):
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": pos,
            "batch_shape": (1, max_len),
        }
        logits = model.forward(cur, params)[:, -1, :].float()
        pos += cur.shape[-1]
        tok = int(torch.argmax(logits, dim=-1).item())
        out_ids.append(tok)
        logits_all.append(logits.cpu())
        cur = torch.tensor([[tok]], dtype=torch.long)
    return out_ids, torch.cat(logits_all, dim=0)


def test_graph_lora_parity():
    model_dir = os.environ.get("EXL3_TEST_MODEL")
    adapter_dir = os.environ.get("EXL3_TEST_ADAPTER")
    if not model_dir or not adapter_dir:
        pytest.skip("set EXL3_TEST_MODEL and EXL3_TEST_ADAPTER to run")

    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.model.lora import LoRA
    from exllamav3.modules import linear as linear_mod

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(device="cuda:0")
    tokenizer = Tokenizer.from_config(config)
    prompt = "The quick brown fox jumps over the lazy dog because"
    n = 32

    # 1. Base, twice (graph warmup + capture happen on the first passes): byte-identical.
    base_ids_a, _ = _decode(model, cache, tokenizer, prompt, n)
    base_ids_b, base_logits = _decode(model, cache, tokenizer, prompt, n)
    assert base_ids_a == base_ids_b

    # 2. Adapted, in-graph vs fallback.
    lora = LoRA.from_directory(model, adapter_dir)
    linear_mod._lora_graph_enable = True
    g_ids_1, g_logits_1 = _decode(model, cache, tokenizer, prompt, n)   # flavour flip + capture
    g_ids, g_logits = _decode(model, cache, tokenizer, prompt, n)       # replay
    linear_mod._lora_graph_enable = False
    f_ids, f_logits = _decode(model, cache, tokenizer, prompt, n)
    linear_mod._lora_graph_enable = True

    assert g_ids_1 == g_ids, "in-graph LoRA: first (capture) and replay decodes differ"
    assert g_ids == f_ids, f"in-graph LoRA tokens differ from fallback:\n{g_ids}\n{f_ids}"
    diff = (g_logits - f_logits).abs().max().item()
    assert diff < 0.05 * max(1.0, f_logits.abs().max().item()), \
        f"in-graph vs fallback logits differ by {diff:.4g}"
    # vacuousness: the adapter must actually change something
    assert (f_logits - base_logits).abs().max().item() > 1e-2, \
        "adapter has no visible effect (vacuous parity)"

    # 3. Unload restores the base bit-for-bit (flavour flips back, graphs recapture).
    lora.unload()
    u_ids_1, _ = _decode(model, cache, tokenizer, prompt, n)
    u_ids, u_logits = _decode(model, cache, tokenizer, prompt, n)
    assert u_ids_1 == u_ids == base_ids_a, "unload did not restore the base generation"
    assert torch.equal(u_logits, base_logits), "unload: base logits not bit-identical"

    # 4. Re-attach: the graphs re-engage and agree with the earlier in-graph run.
    lora2 = LoRA.from_directory(model, adapter_dir)
    r_ids_1, _ = _decode(model, cache, tokenizer, prompt, n)
    r_ids, _ = _decode(model, cache, tokenizer, prompt, n)
    assert r_ids_1 == r_ids == g_ids, "re-attached adapter disagrees with the first in-graph run"
    lora2.unload()
