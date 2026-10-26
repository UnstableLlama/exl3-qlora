"""
Runtime LoRA inside the captured decode graphs (BC_Attention / BC_GatedMLP / BC_MLP), over every registry model
tagged "dense", with random PEFT adapters on q/k/v/o/gate/up/down (testlib.lora).

test_lora_graph_matches_eager: the graphed decode with an adapter attached reproduces the same C++ launch sequence
run eagerly (EXL3_GRAPHS=0), like every other BC path test. The scenario also exercises the adapter lifecycle in one
process: attach, generate, detach, attach a second adapter, a zero-scale adapter, and back to the base model, so the
graphs re-record on every change (graph structure epoch). Within the graphed run: a zero-scale adapter is bit-exact
with the base model, and returning to a state generated earlier (base again, adapter 0 again) is bit-exact too.

test_lora_graph_matches_unfused: the in-graph application against the known-correct unfused path (EXL3_LORA_GRAPH=0:
the modules decline their graphs while an adapter is attached and Linear.forward applies it). The two paths differ by
the decode path's own precision, so the check is relative to the adapter's effect: on the decode steps whose token
prefix agrees, the two adapted runs must be far closer to each other than either is to the base model.
"""

import os

import pytest
import torch

from testlib.env import get_test_device
from testlib.graph import assert_matches_eager, generate_with_logits, load_with_caches, run_with_and_without_graphs, \
    unload_model
from testlib.isolated import device_env, run_isolated

NEW_TOKENS = 12
PROMPT = "The quick brown fox jumps over the lazy dog because"
LABELS = ["base", "lora0", "lora0_zero", "lora1", "base_again", "lora0_again"]


def scenario(model_dir: str, work_dir: str) -> list:
    """Greedy generations through the adapter lifecycle (child process): [(label, tokens, step logits)]"""
    from exllamav3 import LoRA
    from exllamav3.modules import Attention, GatedMLP, MLP
    from testlib.lora import write_random_adapter

    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    dirs = [write_random_adapter(model, os.path.join(work_dir, f"adapter{i}"), rank = 16, seed = i) for i in range(2)]
    expect_graphed = os.environ.get("EXL3_LORA_GRAPH", "1") != "0"

    def gen():
        ((tokens, logits),) = generate_with_logits(model, cache, tok, [PROMPT], NEW_TOKENS, max_batch_size = 1)
        return tokens, [l.cpu() for l in logits]

    def graphed_projections():
        """How many attention / MLP projections currently carry their adapter in the C++ object"""
        n = 0
        for m in model:
            if isinstance(m, Attention):
                n += sum(1 for p in (m.q_proj, m.k_proj, m.v_proj, m.o_proj)
                         if p is not None and p.quant_type == "exl3" and p.inner.bc.has_lora())
            elif isinstance(m, (GatedMLP, MLP)):
                n += sum(1 for p in (*getattr(m, "gates", []), *m.ups, *m.downs)
                         if p.quant_type == "exl3" and p.inner.bc.has_lora())
        return n

    out = []
    out.append(("base", *gen()))

    lora0 = LoRA.from_directory(model, dirs[0])
    out.append(("lora0", *gen()))
    n_graphed = graphed_projections()
    bc_attn_built = any(v for m in model if isinstance(m, Attention) for v in getattr(m, "bc_attn", {}).values())
    bc_mlp_built = any(getattr(m, "bc", None) is not None for m in model if isinstance(m, (GatedMLP, MLP)))
    lora0.unload()

    lora0_zero = LoRA.from_directory(model, dirs[0], lora_scaling = 0.0)
    out.append(("lora0_zero", *gen()))
    lora0_zero.unload()

    lora1 = LoRA.from_directory(model, dirs[1])
    out.append(("lora1", *gen()))
    lora1.unload()

    out.append(("base_again", *gen()))
    assert graphed_projections() == 0, "detached adapter still held by a C++ object"

    lora0 = LoRA.from_directory(model, dirs[0])
    out.append(("lora0_again", *gen()))
    lora0.unload()

    unload_model(model)

    # Vacuity checks: the adapter really went through the graphs (or really stayed out of them)
    assert bc_attn_built or bc_mlp_built, "model built no BC decode path; test is vacuous"
    if expect_graphed:
        assert n_graphed > 0, "no projection handed its adapter to the graph path"
    else:
        assert n_graphed == 0, "EXL3_LORA_GRAPH=0 but a projection handed its adapter to the graph path"
    return out


def _rfn(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().view(-1), b.float().view(-1)
    m = torch.isfinite(a) & torch.isfinite(b)
    return ((a[m] - b[m]).norm() / b[m].norm()).item()


def _by_label(runs) -> dict:
    d = {label: (tokens, logits) for label, tokens, logits in runs}
    assert list(d) == LABELS, list(d)
    return d


def _assert_identical(d, label_a, label_b):
    (ta, la), (tb, lb) = d[label_a], d[label_b]
    assert ta == tb, f"{label_a} vs {label_b}: tokens differ ({ta} vs {tb})"
    for i, (x, y) in enumerate(zip(la, lb)):
        assert torch.equal(x, y), f"{label_a} vs {label_b}: step {i} logits differ (max abs {(x - y).abs().max():.3e})"


@pytest.mark.models("dense")
def test_lora_graph_matches_eager(model_id, model_dir, device, tmp_path):
    graphed, eager = run_with_and_without_graphs(scenario, model_dir, str(tmp_path), device = device)
    assert_matches_eager([(f"{model_id} [{l}]", t, g) for l, t, g in graphed],
                         [(f"{model_id} [{l}]", t, g) for l, t, g in eager], min_steps = 4)

    d = _by_label(graphed)
    # Lifecycle: the graphs re-record on attach/detach and come back to exactly the same kernels
    _assert_identical(d, "base", "base_again")
    _assert_identical(d, "lora0", "lora0_again")
    # Zero scale: B is zero, the delta nodes add exactly nothing
    _assert_identical(d, "base", "lora0_zero")
    # The adapters have an effect (the first step's logits come from the prefill, which applies them too)
    assert _rfn(d["lora0"][1][0], d["base"][1][0]) > 1e-3, "adapter 0 had no visible effect"
    assert _rfn(d["lora1"][1][0], d["lora0"][1][0]) > 1e-3, "adapters 0 and 1 are indistinguishable"


@pytest.mark.models("dense")
def test_lora_graph_matches_unfused(model_id, model_dir, device, tmp_path):
    env = {**device_env(device), "EXL3_GRAPHS": "1"}
    graphed = _by_label(run_isolated(scenario, model_dir, str(tmp_path / "graph"), env = {**env, "EXL3_LORA_GRAPH": "1"}))
    unfused = _by_label(run_isolated(scenario, model_dir, str(tmp_path / "unfused"), env = {**env, "EXL3_LORA_GRAPH": "0"}))

    # Without an adapter the two runs take the same path
    _assert_identical({"g": graphed["base"], "u": unfused["base"]}, "g", "u")

    for label in ("lora0", "lora1"):
        (tok_g, lg_g), (tok_u, lg_u) = graphed[label], unfused[label]
        # Step 0 is the prefill's output (the unfused python path in both runs): the adapter's effect at the
        # same input is the scale the decode-step discrepancy is measured against
        effect = _rfn(lg_u[0], unfused["base"][1][0])
        assert effect > 1e-3, f"{label}: adapter had no visible effect"
        agree = 0
        for i in range(1, min(len(tok_g), len(tok_u))):
            if tok_g[:i] != tok_u[:i]:
                break
            diff = _rfn(lg_g[i], lg_u[i])
            assert diff <= 0.25 * effect, \
                f"{model_id} [{label}] step {i}: in-graph vs unfused LoRA rfn {diff:.4f} (adapter effect {effect:.4f})"
            agree += 1
        assert agree >= 3, f"{model_id} [{label}]: decode tokens diverged too early ({tok_g} vs {tok_u})"
