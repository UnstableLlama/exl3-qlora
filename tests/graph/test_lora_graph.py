"""
Runtime LoRA inside the captured decode graphs (BC_Attention / BC_GatedMLP / BC_MLP / BC_GatedDeltaNetSplit), over
every registry model tagged "dense" and every hybrid tagged "gdn", with random PEFT adapters on the attention, MLP
and GDN projections (testlib.lora). MoE hybrids keep their routed experts unadapted here (the fused expert kernels
have no LoRA stage; an adapter there takes the per-expert path, which is a different test).

test_lora_graph_matches_eager: the graphed decode with an adapter attached reproduces the same C++ launch sequence
run eagerly (EXL3_GRAPHS=0), like every other BC path test. The scenario also exercises the adapter lifecycle in one
process: attach, generate, detach, attach a second adapter, a zero-scale adapter, and back to the base model, so the
graphs re-record on every change (graph structure epoch). Within the graphed run: a zero-scale adapter is bit-exact
with the base model, and returning to a state generated earlier (base again, adapter 0 again) is bit-exact too.

test_lora_graph_matches_unfused: the in-graph application against the known-correct unfused path (EXL3_LORA_GRAPH=0:
the modules decline their graphs while an adapter is attached and Linear.forward applies it), teacher-forced: both
runs prefill the same prompt and decode the same fixed continuation one token at a time through model.forward, so the
per-step logits compare at identical inputs. The two paths differ by their own precision even with no adapter (the
unfused python modules against the C++ graphs: fp16 vs fp32 intermediates, different kernels), measured in the same
run as the discrepancy between a zero-scale adapter on the unfused path and the base model on the graph path. The
adapted runs must agree to within a few times that floor, or a small fraction of the adapter's own effect.
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
    from exllamav3.model.lora import LoRA
    from exllamav3.modules import Attention, GatedMLP, MLP
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from testlib.lora import DEFAULT_TARGETS, write_random_adapter

    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    targets = DEFAULT_TARGETS + ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    dirs = [write_random_adapter(model, os.path.join(work_dir, f"adapter{i}"), rank = 16, seed = i, targets = targets)
            for i in range(2)]
    expect_graphed = os.environ.get("EXL3_LORA_GRAPH", "1") != "0"

    def gen():
        ((tokens, logits),) = generate_with_logits(model, cache, tok, [PROMPT], NEW_TOKENS, max_batch_size = 1)
        return tokens, [l.cpu() for l in logits]

    out = []
    out.append(("base", *gen()))

    lora0 = LoRA.from_directory(model, dirs[0])
    out.append(("lora0", *gen()))
    n_graphed = graphed_projections(model)
    bc_attn_built = any(v for m in model if isinstance(m, Attention) for v in getattr(m, "bc_attn", {}).values())
    bc_mlp_built = any(getattr(m, "bc", None) is not None for m in model if isinstance(m, (GatedMLP, MLP)))
    bc_gdn_built = any(getattr(m, "bc_split", False) for m in model if isinstance(m, GatedDeltaNet))
    lora0.unload()

    lora0_zero = LoRA.from_directory(model, dirs[0], lora_scaling = 0.0)
    out.append(("lora0_zero", *gen()))
    lora0_zero.unload()

    lora1 = LoRA.from_directory(model, dirs[1])
    out.append(("lora1", *gen()))
    lora1.unload()

    out.append(("base_again", *gen()))
    assert graphed_projections(model) == 0, "detached adapter still held by a C++ object"

    lora0 = LoRA.from_directory(model, dirs[0])
    out.append(("lora0_again", *gen()))
    lora0.unload()

    unload_model(model)

    # Vacuity checks: the adapter really went through the graphs (or really stayed out of them)
    assert bc_attn_built or bc_mlp_built or bc_gdn_built, "model built no BC decode path; test is vacuous"
    if expect_graphed:
        assert n_graphed > 0, "no projection handed its adapter to the graph path"
    else:
        assert n_graphed == 0, "EXL3_LORA_GRAPH=0 but a projection handed its adapter to the graph path"
    return out


def graphed_projections(model) -> int:
    """How many attention / MLP / GDN projections currently carry their adapter in the C++ object"""
    from exllamav3.modules import Attention, GatedMLP, MLP
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    n = 0
    for m in model:
        if isinstance(m, Attention):
            n += sum(1 for p in (m.q_proj, m.k_proj, m.v_proj, m.o_proj)
                     if p is not None and p.quant_type == "exl3" and p.inner.bc.has_lora())
        elif isinstance(m, (GatedMLP, MLP)):
            n += sum(1 for p in (*getattr(m, "gates", []), *m.ups, *m.downs)
                     if p.quant_type == "exl3" and p.inner.bc.has_lora())
        elif isinstance(m, GatedDeltaNet):
            n += sum(1 for p in (m.qkv_proj, m.z_proj, m.b_proj, m.a_proj, m.o_proj)
                     if p is not None and p.quant_type in ("exl3", "fp16") and p.inner.bc.has_lora())
    return n


FORCED_PROMPT = "The committee reviewed the proposal in detail and concluded that the lighthouse keeper had, in fact,"
FORCED_STEPS = 8
FORCED_LABELS = ["base", "lora0", "lora0_zero", "lora1"]


def scenario_forced(model_dir: str, work_dir: str) -> dict:
    """Teacher-forced decode (child process): {label: [per-step logits]} for the base model and the adapters, every
    run over the same prompt and the same fixed continuation, one token per forward through the cache"""
    from exllamav3.model.lora import LoRA
    from testlib.lora import DEFAULT_TARGETS, write_random_adapter

    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    targets = DEFAULT_TARGETS + ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    dirs = [write_random_adapter(model, os.path.join(work_dir, f"adapter{i}"), rank = 16, seed = i, targets = targets)
            for i in range(2)]
    expect_graphed = os.environ.get("EXL3_LORA_GRAPH", "1") != "0"
    ids = tok.encode(FORCED_PROMPT, add_bos = True)
    cont = torch.randint(100, 2000, (1, FORCED_STEPS), generator = torch.Generator().manual_seed(1))

    @torch.inference_mode()
    def run():
        n = ids.shape[1]
        state = cache.get_new_state() if cache.num_recurrent_layers else None
        params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096)}
        if state is not None:
            params["recurrent_states"] = [state]
        model.forward(ids, {**params, "past_len": 0})
        out = [model.forward(cont[:, i:i + 1], {**params, "past_len": n + i})[0, -1].float().cpu()
               for i in range(FORCED_STEPS)]
        if state is not None:
            state.free()
        return out

    res = {"base": run()}
    lora0 = LoRA.from_directory(model, dirs[0])
    res["lora0"] = run()
    n_graphed = graphed_projections(model)
    lora0.unload()
    lora0_zero = LoRA.from_directory(model, dirs[0], lora_scaling = 0.0)
    res["lora0_zero"] = run()
    lora0_zero.unload()
    lora1 = LoRA.from_directory(model, dirs[1])
    res["lora1"] = run()
    lora1.unload()
    unload_model(model)
    if expect_graphed:
        assert n_graphed > 0, "no projection handed its adapter to the graph path"
    else:
        assert n_graphed == 0, "EXL3_LORA_GRAPH=0 but a projection handed its adapter to the graph path"
    return res


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


def _check_eager(model_id, model_dir, device, tmp_path):
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


def _check_unfused(model_id, model_dir, device, tmp_path):
    env = {**device_env(device), "EXL3_GRAPHS": "1"}
    graphed = run_isolated(scenario_forced, model_dir, str(tmp_path / "graph"), env = {**env, "EXL3_LORA_GRAPH": "1"})
    unfused = run_isolated(scenario_forced, model_dir, str(tmp_path / "unfused"), env = {**env, "EXL3_LORA_GRAPH": "0"})
    assert list(graphed) == FORCED_LABELS and list(unfused) == FORCED_LABELS

    # Without an adapter both runs take the graph path: identical
    for i, (g, u) in enumerate(zip(graphed["base"], unfused["base"])):
        assert torch.equal(g, u), f"{model_id}: base runs differ at step {i}"

    worst = 0.0
    for i in range(FORCED_STEPS):
        # The paths' own disagreement at this step: unfused python modules (zero delta) vs the graphs (base)
        floor = _rfn(unfused["lora0_zero"][i], graphed["base"][i])
        for label in ("lora0", "lora1"):
            effect = _rfn(unfused[label][i], unfused["base"][i])
            diff = _rfn(graphed[label][i], unfused[label][i])
            assert effect > 10 * floor, f"{model_id} [{label}] step {i}: adapter effect {effect:.4f} within the path noise {floor:.4f}"
            assert diff <= max(3 * floor, 0.1 * effect), \
                f"{model_id} [{label}] step {i}: in-graph vs unfused LoRA rfn {diff:.4f} (effect {effect:.4f}, path floor {floor:.4f})"
            worst = max(worst, diff / max(floor, 1e-9))
    print(f"[{model_id}] in-graph vs unfused LoRA: worst discrepancy {worst:.2f}x the paths' own floor")


@pytest.mark.models("dense")
def test_lora_graph_matches_eager(model_id, model_dir, device, tmp_path):
    _check_eager(model_id, model_dir, device, tmp_path)


@pytest.mark.models("dense")
def test_lora_graph_matches_unfused(model_id, model_dir, device, tmp_path):
    _check_unfused(model_id, model_dir, device, tmp_path)


@pytest.mark.models("gdn")
def test_lora_graph_matches_eager_gdn(model_id, model_dir, device, tmp_path):
    _check_eager(model_id, model_dir, device, tmp_path)


@pytest.mark.models("gdn")
def test_lora_graph_matches_unfused_gdn(model_id, model_dir, device, tmp_path):
    _check_unfused(model_id, model_dir, device, tmp_path)
