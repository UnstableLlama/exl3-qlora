"""
Runtime LoRA inside the captured decode graphs (BC_Attention / BC_GatedMLP / BC_MLP / BC_GatedDeltaNetSplit), over
every registry model tagged "dense", every hybrid tagged "gdn" and every MoE tagged "shared_experts", with random
PEFT adapters on the attention, MLP and GDN projections (testlib.lora). On the MoE models the adapter lands on the
shared expert's gate/up/down (applied inside the shared BC_GatedMLP graph, which BC_BlockSparseMLP::run_bszN takes
in place of its one-expert fused launch while an adapter is attached); the routed experts stay unadapted here (the
fused expert kernels have no LoRA stage; an adapter there takes the per-expert path, which is a different test).

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


def scenario(model_dir: str, work_dir: str, shared: bool = False) -> list:
    """Greedy generations through the adapter lifecycle (child process): [(label, tokens, step logits)]. `shared`:
    the model must run its shared experts inside the MoE decode kernels' path and hand their adapter to it"""
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
    n_sh_graphed, n_sh_embedded = shared_expert_projections(model)
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
    if shared:
        assert n_sh_embedded > 0, "no MoE layer embeds its shared experts in the decode kernels' path; test is vacuous"
        if expect_graphed:
            assert n_sh_graphed > 0, "no embedded shared expert handed its adapter to the graph path"
        else:
            assert n_sh_graphed == 0, "EXL3_LORA_GRAPH=0 but a shared expert handed its adapter to the graph path"
    return out


def graphed_projections(model) -> int:
    """How many attention / MLP / GDN projections currently carry their adapter in the C++ object (the model's
    iteration is recursive, so the shared experts of MoE layers are counted as GatedMLPs)"""
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


def shared_expert_projections(model) -> tuple[int, int]:
    """(graphed, embedded): over the MoE layers whose shared experts are embedded in the BC_BlockSparseMLP decode
    path (bc_sh_exp), how many shared gate/up/down projections carry their adapter in the C++ object, and how many
    such projections there are"""
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    graphed = embedded = 0
    for m in model:
        if isinstance(m, BlockSparseMLP) and getattr(m, "bc_sh_exp", False):
            se = m.shared_experts
            for p in (*se.gates, *se.ups, *se.downs):
                if p.quant_type == "exl3":
                    embedded += 1
                    graphed += int(p.inner.bc.has_lora())
    return graphed, embedded


FORCED_PROMPT = "The committee reviewed the proposal in detail and concluded that the lighthouse keeper had, in fact,"
FORCED_STEPS = 8
FORCED_LABELS = ["base", "lora0", "lora0_zero", "lora1"]


def scenario_forced(model_dir: str, work_dir: str, shared: bool = False) -> dict:
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
    n_sh_graphed, n_sh_embedded = shared_expert_projections(model)
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
    if shared:
        assert n_sh_embedded > 0, "no MoE layer embeds its shared experts in the decode kernels' path; test is vacuous"
        assert (n_sh_graphed > 0) == expect_graphed, \
            f"shared expert projections handed to the graph path: {n_sh_graphed} (EXL3_LORA_GRAPH={int(expect_graphed)})"
    return res


LAYERWISE_PROMPT = FORCED_PROMPT


def scenario_layerwise(model_dir: str, work_dir: str) -> list:
    """One decode step, module by module (child process): for every attention / MLP / MoE module, its output on the
    recorded input with the adapter applied by the graph path and by the unfused python path, and without the adapter.
    Returns [(key, effect_py, effect_graph, graph_vs_python)] as relative norms. A MoE model's whole-model logits are
    chaotic between two numerically different paths (a tiny difference flips an expert choice), so the paths are
    compared where they can be: on the same input, per module"""
    import exllamav3.modules.linear as linmod
    from exllamav3.model.lora import LoRA
    from exllamav3.modules import Linear, Attention, GatedMLP, MLP
    from exllamav3.modules.sliding_attn import SlidingAttention
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from testlib.lora import DEFAULT_TARGETS, write_random_adapter

    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    adapter = write_random_adapter(model, os.path.join(work_dir, "adapter"), rank = 16, seed = 0, targets = DEFAULT_TARGETS)
    ids = tok.encode(LAYERWISE_PROMPT, add_bos = True)
    n = ids.shape[1]
    kinds = (Attention, SlidingAttention, GatedMLP, MLP, BlockSparseMLP)
    mods = [m for m in model if isinstance(m, kinds) and not any(isinstance(o, kinds) and o is not m and m.key.startswith(o.key + ".") for o in model)]
    rec = {}
    for m in mods:
        def wrap(x, params, *a, _orig = m.forward, _key = m.key, **k):
            rec[_key] = (x.clone(), dict(params))
            return _orig(x, params, *a, **k)
        m.forward = wrap

    @torch.inference_mode()
    def step():
        state = cache.get_new_state() if cache.num_recurrent_layers else None
        p = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096)}
        if state is not None:
            p["recurrent_states"] = [state]
        model.forward(ids, {**p, "past_len": 0})
        model.forward(ids[:, -1:], {**p, "past_len": n})
        if state is not None:
            state.free()

    def use_graph(flag):
        linmod._lora_graph_enable = flag
        for m in model:
            if isinstance(m, Linear):
                m._lora_graph_src = ()

    with torch.inference_mode():
        step(); step(); step()                   # graphs captured, inputs recorded on the last decode step
        run = lambda: {m.key: m.__class__.forward(m, *rec[m.key]).clone() for m in mods}
        base = run()
        lora = LoRA.from_directory(model, adapter)
        use_graph(True)
        graph = run()
        use_graph(False)
        python = run()
        use_graph(True)
        lora.unload()
    out = []
    for m in mods:
        k = m.key
        out.append((k, _rfn(python[k], base[k]), _rfn(graph[k], base[k]), _rfn(graph[k], python[k])))
    unload_model(model)
    return out


def _check_layerwise(model_id, model_dir, device, tmp_path):
    rows = run_isolated(scenario_layerwise, model_dir, str(tmp_path), env = {**device_env(device), "EXL3_GRAPHS": "1"})
    adapted = [r for r in rows if r[1] > 1e-4]
    assert adapted, f"{model_id}: no module changed under the adapter"
    worst = max(r[3] / r[1] for r in adapted)
    print(f"[{model_id}] layer-wise in-graph vs unfused LoRA over {len(adapted)} modules: worst {100 * worst:.2f}% of the module's adapter effect")
    for key, eff_py, eff_g, diff in adapted:
        assert diff <= 0.05 * eff_py, f"{model_id} {key}: graph vs python {diff:.4f} (adapter effect {eff_py:.4f})"


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


def _check_eager(model_id, model_dir, device, tmp_path, shared = False):
    graphed, eager = run_with_and_without_graphs(scenario, model_dir, str(tmp_path), shared, device = device)
    assert_matches_eager([(f"{model_id} [{l}]", t, g) for l, t, g in graphed],
                         [(f"{model_id} [{l}]", t, g) for l, t, g in eager], min_steps = 4)

    d = _by_label(graphed)
    # Lifecycle: the graphs re-record on attach/detach and come back to exactly the same kernels
    _assert_identical(d, "base", "base_again")
    _assert_identical(d, "lora0", "lora0_again")
    # Zero scale: B is zero, the delta nodes add exactly nothing. With the shared expert of a MoE layer the adapter
    # also moves it from the one-expert fused launch to the BC_GatedMLP graph (same math, other kernels), so there
    # the check is closeness, not identity
    # (loose: the last-bit differences flip expert choices over the decode steps; the per-module agreement is
    # what test_lora_graph_matches_unfused_shared_experts measures)
    if shared:
        assert d["base"][0][:4] == d["lora0_zero"][0][:4], "base vs lora0_zero: tokens differ within the first 4 steps"
        for i, (x, y) in enumerate(list(zip(d["base"][1], d["lora0_zero"][1]))[:4]):
            assert _rfn(y, x) < 5e-2, f"base vs lora0_zero: step {i} rfn {_rfn(y, x):.4f}"
    else:
        _assert_identical(d, "base", "lora0_zero")
    # The adapters have an effect (the first step's logits come from the prefill, which applies them too)
    assert _rfn(d["lora0"][1][0], d["base"][1][0]) > 1e-3, "adapter 0 had no visible effect"
    assert _rfn(d["lora1"][1][0], d["lora0"][1][0]) > 1e-3, "adapters 0 and 1 are indistinguishable"


def _check_unfused(model_id, model_dir, device, tmp_path, shared = False):
    env = {**device_env(device), "EXL3_GRAPHS": "1"}
    graphed = run_isolated(scenario_forced, model_dir, str(tmp_path / "graph"), shared,
                           env = {**env, "EXL3_LORA_GRAPH": "1"})
    unfused = run_isolated(scenario_forced, model_dir, str(tmp_path / "unfused"), shared,
                           env = {**env, "EXL3_LORA_GRAPH": "0"})
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


# MoE with shared experts: DEFAULT_TARGETS match the shared expert's gate/up/down (adapter_targets leaves the routed
# experts out), so the adapter is applied inside the shared BC_GatedMLP graph, run by BC_BlockSparseMLP::run_bszN in
# place of its one-expert fused launch (sh_coop) while attached; the unfused reference runs the shared experts
# through GatedMLP.forward after the routed sum
@pytest.mark.models("shared_experts")
def test_lora_graph_matches_eager_shared_experts(model_id, model_dir, device, tmp_path):
    _check_eager(model_id, model_dir, device, tmp_path, shared = True)


@pytest.mark.models("shared_experts")
def test_lora_graph_matches_unfused_shared_experts(model_id, model_dir, device, tmp_path):
    # Layer-wise: the whole-model logits of a MoE are chaotic between two numerically different paths
    _check_layerwise(model_id, model_dir, device, tmp_path)
