"""
LoRA initialization strategies for the native QLoRA trainer.

Why initialization matters here
-------------------------------
The trainer's runs are short (tiny SFT sets, a few epochs): with the PEFT
default (kaiming A, B = 0) the adapter spends a meaningful fraction of the
whole run just growing off the ground (|B| is typically still climbing at the
final step). Initialization methods attack exactly that. Two are implemented,
both SVD-based and both computed once at startup from weights the trainer
already reconstructs:

``pissa`` -- Principal Singular values and Singular vectors Adaptation
    (Meng et al. 2024, NeurIPS spotlight). Factor the top-r principal
    component out of the frozen base weight into the *trainable* adapter, so
    training updates the directions that matter most first. The reference
    formulation retrains against a residual base ``W_res = W - principal``;
    our base is a frozen EXL3 trellis that cannot be rewritten, so the
    residual is realized as a frozen rank-r offset instead:

        effective W  =  W_q  +  s·(A@B - A0@B0)
                     = (W_q - s·A0@B0) + s·A@B      # == W_res + s·A@B

    with A/B trainable and initialized to A0/B0 (so the model starts exactly
    at the base). The offset is folded into the frozen-weight closure (see
    ``DiffLinear``), costing one rank-r matmul per weight reconstruction --
    noise next to the trellis dequant it rides on.

``qerr`` -- quantization-error-correction init (the LoftQ idea, single-shot).
    Initialize the adapter to the top-r SVD of the *quantization error*
    ``E = W_ref - W_q`` (W_ref = the original unquantized weight), so at step
    0 the model is the closest rank-r approximation of the ORIGINAL bf16
    model and training proceeds from there. No offset, no bookkeeping: the
    nonzero start is the point. Needs ``--init-ref-model`` (the original HF
    model dir). The lower the bitrate, the bigger E -- this is the natural
    companion of low-bpw EXL3 training.

``eva`` -- Explained Variance Adaptation (Paischer et al. 2024), fixed-rank
    variant. Initialize A to the top-r right-singular vectors of each
    target's INPUT activations, streamed over a short pre-pass of the actual
    training data through the actual quantized forward (so the init adapts
    to the model we ship, not the bf16 original). B stays zero, so step 0 is
    exactly the base model -- no offset, no sidecar, standard save path. The
    data-dependence is the point: gradients through A immediately act in the
    directions the data actually excites, instead of random kaiming ones.
    Rank redistribution from the paper is deliberately skipped (it touches
    adapter config/save/merge for a secondary effect; see the handoff
    decision record). Layers that provably share an input tensor (q/k/v,
    gate/up) share one activation sketch and get the same A.

pissa/qerr divide the factors by sqrt(scale) so the adapter's *scaled*
contribution ``s·A@B`` equals the intended SVD component regardless of
alpha/r/rslora (eva needs no such folding: its B is zero and A is a basis,
not a magnitude).

SVD determinism note: ``svd_niter > 0`` uses randomized ``torch.svd_lowrank``
(seconds per model, the PiSSA "fast SVD" recipe); it is NOT deterministic
across runs, which is why pissa checkpoints persist A0/B0 in a sidecar file
(``pissa_init.safetensors``) instead of recomputing them on resume.
"""

from __future__ import annotations
from typing import Optional
import glob
import json
import os
import time
import torch

from . import backbone


@torch.no_grad()
def principal_factors(
    w: torch.Tensor,
    r: int,
    niter: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Rank-r principal factorization of ``w`` ([in, out], float32).

    Returns ``(a, b, captured, total)`` with ``a`` [in, r], ``b`` [r, out],
    ``a @ b`` = the top-r SVD approximation of ``w``, ``captured`` = the
    approximation's squared Frobenius mass and ``total`` = ``||w||_F^2``
    (their ratio is the explained-variance fraction). ``niter > 0`` selects
    randomized SVD (``torch.svd_lowrank`` with ``niter`` subspace iterations,
    slight oversampling); ``niter == 0`` computes the exact full SVD.
    """
    assert w.dim() == 2
    assert 0 < r <= min(w.shape), f"rank {r} > min dim {min(w.shape)}"
    w = w.float()
    if niter > 0:
        q = min(r + 8, min(w.shape))
        u, s, v = torch.svd_lowrank(w, q=q, niter=niter)
        u, s, v = u[:, :r], s[:r], v[:, :r]
    else:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u, s, v = u[:, :r], s[:r], vh[:r].t()
    sq = s.clamp_min(0).sqrt()
    a = (u * sq).contiguous()                 # [in, r]
    b = (v * sq).t().contiguous()             # [r, out]
    captured = float((s * s).sum())
    total = float((w * w).sum())
    return a, b, captured, total


class EvaSketch:
    """
    Streaming top-k right-singular-subspace sketch of a token-activation
    stream (block incremental PCA, the EVA recipe): maintain ``diag(sv)·Vᵀ``
    of everything streamed so far ([k, in], a few MB), and fold each new
    activation chunk in with one truncated SVD of ``[sketch; chunk]``. Never
    stores the activations or an [in, in] Gram matrix, so the pre-pass adds
    no meaningful memory on top of the no-grad forward.
    """

    def __init__(self, in_features: int, k: int, niter: int = 8):
        self.in_features = in_features
        self.k = min(k, in_features)
        self.niter = niter
        self.v: Optional[torch.Tensor] = None    # [in, k'] right singular vectors
        self.sv: Optional[torch.Tensor] = None   # [k'] singular values
        self.total = 0.0                         # streamed sum of squares
        self.rows = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, max_chunk_rows: int = 8192) -> None:
        """Fold activation rows ``x`` ([..., in]) into the sketch (fp32)."""
        x = x.detach().reshape(-1, x.shape[-1])
        assert x.shape[-1] == self.in_features
        for rows in x.split(max_chunk_rows):
            rows = rows.float()
            self.total += float(rows.pow(2).sum())
            self.rows += rows.shape[0]
            m = rows if self.v is None else torch.cat(
                [self.sv.unsqueeze(1) * self.v.t(), rows])
            q = min(self.k, *m.shape)
            if self.niter > 0:
                _, s, v = torch.svd_lowrank(m, q=q, niter=self.niter)
            else:
                _, s, vh = torch.linalg.svd(m, full_matrices=False)
                s, v = s[:q], vh[:q].t()
            self.sv = s.contiguous()
            self.v = v.contiguous()

    @torch.no_grad()
    def top(self, r: int) -> tuple[torch.Tensor, float, float]:
        """Top-r right-singular vectors ``[in, r]`` (orthonormal columns) plus
        (captured, total) squared-mass accounting for the explained-variance
        fraction."""
        if self.v is None or self.v.shape[1] < r:
            got = 0 if self.v is None else self.v.shape[1]
            raise SystemExit(
                f"--init-lora eva: only {got} activation directions collected "
                f"for rank {r} ({self.rows} rows streamed) -- raise "
                f"--init-eva-tokens / provide more pre-pass data.")
        captured = float((self.sv[:r] ** 2).sum())
        return self.v[:, :r].contiguous(), captured, self.total


# Targets whose input is the same tensor by construction of the native forward
# (q/k/v all consume the post-attn-norm hidden state; gate/up the post-MLP-norm
# one), so they share one activation sketch -- and, since EVA's A depends only
# on the input distribution, the same A init. Verified at runtime by data_ptr.
_EVA_SHARED_INPUT_GROUPS = {
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gate_up", "up_proj": "gate_up",
}


def _eva_site(key: str) -> str:
    prefix, _, leaf = key.rpartition(".")
    return f"{prefix}.{_EVA_SHARED_INPUT_GROUPS.get(leaf, leaf)}"


@torch.no_grad()
def _apply_eva(net, batches, svd_niter: int, verbose: bool) -> None:
    """
    EVA pre-pass + init. ``batches`` yields ``net.forward`` kwargs dicts
    (``input_ids`` plus optional ``attention_mask``/``position_ids``/
    ``seg_ids``) of REAL training data; every target wrapper's input
    activations are streamed into per-site sketches via forward pre-hooks,
    then A is set to each site's top-r right-singular vectors and B stays 0.
    """
    wrappers = [w for w in net._wrappers if w.r > 0]
    if not wrappers:
        return
    if batches is None:
        raise SystemExit(
            "--init-lora eva needs a data pre-pass; pass eva_batches (the "
            "trainer builds it from the training set automatically).")

    t0 = time.time()
    # The incremental update refines the previous sketch on every fold, so it
    # needs far fewer subspace iterations than the one-shot pissa/qerr SVD;
    # cap it to keep the pre-pass math cheap (0 = exact stays available).
    niter = min(svd_niter, 8) if svd_niter > 0 else 0
    oversample = 8
    sketches: dict[str, EvaSketch] = {}
    site_ptr: dict[str, int] = {}       # site -> input data_ptr this forward
    state = {"keep": None}              # [b, t] bool mask of real (non-pad) tokens

    def make_hook(site: str):
        def pre_hook(module, args_):
            x = args_[0]
            ptr = x.data_ptr()
            prev = site_ptr.get(site)
            if prev is not None:
                # A group-mate already streamed this forward's input; the group
                # assumption must actually hold on this architecture.
                if prev != ptr:
                    raise RuntimeError(
                        f"eva: grouped targets at site {site} saw different "
                        f"input tensors -- shared-input assumption violated; "
                        f"fix _EVA_SHARED_INPUT_GROUPS for this architecture.")
                return
            site_ptr[site] = ptr
            sk = sketches.get(site)
            if sk is None:
                sk = sketches[site] = EvaSketch(
                    module.in_features, module.r + oversample, niter)
            keep = state["keep"]
            if keep is not None and x.dim() == 3 and keep.shape == x.shape[:2]:
                x = x[keep.to(x.device)]        # drop pad-token rows
            sk.update(x)
        return pre_hook

    hooks = [w.register_forward_pre_hook(make_hook(_eva_site(w.key)))
             for w in wrappers]
    was_training = net.training
    tokens = 0
    try:
        net.eval()
        for bt in batches:
            site_ptr.clear()
            am = bt.get("attention_mask")
            state["keep"] = am.bool() if am is not None else None
            net.forward(**bt)
            tokens += (int(am.sum()) if am is not None
                       else bt["input_ids"].numel())
    finally:
        for h in hooks:
            h.remove()
        net.train(was_training)

    var_fracs: list[float] = []
    starved_experts = 0
    for w in wrappers:
        try:
            a, captured, total = sketches[_eva_site(w.key)].top(w.r)
        except (KeyError, RuntimeError):
            # Routed MoE experts see only their routed share of the pre-pass
            # tokens; a rarely-hit expert can stream fewer rows than the rank
            # needs (RuntimeError), or none at all (KeyError -- no sketch).
            # That is data sparsity, not a config error: keep the default
            # kaiming/zeros init for that adapter and report the count. For
            # any NON-expert target the old hard failure stands -- there it
            # really means the pre-pass budget is too small.
            if ".experts." in w.key:
                starved_experts += 1
                continue
            raise
        w.lora_a.copy_(a.to(w.lora_a.device, w.lora_a.dtype))
        w.lora_b.zero_()
        w.zero_padding_()       # EXL3 padding region stays exactly zero
        var_fracs.append(captured / max(total, 1e-30))

    net.init_lora = "eva"
    if verbose:
        vf = sorted(var_fracs)
        stats = (f" (rank-{net.r} captured activation variance: "
                 f"median {vf[len(vf) // 2] * 100:.1f}%, "
                 f"min {vf[0] * 100:.1f}%, max {vf[-1] * 100:.1f}%)"
                 if vf else "")
        extra = (f"; {starved_experts} routed-expert adapters kept default "
                 f"init (too few routed tokens in the pre-pass -- raise "
                 f"--init-eva-tokens to cover them)" if starved_experts else "")
        print(f" -- init-lora eva: {len(var_fracs)} adapters initialized from "
              f"{len(sketches)} activation sites / {tokens:,} tokens in "
              f"{time.time() - t0:.1f}s{stats}{extra}")


class RefWeights:
    """
    Lazy reader for the ORIGINAL (unquantized) model's weights, by HF tensor
    name, from a local safetensors model dir (single file or sharded with
    ``model.safetensors.index.json``). Used by qerr init to form the
    quantization error; nothing is loaded until a tensor is requested.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self._name_to_file: dict[str, str] = {}
        self._handles: dict[str, object] = {}
        index = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf8") as f:
                weight_map = json.load(f)["weight_map"]
            for name, fn in weight_map.items():
                self._name_to_file[name] = os.path.join(directory, fn)
        else:
            from safetensors import safe_open
            files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
            if not files:
                raise FileNotFoundError(
                    f"--init-ref-model: no *.safetensors under {directory}")
            for fn in files:
                with safe_open(fn, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        self._name_to_file[name] = fn

    def get(self, name: str) -> Optional[torch.Tensor]:
        from safetensors import safe_open
        fn = self._name_to_file.get(name)
        if fn is None:
            return None
        h = self._handles.get(fn)
        if h is None:
            h = safe_open(fn, framework="pt", device="cpu")
            self._handles[fn] = h
        return h.get_tensor(name)


@torch.no_grad()
def apply_init_lora(
    net,
    mode: str,
    ref_model_dir: Optional[str] = None,
    svd_niter: int = 16,
    verbose: bool = True,
    eva_batches=None,
) -> None:
    """
    Apply an SVD-based LoRA init to every per-linear target wrapper of a
    ``NativeLlamaQLoRA``. ``mode`` is ``"pissa"``, ``"qerr"`` or ``"eva"``
    (``"default"`` is a no-op -- the constructor's kaiming/zeros already
    happened). ``eva`` additionally needs ``eva_batches``, an iterable of
    ``net.forward`` kwargs dicts of real training data for the activation
    pre-pass.

    Embed/head LoRA (``lora_embed``/``lora_head``) keeps the default init:
    pissa/eva on the embedding are ill-defined (token-indexed A) and the head
    is frozen fp16, not trellis-quantized, so qerr has no error to correct.
    """
    if mode in (None, "", "default"):
        return
    assert mode in ("pissa", "qerr", "eva"), f"unknown init_lora mode {mode!r}"

    if mode == "eva":
        _apply_eva(net, eva_batches, svd_niter, verbose)
        return

    refs = None
    if mode == "qerr":
        if not ref_model_dir:
            raise SystemExit(
                "--init-lora qerr needs --init-ref-model (the ORIGINAL "
                "unquantized HF model dir) to form the quantization error.")
        refs = RefWeights(ref_model_dir)

    t0 = time.time()
    n = 0
    var_fracs: list[float] = []
    for w in net._wrappers:
        if w.r <= 0:
            continue
        # Full-precision reconstruction of the frozen weight, on its device.
        base = backbone.frozen_weight_closure(w.linear, torch.float32)()

        if mode == "pissa":
            a0, b0, cap, tot = principal_factors(base, w.r, niter=svd_niter)
            root_s = w.scale ** 0.5
            a0 = a0 / root_s
            b0 = b0 / root_s
            w.lora_a.copy_(a0)
            w.lora_b.copy_(b0)
            # The padded base rows/cols are zero, so the SVD factors are only
            # ~0 there; pin the adapter AND the frozen offset exactly.
            w.zero_padding_()
            w.set_init_offset(a0, b0)
        else:  # qerr
            name = f"{w.key}.weight"
            ref = refs.get(name)
            if ref is None:
                raise SystemExit(
                    f"--init-lora qerr: tensor {name!r} not found in "
                    f"{ref_model_dir} -- is this the original model for "
                    f"this quant?")
            ref = ref.t().to(base.device, torch.float32)   # HF [out,in] -> [in,out]
            in_hf, out_hf = ref.shape
            in_pad, out_pad = base.shape
            if in_hf > in_pad or out_hf > out_pad:
                raise SystemExit(
                    f"--init-lora qerr: reference weight {name} "
                    f"[{in_hf},{out_hf}] larger than the quantized layer "
                    f"[{in_pad},{out_pad}] -- wrong reference model?")
            err = -base
            err[:in_hf, :out_hf] += ref
            # Never "correct" the padding region: padded input rows/output
            # cols are outside the real model and must stay untouched.
            if in_hf < in_pad:
                err[in_hf:, :] = 0
            if out_hf < out_pad:
                err[:, out_hf:] = 0
            a, b, cap, tot = principal_factors(err, w.r, niter=svd_niter)
            root_s = w.scale ** 0.5
            w.lora_a.copy_(a / root_s)
            w.lora_b.copy_(b / root_s)
            w.zero_padding_()
        del base
        var_fracs.append(cap / max(tot, 1e-30))
        n += 1

    net.init_lora = mode
    if verbose and n:
        vf = sorted(var_fracs)
        what = ("principal weight variance" if mode == "pissa"
                else "quantization-error variance")
        print(f" -- init-lora {mode}: {n} adapters initialized in "
              f"{time.time() - t0:.1f}s (rank-{net.r} captured {what}: "
              f"median {vf[len(vf) // 2] * 100:.1f}%, "
              f"min {vf[0] * 100:.1f}%, max {vf[-1] * 100:.1f}%)")
