# EBFT on EXL3 — design, status, and run guide

Energy-Based Fine-Tuning (EBFT) for the exl3-qlora native path. First known
implementation of EBFT on **LoRA / quantized** weights (the paper and its
reference code are full-fine-tuning only).

- **Paper:** "Matching Features, Not Tokens: Energy-Based Fine-Tuning of
  Language Models" (Jelassi et al., arXiv:2603.12248).
- **Reference code:** `sjelassi/ebft_openrlhf` (OpenRLHF fork, full-FT, Ray).
  Cloned and read to pin exact semantics; our reward math is faithful to the
  **code**, not the appendix, where they differ (see "Two deliberate
  deviations" below).
- **Built:** Session 39 (2026-07-17). Status: implementation complete +
  correctness-gated. **First A/B answered (Session 42, 2026-07-18): EBFT-LoRA did
  NOT beat SFT-LoRA** at 1B / 1 epoch on semancy — SFT won held-out CE decisively
  and tied CFM. See "Result: first A/B" below. Not a clean win, not a clean
  refutation either (proxies, untuned EBFT LR, small scale).

---

## ⚠️ Known issue (Session 40): activation offload → host-RAM OOM, whole-machine crash

**Symptom.** A quarter-epoch A/B run (`semancer_llama1b_ebft.yaml`, Llama-3.2-1B
@4bpw, batch 4 / anchors 4 / n 4, `offload_activations: true`) grew host RAM
without bound and killed the whole box (64 GB RAM + swap exhausted) around
step ~23/28. VRAM stayed near-empty (~5 GB) the entire time. The SFT arm on the
same data finished fine (28 steps / 33 s).

**Root cause — NOT model size.** `offload_activations: true` sends
gradient-checkpoint activations to **pinned host RAM** via
`AsyncActivationOffload` (`exllamav3/training/offload.py`). The pinned-buffer
pool is keyed by `(shape, dtype)` (`offload.py:70`, returned at `:164`) with
**no eviction and no total-bytes cap**, and the offloader instance persists for
the whole run (`native_llama.py:1683-1686`). Its design assumption
(`offload.py:24` comment: "steady state holds one buffer") holds for SFT, whose
tensor shapes barely change — but **EBFT picks random anchors every step**
(`pick_anchors(..., rng)`), so the rollout tensor `rows` has a *different padded
length* (and often a different row count) each step. Each step therefore
allocates a fresh ~4 GB set of pinned buffers under a brand-new shape key and
orphans the previous step's buffers in the pool forever → monotonic pinned-RAM
growth → ~step 14-16 crosses 64 GB → pinned pages can't swap, the rest thrashes,
machine locks. `R = batch·anchors·n = 64` rollout rows × up to seq_len makes each
step's offloaded activations large, so the blowup is fast.

**Immediate mitigation (done): `offload_activations: false` in both semancer
llama1b A/B YAMLs.** A 1B @4bpw + LoRA trains in ~5 GB — offload was pure
downside here. This makes the A/B safe to relaunch after a context refresh.
Only reach for offload on a model that genuinely doesn't fit.

**Real fix — BUILT (Session 47).** `offload.py`'s free pool is now an LRU
(`OrderedDict`, MRU last) bounded by total free bytes. Returning a buffer over
the cap evicts least-recently-used shape buckets first (respecting each buffer's
readback event before releasing pinned memory). The effective cap is
`max(EXL3_OFFLOAD_POOL_GIB≈4 GiB, 1.5 × the largest single forward pass)` — the
working-set floor lets a large but shape-STABLE run keep its whole pass pooled
(no re-alloc thrash), while the cross-pass leak of stale shapes is evicted down
to ~1.5 passes. Verified by `tests/test_offload.py` (GPU): (1) value-exact —
grads bit-identical with/without offload; (2) bounded — pool peaks ~154 MB over
40 varying-shape steps (working set ~103 MB) instead of growing ~40×; (3) no
thrash — a fixed shape stays steady at exactly one pass. **Still pending: a
real-run RAM watch on the 128B SFT relaunch** to confirm the synthetic result
holds at scale before treating `offload_activations: true` as safe there again.

**Confidence:** high on the mechanism (pool code read: shape-keyed, no eviction,
persistent instance, EBFT per-step shape variance). The exact crash step is from
live observation (~23/28) — the scratchpad log was wiped on reboot, so this is
inferred from code, not a captured RAM trace.

---

## Result: first EBFT-LoRA vs SFT-LoRA A/B (Session 42)

Llama-3.2-1B @4bpw, semancy, **matched** hyperparameters (r16/α16, rsLoRA, init
default, lr **1e-5**, 1 epoch / 109 steps, batch 4, 7 dense targets, seq 1024;
EBFT objective G8/n4/anchors4, temp 0.6, div 0.5, γ 0.03, native rollout
sampler). Configs `semancer_llama1b_{sft,ebft}.yaml`, overlay `out/sft_vs_ebft.html`.

All three rows measured through the **same EBFT eval path** (fixed 32-example CFM
subset, seed 12345; CE over the full 116-example test split):

| model | held-out CE ↓ | held-out CFM ↓ |
|---|---|---|
| base (no adapter) | 3.765 | 0.901 |
| **SFT-LoRA** | **3.223** | 0.827 |
| **EBFT-LoRA** | 3.547 | **0.822** |

**Read:** EBFT's own signature holds — CE **and** CFM both fall over training
(3.765→3.547 / 0.901→0.822). But it buys no edge over SFT: SFT wins CE by 0.32
nats and matches EBFT's CFM (0.827 vs 0.822 — inside rollout noise) *without ever
optimizing CFM*. As a language model this EBFT-LoRA is worse (higher CE), by
design of γ=0.03.

**Why this isn't a verdict on EBFT:** (1) we compared **CE/CFM proxies**, not the
paper's actual claim (downstream task accuracy at larger scale / full-FT); (2)
**EBFT's LoRA LR is unvalidated** (1e-5 = smoke value) — the 1e-5..1e-4 sweep is
the obvious next lever; (3) single seed, one dataset, 1B/1-epoch. Confidence:
high on the numbers, medium on the conclusion. SFT's CFM was obtained by a
same-axis probe (EBFT trainer `--resume out/ab_sft --reset-optimizer --steps 1`;
its baseline eval reads the SFT-adapter CFM). Next planned: same A/B on
Qwen3.5-4B @4bpw, identical hyperparameters.

---

## Result: Qwen-4B A/B + LR sweep (Session 43)

Qwen3.5-4B @4bpw, semancy, same matched hyperparameters as the 1B A/B (r16/α16,
rsLoRA, init default, 1 epoch / 109 steps, batch 4, 7 dense targets, seq 1024;
EBFT G8/n4/anchors4, temp 0.6, div 0.5, γ 0.03, native sampler), plus a **2× LR
arm for each objective**. Configs `semancer_qwen35_4b_{sft,ebft}[_lr2x].yaml`;
overlays `out/qwen35_4b_sft_vs_ebft.html`, `out/qwen35_4b_full_sweep.html`.

| run | held-out CE ↓ |
|---|---|
| **SFT @2e-5** | **2.272** |
| SFT @1e-5 | 2.305 |
| EBFT @1e-5 | 2.637 |
| EBFT @2e-5 | 2.736 |

**LR-sweep read:** the two objectives want LR in **opposite directions**. SFT
*improved* at 2× LR (2.305→2.272); EBFT *degraded* (2.637→2.736), and EBFT@2e-5's
held-out CE actually **rose above baseline at step 10** before recovering — a
high-LR / high-variance overshoot. So SFT has the wider LR band here; EBFT is the
LR-sensitive one. (Within each objective the LR ranking is clean — same trainer,
same eval path.)

**Measurement caveat (unlike S42):** here the SFT CE comes from the *native*
trainer's eval (`eval_loss`, **macro** = mean over examples) and the EBFT CE from
the *EBFT* trainer's eval (`evaluate`, **micro** = mean over tokens) — different
reductions, so the **SFT-vs-EBFT absolute gap is not apples-to-apples** (S42
avoided this by measuring SFT through the EBFT `--resume` probe). Magnitude is
bounded, though: the no-adapter **baseline** reads 2.987 (macro) vs 2.952 (micro)
= only **~0.035 nats** of reduction difference, so it explains at most ~0.035 of
the 0.33 SFT–EBFT gap. The gap is real; CE also remains SFT's home turf, not
EBFT's objective nor a downstream-quality measure.

**Paper-recipe finding (Appendix G) — we ran EBFT off-recipe.** The paper runs
the RL-style methods (EBFT, RLVR) at a **constant lr 1e-6** (warmup 0.03, **no
decay**); SFT uses lr 1e-5 with warmup+cosine. Our EBFT arms used lr 1e-5/2e-5
(**10–20× too hot**) *and* cosine-decay-to-0 — so none of the EBFT numbers above
is really "the paper's EBFT." This matches the data (EBFT wanted lower, steadier
LR — REINFORCE-variance) and is the next lever, replacing the earlier "sweep LR
*up*" plan. In our trainer `scheduler: none` = warmup-then-constant (verified in
`make_lr_scheduler`), so the faithful run is **lr 1e-6, scheduler none**. Caveat:
at 10× lower LR, 1 epoch underfits (the 1e-5 run was already unconverged), so a
faithful reproduction needs ~10× the step budget (~8–10 epochs) — user-approved
GPU time pending; paper's exact step count not yet found. See
`memory/ebft-paper-lr-recipe.md`.

---

## What EBFT is, in one paragraph

Instead of next-token CE (SFT) or a scalar verifier reward (RLVR), train the
policy so its on-policy rollouts match the *feature statistics* of ground-truth
completions. A frozen feature network φ (a copy of the model at init) embeds
both sampled and ground-truth completions; each rollout gets a dense reward =
whitened alignment-with-ground-truth minus leave-one-out similarity-to-siblings;
the policy is updated by REINFORCE with an RLOO baseline, plus a small CE term.
Paper results: beats SFT on downstream accuracy, matches RLVR with no verifier,
and *lowers* validation CE more than SFT does.

---

## Files (all uncommitted as of Session 39)

| File | Status | What |
|---|---|---|
| `exllamav3/training/ebft.py` | **new** (194 L) | reward math (`ebft_rewards`), whitening (`whiten_features`), exact sampler (`sample_rollouts`) |
| `training/qlora_train_ebft.py` | **new** (983 L) | the trainer (anchors-as-rows, self-test, eval, run-log) |
| `tests/test_ebft.py` | **new** (163 L) | CPU test suite, no GPU needed |
| `exllamav3/training/native_llama.py` | **modified** | `collect_hidden()` + `feature_block_indices()` feature taps in the forward |
| `training/README.md` | **modified** | file listing entry |

---

## Architecture / key decisions

**Frozen feature network φ = the adapter-disabled base.** EBFT's φ is a copy of
the model at init. With LoRA the policy at init *is* the frozen base, so
`net.adapters_disabled()` (the existing DPO/KTO reference trick) gives φ exactly
— zero extra VRAM, no second model. Holds for `--init-lora default/pissa/eva`;
**not** for `qerr` (its step-0 policy ≠ raw base), which the trainer warns about.

**Feature taps.** `collect_hidden(block_indices, positions)` context manager on
`NativeLlamaQLoRA` stashes the residual stream after selected blocks, optionally
gathered at given token positions (always pass positions in training — the full
fp32 stream is hundreds of MB/tap). `feature_block_indices()` picks ~25/50/75%
depth via the HF `hidden_states` convention the reference critic uses. Features
= concat across the 3 taps, last window token, jointly L2-normalized
(reference's `hidden_state_method=concat` + `embed_method=last_token`).

**Exact on-policy sampler (v1 choice).** `sample_rollouts` samples G tokens by
running the differentiable forward G times (no KV cache). Wasteful, but the
rollout is sampled under *exactly* the policy the REINFORCE gradient is later
computed against (`compute_logps` over the same path) — **zero sampling/scoring
mismatch**, unlike a vLLM-vs-trainer RLHF stack.

**Native KV-cached sampler (Session 41, `--rollout-sampler native`).** Rollouts
via the exllamav3 Generator through the runtime-LoRA slots (`apply_to_native`
before each sampling call, removed after): one paged prefill + G single-token
decode steps instead of G full-prefix re-forwards, and the paged cache dedups
the shared context across each anchor's n sibling rollouts (one prefill per
anchor, not per row). Scoring stays on the differentiable path either way.
On-policy modulo inference-kernel numerics (fused fp16 kernels vs the training
forward's dequant matmul — self-test measured max |Δlogp| 0.072 nats over the
training top-20, argmax agreement): the classic RLHF sampler/trainer split
minus the weight staleness. **Measured (Llama-3.2-1B @4bpw, A/B config): 2.06×
step speedup** (17.5–22.5 s/step → 8.4–10.9; fwd share 77% → 53%; +1.07 GB VRAM
for the 32k-token cache, `--sampler-cache-tokens`). **Refused for
routed-expert LoRA** (fused MoE kernels bypass the runtime slots → rollouts
would come from the base experts); the exact sampler remains the default and
the only valid choice for MoE. Generator quirk encoded in the helper: a `Job`
counts the prefill-position sample toward `max_new_tokens` (N yields N−1
tokens), so it requests G+1; the KV `Cache` must be constructed *before*
`model.load()` (same constraint the SFT trainer documents for live samples).

**Anchors-as-rows (v1 choice).** The reference amortizes rollouts across a whole
sequence with a Quiet-STaR custom attention mask (Appendix F). We instead sample
`--anchors` positions per sequence and materialize each `(context, window)` as
its own batch row. Same reward math per group, less prefix sharing. The strided
mask is the planned v2 throughput upgrade, not an algorithmic requirement.

**Loss.** `loss = rl_coef · mean_rows(−advantage · logp_mean) + ce_coef · CE`,
where `logp_mean` is the per-token-mean completion logprob (reference
`EBFTPolicyLoss` semantics; the PPO ratio is 1 for a single-step surrogate), the
advantage is the detached `reward − RLOO_baseline`, and CE is standard next-token
CE on the ground-truth sequences (the paper's γ term).

### Two deliberate deviations from the paper's appendix (both match the reference code, verified numerically)

1. **Ground-truth whitening.** The code whitens the gt feature by applying the
   `[n,n]` row operator to n replicated copies, so each output row is a *scalar
   multiple* of the raw gt vector — **not** the paper's D-space `(Σ⁺)^(1/2) y`.
   Under cosine alignment this reduces to comparing whitened rollouts against the
   raw gt direction. This is what the reference does; we match it.
2. **Diversity scale.** The code's whitened diversity is `1/n` of the appendix
   eq. 48 values (a duplicate pair at n=4 → **1/3**, not 4/3; all-identical n=4
   → 1/2). `tests/test_ebft.py` pins these exact numbers.

**Structural insight worth remembering:** after whitening, the Gram matrix of
*distinct* rollouts is ≈ identity, so the diversity term is essentially a
**duplicate penalty** — it only fires on repeated/near-identical rollouts (n_k>1
groups). That's why temp-0.6/G=8 duplicate rollouts (common at short horizons)
are handled by design rather than needing a special case.

---

## Verified (Session 39)

- **CPU suite** (`python tests/test_ebft.py`) — whitening orthonormalization, gt
  scalar-multiple property, duplicate-gating exact values, RLOO closed forms,
  no-whiten composition, zero-feature finiteness, sampler top-k/top-p. All pass.
- **Model self-test** (`--self-test` on Llama-3.2-1B-Instruct EXL3 @4bpw):
  - tap position-gather == full-stream slice;
  - `compute_logps` == gather(log_softmax(`net.logits`));
  - one full EBFT step is finite with grads on **all 112 adapter tensors and
    nothing else** (φ / base isolation confirmed).
- **30-step smoke train** (Llama-3.2-1B, semancy, r32, lr 1e-5, G8/n4/anchors4,
  temp 0.6, γ 0.03). Held-out eval moved the right way:

  | step | CE ↓ | CFM ↓ | reward ↑ |
  |---|---|---|---|
  | 0 | 3.4294 | 0.8623 | +0.4537 |
  | 10 | 3.3838 | 0.8134 | +0.4980 |
  | 20 | 3.3424 | 0.8165 | +0.4954 |
  | 30 | 3.2965 | 0.8145 | +0.4965 |

  Stable throughout: grad norm 0.9–1.9 (no spikes), |dB| 0→0.22 smooth, RL term
  oscillating ~0 (correct for a zero-mean RLOO advantage), dup 0–9%/step with
  whitened diversity penalizing exactly those steps. **~32 s/step**, 5.08 GB peak,
  76% of step time in forward (the sampler's G re-forwards, as predicted).

  ⚠️ This proves *correct and stable*, not *works*. 30 steps / 231 examples /
  one seed / γ=0.03 means CE could be moving on the CE term alone. Attribution
  needs the A/B.

---

## How to run

```bash
# 1. correctness gate on your model (fast, ~2 min on a 1B)
python training/qlora_train_ebft.py \
    --model /path/to/exl3_model --dataset <id-or-path> \
    --messages-key messages --self-test

# 2. a real run (validation-scale, Qwen2.5-1.5B / Llama-3.2-1B class)
python training/qlora_train_ebft.py \
    --model /path/to/exl3_model --out out/exl3_ebft \
    --dataset <id-or-path> --mode qa \
    --gen-len 8 --n-samples 4 --anchors 4 --temperature 0.6 \
    --lr 1e-5 --epochs 1 --eval-every 50 --val-frac 0.02

# 3. verify the adapter on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3_model --adapter out/exl3_ebft
```

Or launch from a YAML config through the shared launcher (`method: ebft`), which
is how the SFT-vs-EBFT A/B is set up so both arms are one editable pair:

```bash
python training/qlora_train.py --config semancer_llama1b_ebft.yaml
# pairs with semancer_llama1b_sft.yaml (identical data/LoRA/optim); overlay the
# two run reports afterwards:
python training/run_report.py out/ab_sft out/ab_ebft --labels SFT,EBFT -o out/sft_vs_ebft.html
```

The EBFT objective knobs are documented in the "Energy-Based Fine-Tuning"
section of `training/qlora_train_config.yaml`; `method: ebft` forwards them to
this trainer and rejects sft-only keys early.

Reference-code defaults are baked in: G=8, n=4, temp=0.6, align=1.0, div=0.5,
ce(γ)=0.03, betas=(0.9,0.95). `--mode pretrain` trains on raw packed text
(non-verifiable setting). Frozen LM head only; single-process (`--parallel
single|split`); no packing / `--train-head` / `--train-embeddings`.

**Key knobs:** `--lr` (paper full-FT is 1e-6; LoRA range 1e-5..1e-4 is
**unvalidated**), `--anchors`/`--n-samples`/`--batch` (rollout rows =
batch·anchors·n_samples, the real memory/compute knob), `--no-whiten` (ablation
— paper shows clear degradation without it), `--div-coef` (1.0 = pure
proper-scoring-rule; 0.5 = the paper's α=0.5 alignment bias),
`--rollout-sampler native` (KV-cached sampling, ~2× step, dense models only —
see Architecture), `--eval-cfm-samples N` (cap the rollout part of eval at a
fixed-seed subset; CE eval stays full).

---

## Open work (all needs user-approved GPU time — see standing orders in handoff §Session 38)

1. ~~**The real question: EBFT-LoRA vs SFT-LoRA A/B**~~ — **first pass DONE
   (Session 42): EBFT did not beat SFT at 1B/1-epoch** (see "Result: first A/B"
   above). Still open: repeat on **Qwen3.5-4B @4bpw** (configs
   `semancer_qwen35_4b_{sft,ebft}.yaml`, identical hyperparameters) and, if
   pursuing further, a **γ=0 arm** to isolate pure feature-matching and a
   **downstream-accuracy** eval instead of the CE/CFM proxies.
2. **LoRA LR sweep** 1e-5..1e-4 (the one genuinely unknown hyperparameter) —
   now the highest-value next experiment: EBFT ran at its smoke lr 1e-5, so the
   Session-42 null result may just be an untuned LR.
3. ~~**Throughput:** native-generator sampling via `apply_to_native`~~ —
   **DONE Session 41** (`--rollout-sampler native`, 2.06× step + faster eval;
   see the sampler section above and the eval-cost paragraph below). Remaining
   throughput lever: Appendix-F strided-mask rollouts (amortize all anchors
   across one sequence; bigger engineering, only worth it post-validation).
4. **Scale up** only if the A/B validates.

**Eval cost (Session 40 finding, Session 41 fix).** `--eval-cfm` (default on)
runs the full rollout pipeline over the val set each eval — measured ~180 s
per eval on the 116-example A/B val set with the exact sampler, ~half the
quarter-epoch wall clock. Levers now built: the native sampler cuts it to
~95–103 s, and `--eval-cfm-samples N` caps the CFM pass at a fixed-seed subset
(same subset every eval AND every run, so trends stay comparable) — N=32
measured 31–32 s/eval with the CFM trend preserved (capped 0.901→0.866 vs
full 0.888→0.838 over the same 2 steps). CE eval always covers the full val
set. The `[eval]` log line now prints eval seconds.

Cost note: with the exact sampler the A/B over ~5–10k examples is
overnight-class at 1–1.5B on one 3090 (~20–32 s/step, 8 seq/step,
sampler-forward-bound); `--rollout-sampler native` + `--eval-cfm-samples 32`
roughly halves the step time and makes eval near-free.

---

## Provenance / gotchas for a future session

- Reference repo lives in this session's scratchpad only (not in-repo); re-clone
  `sjelassi/ebft_openrlhf` if you need to re-check semantics. Key files read:
  `openrlhf/utils/embedding_utils.py` (whitening, alignment/diversity),
  `.../ebft_experience_maker.py` (`compute_baseline` = RLOO), `models/loss.py`
  (`EBFTPolicyLoss`), `models/critic.py` (feature extraction),
  `scripts/run_ebft_example.sh` (default hyperparameters).
- Naming collision: arXiv:2402.12419 is an *unrelated* 2024 "EBFT" (sparse-LLM
  block fine-tuning) that *does* mention LoRA — not prior art for this.
- Live-sample generations (`--sample-every N` / `--sample-prompt`, YAML keys of
  the same name) are wired into the EBFT trainer (Session 42): between steps it
  installs the current adapter via `apply_to_native()` and generates on the
  native inference path, reusing the `--rollout-sampler native` generator when
  present. Same knobs/behavior as the SFT trainer, so both A/B arms print
  comparable previews. (MoE caveat: like native rollouts, the preview reflects
  the BASE experts — the trainer warns.)
