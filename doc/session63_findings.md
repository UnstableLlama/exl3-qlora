# Session 63 — why the embed/head LoRAs did nothing

**Headline: not a bug. The module LoRAs trained at the same relative rate as the
per-linear adapters that demonstrably worked, both halves of each moved, and the
`lora_head` forward is numerically correct. S62's leading hypothesis — that the
single param group + global grad clip starved them — is REFUTED.**

Everything below was measured from artifacts already on disk. No training run and
no model load was needed.

---

## 0. Method: the optimizer state is a recording of the gradients

`trainer_state.pt` in every v7 checkpoint carries the full `paged_adamw8bit`
state. bitsandbytes stores, blockwise-quantized (block 256, independent absmax
for each):

    state1 = exp_avg     (m, EMA of g)      state2 = exp_avg_sq  (v, EMA of g^2)
    dequant: x = qmap[state] * absmax[block]

From that, per parameter element:

    sqrt(v_hat)                 RMS recent gradient magnitude
    |m_hat| / (sqrt(v_hat)+eps) the Adam step as a FRACTION OF lr  <-- the one that matters

Probe scripts are in the scratchpad (`adam_state_probe.py`, `support_probe.py`,
`probe2.py`); all numbers below are from `checkpoint-00000160`.

Sanity check that the dequant is right: 0% of elements exceed the theoretical
bound `|m_hat|/sqrt(v_hat) <= ~3.2`. An earlier pass that did not mask elements
where `v` quantizes to exact zero while `m` does not produced impossible values
(964, 61290) — that mask is required.

---

## 1. The param-group / clip finding, confirmed and then dismissed

Confirmed directly from the saved state, not just from reading the source:
`param_groups` has **1 group with 836 params** = 832 per-linear + exactly 4
module LoRA tensors (identifiable by shape: `embed_a [202048,32]`,
`embed_b [32,6656]`, `head_a [6656,32]`, `head_b [32,202112]`, appended last by
`lora_parameters()`).

The clip fired on **240 of 240 steps** (median logged `grad_norm` 10.387, clipped
to 1.0 — v6's median was 10.342, so this is normal, not a v7 pathology). The sum
of `v` over all 836 params is **0.994**, which independently confirms the
optimizer sees post-clip gradients normalized to 1.0.

And the per-linear block really does own the gradient norm:

| group | ‖g‖ | % of total |
|---|---|---|
| per-linear LoRA (832 tensors) | 0.9916 | **99.52 %** |
| embed_lora_a | 0.0001 | 0.00 % |
| embed_lora_b | 0.0443 | 0.20 % |
| head_lora_a | 0.0001 | 0.00 % |
| head_lora_b | 0.0532 | 0.29 % |

**This is irrelevant, because Adam is scale-invariant.** The clip multiplies
every gradient by one scalar *c*; that *c* enters both `m` and `√v` and cancels
out of `m/√v`. Raw gradient magnitude does not set Adam's step size. The
quantity that does:

| param | median √v̂ | eps damping | damping if UNCLIPPED | **true step / lr** |
|---|---|---|---|---|
| per-linear param 0 (reference) | 5.46e-06 | 0.998 | 1.000 | **0.1613** |
| embed_lora_a | 1.22e-08 | 0.549 | 0.927 | **0.0303** |
| embed_lora_b | 9.16e-05 | 1.000 | 1.000 | **0.1751** |
| head_lora_a | 1.79e-07 | 0.947 | 0.995 | **0.2719** |
| head_lora_b | 1.17e-06 | 0.991 | 0.999 | **0.0741** |

Three of the four halves move at or above the per-linear reference rate. Nothing
is starved by lr, and nothing is starved by the clip.

### The one real (but minor) clip effect

`embed_lora_a`'s post-clip gradient RMS is 1.2e-8, sitting on Adam's
`eps=1e-8`, so it eats a 0.55 damping factor. That damping IS caused by the
clip: unclipped its √v̂ would be ~10× larger and the factor would be 0.93. So the
global clip roughly halves that one tensor's step rate, via the eps floor rather
than via magnitude.

Judged not worth fixing on its own: a LoRA product `a@b` only needs one factor
moving, and `embed_lora_b` moves at full rate (0.175). Revisit only if the
amplification sweep (§4) says the adapter is undertrained, in which case fold an
eps/clip change into that rerun rather than shipping a standalone knob.

---

## 2. Both halves move (S62 open question #4)

Displacement from the step-40 state to step 240:

| | embed_a | embed_b | head_a | head_b |
|---|---|---|---|---|
| displacement | 0.0432 | 0.0374 | 0.1788 | 0.3671 |
| live fraction | 7.8 % | 100 % | 99.96 % | 20.8 % |

Neither kaiming-init `a` is stuck. The two partial live fractions are both
expected and explained: `embed_lora_a` is token-indexed and only rows for tokens
present in a batch get gradient (15,706 distinct tokens over 798 rows of one
voice is plausible); `head_lora_b`'s 20.8 % is explained in §3.

### The trajectory control — the cleanest single result here

Fitting displacement ~ step^p over the six checkpoints (40/80/120/160/200/240):

- module LoRAs: p = 0.264, 0.241, 0.221, 0.339
- **per-linear control** (6 tensors, layer 0, both A and B): p = 0.231, 0.248,
  0.218, 0.232, 0.206, 0.251

Statistically indistinguishable. The saturating shape (p well below a random
walk's 0.5) is the cosine LR decay, not the module LoRAs failing — the adapters
that produced the entire v6/v7 gain follow the identical curve. Absolute
displacement is well above a random walk in both cases, so this is directional
movement, not noise.

---

## 3. The `lora_head` forward is correct, not merely non-crashing

Reference-checked with the REAL trained `lm_head.lora_a/b` from ckpt-160 against
an fp64 computation of `logits + scale*(hs@A)@B`:

1. **Formula.** Relative error **4.177e-04** — pure fp16 matmul noise. Shapes,
   orientation and the 5.65685 scale are all right. `7e8f2bd` is a correct fix.
2. **The fp16 add.** The delta is added to fp16 `logits`. Across logit rms 2–40
   the delta is 1.3–25 ulps and **~100 % of its norm survives** the add. (70–75 %
   of individual entries round away, but those are the negligible ones; the norm
   is preserved.)
3. **grad_logits fp16 underflow — investigated and benign.** `cross_entropy`'s
   gradient is `(softmax − onehot)/N`, computed in fp16 (min subnormal 5.96e-8).
   For a realistic peaked softmax over 202k classes this is exactly zero for
   99.5 %+ of *entries* — but it retains **98.9–100 % of the L1 gradient mass and
   100 % of L2**. Only the negligible tail dies.

   This also explains `head_lora_b`'s 20.8 % live columns: the truncation
   predicts 20–41 % live columns at top-1 probability 0.3–0.6, which matches.
   An observation that looks alarming in isolation and is not.

---

## 4. What is still open

The audit rules out a bug. It does NOT distinguish:

- **Undertrained** — direction right, magnitude small. 240 steps at lr 1e-5 is a
  small budget; the module LoRAs moved roughly half as far as that budget allows.
- **Dataset ceiling** — 798 rows of one voice need no vocabulary-level
  adaptation; the per-linear adapters already saturate what this data teaches.

`out/muse_glimmer_v6_v7_comparison/amplify_module_scale.py` separates them
without retraining: it amplifies the trained delta's scale (×0.5 … ×16 of
5.65685) on ckpt-160 and re-measures held-out loss, 7 arms off one model load,
~13 min.

- loss keeps improving past ×1 → undertrained → a higher-lr rerun is justified
  (and fold in the §1 eps fix)
- loss degrades immediately past ×1 → the adapter is at its local optimum →
  **0.0011 nats is the real answer for this dataset**, and the question closes

Anchors from S62: per-linear only = 1.8530, ×1 (trained) = 1.8519.

### RESULTS (S64) — undertrained, decisively

Ran 2026-08-28, 7 arms off one model load, ~65 s/arm. Deterministic eval (116
held-out rows, `inference_mode`, no sampling), so the differences between arms
are exact, not sampling noise.

| arm | module scale | held-out loss | vs stripped |
|---|---|---|---|
| per-linear only | — | 1.85299 | 0.0000 |
| ×0.5 | 2.828 | 1.85241 | −0.00058 |
| **×1 (as trained)** | 5.657 | 1.85194 | **−0.00105** |
| ×2 | 11.314 | 1.85110 | −0.00190 |
| ×4 | 22.627 | 1.84997 | −0.00303 |
| **×8** | 45.255 | 1.84934 | **−0.00365** ← best |
| ×16 | 90.510 | 1.85547 | +0.00248 |

Monotone improvement from ×0.5 through ×8, then a sharp break at ×16. The
minimum along this line sits around ×8–10.

Two things follow:

1. **The direction is right and the magnitude is the limit.** Compare against a
   purely linear response (−0.00058 per ×0.5): observed is 89 % of linear at ×1,
   81 % at ×2, 65 % at ×4, 39 % at ×8. At the trained point the loss is still
   *nearly first-order* in the delta magnitude — the adapter never entered the
   curved region of its own descent direction. It stopped ~8× short of a plain
   line search along the direction it had already chosen. That is the textbook
   undertrained signature, and it rules out "the adapter sits at its local
   optimum".

2. **But the prize along this line is still small.** The best attainable is
   −0.0036 nats, 3.5× the trained −0.0011 and no more. A retrain would co-adapt
   the per-linear adapters rather than ride this exact ray, so −0.0036 is not a
   hard ceiling — but nothing here suggests a large win is hiding.

### It is not logit calibration, and the embedding LoRA is dead weight

Two follow-ups settled what the amplified delta actually is. Scripts and result
JSONs are in `science/module_lora_v7/` (`head_delta_spectrum.py`,
`split_head_embed.py`); the 70 KB generation dump stays in
`out/muse_glimmer_v6_v7_comparison/gens_x8.jsonl` with the other run outputs.

**Spectrum + frequency, no model load.** Exact rank-32 SVD of `scale*(a@b)` via
QR, never materializing the [6656, 202112] product:

| | head delta | embed delta |
|---|---|---|
| Frobenius norm | 22.78 | 12.64 |
| energy in top 1 direction | 25.7 % | 13.1 % |
| effective rank (participation ratio) | 10.07 / 32 | 21.27 / 32 |
| mean per-token ‖Δ‖, seen vs unseen | 0.0491 vs 0.0189 = **2.59×** | 0.0280 vs 0.0279 = **1.00×** |
| Spearman(‖Δ‖, unigram count) | **−0.240** | +0.010 |
| ‖Δ‖ by frequency decile (low→high) | 0.056 … 0.037 | 0.0280 … 0.0281 (flat) |

The head is **not** a calibration shift: a frequency-prior correction would be
one or two dominant directions with ‖Δ‖ *rising* in frequency. Observed is a
spread spectrum, 2.59× more movement on tokens the data actually contains, and a
−0.24 anti-correlation — rarer seen tokens move more. Real, targeted adaptation.

The embed delta is flat to three decimals across every frequency decile, which
has a mechanical cause worth recording: **whichever half carries the token axis
decides whether per-token structure can exist.** For the head that is `head_b`,
the zero-init half, fully trained (100 % live). For the embedding it is
`embed_a`, the *kaiming-init* half, at 7.8 % live rows (§2). The live rows are
the seen tokens (7.8 % live vs 6.7 % of vocab seen — they match), but they moved
too little relative to their random init to shift the norms. So the embed LoRA is
a **random hash** of token id into a learned shared direction: all the learning
sits in `b`, and the token→subspace assignment is still whatever kaiming handed
it. `embed_a` is also the tensor eating the 0.55 eps damping from §1.

**Attribution sweep** (same eval, floor 1.85299; the loader keys off each module
independently so a one-module checkpoint loads exactly that module):

| arm | loss | vs floor |
|---|---|---|
| head only ×1 | 1.85195 | −0.00104 |
| head only ×8 | 1.84958 | **−0.00342** |
| embed only ×1 | 1.85298 | −0.00002 |
| embed only ×8 | 1.85282 | −0.00017 |
| both ×8 (harness check) | 1.84934 | −0.00365 |

The harness check reproduced `amplify_results.json`'s ×8 arm bit-for-bit
(1.849344963470009). Contributions are additive to within 1.7 %, and the
embedding LoRA is **1.8 % of the effect at ×1, 4.7 % at ×8** — a no-op inside a
no-op. The head LoRA is the entire story.

### Generation check at ×8 — no degeneration (`gens_x8.jsonl`)

A held-out loss win from amplifying an adapter 8× past its trained scale does not
by itself license shipping at ×8, so the 9-question set was re-run at ×1, ×8
head-only and ×8 both (plus a free base arm), one model load, seed 1234,
temp 0.95 / min_p 0.04 / top_k 50.

| arm | mean words | 3-gram repetition (avg / max) |
|---|---|---|
| base | 323.7 | 0.055 / 0.128 |
| ×1 as trained | 241.6 | 0.014 / 0.057 |
| ×8 head only | 229.3 | **0.013 / 0.029** |
| ×8 both | 212.0 | 0.011 / 0.040 |

All 27 adapted generations are coherent; ×8 repeats *less* than ×1, not more, and
the shortest ×8 output (a 40-word birthday message) is simply concise, not
truncated or broken. ×8 head-only and ×8 both are near-indistinguishable, as the
attribution table predicts.

One honest caveat, weak evidence: on the TCP/UDP OOD prompt the ×8 answer
editorializes in-voice ("embarrassing that the old answer has become tribal
lore") where ×1 makes the same corrective opening move without the flourish. That
reads as ×8 *amplifying* register bleed that already exists at ×1 rather than
introducing a new failure — but it is 1 of 3 OOD prompts at a single seed, and
should not be treated as established. ×4 keeps 83 % of the loss gain (−0.0030)
if a more conservative operating point is wanted.

---

## 5. Verdict on S62's five candidate explanations

| # | hypothesis | status |
|---|---|---|
| 1 | Magnitude / undertrained | **CONFIRMED** (§4). Loss improves monotonically out to ×8 and is still near-linear in the delta at ×1 — the adapter stopped ~8× short of a line search along its own direction |
| 2 | LR / no separate group / global clip | **REFUTED** (§1). Adam is scale-invariant; the clip cancels out of m/√v. Only residue is a 2× eps-floor damping on `embed_lora_a` alone |
| 3 | B=0 init + short run | **PARTIALLY REFUTED** — trajectory is indistinguishable from the per-linear control (§2); folded into #1 |
| 4 | Gradient actually flowing? | **REFUTED** (§2, §3). Both halves move, forward matches fp64 to 4.2e-4 |
| 5 | Dataset ceiling | **REFUTED as the explanation for 0.0011** (§4) — more of the same delta keeps helping, so the data was not exhausted at ×1. Partially reinstated as a *magnitude* statement: even the best point on this line is only −0.0036 nats |

## 6. Commits

Everything from S62 landed on `master` and was pushed (`340253a..1edab55`):

- `cb8177b` module LoRA scale lost on load after a PiSSA export (the real bug fix)
- `7c6464a` inference: jinja templates, multiple adapters off one model load
- `cae4c4f` keep the final-step adapter under `--save-best`; live_report via YAML
- `1edab55` run log: Muse-Glimmer semancy v5/v6/v7

(`7e8f2bd`, the lora_head dtype fix, was already committed in S62.)

## 7. Fans

Set to 66/66 at session start, verified 66/66 at 39/36 °C. `out/dim_fans_30.sh`
drops them to 30.
