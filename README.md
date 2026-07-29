# exl3-qlora

**QLoRA fine-tuning directly on EXL3-quantized models.** This repo began as a fork of [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) and still contains the full inference library (synced to upstream v1.2.1 — see the [upstream README](https://github.com/turboderp-org/exllamav3#readme) for inference, conversion, and installation documentation). On top of it there is a self-contained, **transformers-free** training path: a differentiable forward built over the EXL3 trellis (validated against the native inference forward to 100% argmax agreement), a plain-PyTorch trainer, and adapters that save in standard PEFT format.

### Why train on EXL3 instead of bitsandbytes?

- Train on 2-8bpw quants, including non-integer. BNB is 4 or 8 bit.
- EXL3 is higher accuracy than BNB, theoretically this should help? - still needs benchmarking
- EXL3 is more performant in the lower and mid-size batches that are often used when squeezing big models onto consumer cards.

### Quick start

```bash
# install: PyTorch (CUDA 12.4+) first, then this repo (builds the CUDA extension),
# plus the one training dep
pip install -r requirements.txt
pip install .
pip install datasets            # optional extras: flash-attn, liger-kernel, bitsandbytes

# 1. prove the differentiable forward is correct for YOUR model (run this first)
python training/qlora_validate_native.py --model /path/to/exl3-model --compute-dtype bfloat16

# 2. edit the config, then train
python training/qlora_train.py --config training/qlora_train_config.yaml

# 3. before/after comparison on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3-model --adapter out/my_adapter
```

Everything is driven by one YAML file ([`training/qlora_train_config.yaml`](training/qlora_train_config.yaml) is the fully-commented reference — its keys mirror the CLI flags of `training/qlora_train_native.py` one-to-one). A minimal config looks like:

```yaml
model: /models/Llama-3.2-3B-Instruct-exl3-4bpw
out: out/my_adapter
parallel: split           # single | split (layer-split across GPUs) | ddp (torchrun)

r: 64
alpha: 64.0               # use alpha = r with init_lora: pissa
lr: 5e-6
epochs: 1.0
batch: 3
seq_len: 8192
pack: true                # sample packing (best-fit-decreasing, ~98% fill)

dataset: /data/my-set.jsonl
messages_key: messages    # OpenAI-style chats; or instruction/context/response keys
prompt_format: auto

init_lora: pissa          # default | pissa | qerr | eva  (see below)
eval_split: test          # held-out eval from the dataset's own split
eval_every: 50
compute_dtype: bfloat16
use_liger: true
optim: paged_adamw8bit
```

### What's in the box

- **Single-GPU, multi-GPU layer-split (`parallel: split`), and DDP (`parallel: ddp`)** training of LoRA adapters over a frozen EXL3 base — plus optional embedding/LM-head training (full or low-rank).
- **Preference optimization: DPO, KTO, and SimPO** (`training/qlora_train_pref.py --method dpo|kto|simpo`, or via the YAML launcher with `method: dpo|kto|simpo` and the `qlora_train_pref_config.yaml` template). DPO/KTO use the frozen quantized base as the reference model (adapter-disable trick — no second model copy); SimPO is reference-free (length-normalized rewards + target margin γ — no reference forward, so a step costs about half a DPO step). Loss semantics follow [HuggingFace TRL](https://github.com/huggingface/trl)'s stable `DPOTrainer`/`KTOTrainer`/`CPOTrainer` (with credit — see below), so β/loss-variant hyperparameters transfer directly; variants: sigmoid/cDPO, hinge (SLiC), IPO, KTO, APO-zero-unpaired, SimPO (+ optional CPO-style SFT mix).
- **Memory levers** for long context on consumer cards: gradient checkpointing, activation offload to CPU RAM, fused/chunked cross-entropy (chunked over the vocab too, for 256k-vocab models), 8-bit and paged optimizers, Liger kernels (RMSNorm/RoPE/SwiGLU).
- **A real eval harness**: held-out loss from your dataset's own split (`eval_split`) or a carved fraction (`val_frac`), an optional second monitor set (`eval2_*`, e.g. wikitext LM loss watched next to your task loss), `save_best` checkpointing, periodic live sample generations, and a per-run CSV log of hyperparameters/losses/VRAM/throughput.
- **Correctness gates, not vibes**: `qlora_validate_native.py` checks the differentiable forward against the native inference forward, the Liger backward against plain torch, packing isolation, and each adapter init's step-0 math — before you spend GPU-days. A CPU test suite covers the gradient path end-to-end.
- **Standard outputs**: adapters save as PEFT-format safetensors, loadable by exllamav3's native LoRA loader (TabbyAPI), PEFT, or merge scripts. Runtime adapters apply correctly on every inference path; note that decode is slower **while an adapter is loaded** (the graph-fused decode paths can't run under one — `unload()` restores full speed), and deploying via merge-and-requantize has no hit at all.

### Supported architectures (training)

The differentiable forward reads every norm/activation/scale from the loaded modules and is validated against the native inference forward per architecture. Unsupported layouts are **rejected loudly at construction** — nothing silently mistrains.

| Architecture family | Examples | Status |
|---|---|---|
| Llama (plain pre-norm dense) | Llama 3.x, DeciLM-lite | **Box-proven** (SFT, DPO/KTO, packing, DDP, split) |
| Mistral dense | Mistral 7B v0.3, Mistral-Nemo (Rocinante-XL-16B), Mistral Small/Medium 3.x (`mistral3`) | **Box-proven** (16B metharme SFT; Medium-3.5-128B) |
| Qwen2 dense | Qwen2/2.5 | Accepted (same plain path as Llama) |
| Qwen3 dense | Qwen3 4B/8B/14B | **Box-proven** (q/k-norm path) |
| Qwen3-MoE / Qwen3.5-MoE | Qwen3-30B-A3B, Qwen3.6-35B-A3B | **Box-proven** (std softmax router; shared expert + sigmoid shared gate; routed-expert adapters opt-in via `expert_*` targets) |
| Qwen3.5/3.6 hybrids | Qwen3.5 0.8B–4B, Qwen3.6-27B | **Box-proven** (differentiable Gated DeltaNet + gated attention; no sample packing on GDN models) |
| Gemma 3/4 | Gemma3, Gemma4-12B, Gemma4 MoE (MeroMero-26B) | **Box-proven** (sandwich norms, GeGLU, sliding/full, softcap, big-head, Gemma4 MoE alt-residual layout) |
| AFMoE | **Trinity-Nano** (Arcee), dots.llm1-style sigmoid routers | **Box-proven** (10-step SFT + fast-vs-legacy A/B + adapter steering generation at inference) — dots sigmoid router (selection bias, normalize-over-selected, route scale), full-width attention output gate, NoPE full-attention layers, muP embedding, ungated shared expert, dense-first-N layers |
| Mixtral | Mixtral 8x7B | Accepted (std router, no shared expert) — not yet box-tested |
| Qwen-VL text towers | Qwen2.5/3-VL | **Box-proven**, text-only (mRoPE collapses to 1D RoPE; vision tower not trained) |
| Rejected loudly | Qwen3-Next (fused-qkvz GDN), grouped ds3-router MoE (DeepSeek-V3), headwise attention gating, non-NeoX RoPE | — |

MoE note: the plain `gate_proj`/`up_proj`/`down_proj` targets adapt dense MLPs and the always-active shared expert; routed experts are opt-in (`expert_gate_proj` etc., with `--expert-r` for rank). Routers stay frozen. On AFMoE the *attention* gate is keyed `self_attn.gate_proj` in the checkpoint and rides the `gate_proj` target (or `attn_gate_proj` to adapt it alone).

### Prompt formats (`--prompt-format` / `prompt_format:`)

| Format | Template | Use with |
|---|---|---|
| `auto` (default) | The model's own architecture template (`default_chat_prompt`) + arch-correct turn-end token | Any supported base |
| `llama3` | `<\|start_header_id\|>…<\|eot_id\|>` headers | Llama-3 family (explicit / cross-arch) |
| `mistral` | `<s>[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]` (V7+/V13, no spaces) | Mistral instruct family |
| `chatml` (= `qwen3.5`) | `<\|im_start\|>role\n…<\|im_end\|>` | Qwen, **Trinity/AFMoE**, any ChatML base |
| `qwen3.5-nothink` | ChatML with the `<think>` block pre-closed empty | Qwen3.5/3.6 reasoning bases, trained to answer directly |
| `gemma4-nothink` | Gemma4 turns with the thought channel pre-closed | Gemma4 |
| `metharme` | `<\|system\|>/<\|user\|>/<\|model\|>` markers | Pygmalion-style tunes on any base |

All formats do exact prompt/response boundary masking (prompt and response are tokenized separately) and single-BOS normalization; verify any new base with `--inspect 3` before training.

### Modern PEFT: SVD adapter initializations

Short SFT runs spend a large fraction of their steps just growing the adapter off the ground (the default zero-init of B). This fork implements the current crop of SVD-based initializations, adapted to an immutable quantized base — select with one config key, `init_lora`:

- **`pissa`** ([PiSSA](https://arxiv.org/abs/2404.02948)) — the adapter starts as the top-r principal components of the base weights, trained against a residual base realized as a frozen offset (the trellis itself is never rewritten). Exports as a converted rank-2r standard LoRA so any consumer loads it correctly. **Current default recommendation: it won its first A/B clearly** (use `alpha = r`).
- **`eva`** ([EVA](https://arxiv.org/abs/2410.07170)) — A is initialized to the top-r right-singular vectors of each layer's *input activations*, streamed from your actual training data through the actual quantized forward in a short pre-pass. Function-preserving at step 0. Freshly built; being evaluated against pissa now.
- **`qerr`** (LoftQ-style, single-shot) — the adapter starts as the closest rank-r repair of the *quantization error* vs the original bf16 weights, aimed at the low-bpw regime where that error is large.
- **`use_rslora`** — rank-stabilized scaling (`alpha/sqrt(r)`) for rank sweeps.

Each init has a hard step-0 gate in `qlora_validate_native.py --init-lora <mode>`.

### Credits

The DPO/KTO/SimPO preference-training implementation follows the loss semantics of **[HuggingFace TRL](https://github.com/huggingface/trl)**'s stable `DPOTrainer`, `KTOTrainer` (KTO stabilized in [trl#6175](https://github.com/huggingface/trl/pull/6175)), and `CPOTrainer` (`loss_type="simpo"`). TRL is Apache-2.0 licensed, Copyright The HuggingFace Team; this fork reimplements the formulations independently against the EXL3 native training path rather than reusing TRL code. Underlying methods: DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)), KTO ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)), IPO, SLiC, SimPO ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)), CPO ([Xu et al. 2024](https://arxiv.org/abs/2401.08417)).

### Project status

Research project under active development. It started as an exllamav3 fork, but has diverged enough to stand alone (upstream is still merged in periodically — currently at v1.0.0 parity — so the inference side stays current). The core mechanism is proven end-to-end: validated forward parity on the quantized weights, healthy trainings from 1B to 16B models on 1–2× RTX 3090 (including 8k-context packed runs on a 12B), adapters that load and steer generation on the native inference path. Training-side architecture support currently covers **Llama-family, Gemma 3/4 (incl. Gemma4 MoE), Qwen3-dense, Qwen3-MoE, Qwen3.5/3.6 hybrids (differentiable Gated DeltaNet + gated attention) incl. Qwen3.5-MoE, AFMoE (Trinity-Nano — dots sigmoid router, gated attention, NoPE), and Mistral(-Nemo) dense models** (see the supported-architectures table above; no sample packing on Gated DeltaNet models); unsupported features are rejected loudly rather than silently mistrained. Interfaces may still move between sessions — the full engineering log with per-session results and rationale lives in [`doc/qlora_handoff.md`](doc/qlora_handoff.md), and experiment-specific tooling is quarantined in [`training/experiments/`](training/experiments/).
