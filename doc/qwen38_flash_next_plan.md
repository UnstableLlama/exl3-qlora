# Plan: Qwen3.8-Flash-Next (qwen4_exp) on the native QLoRA path, down to a 16 GB card

Written 2026-10-09 (Session 57, after the v1.6.0 sync, PR #167). Status: **Phase A built and
CPU-tested in Session 58 (branch `feat/qwen38-flash-next`); box validation pending. Phases B–D
not started.** Session 58 closed open question 6: the hc `up` table is recovered from the resident
`upx_h` repack, so no `keep_source_weights` load is needed.
Goal: a short SFT run (~30 steps) of `Qwen/Qwen3.8-Flash-Next` through `qlora_train_native.py`,
first on a box that holds the model, then on a single 16 GB GPU with the routed experts streamed
from host RAM.

## 0. What the model is (from its config + the inference architecture file)

`exllamav3/architecture/qwen4_exp.py`, `Qwen4ExpModel`. Module list in forward order:

```
Embedding -> ExpandStreams -> [block 0, block 1, PLELayer, block 2 ... block 47]
          -> GatedResidual(use_combine=False, "hyper_connection_mixer") -> Linear lm_head
```

- 48 blocks: 12 x (3 x GatedDeltaNet -> MoE, 1 x Attention(QSA) -> MoE). Every block is a
  `TransformerBlock` with `attn_hc` / `mlp_hc` = `GatedResidual` sites in place of the
  input/post norms. The residual between blocks is a **(bsz, seq, 4, 2560) fp32 stream stack**.
- `ExpandStreams`: broadcast the embedding into the 4 streams (stateless).
- `GatedResidual` site (`hyperconnections.py:305`): per-stream grouped RMSNorm (weight applied
  as 1 + w), low-rank gate `sigmoid(up(silu(down(normed_flat) / 4)))`, elementwise MEAN of the
  gated normed streams feeds the sublayer; sublayer output injected back with a per-stream
  scalar `2 * sigmoid(inject(normed_flat) / 4)`. No Sinkhorn, no combine matrix. `_mix_ref()`
  is a pure-torch fp32 reference. Tensors: `hc_norm.weight`, `input_mix_weight_down.weight`,
  `input_mix_weight_up.weight`, `block_inject_weight.weight` (rank 320).
- Final mixer: same module with `use_combine=False`; collapses the stack to (b, s, 2560). **There
  is no final RMSNorm.**
- `PLELayer` (`ple.py:128`, sits before block index 2, 1-based `ple_layer_ids=[2]`): hashed
  n-gram lookup (`NGramEmbedding`, 51.2B-param table, (ngram_size-1)*heads_per_ngram = 16 rows
  per token, hashing on CPU, rows gathered from RAM or disk) -> key/value projections to
  4 x 2560 -> grouped RMSNorms -> signed-sqrt dot-product gate scaled by 1/sqrt(hidden) ->
  depthwise dilated causal conv (kernel 4, dilation 3) -> delta added into the streams.
  `forward_streams_reference` is the op-by-op form. Carries `ngram_size-1 = 2` previous token
  ids and `(k-1)*dilation = 9` conv positions of state: **no sample packing** (same rule as GDN).
- GDN blocks: Qwen3.5 split-projection layout (in_proj_qkv/z/b/a), already supported, EXCEPT the
  gated output norm uses `gate_activation = "sigmoid"` (`output_gate_type`), where
  `training/gdn.py:139` hardcodes `silu(z)`.
- Attention blocks: GQA 24q/2kv, head_dim 256, interleaved gate (supported), q/k RMSNorm with
  constant_bias 1.0 (supported), partial rotary 0.25 -> rotary_dim 64 (supported), mRoPE
  interleaved sections [11,11,10] (text-only collapses to 1D, supported), plus a `QSAIndexer`.
  **QSA is exactly dense attention for query positions below `sparse_threshold() = 4*block_topk+3
  = 2051`** (`qsa_indexer.py:473`; block_topk = 2048/4 = 512). At seq_len <= 2048 the indexer
  never engages. The indexer's own projections (`self_attn.indexer.*`) are then dead weight.
- MoE: 512 experts, top-10, softmax std router, shared expert + sigmoid shared gate: the
  Qwen3.5-MoE path, box-proven.
- Params: ~121B routed experts, ~4B dense (incl. 248320 x 2560 embed and untied head), 51B
  n-gram table, 4B MTP head (ignored), ~0.4B vision tower (ignored, text-only run).

Weight bytes of the available quants (turboderp/Qwen3.8-Flash-Next-exl3; `_hN_ngN` = head bits,
n-gram bits):

| | 2.05bpw_h4_ng4 | 3.05bpw_h5_ng5 |
|---|---|---|
| routed experts | ~31 GB | ~46 GB |
| n-gram table | ~26 GB | ~32 GB |
| dense trunk + embed + head | ~2 GB | ~2.5 GB |

(3.05 pack is reported as ~79 GiB total, which matches.) Requires exllamav3 >= 1.4.5; we are at
v1.6.0 parity.

## 1. Why the training path rejects it today

`training/backbone.py::_decoder_layout` asserts: `mods[1..first_block)` must be empty or a
known pre-module, nothing but DeepstackEmbed between blocks, `mods[-2]` is an `RMSNorm`. Here
`mods[1]` is `ExpandStreams`, a `PLELayer` sits between blocks 1 and 2, and `mods[-2]` is a
`GatedResidual`. `assert_block_supported` would then stop at the hc sites (block has no
`attn_norm`/`mlp_norm` norm_spec) and, if it got that far, the GDN gate activation would be
silently wrong (silu vs sigmoid). The trainer's `_block_forward`/`_gdn_forward`/`_moe_out`
all assume a (b, t, d) residual with pre-norms.

## 2. Phase A: architecture port (box with >= 48 GB VRAM for validation; code here)

Deliverable: `qlora_validate_native.py` argmax-agreement gate passes on the 2.05bpw quant
(`--parallel split` over 2 x 3090: ~33 GB of weights + n-gram table in host RAM via the loader's
`ngram_ram`/disk streaming default; QSA layers load whole on one device per the model caps).

Work items, in order:

1. **Layout acceptance** (`backbone.py`): accept `ExpandStreams` as the one allowed pre-module
   (record `hc_mult`), accept `PLELayer` between blocks (record its index and module, like
   `deepstack_layout`), accept `GatedResidual(use_combine=False)` as the "final norm" slot
   and expose it as `final_mixer`. Keep everything else rejected. `block_metadata` gains
   `hc=True`, and `attn_hc`/`mlp_hc` entries carrying `norm_w (4,2560) as 1+w`, `down`, `up`,
   `inject` as frozen fp32/bf16 tensors (read with `keep_source_weights=True`, or keep the
   fp16 `proj_h` rows: `down_h = proj_h[:rank]`, `inject_h = proj_h[rank:M]`, `up_h` is
   released after load when the tiled path is on, so load with `keep_source_weights=True`
   or re-read from the safetensors as `embed_weight` does).
2. **Stream-stack residual in the trainer** (`native_llama.py`): `_forward_trunk` starts with
   `hidden = embed.float().unsqueeze(2).expand(-1,-1,4,-1)` when `hc`; `_run_block` passes the
   stack; `_block_forward`/`_gdn_forward` replace `normed = self._norm(hidden, spec)` with
   `post, y = hc_mix(hidden, site)`, run the sublayer on `y`, and replace `hidden + out` with
   `hidden + post.unsqueeze(-1) * out.float().unsqueeze(-2)`. Port `_mix_ref` literally
   (fp32, per-stream RMSNorm over the last dim, `silu(down/4)`, `sigmoid(up)`, mean over
   streams, `2*sigmoid(inject/4)`). Same for the mlp site. Checkpoint boundaries save the
   stack: 4 x 2560 x 4 B = 40 KB/token/block, ~1 GB at 512 tokens over 48 blocks, fine with
   `--offload-activations`.
3. **Final mixer + head**: after the last block, `hidden = hc_mix(hidden, final_mixer)[1]`
   (mixed only, no inject), then the existing head/fused-CE path on (b, t, 2560). The fused CE
   head needs no change (248320 vocab: use `--head-vocab-chunk 32768`).
4. **GDN sigmoid gate**: `gdn_norm_spec` reads `norm.gate_activation` ("silu"/"sigmoid");
   `gdn_gated_rmsnorm` applies the matching function. Add the assert so an unknown activation
   is rejected.
5. **QSA as dense**: in `assert_block_supported`, when `attn.qsa_indexer is not None`, record
   `qsa_threshold = indexer.sparse_threshold()` in metadata; the trainer refuses any batch with
   `t >= qsa_threshold` (or seq_len >= threshold at config time) with a clear message. No
   indexer math. Also `--pack` is already rejected by GDN; keep it so.
6. **PLE layer**: port `forward_streams_reference` to a differentiable `_ple_forward(streams,
   input_ids)` in a new `training/ple.py`: (a) n-gram rows: call the inference module's own
   `NGramEmbedding.forward(token_history, params)` under `torch.no_grad()` (frozen, host-side
   hashing, gathers from RAM/disk; the token history for a fresh sequence is the module's
   fresh-state convention: check `PLELayerState.clear` / `ple_eos_token_id` padding and
   reproduce it); (b) key/value projections as frozen `DiffLinear`s (not LoRA targets);
   (c) grouped RMSNorms (`norm_key`, `norm_query`, `norm_conv`, weight as 1+w, groups=4);
   (d) signed-sqrt gate and the dilated depthwise causal conv in plain torch (`F.conv1d` with
   `groups=hc_hidden`, `dilation=3`, left-pad 9 zeros for a fresh sequence); (e) add the delta
   into the stack. Run it in `_forward_trunk` between blocks at the recorded index, under
   checkpoint. Compare against the inference module on a short CPU sequence (fp16 table
   fixture) in a unit test.
7. **Loader side**: nothing new for Phase A: `Model.from_config(config)` builds the text
   trunk only, `has_vision` is irrelevant without `--vision`, MTP not loaded without
   `--mtp-targets`. Make sure the n-gram table defaults to the loader's streaming/RAM mode
   (`config.infer_params.ngram_stream_from_disk`); add a trainer flag `--ngram-ram` mirroring
   `model_init`'s.
8. **Targets**: `--targets` on attention (q/k/v/o), GDN (qkv/z/b/a/o), shared expert. Routed
   `expert_*` adapters stay possible on the big box but are excluded in Phase C. The hc site
   tensors, PLE projections, router and indexer are never targets.
9. **Validation gate**: extend `qlora_validate_native.py` if it assumes a final RMSNorm or a
   (b,t,d) residual at its hidden-state taps (`collect_hidden`/`feature_block_indices`: tap the
   stream MEAN, which is what the inference block exports as the collapsed hidden state,
   `transformer.py:210`). Pass criterion unchanged: 100% argmax agreement on the validation
   prompts, text-only.

CPU-testable here (scratch CPU-torch venv): items 1, 2 (against `_mix_ref` on random weights),
4, 5, 6(c-e). Box-only: the real-quant forward, items 7-9.

## 3. Phase B: box run on the 2 x 3090 (proves the forward trains)

- `qlora_train_native.py --parallel split`, 2.05bpw, seq-len 512, batch 1, grad-accum 4,
  r 16, targets attention+GDN+shared, `--optim adamw8bit`, `--offload-activations`,
  `--head-vocab-chunk 32768`, `--lora-head`, 30 steps, `--sample-every 10`.
- Record VRAM per device, tok/s, loss curve, and a before/after sample in a Session entry.
- This is where "does a 30-step attention-only LoRA do anything visible on a 125B MoE" gets
  answered; the architecture port is done regardless of that answer.

## 4. Phase C: streamed frozen experts (the 16 GB target)

The differentiable MoE path (`_moe_out`) reconstructs each touched expert's inner trellis
weight on the fly via `frozen_trellis_parts` -> `inner.get_inner_weight_tensor()`, a CUDA
kernel over `inner.trellis` (+ `suh`/`svh`, codebook). The only thing that has to change is
**where the packed trellis tensors live between uses**.

Prior art already in the tree:
- `LinearEXL3.swap_cpu()` / `unswap_cpu()` (`modules/quant/exl3.py:312`): moves
  trellis/su/sv/suh/svh/bias to CPU and back, used by the quantizer to free VRAM.
- `training/offload.py` (S36): async double-buffered H2D/D2H over a side stream with pinned
  pooled buffers; the pattern to copy for prefetch.
- `training/aux_offload.py` `ModelParker`: evicts whole components via `unload()`/`load()`.

Design:
1. **Residency manager** `training/expert_stream.py`: at setup, for every MoE block, pin the
   expert `inner` tensors in host RAM (contiguous per-layer slab per tensor kind; `trellis`
   dominates, ~0.65 GB/layer at 2.05bpw) and leave the GPU copies unallocated. Dense trunk,
   router, shared expert, hc sites, PLE projections, embed and head stay resident on the GPU.
   Memory check up front via `util/memory.check_host_memory`.
2. **Per-block materialize/evict** around `_moe_out`: `ensure_resident(layer)` copies the
   layer's slabs H2D on the side stream into a 2-slot ring (current + next) and rebinds
   `inner.trellis`/`suh`/`svh` views; `release(layer)` drops the slot after the block's
   backward. Forward walks layers 0..47, backward 47..0: the prefetch direction follows a
   `phase` flag the trainer sets (`backward_dequant_cache` already marks the backward phase).
   Under grad checkpointing each layer's recompute and backward run back-to-back, so one
   residency serves both.
3. **Cost model**: every pass streams nearly all 512 experts per layer (top-10 of 512 over a
   few hundred tokens touches ~all), so ~31 GB per pass x 3 passes/step at 2.05bpw, ~4-8 s/step
   on PCIe 4 x16, ~1-2 s/step less with `--dequant-cache` (2 passes). 30 steps: minutes. The
   reconstruct kernel itself is unchanged.
4. **VRAM budget on 16 GB** (2.05bpw, seq 512, bsz 1): dense weights ~2 GB, two expert slots
   ~1.3 GB, reconstructed expert temporaries < 0.5 GB, stream-stack activations ~1 GB
   (offloadable), LoRA + 8-bit Adam < 0.5 GB, head logits chunked. Headroom is real.
5. **Host RAM**: 31 GB pinned experts + 26 GB n-gram (or disk-streamed) + 2 GB + working set:
   **>= 64 GB with the n-gram table in RAM, ~48 GB with it on disk**. Pinned pages can't
   swap, so this is a hard floor, checked at startup.
6. **Flags**: `--stream-experts` (on/off), `--stream-experts-slots N` (default 2),
   `--ngram-ram`. `--parallel split` and `--vram-spillover` are rejected with it.
7. **Gate**: same-seed loss bit-identical to the resident run of Phase B (value-exact by
   construction: copies don't round), then the 16 GB run.

## 5. Phase D: the recipe for the 16 GB user

```
python training/qlora_train_native.py --model <Qwen3.8-Flash-Next-exl3 2.05bpw_h4_ng4> \
  --stream-experts --ngram-ram \
  --targets q_proj k_proj v_proj o_proj qkv_proj z_proj b_proj a_proj shared_up shared_gate shared_down \
  --r 16 --alpha 32 --optim adamw8bit --offload-activations --attn-impl flash \
  --seq-len 512 --batch 1 --grad-accum 4 --steps 30 --lr 1e-4 \
  --head-vocab-chunk 32768 --lora-head --sample-every 10
```
(exact target leaf names per `backbone.gdn_projections` / `moe_shared_projections`.)
Requirements: Linux, >= 64 GB RAM (48 GB with the n-gram table on NVMe), the 2.05bpw pack
(~60 GB on disk), flash-attn for head_dim 256.

## 6. Risks and open questions

- **Numerics of the stream stack in bf16 compute**: inference keeps the streams fp32 and only
  the sublayer inputs/outputs fp16. The port keeps the stack fp32 and casts `y` to compute
  dtype, matching inference. The argmax gate decides.
- **Hyper-connection weights after load**: `up_h` is freed when the tiled kernel path is
  active. Load the trunk with `keep_source_weights=True` for training, or read the four
  tensors from the safetensors directly (preferred: no dependence on inference's kernel
  tables).
- **PLE fresh-sequence state**: confirm the token-history padding (`ple_eos_token_id`) and
  conv zero-state match what the generator does for a new sequence; a wrong convention would
  be invisible to the gate on single-prompt validation only if the validate prompts start the
  same way. Validate on >= 3 prompts with different first tokens.
- **Indexer weights**: dead at training lengths but still loaded (small). Fine.
- **Expert streaming + `--dequant-cache`**: the backward cache holds reconstructed weights,
  not trellis; compatible. Measure both.
- **30 steps of attention-only LoRA on a 125B MoE may not move the samples much.** That is a
  product question, not a correctness one; the box run answers it before the streaming tier
  is built, so the user can stop after Phase B if the answer is "not worth it".
- **Not in scope**: QSA beyond 2048 tokens, MTP head training on this model (hc head is a
  different computation, S53), vision, sample packing, routed-expert adapters on 16 GB.

## 7. Files to touch

- `exllamav3/training/backbone.py`: layout, block asserts, metadata (hc, ple, final_mixer,
  qsa_threshold, gdn gate activation)
- `exllamav3/training/native_llama.py`: stream-stack residual, hc mix/apply, final mixer,
  PLE call site, seq-len guard
- `exllamav3/training/gdn.py`: gate activation
- `exllamav3/training/ple.py` (new): differentiable PLE
- `exllamav3/training/expert_stream.py` (new, Phase C)
- `training/qlora_train_native.py`, `training/qlora_validate_native.py`: flags, taps
- `tests/test_hyperconnections_train.py`, `tests/test_ple_train.py` (new, CPU)
- `README.md` supported-architectures table; `doc/qlora_handoff.md` Session entries

## 8. Session plan

1. Session 58: Phase A items 1-6 with CPU tests. Box handoff: validate gate.
2. Session 59: fix what the gate finds; Phase B run; Session entry with numbers.
3. Session 60: Phase C build + bit-exact gate on the box; 16 GB run; Phase D recipe into the
   README.
