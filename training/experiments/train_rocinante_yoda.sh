#!/usr/bin/env bash
#
# Overnight 2-GPU QLoRA fine-tune of TheDrummer/Rocinante-XL-16B-v1 (a 54-layer
# Mistral-Nemo-12B depth merge) on a generated Yoda style set.
#
# Two multi-GPU modes (PARALLEL env, default `split`):
#   split  -- layer-split the frozen base across both cards in one python process.
#             Frees memory for a bigger batch / longer sequence, and the cards
#             alternate (~50% duty each) so they run cooler on a multi-day run.
#             Use when the model + training state won't comfortably fit one card.
#   ddp    -- replicate the model per card and shard the batch (torchrun). Use
#             when the model fits one card and you want raw throughput.
#
# IMPORTANT (split): exllamav3's autosplit is GREEDY -- it fills cuda:0 until that
# card's budget is exhausted, THEN spills to cuda:1 (model_ls.py). Rocinante (~9 GB
# at 4bpw) fits one card, so WITHOUT a budget cap the whole model lands on cuda:0
# and you're back at the memory wall. USE_PER_DEVICE caps each card so the layers
# actually split. That cap only sizes the LOAD (base weights + a reference
# forward); training overhead (optimizer 2x moments, grads, activations) is
# allocated at RUNTIME in the leftover headroom -- so cap cuda:0 near HALF THE BASE
# SIZE (~5 GB), not half the card. Watch the printed "decoder block devices:" line
# for a roughly even split and tune; if it OOMs during training (not load), shift
# more layers off cuda:0 (lower its cap) or drop BATCH.
#
# Usage:
#   bash training/experiments/train_rocinante_yoda.sh MODEL_DIR DATA_JSONL [OUT_DIR]
#
# Override with env vars:
#   PARALLEL         split | ddp                      (default split)
#   USE_PER_DEVICE   (split) per-card GB budget list  (default "5 24"; tune!)
#   BATCH            micro-batch                       (default 4)
#   ACCUM            grad-accum steps                  (default 4)
#   STEPS            training steps                    (default 2000)
#   NGPU             GPUs (ddp processes)              (default 2)
#   SKIP_VALIDATE=1  skip the forward-correctness gate
#
# Effective batch:  split = BATCH*ACCUM   ;   ddp = BATCH*NGPU*ACCUM
#
# Why these defaults: lr 1e-4 + --save-best
# (lr 2e-4 overfit: train 0.09 / held-out 3.11); r64/alpha64 (ratio 1.0). The
# validate gate runs first because this is the first Mistral-arch run of the
# transformers-free forward -- prove it matches native before training all night.

set -euo pipefail

MODEL="${1:?usage: train_rocinante_yoda.sh MODEL_DIR DATA_JSONL [OUT_DIR]}"
DATA="${2:?need the Yoda .jsonl dataset path (Alpaca schema)}"
OUT="${3:-${MODEL%/}/yoda}"

PARALLEL="${PARALLEL:-split}"
USE_PER_DEVICE="${USE_PER_DEVICE:-5 24}"
NGPU="${NGPU:-2}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
STEPS="${STEPS:-2000}"

# Run from the repo root so the example paths below resolve.
cd "$(dirname "$0")/.."

mkdir -p "$OUT"
LOG="${OUT%/}.log"

if [[ "$PARALLEL" == "split" ]]; then
    EFF=$((BATCH * ACCUM))
elif [[ "$PARALLEL" == "ddp" ]]; then
    EFF=$((BATCH * NGPU * ACCUM))
else
    echo "unknown PARALLEL=$PARALLEL (expected 'split' or 'ddp')"; exit 1
fi

echo "=============================================================================="
echo " Rocinante-XL-16B QLoRA (Yoda)  |  PARALLEL=$PARALLEL  |  eff-batch $EFF"
echo " model : $MODEL"
echo " data  : $DATA"
echo " out   : $OUT"
echo " log   : $LOG"
echo "=============================================================================="

# Split-mode args shared by the validate gate and the training run (empty in ddp,
# so the gate runs single-device -- Rocinante fits one card for inference).
SPLIT_ARGS=()
if [[ "$PARALLEL" == "split" ]]; then
    SPLIT_ARGS=(--parallel split)
    [[ -n "$USE_PER_DEVICE" ]] && SPLIT_ARGS+=(--use-per-device $USE_PER_DEVICE)
fi

# 1. Correctness gate: the differentiable forward must match native exllamav3 on
#    this (Mistral) architecture. Under split this gates the actual cross-device
#    forward. set -e aborts the whole run if it exits non-zero.
if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
    echo "== [1/2] validating differentiable forward (gates the run) =="
    python training/qlora_validate_native.py --model "$MODEL" "${SPLIT_ARGS[@]}"
else
    echo "== [1/2] SKIP_VALIDATE=1 -- skipping the forward-correctness gate =="
fi

# 2. The overnight run. --save-best keeps the best held-out checkpoint at $OUT.
echo "== [2/2] launching $PARALLEL QLoRA run =="
if [[ "$PARALLEL" == "ddp" ]]; then
    # torchrun spawns one process per GPU; only rank 0 prints/saves.
    torchrun --standalone --nproc_per_node="$NGPU" training/qlora_train_native_ddp.py \
        --model "$MODEL" --out "$OUT" --dataset "$DATA" \
        --lora-r 64 --alpha 64 --lr 1e-4 \
        --batch "$BATCH" --grad-accum "$ACCUM" --seq-len 512 --steps "$STEPS" \
        --val-frac 0.02 --eval-every 100 --save-best \
        2>&1 | tee "$LOG"
else
    # Single process spanning both cards (layer-split). Note: --r here, not
    # --lora-r (the torchrun abbrev collision only affects the ddp launcher).
    python training/qlora_train_native.py \
        --model "$MODEL" --out "$OUT" --dataset "$DATA" "${SPLIT_ARGS[@]}" \
        --r 64 --alpha 64 --lr 1e-4 \
        --batch "$BATCH" --grad-accum "$ACCUM" --seq-len 512 --steps "$STEPS" \
        --val-frac 0.02 --eval-every 100 --save-best \
        2>&1 | tee "$LOG"
fi

echo "== done. adapter -> $OUT ; full log -> $LOG =="
echo "   verify (on bf16 base for a fair read; the 4bpw base attenuates LoRAs):"
echo "   python training/qlora_infer_native.py --model $MODEL --adapter $OUT"
