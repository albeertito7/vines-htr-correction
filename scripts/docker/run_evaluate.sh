#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Evaluate the fine-tuned model on the test set or a subset.
#
# Usage:
#   bash scripts/docker/run_evaluate.sh                          # Full test set
#   bash scripts/docker/run_evaluate.sh test 5                   # Only 5 test pages
#   bash scripts/docker/run_evaluate.sh train 10                 # 10 train pages (sanity check)
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

SPLIT="${1:-test}"
MAX_PAGES="${2:-}"

SPLIT_FILE="data/finetuning/${SPLIT}.jsonl"
EXTRA_ARGS=""

if [ -n "$MAX_PAGES" ]; then
    EXTRA_ARGS="--max-pages $MAX_PAGES"
fi

echo "=== Evaluating fine-tuned model ==="
echo "Split: $SPLIT_FILE"
echo ""

docker run --gpus all --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    -e HF_HOME=/app/.cache/huggingface \
    vines-finetune scripts/evaluate_finetuned.py \
    --model models/finetuned \
    --gt /gt \
    --htr /htr_raw \
    --split "$SPLIT_FILE" \
    $EXTRA_ARGS
