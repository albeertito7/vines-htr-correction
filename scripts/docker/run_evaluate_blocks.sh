#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Evaluate the fine-tuned model with block-based inference.
#
# Usage:
#   bash scripts/docker/run_evaluate_blocks.sh                    # Full test, blocks of 10
#   bash scripts/docker/run_evaluate_blocks.sh 10 3               # 3 pages, blocks of 10
#   bash scripts/docker/run_evaluate_blocks.sh 5                  # Full test, blocks of 5
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

BLOCK_SIZE="${1:-10}"
MAX_PAGES="${2:-}"

EXTRA_ARGS=""
if [ -n "$MAX_PAGES" ]; then
    EXTRA_ARGS="--max-pages $MAX_PAGES"
fi

echo "=== Evaluating with blocks of $BLOCK_SIZE lines ==="
echo ""

docker run --gpus all --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    -e HF_HOME=/app/.cache/huggingface \
    vines-finetune scripts/evaluate_finetuned_blocks.py \
    --model models/finetuned \
    --gt /gt \
    --htr /htr_raw \
    --split data/finetuning/test.jsonl \
    --block-size "$BLOCK_SIZE" \
    $EXTRA_ARGS
