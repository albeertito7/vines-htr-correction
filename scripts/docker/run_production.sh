#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Apply the fine-tuned model to the entire corpus.
#
# Includes checkpointing — can be interrupted and resumed.
# Recommended to run in background with nohup.
#
# Usage:
#   bash scripts/docker/run_production.sh                # Normal
#   nohup bash scripts/docker/run_production.sh &        # Background
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

echo "=== Aplying fine-tuned model to the whole corpus ==="
echo ""

docker run --gpus all --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -e HF_HOME=/app/.cache/huggingface \
    vines-finetune scripts/run_production.py \
    --model models/finetuned \
    --htr data/htr_raw \
    --output data/corrected
