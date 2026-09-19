#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Prepare fine-tuning data.
#
# Generates training pairs (HTR raw -> GT) in JSONL format,
# split by year into train/val/test.
#
# Usage:
#   bash scripts/docker/run_prepare_finetuning.sh
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

echo "Preparing fine-tuning data..."

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    -v ~/vines-htr-correction/data:/data \
    vines-pipeline -m src.pipeline.prepare_finetuning_data /gt /htr_raw /data/finetuning
