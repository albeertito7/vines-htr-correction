#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Analyse outliers in the fine-tuning dataset.
#
# Shows pages with highest CER and shortest GT to identify covers,
# blank pages, or anomalous pairs to filter before fine-tuning.
#
# Usage:
#   bash scripts/docker/run_analyze_outliers.sh
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/vines-htr-correction/data:/data \
    vines-pipeline scripts/analyze_finetuning_outliers.py /data/finetuning
