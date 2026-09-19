#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Calculate baseline CER/WER for the entire corpus.
#
# Compares raw HTR output (data/htr_raw) against ground truth
# (RV-training) and saves results to benchmark/evaluation/baseline/
#
# Usage:
#   bash scripts/docker/run_baseline.sh              # Default: Balakirev
#   bash scripts/docker/run_baseline.sh Chopin        # Specify model name
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

MODEL="${1:-Balakirev}"

echo "Calculating model baseline: $MODEL"

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    -v ~/vines-htr-correction/benchmark:/benchmark \
    vines-pipeline scripts/calculate_baseline_cer.py /gt /htr_raw /benchmark/evaluation "$MODEL"
