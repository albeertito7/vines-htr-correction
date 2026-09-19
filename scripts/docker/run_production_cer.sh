#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Calculate CER/WER of the production-corrected corpus vs baseline vs GT.
#
# Usage:
#   bash scripts/docker/run_production_cer.sh
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    -v ~/vines-htr-correction/data/corrected:/corrected \
    -v ~/vines-htr-correction/benchmark:/benchmark \
    vines-pipeline scripts/calculate_production_cer.py /gt /htr_raw /corrected /benchmark/evaluation/production
