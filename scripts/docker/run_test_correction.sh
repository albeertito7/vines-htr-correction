#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run a single-page correction test with a general LLM.
#
# Compares HTR raw vs GT, corrects with Llama 70B via Ollama,
# and shows CER before and after for two strategies (line by line
# and context blocks).
#
# Requires Ollama running with the model downloaded.
#
# Usage:
#   bash scripts/docker/run_test_correction.sh RV_1890 page_00005
# ─────────────────────────────────────────────────────────────────────

set -e

YEAR="${1:?Usage: $0 <year> <page>  (ex: RV_1890 page_00005)}"
PAGE="${2:?Usage: $0 <year> <page>  (ex: RV_1890 page_00005)}"

echo "Executing post-correction: $YEAR/$PAGE"
echo ""

docker run --rm --network host \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training/data:/gt \
    -v ~/vines-htr-correction/data/htr_raw:/htr_raw \
    vines-pipeline scripts/legacy/test_correction.py \
    /gt/$YEAR/${PAGE}.xml \
    /htr_raw/$YEAR/${PAGE}.xml \
    http://localhost:11434
