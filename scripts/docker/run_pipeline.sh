#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run any Python script from the pipeline inside Docker.
#
# Usage:
#   bash scripts/docker/run_pipeline.sh src/parsers/alto_parser.py /rv-training/data/RV_1890/page_00005.xml
#   bash scripts/docker/run_pipeline.sh src/evaluation/metrics.py
#   bash scripts/docker/run_pipeline.sh -c "print('hello')"
# ─────────────────────────────────────────────────────────────────────

set -e

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training:/rv-training \
    -v ~/vines-htr-correction/data:/data \
    -v ~/vines-htr-correction/benchmark:/benchmark \
    -v ~/vines-htr-correction/logs:/logs \
    vines-pipeline "$@"
