#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run the ALTO XML parser inside Docker.
#
# Usage:
#   bash scripts/docker/run_parser.sh /rv-training/data/RV_1890/page_00005.xml
#   bash scripts/docker/run_parser.sh /rv-training/data/RV_1890  (directory)
# ─────────────────────────────────────────────────────────────────────

set -e

INPUT="${1:?Usage: $0 <file.xml or directory>}"

docker run --rm \
    -v ~/vines-htr-correction:/app \
    -v ~/RV-training:/rv-training \
    vines-pipeline src/parsers/alto_parser.py "$INPUT"
