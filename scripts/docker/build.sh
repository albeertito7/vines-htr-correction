#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Build all Docker images for the project.
#
# Images:
#   vines-pipeline  — Python pipeline (parser, correctors, metrics)
#   vines-kraken    — Kraken + PyTorch (runs HTR models like Balakirev)
#   vines-finetune  — NVIDIA NGC (fine-tuning and inference with GPU)
#
# Usage:
#   bash scripts/docker/build.sh
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

echo "=== Building vines-pipeline ==="
docker build -t vines-pipeline .

echo ""
echo "=== Building vines-kraken ==="
docker build -t vines-kraken -f Dockerfile.kraken .

echo ""
echo "=== Building vines-finetune ==="
docker build -t vines-finetune -f Dockerfile.finetune .

echo ""
echo "=== Images built ==="
docker images | grep vines
