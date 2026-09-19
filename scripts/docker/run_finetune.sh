#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Launch LoRA fine-tuning on the DGX Spark.
#
# Requires:
#   - vines-finetune Docker image built
#   - Dataset prepared in data/finetuning/
#   - Config at configs/finetune_config.yaml
#
# Usage:
#   bash scripts/docker/run_finetune.sh                        # Default config
#   bash scripts/docker/run_finetune.sh configs/custom.yaml    # Custom config
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/finetune_config.yaml}"

echo "=== Starting fine-tuning ==="
echo "Config: $CONFIG"
echo ""

docker run --gpus all --rm \
    -v ~/vines-htr-correction:/app \
    vines-finetune scripts/finetune_lora.py --config "$CONFIG"
