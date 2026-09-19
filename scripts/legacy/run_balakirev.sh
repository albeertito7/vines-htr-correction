#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run Balakirev on all images in the RV-training repository.
#
# Generates the raw HTR output (ALTO XML) for each page image,
# organised in the same directory structure as the original repo.
#
# This output is the "text with errors" needed to create the
# (HTR erroneous -> GT correct) pairs for benchmarking and fine-tuning.
#
# Includes checkpointing: skips pages that already have an XML output
# file, so the script can be interrupted and resumed safely.
#
# Usage (from DGX Spark):
#   bash scripts/run_balakirev.sh /path/to/RV-training /path/to/output
#
# Example:
#   bash scripts/legacy/run_balakirev.sh ~/RV-training ~/vines-htr-correction/data/htr_raw
# ─────────────────────────────────────────────────────────────────────

set -e

RV_TRAINING="${1:?Ús: $0 <path-rv-training> <path-output>}"
OUTPUT_DIR="${2:?Ús: $0 <path-rv-training> <path-output>}"
MODEL="$RV_TRAINING/models/FoNDUE-GD_es_v3_Balakirev.mlmodel"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Balakirev model not found at $MODEL"
    exit 1
fi

echo "=== Running Balakirev on RV-training ==="
echo "Input:  $RV_TRAINING/data/"
echo "Output: $OUTPUT_DIR/"
echo "Model:  $MODEL"
echo ""

total=0
errors=0

for year_dir in "$RV_TRAINING"/data/RV_*; do
    year=$(basename "$year_dir")
    out_dir="$OUTPUT_DIR/$year"
    mkdir -p "$out_dir"

    # Find images (PNG and JPG)
    images=$(find "$year_dir" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) | sort)
    count=$(echo "$images" | grep -c . || true)

    if [ "$count" -eq 0 ]; then
        echo "[$year] No images found, skipping..."
        continue
    fi

    echo "[$year] Processing $count images..."

    for img in $images; do
        page=$(basename "${img%.*}")
        out_file="$out_dir/${page}.xml"

        # Skip if already processed (allows resume after interruption)
        if [ -f "$out_file" ]; then
            continue
        fi

        docker run --rm \
            -v "$RV_TRAINING":/rv-training \
            -v "$out_dir":/output \
            vines-kraken \
            -i "/rv-training/data/$year/$(basename "$img")" "/output/${page}.xml" \
            -a \
            segment -bl \
            ocr -m /rv-training/models/FoNDUE-GD_es_v3_Balakirev.mlmodel \
            2>/dev/null

        if [ $? -eq 0 ]; then
            total=$((total + 1))
        else
            echo "  ERROR: $page"
            errors=$((errors + 1))
        fi
    done

    processed=$(find "$out_dir" -name "*.xml" | wc -l)
    echo "  -> $processed XMLs generated in $out_dir"
done

echo ""
echo "=== COMPLETE ==="
echo "Total processed: $total"
echo "Errors: $errors"
