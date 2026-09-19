#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run Balakirev on all images in the RV-training repository.
#
# Generates raw HTR output (ALTO XML) for each page image, organised
# in the same directory structure as the original repo.
#
# Checkpointing: skips pages that already have an XML output file.
# Can be interrupted and resumed safely.
#
# Usage:
#   bash scripts/docker/run_balakirev_all.sh                    # Full corpus
#   bash scripts/docker/run_balakirev_all.sh RV_1906            # Single year
#   bash scripts/docker/run_balakirev_all.sh RV_1906 --force    # Force reprocess
# ─────────────────────────────────────────────────────────────────────

set -e

RV_TRAINING="$HOME/RV-training"
OUTPUT_DIR="$HOME/vines-htr-correction/data/htr_raw"
MODEL="models/FoNDUE-GD_es_v3_Balakirev.mlmodel"
FILTER_YEAR="${1:-}"
FORCE="${2:-}"

if [ ! -f "$RV_TRAINING/$MODEL" ]; then
    echo "ERROR: No es troba el model Balakirev a $RV_TRAINING/$MODEL"
    exit 1
fi

echo "=== Executant Balakirev sobre RV-training ==="
echo "Input:  $RV_TRAINING/data/"
echo "Output: $OUTPUT_DIR/"
echo "Model:  $MODEL"
if [ -n "$FILTER_YEAR" ] && [ "$FILTER_YEAR" != "--force" ]; then
    echo "Filtre: només $FILTER_YEAR"
fi
if [ "$FORCE" = "--force" ] || [ "$FILTER_YEAR" = "--force" ]; then
    echo "Mode:   FORÇAT (re-processa tot)"
fi
echo ""

total=0
skipped=0
errors=0

for year_dir in "$RV_TRAINING"/data/RV_*; do
    year=$(basename "$year_dir")

    if [ -n "$FILTER_YEAR" ] && [ "$FILTER_YEAR" != "--force" ] && [ "$year" != "$FILTER_YEAR" ]; then
        continue
    fi

    out_dir="$OUTPUT_DIR/$year"
    mkdir -p "$out_dir"

    # Found images (PNG i JPG)
    images=$(find "$year_dir" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) | sort)
    count=$(echo "$images" | grep -c . || true)

    if [ "$count" -eq 0 ]; then
        echo "[$year] No image found, skipping..."
        continue
    fi

    echo "[$year] $count images found"

    for img in $images; do
        page=$(basename "${img%.*}")
        ext="${img##*.}"
        out_file="$out_dir/${page}.xml"

        # Checkpointing: skip if exists (unless --force)
        if [ -f "$out_file" ] && [ "$FORCE" != "--force" ] && [ "$FILTER_YEAR" != "--force" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        docker run --rm \
            -v "$RV_TRAINING":/rv-training \
            -v "$out_dir":/output \
            vines-kraken \
            -i "/rv-training/data/$year/${page}.${ext}" "/output/${page}.xml" \
            -a \
            segment -bl \
            ocr -m "/rv-training/$MODEL" \
            2>/dev/null

        if [ $? -eq 0 ]; then
            total=$((total + 1))
            echo "  ✓ $page"
        else
            errors=$((errors + 1))
            echo "  ✗ $page (ERROR)"
        fi
    done

    processed=$(find "$out_dir" -name "*.xml" | wc -l)
    echo "  -> $processed XMLs a $out_dir"
    echo ""
done

echo "=== COMPLETE ==="
echo "Processed: $total"
echo "Skipped (already exists): $skipped"
echo "Errors: $errors"
