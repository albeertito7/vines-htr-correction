#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Run Balakirev on a single page.
#
# Usage:
#   bash scripts/docker/run_balakirev_page.sh RV_1890 page_00005
# ─────────────────────────────────────────────────────────────────────

set -e

YEAR="${1:?Usage: $0 <any> <page>  (ex: RV_1890 page_00005)}"
PAGE="${2:?Usage: $0 <any> <page>  (ex: RV_1890 page_00005)}"

# Detection (png o jpg)
if [ -f ~/RV-training/data/$YEAR/${PAGE}.png ]; then
    EXT="png"
elif [ -f ~/RV-training/data/$YEAR/${PAGE}.jpg ]; then
    EXT="jpg"
else
    echo "ERROR: Not found ~/RV-training/data/$YEAR/${PAGE}.png neither .jpg"
    exit 1
fi

mkdir -p ~/vines-htr-correction/data/htr_raw/$YEAR

echo "Processing $YEAR/$PAGE.$EXT..."
docker run --rm \
    -v ~/RV-training:/rv-training \
    -v ~/vines-htr-correction/data/htr_raw/$YEAR:/output \
    vines-kraken \
    -i "/rv-training/data/$YEAR/${PAGE}.$EXT" "/output/${PAGE}.xml" \
    -a \
    segment -bl \
    ocr -m /rv-training/models/FoNDUE-GD_es_v3_Balakirev.mlmodel

echo "Result: ~/vines-htr-correction/data/htr_raw/$YEAR/${PAGE}.xml"
