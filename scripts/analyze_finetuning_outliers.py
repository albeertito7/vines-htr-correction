"""
Analyse outliers in the fine-tuning dataset.

Shows the pages with the highest CER and shortest GT to confirm they
are cover pages or blank pages that should be filtered out before
fine-tuning.

Why high CER indicates anomalous pages:
    A CER > 80% (especially > 100%) means the edit distance between
    HTR raw and GT is comparable to or exceeds the GT length. This
    happens when Balakirev generates long text from a page where the
    GT is very short (covers, page numbers), because it detects ink
    stains, margins, or annotations that the human team did not
    transcribe. The result is a pair where input and output have
    almost nothing in common — noise for fine-tuning.

Why short GT indicates low-value pages:
    A GT shorter than 50 characters corresponds to covers ("1890"),
    standalone page numbers ("4"), or near-blank pages. These contain
    no narrative diary text with correctable errors, so they add
    nothing to training. Additionally, CER on very short texts is
    highly unstable (one character difference in 4 characters = 25%
    CER), distorting the statistics.

Usage:
    python scripts/analyze_finetuning_outliers.py [path-finetuning-dir]
"""

import json
import sys
from pathlib import Path


def analyze(finetuning_dir):
    """Load all splits and identify outlier pages."""
    finetuning_dir = Path(finetuning_dir)

    pairs = []
    for split in ["train", "val", "test"]:
        path = finetuning_dir / f"{split}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                pairs.extend([json.loads(l) for l in f])

    if not pairs:
        print("No pairs found.")
        return

    # Top 10 highest CER
    print("=== TOP 10 HIGHEST CER ===")
    worst = sorted(pairs, key=lambda p: p["metadata"]["cer"], reverse=True)[:10]
    for p in worst:
        m = p["metadata"]
        print(f"  {m['page_id']}: CER={m['cer']:.2%} GT={m['gt_chars']}chars HTR={m['htr_chars']}chars")

    print()

    # Top 10 shortest GT
    print("=== TOP 10 SHORTEST GT ===")
    shortest = sorted(pairs, key=lambda p: p["metadata"]["gt_chars"])[:10]
    for p in shortest:
        m = p["metadata"]
        print(f"  {m['page_id']}: GT={m['gt_chars']}chars HTR={m['htr_chars']}chars CER={m['cer']:.2%}")

    print()

    # Count pages affected by each filter
    gt_short = sum(1 for p in pairs if p["metadata"]["gt_chars"] < 50)
    cer_high = sum(1 for p in pairs if p["metadata"]["cer"] > 0.80)
    either = sum(
        1 for p in pairs
        if p["metadata"]["gt_chars"] < 50 or p["metadata"]["cer"] > 0.80
    )

    print(f"Total pairs: {len(pairs)}")
    print(f"Pairs with GT < 50 chars: {gt_short}")
    print(f"Pairs with CER > 80%: {cer_high}")
    print(f"Pairs excluded by filters (GT<50 or CER>80%): {either}")
    print(f"Pairs remaining after filtering: {len(pairs) - either}")


if __name__ == "__main__":
    finetuning_dir = sys.argv[1] if len(sys.argv) > 1 else "data/finetuning"
    analyze(finetuning_dir)
