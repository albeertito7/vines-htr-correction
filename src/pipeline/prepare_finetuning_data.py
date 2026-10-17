"""
Prepare fine-tuning data for LLM post-correction.

From the ground truth (RV-training repo) and raw HTR outputs (generated
by Balakirev), generates full-page training pairs in JSONL format for
instruction-tuning with LoRA/QLoRA.

Design decisions:
    - Full-page pairs (not per line or per block) to avoid the segmentation
      alignment problem between HTR raw and GT, and because full-page context
      helps the model learn when NOT to correct (Viñes' orthography vs
      Balakirev errors).
    - Split by year (not randomly by page) to prevent data leakage between
      train and test sets.
    - Outlier filtering: pages with GT < 50 chars (covers, page numbers) or
      CER > 80% (structural mismatches) are excluded.

Usage:
    python -m src.pipeline.prepare_finetuning_data <path-gt> <path-htr-raw> [path-output]

Example:
    python -m src.pipeline.prepare_finetuning_data ~/RV-training/data data/htr_raw data/finetuning
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parsers.alto_parser import parse_alto_file
from src.evaluation.metrics import calculate_cer


# ─── Configuration ─────────────────────────────────────────────────────────────

# System prompt kept in Catalan to match the fine-tuned model's training
SYSTEM_PROMPT = (
    "Ets un expert en post-correcció d'HTR dels diaris de Ricard Viñes. "
    "Corregeix els errors de reconeixement del text manuscrit. "
    "Només corregeix errors d'HTR evidents. No modifiquis l'ortografia "
    "de l'autor, noms propis, ni símbols especials (¶, ¬, ⟦ ⟧)."
)

# Split by year to prevent data leakage (pages from the same year share
# handwriting style, ink, and paper characteristics)
SPLIT_CONFIG = {
    "train": [
        "RV_1890", "RV_1894", "RV_1895", "RV_1898",
        "RV_1900", "RV_1901", "RV_1902", "RV_1903",
        "RV_1904", "RV_1905", "RV_1906", "RV_1907", "RV_1913",
    ],
    "val": ["RV_1892-1893", "RV_1909"],
    "test": ["RV_1908", "RV_1915"],
}


# ─── Pair generation ──────────────────────────────────────────────────────────

def create_pair(htr_text, gt_text, page_id):
    """
    Create a training pair in chat JSONL format.

    Args:
        htr_text: full-page HTR text (with errors).
        gt_text: full-page GT text (correct).
        page_id: page identifier (year/page).

    Returns:
        dict in JSONL format with messages and metadata, or None if empty.
    """
    htr_text = htr_text.strip()
    gt_text = gt_text.strip()

    if not htr_text or not gt_text:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Corregeix el següent text HTR:\n{htr_text}"},
            {"role": "assistant", "content": gt_text},
        ],
        "metadata": {
            "page_id": page_id,
            "cer": round(calculate_cer(gt_text, htr_text), 6),
            "htr_chars": len(htr_text),
            "gt_chars": len(gt_text),
        },
    }


def get_split(year):
    """Return the split (train/val/test) for a given year."""
    for split_name, years in SPLIT_CONFIG.items():
        if year in years:
            return split_name
    return "train"  # fallback


# ─── Main pipeline ────────────────────────────────────────────────────────────

def prepare_data(gt_dir, htr_dir, output_dir):
    """Generate training pairs from GT and HTR raw, split by year."""
    gt_dir = Path(gt_dir)
    htr_dir = Path(htr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    skipped = 0
    total = 0
    identical = 0

    print("=== Generating training pairs ===\n")

    for year_dir in sorted(gt_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        split = get_split(year)
        year_count = 0
        year_identical = 0

        for gt_file in sorted(year_dir.glob("*.xml")):
            htr_file = htr_dir / year / gt_file.name
            if not htr_file.exists():
                skipped += 1
                continue

            try:
                gt_parsed = parse_alto_file(gt_file)
                htr_parsed = parse_alto_file(htr_file)

                gt_text = " ".join([l["text"] for l in gt_parsed["lines"]])
                htr_text = " ".join([l["text"] for l in htr_parsed["lines"]])

                page_id = f"{year}/{gt_file.stem}"
                pair = create_pair(htr_text, gt_text, page_id)

                if pair is None:
                    skipped += 1
                    continue

                # Filter outliers
                cer = pair["metadata"]["cer"]
                gt_chars = pair["metadata"]["gt_chars"]

                if gt_chars < 50:
                    skipped += 1
                    continue

                if cer > 0.80:
                    skipped += 1
                    continue

                splits[split].append(pair)
                year_count += 1
                total += 1

                if pair["metadata"]["cer"] == 0:
                    year_identical += 1
                    identical += 1

            except Exception as e:
                print(f"  Error {year}/{gt_file.stem}: {e}", flush=True)
                skipped += 1

        if year_count > 0:
            print(
                f"  [{year}] {year_count} parells -> {split} "
                f"({year_identical} idèntics)",
                flush=True,
            )

    # ─── Save datasets ────────────────────────────────────────────────

    print(f"\n=== Saving datasets ===\n")

    for split_name, pairs in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(pairs)} parells -> {path}")

    # ─── Statistics ───────────────────────────────────────────────────

    all_pairs = splits["train"] + splits["val"] + splits["test"]
    cers = [p["metadata"]["cer"] for p in all_pairs]
    chars = [p["metadata"]["gt_chars"] for p in all_pairs]

    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_pairs": total,
        "skipped": skipped,
        "identical_pairs": identical,
        "split": {
            name: {
                "count": len(pairs),
                "years": SPLIT_CONFIG.get(name, []),
            }
            for name, pairs in splits.items()
        },
        "cer": {
            "mean": round(sum(cers) / len(cers), 6) if cers else 0,
            "min": round(min(cers), 6) if cers else 0,
            "max": round(max(cers), 6) if cers else 0,
        },
        "chars": {
            "mean": round(sum(chars) / len(chars)) if chars else 0,
            "min": min(chars) if chars else 0,
            "max": max(chars) if chars else 0,
            "total": sum(chars),
        },
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ─── Summary ──────────────────────────────────────────────────────

    print(f"\n=== SUMMARY ===\n")
    print(f"Total pairs: {total}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")
    print(f"Identical (CER=0): {identical}")
    print(f"Skipped: {skipped}")
    print(f"\nMean CER: {stats['cer']['mean']:.2%}")
    print(f"CER range: {stats['cer']['min']:.2%} - {stats['cer']['max']:.2%}")
    print(f"\nGT chars per page: {stats['chars']['mean']} mean ({stats['chars']['min']}-{stats['chars']['max']})")
    print(f"GT chars total: {stats['chars']['total']:,}")
    print(f"\nStatistics: {stats_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.pipeline.prepare_finetuning_data <path-gt> <path-htr-raw> [path-output]")
        sys.exit(1)

    output = sys.argv[3] if len(sys.argv) > 3 else "data/finetuning"
    prepare_data(sys.argv[1], sys.argv[2], output)
