"""
Calculate baseline CER/WER for the Balakirev HTR model across the corpus.

Compares the raw HTR output (data/htr_raw) against the ground truth
(RV-training repository) to establish the baseline performance before
any post-correction.

Computes metrics per page (not concatenating entire years) to avoid
running Levenshtein on extremely long texts.

Outputs:
    - baseline_detail.csv: CER/WER per page
    - baseline_summary.csv: CER/WER per year and global
    - baseline_meta.json: run metadata

Usage:
    python scripts/calculate_baseline_cer.py <path-gt> <path-htr-raw> [path-output] [model-name]

Example:
    python scripts/calculate_baseline_cer.py ~/RV-training/data data/htr_raw benchmark/evaluation Balakirev
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.alto_parser import parse_alto_file
from src.evaluation.metrics import calculate_cer, calculate_wer


def calculate_baseline(gt_dir, htr_dir, output_dir):
    """Compute baseline CER/WER for all pages with both GT and HTR raw."""
    gt_dir = Path(gt_dir)
    htr_dir = Path(htr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    skipped = 0
    per_year = {}
    all_results = []

    for year_dir in sorted(gt_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        year_results = []

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

                if len(gt_text.strip()) == 0:
                    continue

                cer = calculate_cer(gt_text, htr_text)
                wer = calculate_wer(gt_text, htr_text)

                result = {
                    "year": year,
                    "page": gt_file.stem,
                    "cer": cer,
                    "wer": wer,
                    "gt_chars": len(gt_text),
                    "gt_words": len(gt_text.split()),
                    "htr_chars": len(htr_text),
                    "htr_words": len(htr_text.split()),
                    "gt_lines": len(gt_parsed["lines"]),
                    "htr_lines": len(htr_parsed["lines"]),
                }

                year_results.append(result)
                all_results.append(result)
                total_pages += 1
                print(f"  {year}/{gt_file.stem}: CER={cer:.2%} WER={wer:.2%}", flush=True)

            except Exception as e:
                print(f"  Error {gt_file.name}: {e}", flush=True)
                skipped += 1

        if year_results:
            total_chars = sum(r["gt_chars"] for r in year_results)
            avg_cer = sum(r["cer"] * r["gt_chars"] for r in year_results) / total_chars
            avg_wer = sum(r["wer"] * r["gt_chars"] for r in year_results) / total_chars
            per_year[year] = {
                "pages": len(year_results),
                "cer": avg_cer,
                "wer": avg_wer,
                "total_chars": total_chars,
            }
            print(f"[{year}] {len(year_results)} pàgines -> CER={avg_cer:.2%} WER={avg_wer:.2%}", flush=True)
            print(flush=True)

    # ─── Per-page detail (CSV) ────────────────────────────────────────

    detail_path = output_dir / "baseline_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "year", "page", "cer", "wer",
            "gt_chars", "gt_words", "htr_chars", "htr_words",
            "gt_lines", "htr_lines",
        ])
        writer.writeheader()
        for r in all_results:
            row = r.copy()
            row["cer"] = round(r["cer"], 6)
            row["wer"] = round(r["wer"], 6)
            writer.writerow(row)

    print(f"\nPer-page detail saved to {detail_path}")

    # ─── Per-year summary (CSV) ───────────────────────────────────────

    total_chars = sum(d["total_chars"] for d in per_year.values())
    global_cer = sum(d["cer"] * d["total_chars"] for d in per_year.values()) / total_chars
    global_wer = sum(d["wer"] * d["total_chars"] for d in per_year.values()) / total_chars

    summary_path = output_dir / "baseline_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "pages", "cer", "wer", "total_chars"])
        writer.writeheader()
        for year, data in sorted(per_year.items()):
            writer.writerow({
                "year": year,
                "pages": data["pages"],
                "cer": round(data["cer"], 6),
                "wer": round(data["wer"], 6),
                "total_chars": data["total_chars"],
            })
        writer.writerow({
            "year": "TOTAL",
            "pages": total_pages,
            "cer": round(global_cer, 6),
            "wer": round(global_wer, 6),
            "total_chars": total_chars,
        })

    print(f"Per-year summary saved to {summary_path}")

    # ─── Metadata (JSON) ──────────────────────────────────────────────

    meta_path = output_dir / "baseline_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "FoNDUE-GD_es_v3_Balakirev",
            "total_pages": total_pages,
            "skipped": skipped,
            "global_cer": round(global_cer, 6),
            "global_wer": round(global_wer, 6),
            "total_chars": total_chars,
        }, f, indent=2)

    print(f"Metadata saved to {meta_path}")

    # ─── Print summary ────────────────────────────────────────────────

    print()
    print(f"{'Year':<15} {'Pages':>5} {'CER':>8} {'WER':>8}")
    print("-" * 40)
    for year, data in sorted(per_year.items()):
        print(f"{year:<15} {data['pages']:>5} {data['cer']:>7.2%} {data['wer']:>7.2%}")
    print("-" * 40)
    print(f"{'TOTAL':<15} {total_pages:>5} {global_cer:>7.2%} {global_wer:>7.2%}")
    print(f"\nSkipped: {skipped}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/calculate_baseline_cer.py <path-gt> <path-htr-raw> [path-output] [model-name]")
        sys.exit(1)

    base_output = sys.argv[3] if len(sys.argv) > 3 else "benchmark/evaluation"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "Balakirev"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = str(Path(base_output) / "baseline" / f"{timestamp}_{model_name}")

    calculate_baseline(sys.argv[1], sys.argv[2], output)
