"""
Calculate CER/WER of the production-corrected corpus.

Compares the production output (data/corrected) against the ground truth
(RV-training) and the baseline (HTR raw) to measure the real improvement
across the entire corpus.

Outputs a per-year summary table and a JSON with detailed results.

Usage:
    python scripts/calculate_production_cer.py <path-gt> <path-htr-raw> <path-corrected> [path-output]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.alto_parser import parse_alto_file
from src.evaluation.metrics import calculate_cer, calculate_wer


def calculate_production(gt_dir, htr_dir, corrected_dir, output_dir):
    """Compare corrected corpus against GT and HTR baseline per year."""
    gt_dir = Path(gt_dir)
    htr_dir = Path(htr_dir)
    corrected_dir = Path(corrected_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_year = {}
    all_results = []
    skipped = 0
    total = 0
    improved = 0
    worsened = 0
    fallbacks = 0
    unchanged = 0

    for year_dir in sorted(gt_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        year_results = []

        for gt_file in sorted(year_dir.glob("*.xml")):
            page = gt_file.stem
            htr_file = htr_dir / year / f"{page}.xml"
            cor_file = corrected_dir / year / f"{page}.json"

            if not htr_file.exists() or not cor_file.exists():
                skipped += 1
                continue

            try:
                gt_parsed = parse_alto_file(gt_file)
                htr_parsed = parse_alto_file(htr_file)
                gt_text = " ".join([l["text"] for l in gt_parsed["lines"]])
                htr_text = " ".join([l["text"] for l in htr_parsed["lines"]])

                if len(gt_text.strip()) < 50:
                    skipped += 1
                    continue

                with open(cor_file, encoding="utf-8") as f:
                    cor_data = json.load(f)

                cor_text = cor_data["fulltext_corrected"]
                is_fallback = cor_data.get("fallback", False)

                cer_before = calculate_cer(gt_text, htr_text)
                cer_after = calculate_cer(gt_text, cor_text)
                wer_before = calculate_wer(gt_text, htr_text)
                wer_after = calculate_wer(gt_text, cor_text)

                result = {
                    "year": year,
                    "page": page,
                    "cer_before": round(cer_before, 6),
                    "cer_after": round(cer_after, 6),
                    "wer_before": round(wer_before, 6),
                    "wer_after": round(wer_after, 6),
                    "gt_chars": len(gt_text),
                    "fallback": is_fallback,
                    "improved": cer_after < cer_before,
                    "worsened": cer_after > cer_before and not is_fallback,
                }

                year_results.append(result)
                all_results.append(result)
                total += 1

                if is_fallback:
                    fallbacks += 1
                elif cer_after < cer_before:
                    improved += 1
                elif cer_after > cer_before:
                    worsened += 1
                else:
                    unchanged += 1

            except Exception as e:
                print(f"  Error {year}/{page}: {e}", flush=True)
                skipped += 1

        if year_results:
            total_chars = sum(r["gt_chars"] for r in year_results)
            avg_cer_before = sum(r["cer_before"] * r["gt_chars"] for r in year_results) / total_chars
            avg_cer_after = sum(r["cer_after"] * r["gt_chars"] for r in year_results) / total_chars
            avg_wer_before = sum(r["wer_before"] * r["gt_chars"] for r in year_results) / total_chars
            avg_wer_after = sum(r["wer_after"] * r["gt_chars"] for r in year_results) / total_chars
            yr_improved = sum(1 for r in year_results if r["improved"])
            yr_fallbacks = sum(1 for r in year_results if r["fallback"])
            yr_worsened = sum(1 for r in year_results if r["worsened"])

            per_year[year] = {
                "pages": len(year_results),
                "cer_before": avg_cer_before,
                "cer_after": avg_cer_after,
                "wer_before": avg_wer_before,
                "wer_after": avg_wer_after,
                "improved": yr_improved,
                "fallbacks": yr_fallbacks,
                "worsened": yr_worsened,
                "total_chars": total_chars,
            }
            improvement = ((avg_cer_before - avg_cer_after) / avg_cer_before * 100) if avg_cer_before > 0 else 0
            print(
                f"  [{year}] {len(year_results)} pages: "
                f"CER {avg_cer_before:.2%} → {avg_cer_after:.2%} ({improvement:+.1f}%) "
                f"[{yr_improved}✓ {yr_fallbacks}⟲ {yr_worsened}✗]",
                flush=True,
            )

    # ─── Global metrics ───────────────────────────────────────────────

    total_chars = sum(d["total_chars"] for d in per_year.values())
    global_cer_before = sum(d["cer_before"] * d["total_chars"] for d in per_year.values()) / total_chars
    global_cer_after = sum(d["cer_after"] * d["total_chars"] for d in per_year.values()) / total_chars
    global_wer_before = sum(d["wer_before"] * d["total_chars"] for d in per_year.values()) / total_chars
    global_wer_after = sum(d["wer_after"] * d["total_chars"] for d in per_year.values()) / total_chars
    global_improvement = ((global_cer_before - global_cer_after) / global_cer_before * 100) if global_cer_before > 0 else 0

    # ─── Print summary ────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"PRODUCTION RESULTS — Full corpus")
    print(f"{'='*70}")
    print(f"\n{'Year':<15} {'Pages':>5} {'CER before':>10} {'CER after':>10} {'Change':>10} {'✓':>4} {'⟲':>4} {'✗':>4}")
    print("-" * 70)
    for year, data in sorted(per_year.items()):
        imp = ((data["cer_before"] - data["cer_after"]) / data["cer_before"] * 100) if data["cer_before"] > 0 else 0
        print(
            f"{year:<15} {data['pages']:>5} {data['cer_before']:>9.2%} {data['cer_after']:>9.2%} "
            f"{imp:>+9.1f}% {data['improved']:>4} {data['fallbacks']:>4} {data['worsened']:>4}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<15} {total:>5} {global_cer_before:>9.2%} {global_cer_after:>9.2%} "
        f"{global_improvement:>+9.1f}% {improved:>4} {fallbacks:>4} {worsened:>4}"
    )

    print(f"\nWER: {global_wer_before:.2%} -> {global_wer_after:.2%}")
    print(f"Skipped: {skipped}")

    # ─── Save results ─────────────────────────────────────────────────

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_pages": total,
        "improved": improved,
        "fallbacks": fallbacks,
        "worsened": worsened,
        "unchanged": unchanged,
        "skipped": skipped,
        "global_cer_before": round(global_cer_before, 6),
        "global_cer_after": round(global_cer_after, 6),
        "global_cer_improvement_pct": round(global_improvement, 2),
        "global_wer_before": round(global_wer_before, 6),
        "global_wer_after": round(global_wer_after, 6),
        "per_year": {
            year: {k: round(v, 6) if isinstance(v, float) else v for k, v in data.items()}
            for year, data in sorted(per_year.items())
        },
    }

    with open(output_dir / "production_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_dir}/production_summary.json")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/calculate_production_cer.py <path-gt> <path-htr-raw> <path-corrected> [path-output]")
        sys.exit(1)

    output = sys.argv[4] if len(sys.argv) > 4 else "benchmark/evaluation/production"
    calculate_production(sys.argv[1], sys.argv[2], sys.argv[3], output)
