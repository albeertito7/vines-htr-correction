"""
Benchmark report generator.

Reads the JSON result files produced by the benchmark runner and generates:
    - Comparative table printed to console
    - CSV file with all metrics
    - Markdown report with detailed analysis

Usage:
    python report.py [results_dir]
"""

import json
import csv
from pathlib import Path


def load_results(results_dir):
    """
    Load all benchmark results from a directory.

    Args:
        results_dir: directory containing per-strategy JSON files.

    Returns:
        list of dicts with each strategy's results.
    """
    results_dir = Path(results_dir)
    results = []

    for f in sorted(results_dir.glob("*.json")):
        # Skip full summary files
        if f.stem.startswith("benchmark_summary"):
            continue
        with open(f, encoding="utf-8") as fh:
            results.append(json.load(fh))

    return results


def aggregate_metrics(strategy_result):
    """
    Aggregate per-page metrics into strategy-level summary.

    Args:
        strategy_result: dict with {strategy, model, pages: [{evaluation}]}

    Returns:
        dict with aggregated metrics, or None if no pages.
    """
    pages = strategy_result["pages"]
    if not pages:
        return None

    n = len(pages)

    avg = lambda key: sum(p["evaluation"][key] for p in pages) / n

    total_overcor = sum(p["evaluation"]["overcorrections"]["count"] for p in pages)
    total_tp = sum(p["evaluation"]["corrections"]["true_positives"] for p in pages)
    total_fp = sum(p["evaluation"]["corrections"]["false_positives"] for p in pages)
    total_fn = sum(p["evaluation"]["corrections"]["false_negatives"] for p in pages)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cer_before = avg("cer_before")
    cer_after = avg("cer_after")
    improvement = ((cer_before - cer_after) / cer_before * 100) if cer_before > 0 else 0.0

    return {
        "strategy": strategy_result["strategy"],
        "model": strategy_result["model"],
        "pages": n,
        "cer_before": round(cer_before, 4),
        "cer_after": round(cer_after, 4),
        "cer_improvement": round(improvement, 1),
        "wer_before": round(avg("wer_before"), 4),
        "wer_after": round(avg("wer_after"), 4),
        "overcorrections": total_overcor,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def print_table(results_dir):
    """Print a comparative table to the console."""
    results = load_results(results_dir)

    if not results:
        print("No results found in the directory.")
        return

    metrics = [aggregate_metrics(r) for r in results]
    metrics = [m for m in metrics if m is not None]

    # Capçalera
    header = (
        f"{'Strategy':<25} {'Model':<20} {'CER before':>10} {'CER after':>12} "
        f"{'Improv':>8} {'Overcor.':>10} {'Precision':>9} {'Recall':>8} {'F1':>6}"
    )
    separator = "─" * len(header)

    print(f"\n{separator}")
    print("BENCHMARK RESULTS")
    print(f"{separator}")
    print(header)
    print(separator)

    for m in metrics:
        print(
            f"{m['strategy']:<25} {m['model']:<20} {m['cer_before']:>10.4f} "
            f"{m['cer_after']:>12.4f} {m['cer_improvement']:>7.1f}% "
            f"{m['overcorrections']:>10} {m['precision']:>9.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>6.4f}"
        )

    print(separator)
    print(f"Pages evaluated: {metrics[0]['pages'] if metrics else 0}")
    print()


def export_csv(results_dir, output_path=None):
    """Export aggregated metrics to CSV."""
    results = load_results(results_dir)
    metrics = [aggregate_metrics(r) for r in results]
    metrics = [m for m in metrics if m is not None]

    if not metrics:
        print("No results to export.")
        return

    if output_path is None:
        output_path = Path(results_dir) / "benchmark_comparison.csv"

    fieldnames = [
        "strategy", "model", "pages",
        "cer_before", "cer_after", "cer_improvement",
        "wer_before", "wer_after",
        "overcorrections", "true_positives", "false_positives", "false_negatives",
        "precision", "recall", "f1",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"CSV exported to {output_path}")


def generate_markdown_report(results_dir, output_path=None):
    """Generate a complete markdown report."""
    results = load_results(results_dir)
    metrics = [aggregate_metrics(r) for r in results]
    metrics = [m for m in metrics if m is not None]

    if not metrics:
        print("No results to generate report.")
        return

    if output_path is None:
        output_path = Path(results_dir) / "benchmark_report.md"

    # Look for the best strategy
    best = min(metrics, key=lambda m: m["cer_after"])

    lines = []
    lines.append("# Benchmark Report — HTR Post-correction")
    lines.append("")
    lines.append(f"Pages evaluated: {metrics[0]['pages']}")
    lines.append("")

    lines.append("## Comparative Table")
    lines.append("")
    lines.append("| Strategy | Model | CER before | CER after | Improvement | Overcor. | Precision | Recall | F1 |")
    lines.append("|-----------|-------|-----------|-------------|---------|-----------|----------|--------|------|")

    for m in metrics:
        marker = " **←**" if m["strategy"] == best["strategy"] else ""
        lines.append(
            f"| {m['strategy']} | {m['model']} | {m['cer_before']:.4f} | "
            f"{m['cer_after']:.4f} | {m['cer_improvement']:.1f}% | "
            f"{m['overcorrections']} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} |{marker}"
        )

    lines.append("")

    # Best strategy conclusion
    lines.append("## Best Strategy")
    lines.append("")
    lines.append(f"**{best['strategy']}** ({best['model']})")
    lines.append(f"- CER: {best['cer_before']:.4f} → {best['cer_after']:.4f} ({best['cer_improvement']:.1f}% improvement)")
    lines.append(f"- Overcorrections: {best['overcorrections']}")
    lines.append(f"- F1: {best['f1']:.4f} (Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f})")
    lines.append("")

    # Per-page detail for best strategy
    best_result = next(r for r in results if r["strategy"] == best["strategy"])
    lines.append("## Per-page Detail (best strategy)")
    lines.append("")
    lines.append("| Page | Lines | CER before | CER after | Overcor. |")
    lines.append("|--------|--------|-----------|-------------|-----------|")

    for page in best_result["pages"]:
        ev = page["evaluation"]
        lines.append(
            f"| {page['page']} | {page['num_lines']} | "
            f"{ev['cer_before']:.4f} | {ev['cer_after']:.4f} | "
            f"{ev['overcorrections']['count']} |"
        )

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report generated at {output_path}")


if __name__ == "__main__":
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark/results"

    print_table(results_dir)
    export_csv(results_dir)
    generate_markdown_report(results_dir)
