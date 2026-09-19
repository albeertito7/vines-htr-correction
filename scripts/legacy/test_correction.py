"""
Single-page correction test with metrics.

Quick evaluation script for testing a correction strategy on one page
before running the full benchmark. Corrects the HTR text with a general
LLM via Ollama and compares CER before and after against the ground truth.

This script was used during the initial evaluation to demonstrate that
general-purpose LLMs overcorrect historical text. The fine-tuned model
evaluation uses scripts/evaluate_finetuned.py instead.

Usage:
    python scripts/legacy/test_correction.py <gt-xml> <htr-xml> [ollama-url]

Example:
    python scripts/legacy/test_correction.py \\
        /gt/RV_1890/page_00005.xml \\
        /htr_raw/RV_1890/page_00005.xml \\
        http://host.docker.internal:11434
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.parsers.alto_parser import parse_alto_file
from src.correction.prompts import PROMPT_TEXT_ONLY, PROMPT_CONTEXT
from src.evaluation.metrics import calculate_cer, calculate_wer


def call_ollama(prompt, model="llama3.1:70b", base_url="http://localhost:11434"):
    """Call Ollama directly without depending on the corrector class."""
    import requests
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def test_line_by_line(htr_lines, model, base_url):
    """Strategy B/C: correct each line independently."""
    corrected = []
    for i, line in enumerate(htr_lines):
        prompt = PROMPT_TEXT_ONLY.format(text=line)
        result = call_ollama(prompt, model, base_url)
        corrected.append(result.strip())
        print(f"  [{i+1}/{len(htr_lines)}] {line[:50]}...", flush=True)
        if line != result.strip():
            print(f"         -> {result.strip()[:50]}...", flush=True)
        time.sleep(0.5)
    return corrected


def test_context_blocks(htr_lines, model, base_url, block_size=5):
    """Strategy E: correct in blocks of N consecutive lines."""
    corrected = []
    for i in range(0, len(htr_lines), block_size):
        block = htr_lines[i:i + block_size]
        block_text = "\n".join(block)
        prompt = PROMPT_CONTEXT.format(text=block_text)
        result = call_ollama(prompt, model, base_url)
        result_lines = result.strip().split("\n")

        if len(result_lines) != len(block):
            print(f"  Block {i}: {len(result_lines)} lines vs {len(block)} expected. Keeping original.", flush=True)
            corrected.extend(block)
        else:
            corrected.extend(result_lines)

        print(f"  Block {i+1}-{i+len(block)}/{len(htr_lines)} processed", flush=True)
        time.sleep(0.5)
    return corrected


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_correction.py <gt-xml> <htr-xml> [ollama-url]")
        sys.exit(1)

    gt_path = sys.argv[1]
    htr_path = sys.argv[2]
    base_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:11434"
    model = "llama3.1:70b"

    # Parse
    print(f"GT:  {gt_path}")
    print(f"HTR: {htr_path}")
    print(f"Model: {model}")
    print(f"Ollama: {base_url}")
    print()

    gt_parsed = parse_alto_file(gt_path)
    htr_parsed = parse_alto_file(htr_path)

    gt_lines = [l["text"] for l in gt_parsed["lines"]]
    htr_lines = [l["text"] for l in htr_parsed["lines"]]

    gt_full = " ".join(gt_lines)
    htr_full = " ".join(htr_lines)

    cer_before = calculate_cer(gt_full, htr_full)
    wer_before = calculate_wer(gt_full, htr_full)

    print(f"=== BASELINE (no correction) ===")
    print(f"GT:  {len(gt_lines)} lines, {len(gt_full)} chars")
    print(f"HTR: {len(htr_lines)} lines, {len(htr_full)} chars")
    print(f"CER: {cer_before:.2%}")
    print(f"WER: {wer_before:.2%}")
    print()

    # Strategy B/C: line by line
    print(f"=== STRATEGY B/C: TEXT-ONLY LINE BY LINE ({model}) ===")
    corrected_lbl = test_line_by_line(htr_lines, model, base_url)
    cor_lbl_full = " ".join(corrected_lbl)
    cer_lbl = calculate_cer(gt_full, cor_lbl_full)
    wer_lbl = calculate_wer(gt_full, cor_lbl_full)
    print(f"\n  CER: {cer_before:.2%} -> {cer_lbl:.2%} ({(cer_before-cer_lbl)/cer_before*100:.1f}%)")
    print(f"  WER: {wer_before:.2%} -> {wer_lbl:.2%}")
    print()

    # Strategy E: context blocks
    print(f"=== STRATEGY E: CONTEXT BLOCKS OF 5 ({model}) ===")
    corrected_ctx = test_context_blocks(htr_lines, model, base_url)
    cor_ctx_full = " ".join(corrected_ctx)
    cer_ctx = calculate_cer(gt_full, cor_ctx_full)
    wer_ctx = calculate_wer(gt_full, cor_ctx_full)
    print(f"\n  CER: {cer_before:.2%} -> {cer_ctx:.2%} ({(cer_before-cer_ctx)/cer_before*100:.1f}%)")
    print(f"  WER: {wer_before:.2%} -> {wer_ctx:.2%}")
    print()

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY — {Path(htr_path).stem}")
    print(f"{'='*60}")
    print(f"{'Strategy':<30} {'CER':>8} {'Change':>10}")
    print(f"{'-'*50}")
    print(f"{'Baseline (HTR only)':<30} {cer_before:>7.2%} {'—':>10}")
    print(f"{'B/C text-only line×line':<30} {cer_lbl:>7.2%} {(cer_before-cer_lbl)/cer_before*100:>9.1f}%")
    print(f"{'E context blocks (5)':<30} {cer_ctx:>7.2%} {(cer_before-cer_ctx)/cer_before*100:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
