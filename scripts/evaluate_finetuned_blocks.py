"""
Evaluate the fine-tuned model using block-based inference.

Instead of sending the full page to the model (which can cause truncation
on long pages), this script splits each page into blocks of N lines,
corrects each block independently, and concatenates the results.

Includes post-processing to detect hallucinations: if a word changes
radically (edit distance > 50% of its length), it is reverted to the
original.

IMPORTANT: This experiment demonstrated that block inference does NOT work
with a model trained on full pages (distribution shift). The CER worsened
from 8.72% (full page) to 48.28% (blocks of 10). Block inference would
require retraining the model on block-level pairs. See Chapter 5 of the
thesis for the full analysis.

Usage:
    python scripts/evaluate_finetuned_blocks.py \\
        --model models/finetuned \\
        --gt /gt \\
        --htr /htr_raw \\
        --split data/finetuning/test.jsonl \\
        --block-size 10
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.alto_parser import parse_alto_file
from src.evaluation.metrics import calculate_cer, calculate_wer


# ─── Prompt (must match fine-tuning; kept in Catalan) ────────

SYSTEM_PROMPT = (
    "Ets un expert en post-correcció d'HTR dels diaris de Ramon Viñes. "
    "Corregeix els errors de reconeixement del text manuscrit. "
    "Només corregeix errors d'HTR evidents. No modifiquis l'ortografia "
    "de l'autor, noms propis, ni símbols especials (¶, ¬, ⟦ ⟧). "
    "Si no estàs segur d'una correcció, deixa la paraula original."
)


# ─── Model loading ─────────────────────────────────────────────────────────────────────

def load_model(model_path):
    """Load the base model + LoRA adapters."""
    print(f"Loading model from {model_path}...", flush=True)

    config_path = Path(model_path) / "finetune_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        base_model_name = config.get("config", {}).get("model", {}).get("name", "")
        token = config.get("config", {}).get("model", {}).get("access_token", "") or None
    else:
        base_model_name = ""
        token = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model_name:
        print(f"  Base model: {base_model_name}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


# ─── Block correction ─────────────────────────────────────────────────────────

def correct_block(model, tokenizer, block_text, max_new_tokens=2048):
    """Correct a block of HTR text using the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Corregeix el següent text HTR:\n{block_text}"},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def correct_page_by_blocks(model, tokenizer, htr_lines, block_size=10):
    """
    Correct a page by splitting it into blocks of N lines.

    Each block is corrected independently, and the results are concatenated.
    """
    corrected_blocks = []

    for i in range(0, len(htr_lines), block_size):
        block = htr_lines[i:i + block_size]
        block_text = "\n".join(block)

        corrected = correct_block(model, tokenizer, block_text)
        corrected_blocks.append(corrected)

    return " ".join(corrected_blocks)


# ─── Post-processing ──────────────────────────────────────────────────────────

def levenshtein_word(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_word(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def post_process(htr_text, corrected_text, max_change_ratio=0.5):
    """
    Revert hallucinated word changes.

    Compares words positionally. If a word's edit distance exceeds
    max_change_ratio of its length, the original word is kept.

    Returns:
        tuple (post-processed text, number of reverted words)
    """
    htr_words = htr_text.split()
    cor_words = corrected_text.split()

    # If word counts differ too much, skip positional comparison
    if abs(len(htr_words) - len(cor_words)) > len(htr_words) * 0.2:
        return corrected_text, 0

    result = []
    reverted = 0

    min_len = min(len(htr_words), len(cor_words))
    for i in range(min_len):
        if htr_words[i] != cor_words[i]:
            word_len = max(len(htr_words[i]), 1)
            dist = levenshtein_word(htr_words[i], cor_words[i])

            if dist / word_len > max_change_ratio:
                result.append(htr_words[i])
                reverted += 1
            else:
                result.append(cor_words[i])
        else:
            result.append(cor_words[i])

    # Append remaining words from the longer list
    if len(cor_words) > min_len:
        result.extend(cor_words[min_len:])
    elif len(htr_words) > min_len:
        result.extend(htr_words[min_len:])

    return " ".join(result), reverted


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path, gt_dir, htr_dir, split_path, block_size=10,
             max_pages=None, use_postprocess=True):
    """Evaluate the fine-tuned model with block-based inference."""

    model, tokenizer = load_model(model_path)

    with open(split_path, encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f]

    if max_pages:
        pairs = pairs[:max_pages]

    print(f"\nEvaluating {len(pairs)} pages (blocks of {block_size} lines)...\n", flush=True)

    results = []
    gt_dir = Path(gt_dir)
    htr_dir = Path(htr_dir)

    for i, pair in enumerate(pairs):
        page_id = pair["metadata"]["page_id"]
        year, page = page_id.split("/")

        gt_file = gt_dir / year / f"{page}.xml"
        htr_file = htr_dir / year / f"{page}.xml"

        if not gt_file.exists() or not htr_file.exists():
            print(f"  [{i+1}/{len(pairs)}] {page_id}: FILES NOT FOUND", flush=True)
            continue

        gt_parsed = parse_alto_file(gt_file)
        htr_parsed = parse_alto_file(htr_file)

        gt_text = " ".join([l["text"] for l in gt_parsed["lines"]])
        htr_lines = [l["text"] for l in htr_parsed["lines"]]
        htr_text = " ".join(htr_lines)

        cer_before = calculate_cer(gt_text, htr_text)
        wer_before = calculate_wer(gt_text, htr_text)

        # Block correction
        start = time.time()
        corrected_text = correct_page_by_blocks(model, tokenizer, htr_lines, block_size)
        elapsed = time.time() - start

        # Post-processing
        reverted = 0
        if use_postprocess:
            corrected_text, reverted = post_process(htr_text, corrected_text)

        cer_after = calculate_cer(gt_text, corrected_text)
        wer_after = calculate_wer(gt_text, corrected_text)

        improvement = ((cer_before - cer_after) / cer_before * 100) if cer_before > 0 else 0
        num_blocks = (len(htr_lines) + block_size - 1) // block_size

        result = {
            "page_id": page_id,
            "cer_before": round(cer_before, 4),
            "cer_after": round(cer_after, 4),
            "improvement": round(improvement, 1),
            "wer_before": round(wer_before, 4),
            "wer_after": round(wer_after, 4),
            "gt_chars": len(gt_text),
            "htr_chars": len(htr_text),
            "corrected_chars": len(corrected_text),
            "blocks": num_blocks,
            "reverted_words": reverted,
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

        status = "✓" if cer_after < cer_before else "✗"
        revert_info = f" [{reverted} reverted]" if reverted > 0 else ""
        print(
            f"  [{i+1}/{len(pairs)}] {page_id}: "
            f"CER {cer_before:.2%} → {cer_after:.2%} ({improvement:+.1f}%) "
            f"[{num_blocks} blocks, {elapsed:.1f}s]{revert_info} {status}",
            flush=True,
        )

    # ─── Summary ──────────────────────────────────────────────────────

    if not results:
        print("No pages evaluated.")
        return

    total_gt = sum(r["gt_chars"] for r in results)
    avg_cer_before = sum(r["cer_before"] * r["gt_chars"] for r in results) / total_gt
    avg_cer_after = sum(r["cer_after"] * r["gt_chars"] for r in results) / total_gt
    avg_wer_before = sum(r["wer_before"] * r["gt_chars"] for r in results) / total_gt
    avg_wer_after = sum(r["wer_after"] * r["gt_chars"] for r in results) / total_gt

    improved = sum(1 for r in results if r["cer_after"] < r["cer_before"])
    worsened = sum(1 for r in results if r["cer_after"] > r["cer_before"])
    total_reverted = sum(r["reverted_words"] for r in results)
    total_time = sum(r["time_seconds"] for r in results)
    avg_improvement = ((avg_cer_before - avg_cer_after) / avg_cer_before * 100) if avg_cer_before > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS — Blocks of {block_size} lines + post-processing")
    print(f"{'='*60}")
    print(f"Pages evaluated: {len(results)}")
    print(f"  Improved:  {improved}")
    print(f"  Worsened:  {worsened}")
    print(f"\nWeighted CER:")
    print(f"  Before (HTR only):    {avg_cer_before:.2%}")
    print(f"  After (fine-tuned):   {avg_cer_after:.2%}")
    print(f"  Improvement:          {avg_improvement:.1f}%")
    print(f"\nWeighted WER:")
    print(f"  Before (HTR only):    {avg_wer_before:.2%}")
    print(f"  After (fine-tuned):   {avg_wer_after:.2%}")
    print(f"\nPost-processing: {total_reverted} words reverted")
    print(f"Total time: {total_time:.0f}s ({total_time/len(results):.1f}s/page)")
    print(f"\nComparison:")
    print(f"  HTR only (global baseline):  17.54% CER")
    print(f"  General model (Llama 70B):   21.20% CER (worsens)")
    print(f"  Fine-tuned full page:         8.72% CER")
    print(f"  Fine-tuned blocks of {block_size}:      {avg_cer_after:.2%} CER")

    # ─── Save ─────────────────────────────────────────────────────────

    output_dir = Path("benchmark/evaluation/finetuned_blocks")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{timestamp}_blocks{block_size}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "detail.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "block_size": block_size,
        "post_processing": use_postprocess,
        "pages_evaluated": len(results),
        "pages_improved": improved,
        "pages_worsened": worsened,
        "cer_before": round(avg_cer_before, 6),
        "cer_after": round(avg_cer_after, 6),
        "cer_improvement_pct": round(avg_improvement, 2),
        "wer_before": round(avg_wer_before, 6),
        "wer_after": round(avg_wer_after, 6),
        "total_reverted_words": total_reverted,
        "total_time_seconds": round(total_time, 1),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model with block inference")
    parser.add_argument("--model", default="models/finetuned")
    parser.add_argument("--gt", default="/gt")
    parser.add_argument("--htr", default="/htr_raw")
    parser.add_argument("--split", default="data/finetuning/test.jsonl")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()

    evaluate(args.model, args.gt, args.htr, args.split,
             args.block_size, args.max_pages, not args.no_postprocess)
