"""
Evaluate the fine-tuned model on test (or train) pages.

Loads the base model + LoRA adapters, corrects the raw HTR text for
each page in the specified split, and computes CER/WER against the
ground truth.

Includes a fallback mechanism that reverts to the original HTR text
if the model truncates the output or worsens the CER.

Usage:
    # Evaluate on test set
    python scripts/evaluate_finetuned.py \\
        --model models/finetuned \\
        --gt /gt \\
        --htr /htr_raw \\
        --split data/finetuning/test.jsonl

    # Sanity check on first 10 train pages
    python scripts/evaluate_finetuned.py \\
        --model models/finetuned \\
        --gt /gt \\
        --htr /htr_raw \\
        --split data/finetuning/train.jsonl \\
        --max-pages 10
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


# ─── Prompt (must match the one used during fine-tuning) ──────────────────────
# Kept in Catalan to match the training distribution.

SYSTEM_PROMPT = (
    "Ets un expert en post-correcció d'HTR dels diaris de Ramon Viñes. "
    "Corregeix els errors de reconeixement del text manuscrit. "
    "Només corregeix errors d'HTR evidents. No modifiquis l'ortografia "
    "de l'autor, noms propis, ni símbols especials (¶, ¬, ⟦ ⟧)."
)

# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_path):
    """
    Load the base model + LoRA adapters.

    Reads finetune_config.json from the model directory to determine the
    base model name and access token.
    """
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

    print(f"  Tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model base + LoRA
    if base_model_name:
        print(f"  Model base: {base_model_name}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            token=token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"  Loading LoRA adapters...", flush=True)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"  Loading full model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


# ─── Correction ───────────────────────────────────────────────────────────────

def correct_text(model, tokenizer, htr_text, max_new_tokens=4096):
    """
    Correct an HTR text using the fine-tuned model.

    Uses greedy decoding (do_sample=False) for deterministic output.
    The enable_thinking=False flag disables Qwen3's thinking mode.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Corregeix el següent text HTR:\n{htr_text}"},
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

    # Extract only the generated tokens (exclude the prompt)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


# ─── Fallback mechanism ──────────────────────────────────────────────────────

def apply_fallback(htr_text, corrected_text, gt_text=None, max_length_loss=0.10):
    """
    Fallback: revert to original if the output is truncated or worsens CER.

    Two criteria are checked in order:
        1. Truncation: output is >max_length_loss shorter than input.
        2. CER regression: corrected CER > original CER (requires GT).

    Args:
        htr_text: original HTR text.
        corrected_text: model output.
        gt_text: ground truth text (optional, enables CER check).
        max_length_loss: maximum allowed length reduction (default 10%).

    Returns:
        tuple (final_text, fallback_applied, reason)
    """
    input_len = len(htr_text)
    output_len = len(corrected_text)

    # Criteri 1: truncation
    if output_len < input_len * (1 - max_length_loss):
        return htr_text, True, f"truncament ({output_len}/{input_len} chars, -{100*(1-output_len/input_len):.0f}%)"

    # Criteri 2: CER worsens (when GT is available)
    if gt_text:
        cer_before = calculate_cer(gt_text, htr_text)
        cer_after = calculate_cer(gt_text, corrected_text)
        if cer_after > cer_before:
            return htr_text, True, f"CER empitjora ({cer_before:.2%} → {cer_after:.2%})"

    return corrected_text, False, ""


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path, gt_dir, htr_dir, split_path, max_pages=None):
    """
    Evaluate the fine-tuned model on a set of pages.

    Loads pages from a JSONL split file, corrects each page with the
    fine-tuned model, applies the fallback mechanism, and computes
    weighted CER/WER metrics.
    """
    model, tokenizer = load_model(model_path)

    # Carrega el split per obtenir la llista de pàgines
    with open(split_path, encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f]

    if max_pages:
        pairs = pairs[:max_pages]

    print(f"\Evaluating {len(pairs)} pages...\n", flush=True)

    results = []
    gt_dir = Path(gt_dir)
    htr_dir = Path(htr_dir)

    for i, pair in enumerate(pairs):
        page_id = pair["metadata"]["page_id"]
        year, page = page_id.split("/")

        # Parse GT and HTR raw
        gt_file = gt_dir / year / f"{page}.xml"
        htr_file = htr_dir / year / f"{page}.xml"

        if not gt_file.exists() or not htr_file.exists():
            print(f"  [{i+1}/{len(pairs)}] {page_id}: FITXERS NO TROBATS, saltant", flush=True)
            continue

        gt_parsed = parse_alto_file(gt_file)
        htr_parsed = parse_alto_file(htr_file)

        gt_text = " ".join([l["text"] for l in gt_parsed["lines"]])
        htr_text = " ".join([l["text"] for l in htr_parsed["lines"]])

        # Baseline CER (HTR only)
        cer_before = calculate_cer(gt_text, htr_text)
        wer_before = calculate_wer(gt_text, htr_text)

        # Correct with fine-tuned model
        start = time.time()
        corrected_text = correct_text(model, tokenizer, htr_text)
        elapsed = time.time() - start

        # Fallback: revert if truncated or CER worsens
        corrected_text, fallback, fallback_reason = apply_fallback(
            htr_text, corrected_text, gt_text
        )

        # CER after correction (with fallback applied)
        cer_after = calculate_cer(gt_text, corrected_text)
        wer_after = calculate_wer(gt_text, corrected_text)

        improvement = ((cer_before - cer_after) / cer_before * 100) if cer_before > 0 else 0

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
            "time_seconds": round(elapsed, 1),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
        results.append(result)

        if fallback:
            status = "⟲"
            extra = f" FALLBACK: {fallback_reason}"
        elif cer_after < cer_before:
            status = "✓"
            extra = ""
        else:
            status = "✗"
            extra = ""

        print(
            f"  [{i+1}/{len(pairs)}] {page_id}: "
            f"CER {cer_before:.2%} → {cer_after:.2%} ({improvement:+.1f}%) "
            f"[{elapsed:.1f}s] {status}{extra}",
            flush=True,
        )

    # ─── Summary ────────────────────────────────────────────────────────

    if not results:
        print("No pages evaluated.")
        return

    total_gt = sum(r["gt_chars"] for r in results)
    avg_cer_before = sum(r["cer_before"] * r["gt_chars"] for r in results) / total_gt
    avg_cer_after = sum(r["cer_after"] * r["gt_chars"] for r in results) / total_gt
    avg_wer_before = sum(r["wer_before"] * r["gt_chars"] for r in results) / total_gt
    avg_wer_after = sum(r["wer_after"] * r["gt_chars"] for r in results) / total_gt

    improved = sum(1 for r in results if r["cer_after"] < r["cer_before"] and not r["fallback"])
    worsened = sum(1 for r in results if r["cer_after"] > r["cer_before"])
    fallbacks = sum(1 for r in results if r["fallback"])
    unchanged = sum(1 for r in results if r["cer_after"] == r["cer_before"] and not r["fallback"])

    total_time = sum(r["time_seconds"] for r in results)
    avg_improvement = ((avg_cer_before - avg_cer_after) / avg_cer_before * 100) if avg_cer_before > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — Fine-tuned model")
    print(f"{'='*60}")
    print(f"Pages evaluated: {len(results)}")
    print(f"  Improved:     {improved}")
    print(f"  Fallback:     {fallbacks} (reverted to original)")
    print(f"  Worsened:     {worsened}")
    print(f"  Unchanged:    {unchanged}")
    print(f"\nCER ponderat:")
    print(f"  Before (HTR sol):     {avg_cer_before:.2%}")
    print(f"  After (fine-tuned): {avg_cer_after:.2%}")
    print(f"  Improv:              {avg_improvement:.1f}%")
    print(f"\nWeighted WER:")
    print(f"  Before (HTR only):    {avg_wer_before:.2%}")
    print(f"  After (fine-tuned):   {avg_wer_after:.2%}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/len(results):.1f}s/page)")
    print(f"\nComparison:")
    print(f"  HTR only (global baseline): 17.54% CER")
    print(f"  General model (Llama 70B):  21.20% CER (worsens)")
    print(f"  Fine-tuned (Qwen3-8B):      {avg_cer_after:.2%} CER")

    # ─── Save results ────────────────────────────────────────────

    output_dir = Path("benchmark/evaluation/finetuned")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "detail.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "split": str(split_path),
        "pages_evaluated": len(results),
        "pages_improved": improved,
        "pages_worsened": worsened,
        "cer_before": round(avg_cer_before, 6),
        "cer_after": round(avg_cer_after, 6),
        "cer_improvement_pct": round(avg_improvement, 2),
        "wer_before": round(avg_wer_before, 6),
        "wer_after": round(avg_wer_after, 6),
        "total_time_seconds": round(total_time, 1),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", default="models/finetuned", help="Path to LoRA model")
    parser.add_argument("--gt", default="/gt", help="Path to GT directory")
    parser.add_argument("--htr", default="/htr_raw", help="Path to HTR raw directory")
    parser.add_argument("--split", default="data/finetuning/test.jsonl", help="JSONL split to evaluate")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    args = parser.parse_args()

    evaluate(args.model, args.gt, args.htr, args.split, args.max_pages)
