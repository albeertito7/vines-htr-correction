"""
Production runner: apply the selected correction strategy to the
entire corpus with checkpointing, logging, and retry.

This is the original production runner that uses the BaseCorrector
interface (Ollama/vLLM backends). For production with the fine-tuned
model (Transformers + LoRA), see scripts/run_production.py instead.

Workflow:
    1. Load configuration and selected strategy
    2. List all pages in the corpus
    3. For each page (skipping already processed):
       a. Parse the ALTO XML
       b. Correct with the selected model
       c. Save the result
       d. Update the checkpoint
    4. Generate a final summary

Usage:
    python -m src.pipeline.production_runner --config config.yaml
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.parsers.alto_parser import parse_alto_file
from src.correction.prompts import PROMPT_TEXT_ONLY, PROMPT_CONTEXT, PROMPT_MULTIMODAL
from src.correction.local_corrector import OllamaCorrector, VLLMCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/production.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Checkpoint ────────────────────────────────────────────────────────────────

CHECKPOINT_FILE = "data/production_checkpoint.json"


def load_checkpoint():
    """Load the set of already-processed page names."""
    try:
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed", []))
    except FileNotFoundError:
        return set()


def save_checkpoint(processed):
    """Save the current checkpoint state."""
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "processed": sorted(processed),
            "last_updated": datetime.now().isoformat(),
            "total": len(processed),
        }, f, indent=2)


# ─── Corrector build ─────────────────────────────────────────────────

def build_corrector(config):
    """
    Build a corrector from the production section of config.yaml.

    Expected config keys under 'production':
        strategy: "text_only", "context", or "multimodal"
        model: model name (e.g. "llama3.1:70b")
        backend: "ollama" or "vllm"
        context_lines: 0 or N (for context strategy)
    """
    prod = config.get("production", {})
    strategy = prod.get("strategy", "text_only")
    model = prod.get("model", "llama3.1:70b")
    backend = prod.get("backend", "ollama")
    context_lines = prod.get("context_lines", 0)

    # Select prompt template
    if strategy == "multimodal":
        prompt = PROMPT_MULTIMODAL
    elif strategy == "context":
        prompt = PROMPT_CONTEXT
    else:
        prompt = PROMPT_TEXT_ONLY

    # Select backend
    if backend == "vllm":
        base_url = prod.get("vllm_url", "http://localhost:8000")
        corrector = VLLMCorrector(prompt, model=model, base_url=base_url)
    else:
        corrector = OllamaCorrector(prompt, model=model)

    return corrector, context_lines, strategy


# ─── Page correction ───────────────────────────────────────────────────

def correct_page(corrector, parsed_page, context_lines=0,
                 line_images_dir=None, use_images=False, delay=0.5,
                 max_retries=3):
    """
    Correct an entire page with per-line/block retry.

    Returns:
        list of dicts with {line_id, esc_id, text_original, text_corrected}
    """
    lines = parsed_page["lines"]
    results = []

    if context_lines > 0:
        # Block correction
        for i in range(0, len(lines), context_lines):
            block = lines[i:i + context_lines]
            texts = [l["text"] for l in block]
            block_text = "\n".join(texts)

            corrected_text = _call_with_retry(
                corrector.correct_line, block_text,
                max_retries=max_retries
            )
            corrected_lines = corrected_text.split("\n")

            if len(corrected_lines) != len(texts):
                logger.warning(
                    f"Block {i}: {len(corrected_lines)} lines returned "
                    f"vs {len(texts)} expected. Keeping original."
                )
                corrected_lines = texts

            for line, corrected in zip(block, corrected_lines):
                results.append({
                    "line_id": line["line_id"],
                    "esc_id": line.get("esc_id", ""),
                    "text_original": line["text"],
                    "text_corrected": corrected,
                })

            time.sleep(delay)
    else:
        # Line-by-line correction
        for line in lines:
            image_path = None
            if use_images and line_images_dir:
                img = Path(line_images_dir) / f"{line['line_id']}.png"
                if img.exists():
                    image_path = str(img)

            corrected = _call_with_retry(
                corrector.correct_line, line["text"], image_path,
                max_retries=max_retries
            )

            results.append({
                "line_id": line["line_id"],
                "esc_id": line.get("esc_id", ""),
                "text_original": line["text"],
                "text_corrected": corrected,
            })

            time.sleep(delay)

    return results


def _call_with_retry(func, *args, max_retries=3):
    """Execute a function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Error (attempt {attempt + 1}/{max_retries}): {e}. "
                          f"Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"Failed after {max_retries} attempts. Returning original.")
    return args[0]  # return the original text


# ─── Main pipeline ───────────────────────────────────────────────────────

def run_production(config_path="config.yaml"):
    """Run production correction on the entire corpus."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    htr_dir = Path(paths["htr_output"])
    images_dir = Path(paths["line_images"])
    output_dir = Path("data/corrected")
    output_dir.mkdir(parents=True, exist_ok=True)

    prod = config.get("production", {})
    delay = prod.get("delay", 0.5)
    max_retries = prod.get("max_retries", 3)
    use_images = prod.get("strategy") == "multimodal"

    # Build corrector
    corrector, context_lines, strategy = build_corrector(config)
    logger.info(f"Corrector: {corrector.model_name} | Strategy: {strategy}")

    # Load checkpoint
    processed = load_checkpoint()
    logger.info(f"Already processed: {len(processed)} pages")

    # List pages
    xml_files = sorted(htr_dir.glob("*.xml"))
    pending = [f for f in xml_files if f.stem not in processed]
    logger.info(f"Total pages: {len(xml_files)} | Pending: {len(pending)}")

    if not pending:
        logger.info("All pages already processed.")
        return

    # Process
    start_time = datetime.now()
    errors = 0

    for i, xml_file in enumerate(pending):
        page_name = xml_file.stem
        logger.info(f"[{i + 1}/{len(pending)}] Processing: {page_name}")

        try:
            parsed = parse_alto_file(xml_file)

            line_imgs_dir = images_dir / page_name
            results = correct_page(
                corrector, parsed,
                context_lines=context_lines,
                line_images_dir=line_imgs_dir,
                use_images=use_images,
                delay=delay,
                max_retries=max_retries,
            )

            # Save result
            output_file = output_dir / f"{page_name}.json"
            output = {
                "page": page_name,
                "model": corrector.model_name,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat(),
                "lines": results,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            # Update checkpoint
            processed.add(page_name)
            save_checkpoint(processed)

            changes = sum(
                1 for r in results
                if r["text_original"] != r["text_corrected"]
            )
            logger.info(f"  {len(results)} línies, {changes} canvis")

        except Exception as e:
            errors += 1
            logger.error(f"  ERROR a {page_name}: {e}")

    # Final summary
    elapsed = datetime.now() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"PRODUCTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Pages processed: {len(pending) - errors}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Average per page: {elapsed / max(len(pending) - errors, 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production correction")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    run_production(args.config)
