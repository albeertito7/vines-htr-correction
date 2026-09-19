"""
Benchmark runner for HTR post-correction strategies.

Executes all configured correction strategies on the ground truth pages
and saves the results for comparison.

Workflow:
    1. Load GT pages (ALTO XML from RV-training) and HTR raw (Balakirev output)
    2. For each strategy/model:
       a. Parse the ALTO XML
       b. Correct with the corresponding corrector
       c. Compute metrics against GT
       d. Save results
    3. Generate a comparative summary

Usage:
    python -m src.pipeline.benchmark_runner --config config.yaml
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.parsers.alto_parser import parse_alto_file
from src.correction.prompts import (
    PROMPT_TEXT_ONLY,
    PROMPT_CONTEXT,
    PROMPT_MULTIMODAL,
    PROMPT_DETECTION,
)
from src.correction.cloud_corrector import ClaudeCorrector, GeminiCorrector, OpenAICorrector
from src.correction.local_corrector import OllamaCorrector, VLLMCorrector
from src.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
    calculate_overcorrections,
    calculate_corrections,
    evaluate_page,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/benchmark.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Strategy definitions ──────────────────────────────────────────────

def build_strategies(config):
    """
    Build the list of strategies to execute.

    Each strategy is a dict with:
        - name: unique identifier
        - corrector: BaseCorrector instance (None for baseline)
        - context_lines: 0 (line by line) or N (blocks of N lines)
        - use_images: True/False (multimodal)

    Only strategies with the required credentials are included.
    """
    strategies = []

    # A — Baseline (no correction)
    strategies.append({
        "name": "A_baseline",
        "corrector": None,
        "context_lines": 0,
        "use_images": False,
    })

    # B — LLM text-only (cloud)
    if config["llm"].get("anthropic_api_key"):
        strategies.append({
            "name": "B_claude_text",
            "corrector": ClaudeCorrector(PROMPT_TEXT_ONLY),
            "context_lines": 0,
            "use_images": False,
        })

    if config["llm"].get("google_api_key"):
        strategies.append({
            "name": "B_gemini_text",
            "corrector": GeminiCorrector(PROMPT_TEXT_ONLY),
            "context_lines": 0,
            "use_images": False,
        })

    # C — LLM text-only (local via Ollama)
    ollama_url = config["llm"].get("ollama_url", "http://localhost:11434")

    strategies.append({
        "name": "C_llama8b_text",
        "corrector": OllamaCorrector(PROMPT_TEXT_ONLY, model="llama3.1:8b"),
        "context_lines": 0,
        "use_images": False,
    })

    strategies.append({
        "name": "C_llama70b_text",
        "corrector": OllamaCorrector(PROMPT_TEXT_ONLY, model="llama3.1:70b"),
        "context_lines": 0,
        "use_images": False,
    })

    strategies.append({
        "name": "C_mistral_text",
        "corrector": OllamaCorrector(PROMPT_TEXT_ONLY, model="mistral:latest"),
        "context_lines": 0,
        "use_images": False,
    })

    # D — Multimodal (text + line image)
    if config["llm"].get("anthropic_api_key"):
        strategies.append({
            "name": "D_claude_multimodal",
            "corrector": ClaudeCorrector(PROMPT_MULTIMODAL),
            "context_lines": 0,
            "use_images": True,
        })

    strategies.append({
        "name": "D_llama_vision",
        "corrector": OllamaCorrector(PROMPT_MULTIMODAL, model="llama3.2-vision"),
        "context_lines": 0,
        "use_images": True,
    })

    # E — Context aware (blocks of consecutive lines)
    if config["llm"].get("anthropic_api_key"):
        strategies.append({
            "name": "E_claude_context5",
            "corrector": ClaudeCorrector(PROMPT_CONTEXT),
            "context_lines": 5,
            "use_images": False,
        })

    strategies.append({
        "name": "E_llama70b_context5",
        "corrector": OllamaCorrector(PROMPT_CONTEXT, model="llama3.1:70b"),
        "context_lines": 5,
        "use_images": False,
    })

    # F — Two pass (detection + correction)
    if config["llm"].get("anthropic_api_key"):
        strategies.append({
            "name": "F_claude_twopass",
            "corrector": ClaudeCorrector(PROMPT_DETECTION),
            "context_lines": 0,
            "use_images": False,
        })

    return strategies


# ─── Single-page strategy execution ───────────────────────────────

def run_baseline(parsed_page):
    """Strategy A: return the HTR text unmodified."""
    return [line["text"] for line in parsed_page["lines"]]


def run_line_by_line(corrector, parsed_page, line_images_dir=None, use_images=False, delay=1):
    """Strategies B, C, D: correct each line independently."""
    corrected = []
    for line in parsed_page["lines"]:
        image_path = None
        if use_images and line_images_dir:
            img = Path(line_images_dir) / f"{line['line_id']}.png"
            if img.exists():
                image_path = str(img)

        result = corrector.correct_line(line["text"], image_path)
        corrected.append(result)
        time.sleep(delay)

    return corrected


def run_context_blocks(corrector, parsed_page, block_size=5, delay=2):
    """Strategy E: correct in blocks of N consecutive lines."""
    lines = parsed_page["lines"]
    corrected = []

    for i in range(0, len(lines), block_size):
        block = lines[i:i + block_size]
        texts = [l["text"] for l in block]
        block_text = "\n".join(texts)

        result = corrector.correct_line(block_text)
        result_lines = result.split("\n")

        # If line count doesn't match, keep the original
        if len(result_lines) != len(texts):
            logger.warning(
                f"Block {i}: returned {len(result_lines)} lines "
                f"instead of {len(texts)}. Keeping original."
            )
            corrected.extend(texts)
        else:
            corrected.extend(result_lines)

        time.sleep(delay)

    return corrected


def run_two_pass(corrector, parsed_page, delay=1):
    """
    Strategy F: first pass detects errors, second pass corrects
    only the high-confidence detections.
    """
    corrected = []
    for line in parsed_page["lines"]:
        # Pass 1: detection
        errors = corrector.detect_errors(line["text"])
        time.sleep(delay)

        if errors:
            # Pass 2: apply high-confidence suggestions
            text = line["text"]
            for error in errors:
                if error.get("confidence") == "alta" and error.get("suggestion"):
                    text = text.replace(error["original"], error["suggestion"])
            corrected.append(text)
        else:
            corrected.append(line["text"])

        time.sleep(delay)

    return corrected


def run_strategy(strategy, parsed_page, line_images_dir=None, delay=1):
    """Execute a strategy on a page and return the corrected lines."""
    name = strategy["name"]

    if strategy["corrector"] is None:
        return run_baseline(parsed_page)

    if name.startswith("F_"):
        return run_two_pass(strategy["corrector"], parsed_page, delay)

    if strategy["context_lines"] > 0:
        return run_context_blocks(
            strategy["corrector"], parsed_page,
            block_size=strategy["context_lines"], delay=delay,
        )

    return run_line_by_line(
        strategy["corrector"], parsed_page,
        line_images_dir=line_images_dir,
        use_images=strategy["use_images"],
        delay=delay,
    )


# ─── Main Pipeline ───────────────────────────────────────────────────────

"""def load_ground_truth(gt_path):
    gt_path = Path(gt_path)
    if not gt_path.exists():
        return None
    with open(gt_path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f.readlines()]"""


def run_benchmark(config_path="config.yaml"):
    """Run the complete benchmark across all strategies and pages."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    htr_dir = Path(paths["htr_output"])
    htr_raw_dir = Path(paths.get("htr_raw", "data/htr_raw"))
    gt_dir = Path(paths["ground_truth"])
    images_dir = Path(paths["line_images"])
    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    delay = config["benchmark"].get("delay", 2)

    # Find pages that have both GT (from RV-training repo) and HTR raw (from Balakirev)
    # GT: extracted from the RV-training repository.
    # HTR raw: XMLs generated by run_balakirev.sh
    pages_with_gt = []
    for year_dir in sorted(gt_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for gt_file in sorted(year_dir.glob("*.xml")):
            htr_raw_file = htr_raw_dir / year_dir.name / gt_file.name
            if htr_raw_file.exists():
                pages_with_gt.append((gt_file, htr_raw_file))

    if not pages_with_gt:
        logger.error(
            f"No pages with ground truth found. "
            f"Expected .xml files in {gt_dir} matching HTR raw files in {htr_raw_dir}."
        )
        return

    logger.info(f"Found {len(pages_with_gt)} pages with ground truth")

    # Build strategies
    strategies = build_strategies(config)
    logger.info(f"Configured strategies: {[s['name'] for s in strategies]}")

    # Execute
    all_results = []

    for strategy in strategies:
        strategy_name = strategy["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running strategy: {strategy_name}")
        logger.info(f"{'='*60}")

        strategy_results = {
            "strategy": strategy_name,
            "model": strategy["corrector"].model_name if strategy["corrector"] else "cap",
            "pages": [],
        }

        for gt_file, htr_raw_file in pages_with_gt:
            page_name = f"{gt_file.parent.name}/{gt_file.stem}"
            logger.info(f"  Page: {page_name}")

            # Parse GT (ALTO XML from RV-training repo)
            parsed_gt = parse_alto_file(gt_file)
            gt_lines = [line["text"] for line in parsed_gt["lines"]]

            # Parse HTR raw (Balakirev raw output)
            parsed_htr = parse_alto_file(htr_raw_file)
            htr_lines = [line["text"] for line in parsed_htr["lines"]]

            # Line images directory (for multimodal strategies)
            line_imgs_dir = images_dir / page_name

            # Execute strategy
            try:
                corrected_lines = run_strategy(
                    strategy, parsed_htr,
                    line_images_dir=line_imgs_dir,
                    delay=delay,
                )
            except Exception as e:
                logger.error(f"  Error running {strategy_name} a {page_name}: {e}")
                corrected_lines = htr_lines  # fallback to original

            # Evaluate
            if len(corrected_lines) != len(htr_lines):
                logger.warning(
                    f"  {page_name}: correction returned {len(corrected_lines)} lines "
                    f"instead of {len(htr_lines)}. Using original."
                )
                corrected_lines = htr_lines

            evaluation = evaluate_page(gt_lines, htr_lines, corrected_lines)

            page_result = {
                "page": page_name,
                "num_lines": len(gt_lines),
                "evaluation": evaluation,
                "lines": [
                    {
                        "gt": gt,
                        "htr": htr,
                        "corrected": cor,
                    }
                    for gt, htr, cor in zip(gt_lines, htr_lines, corrected_lines)
                ],
            }

            strategy_results["pages"].append(page_result)

            cer_before = evaluation["cer_before"]
            cer_after = evaluation["cer_after"]
            overcor = evaluation["overcorrections"]["count"]
            logger.info(
                f"  CER: {cer_before:.4f} → {cer_after:.4f} | "
                f"Overcorrections: {overcor}"
            )

        all_results.append(strategy_results)

        # Save partial results
        output_file = results_dir / f"{strategy_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Results saved to {output_file}")

    # Save complete summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = results_dir / f"benchmark_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nComplete summary saved to {summary_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTR post-correction benchmark")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    run_benchmark(args.config)
