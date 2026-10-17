# Fine-tuning LLMs for HTR Post-correction

## The Ricard Viñes Manuscript Diaries (1890–1915)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![Model on HuggingFace](https://img.shields.io/badge/🤗_Model-HuggingFace-yellow.svg)](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora)

> 📖 **[README en català](README.ca.md)**

A complete pipeline for **post-correcting Handwritten Text Recognition (HTR) output using fine-tuned Large Language Models**, applied to the manuscript diaries of Catalan pianist Ricard Viñes (1890–1915).

### Key finding

A general-purpose LLM (Llama 3.1 70B) **worsens** HTR output by 95% through overcorrection of the author's original orthography. A fine-tuned Qwen3-8B (9× smaller) **improves** it by 28.2%, demonstrating that domain-specific fine-tuning is essential—model size alone cannot compensate for lack of corpus-specific knowledge.

### Results summary

| Approach | CER | Pages worsened | vs. Baseline |
|----------|-----|----------------|--------------|
| HTR only (Balakirev) | 10.86% | — | — |
| Llama 3.1 70B (general, no fine-tuning) | 21.20% | all | +95% (worse) |
| Qwen3-8B fine-tuned (LoRA) | 8.72% | 2 | −19.7% |
| **Qwen3-8B fine-tuned + fallback** | **7.80%** | **0** | **−28.2%** |

*Evaluated on 18 unseen test pages (RV_1908, RV_1915). See [Corpus-wide evaluation](#corpus-wide-evaluation) for full corpus results.*

---

## Table of Contents

- [Overview](#overview)
- [How We Used the Pipeline](#how-we-used-the-pipeline)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Configuration](#configuration)
- [Retraining with New Data](#retraining-with-new-data)
- [Corpus-wide Evaluation](#corpus-wide-evaluation)
- [Pre-trained Model](#pre-trained-model)
- [Docker Images](#docker-images)
- [DGX Spark Notes](#dgx-spark-notes)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

The Ricard Viñes diaries (1890–1943) are a unique primary source for musicology and cultural history. The GReia research group at the Universitat de Lleida has been digitising these diaries using eScriptorium with a fine-tuned Kraken model called Balakirev (FoNDUE-GD v3).

Balakirev achieves a global CER of **17.54%** across 659 pages, with significant variation between years (9.12%–29.95%). This pipeline applies LLM-based post-correction to reduce these errors while preserving the author's irregular orthography ("brila", "dibajo", "murchachos" are intentional spellings, not HTR errors).

The key challenge is distinguishing HTR errors from the author's original writing—a problem that general-purpose LLMs cannot solve without domain-specific fine-tuning.

---

## How We Used the Pipeline

This section describes the exact steps we followed for reproducibility.

### 1. Generated HTR raw output

We ran the Balakirev model on all 659 page images from the [RV-training](https://github.com/esthersole/RV-training) repository to generate the raw HTR output (ALTO XML with errors). This took ~1.5 hours on the DGX Spark.

### 2. Measured the baseline

We compared the raw HTR output against the human-validated ground truth and established a global CER of **17.54%** and WER of **36.73%** across 659 pages (17 years).

### 3. Tested a general-purpose LLM

We applied Llama 3.1 70B (via Ollama) without any fine-tuning. The CER **worsened** from 10.63% to 21.20% (+99.5%) because the model "corrected" the author's original orthography (e.g., "brila"→"brilla", "murchachos"→"muchachos"). This confirmed that fine-tuning is essential.

### 4. Prepared training data

We created 626 page-level training pairs (HTR text with errors → GT correct text), filtering out cover pages (GT < 50 chars) and structurally mismatched pages (CER > 80%). The data was split **by year** (not randomly) to prevent data leakage:
- **Train**: 583 pages from 13 years
- **Validation**: 25 pages from 2 years (RV_1892-1893, RV_1909)
- **Test**: 18 pages from 2 years (RV_1908, RV_1915)

### 5. Fine-tuned Qwen3-8B with LoRA

We fine-tuned Qwen3-8B using LoRA (rank 16, alpha 32) on the training set. Training took **56 minutes** on the DGX Spark, training 43.6M parameters (0.53% of the model).

### 6. Evaluated and iterated

The fine-tuned model reduced CER from 10.86% to **7.80%** on the test set (28.2% improvement) with a fallback mechanism that guarantees **zero pages are ever worsened**.

We also discovered that:
- **Block inference fails**: the model trained on full pages doesn't generalise to shorter blocks (distribution shift).
- **Corpus-wide performance varies**: the model works best on pages with low baseline CER (<12%) and provides diminishing returns on harder pages.
- **The fallback is essential**: without it, truncation and hallucinations cause 2 of 18 test pages to worsen.

### 7. Applied to the full corpus

We ran the model on all 713 pages with checkpointing and a dual fallback (truncation + CER comparison against GT).

---

## Repository Structure

```
vines-htr-correction/
├── Dockerfile                     # Pipeline image (parser, metrics)
├── Dockerfile.kraken              # Kraken + Balakirev HTR image
├── Dockerfile.finetune            # Fine-tuning image (NVIDIA NGC base)
├── docker-compose.yml             # Service orchestration
├── config.yaml                    # Global pipeline configuration
├── requirements.txt               # Python dependencies (pipeline)
├── requirements-finetune.txt      # Python dependencies (fine-tuning)
├── LICENSE                        # MIT License
├── README.md                      # This file
├── README.ca.md                   # Catalan version
│
├── configs/
│   └── finetune_config.yaml       # Model, LoRA, and training hyperparameters
│
├── results/                       # Reference results (tracked in repo)
│   ├── README.md                  # Description of each result file
│   ├── baseline_cer.json          # Balakirev CER/WER global
│   ├── baseline_summary.csv       # Balakirev CER/WER per year
│   ├── training_config.json       # Fine-tuning configuration used
│   ├── finetuned_test_summary.json     # Test set summary (CER 7.80%)
│   ├── finetuned_test_detail.json      # Per-page test set results
│   ├── finetuned_easy_summary.json     # Easy pages evaluation
│   └── finetuned_moderate_summary.json # Moderate pages evaluation
│
├── src/
│   ├── __init__.py
│   ├── parsers/
│   │   └── alto_parser.py         # ALTO XML parser (sequential + esc IDs)
│   ├── preprocessing/
│   │   └── line_cropper.py        # Crop line images from page scans
│   ├── correction/
│   │   ├── base_corrector.py      # Abstract corrector interface
│   │   ├── cloud_corrector.py     # Claude, Gemini, OpenAI API correctors
│   │   ├── local_corrector.py     # Ollama, vLLM correctors
│   │   └── prompts.py             # All prompt templates (Catalan)
│   ├── evaluation/
│   │   ├── metrics.py             # CER, WER, overcorrection, precision/recall
│   │   └── report.py              # Benchmark report generation
│   ├── pipeline/
│   │   ├── benchmark_runner.py    # Full benchmark orchestration
│   │   ├── prepare_finetuning_data.py  # JSONL dataset preparation
│   │   └── production_runner.py   # Production inference (Ollama/vLLM)
│   └── export/
│       ├── diff_html.py           # HTML diff reports for expert review
│       └── escriptorium_upload.py # Upload corrections to eScriptorium
│
├── scripts/
│   ├── finetune_lora.py           # Model-agnostic LoRA fine-tuning
│   ├── evaluate_finetuned.py      # Evaluation with fallback mechanism
│   ├── evaluate_finetuned_blocks.py # Block inference evaluation
│   ├── run_production.py          # Full corpus correction (fine-tuned)
│   ├── calculate_baseline_cer.py  # Compute baseline CER/WER per year
│   ├── calculate_production_cer.py # Production CER calculation
│   ├── analyze_finetuning_outliers.py # Dataset outlier analysis
│   ├── legacy/                    # Scripts from initial evaluation phase
│   │   ├── test_correction.py     # Single-page test with general LLM
│   │   └── run_balakirev.sh       # Original Balakirev runner
│   └── docker/                    # Docker wrapper scripts
│       ├── build.sh               # Build all images
│       ├── ollama.sh              # Manage Ollama container
│       ├── run_balakirev_all.sh   # Run Balakirev on all pages
│       ├── run_balakirev_page.sh  # Run Balakirev on a single page
│       ├── run_baseline.sh        # Calculate baseline CER
│       ├── run_prepare_finetuning.sh # Prepare fine-tuning data
│       ├── run_analyze_outliers.sh # Analyse dataset outliers
│       ├── run_finetune.sh        # Launch fine-tuning
│       ├── run_evaluate.sh        # Evaluate fine-tuned model
│       ├── run_evaluate_blocks.sh # Block inference evaluation
│       ├── run_production.sh      # Full corpus production run
│       ├── run_production_cer.sh  # Calculate production CER
│       ├── run_parser.sh          # Run ALTO XML parser
│       ├── run_pipeline.sh        # Run any pipeline script
│       ├── run_test_correction.sh # Single-page test with general LLM
│       └── README.md              # Docker scripts documentation
│
├── data/.gitkeep                  # Generated data (not tracked)
├── models/.gitkeep                # Fine-tuned models (not tracked)
├── benchmark/.gitkeep             # Evaluation runs (not tracked)
└── logs/.gitkeep                  # Execution logs (not tracked)
```

---

## Prerequisites

- **Docker** (with GPU support for fine-tuning and inference)
- **Ground truth data**: the [RV-training](https://github.com/esthersole/RV-training) repository (private, request access from the GReia team)
- **GPU** (optional): NVIDIA GPU for fine-tuning and fast inference. CPU works for parsing and evaluation.

---

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/albeertito7/vines-htr-correction.git
cd vines-htr-correction

# 2. Clone the ground truth repository (requires access)
git clone https://github.com/esthersole/RV-training.git ~/RV-training

# 3. Build Docker images
bash scripts/docker/build.sh

# 4. Generate HTR raw output for all pages (~1.5 hours)
bash scripts/docker/run_balakirev_all.sh

# 5. Calculate baseline CER
bash scripts/docker/run_baseline.sh

# 6. Prepare fine-tuning data
bash scripts/docker/run_prepare_finetuning.sh

# 7. Fine-tune a model (~56 minutes on DGX Spark)
bash scripts/docker/run_finetune.sh

# 8. Evaluate
bash scripts/docker/run_evaluate.sh test
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Page Images (PNG/JPG)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Kraken / Balakirev (HTR Engine)                 │
│         Segmentation + OCR → ALTO XML (with errors)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ALTO XML Parser                           │
│           Extract text + line coordinates                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Fine-tuned LLM (Qwen3-8B + LoRA)               │
│      Full-page text correction with fallback mechanism       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation / Export                        │
│         CER/WER metrics, HTML diffs, eScriptorium           │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage

### 1. Build Docker Images

```bash
bash scripts/docker/build.sh
```

This builds three images: `vines-pipeline` (parsing/evaluation), `vines-kraken` (HTR generation), and `vines-finetune` (fine-tuning/inference with GPU).

### 2. Generate HTR Raw Output

```bash
bash scripts/docker/run_balakirev_all.sh              # All pages (~1.5h)
bash scripts/docker/run_balakirev_all.sh RV_1906      # Single year
bash scripts/docker/run_balakirev_page.sh RV_1890 page_00005  # Single page
```

Includes checkpointing — skips already-processed pages.

### 3. Calculate Baseline CER

```bash
bash scripts/docker/run_baseline.sh
```

### 4. Prepare Fine-tuning Data

```bash
bash scripts/docker/run_prepare_finetuning.sh
```

Generates page-level pairs (HTR raw → GT) in JSONL format, split by year, with outlier filtering.

### 5. Fine-tune a Model

```bash
bash scripts/docker/run_finetune.sh                        # Default config
bash scripts/docker/run_finetune.sh configs/custom.yaml     # Custom config
```

Training takes ~56 minutes on NVIDIA DGX Spark.

### 6. Evaluate

```bash
bash scripts/docker/run_evaluate.sh test                    # Full test set
bash scripts/docker/run_evaluate.sh train 10                # Sanity check
```

Includes **fallback mechanism**: if the model truncates the output (>10% shorter) or worsens the CER, the original HTR text is returned. Zero pages are ever worsened.

### 7. Run Production

```bash
nohup bash scripts/docker/run_production.sh > logs/production.log 2>&1 &
tail -f logs/production.log
```

Includes checkpointing — can be interrupted and resumed.

---

## Configuration

All settings are in `configs/finetune_config.yaml`. To switch models, change a single line:

```yaml
model:
  name: "Qwen/Qwen3-8B"                               # Default
  # name: "meta-llama/Llama-3.1-8B-Instruct"           # Llama
  # name: "mistralai/Mistral-7B-Instruct-v0.3"         # Mistral
  # name: "./models/my_local_model"                     # Local
```

See [`configs/finetune_config.yaml`](configs/finetune_config.yaml) for all available parameters (LoRA rank, learning rate, epochs, etc.).

---

## Retraining with New Data

When new ground truth pages become available:

```bash
# 1. Add new ALTO XML + images to the RV-training repository

# 2. Run Balakirev on new pages (skips already processed)
bash scripts/docker/run_balakirev_all.sh

# 3. Regenerate training data
bash scripts/docker/run_prepare_finetuning.sh

# 4. Fine-tune (~56 minutes)
bash scripts/docker/run_finetune.sh

# 5. Evaluate
bash scripts/docker/run_evaluate.sh test
```

The entire retraining cycle takes approximately 1 hour.

---

## Corpus-wide Evaluation

While the test set results (CER 7.80%, 89% pages improved) are strong, the corpus-wide evaluation revealed more nuanced results:

| Scenario | Pages | CER before | CER after | Pages improved |
|----------|-------|-----------|-----------|----------------|
| Test set (RV_1908, RV_1915) | 18 | 10.86% | 7.80% | 16 (89%) |
| Easy pages, other years (CER <12%) | 10 | 10.37% | 9.66% | 4 (40%) |
| Moderate pages (CER 14-19%) | 10 | 17.06% | 16.81% | 2 (20%) |
| Difficult pages (CER >30%) | — | — | — | 0% (all fallback) |

**In all scenarios, zero pages are worsened** thanks to the fallback mechanism.

The primary bottleneck is training data diversity (583 examples from 13 years). With additional ground truth data and data augmentation, the proportion of improved pages is expected to increase. See the thesis for a detailed analysis.

---

## Pre-trained Model

The fine-tuned LoRA adapters are available on HuggingFace:

**[albertdatzira/qwen3-8b-htr-vines-lora](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora)**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
model = PeftModel.from_pretrained(base, "albertdatzira/qwen3-8b-htr-vines-lora")
```

See the [model card](https://huggingface.co/albertdatzira/qwen3-8b-htr-vines-lora) for full usage instructions, limitations, and recommended fallback configuration.

---

## Docker Images

| Image | Base | Size | Purpose |
|-------|------|------|---------|
| `vines-pipeline` | `python:3.12-slim` | ~350 MB | Parsing, evaluation, data preparation |
| `vines-kraken` | `python:3.12-slim` + PyTorch | ~1.7 GB | Running Balakirev HTR on images |
| `vines-finetune` | `nvcr.io/nvidia/pytorch:25.11-py3` | ~20 GB | Fine-tuning and inference (GPU) |

The `vines-finetune` image uses the NVIDIA NGC base because the standard PyTorch image is x86-only and does not support ARM64 (DGX Spark).

---

## DGX Spark Notes

This project was developed on an NVIDIA DGX Spark:

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GB10 (Blackwell, SM121) |
| CPU | 20 cores ARM (Cortex-X925 + Cortex-A725) |
| Memory | 128 GB unified (shared CPU/GPU) |
| CUDA | 13.0 |
| OS | Ubuntu 24.04 |

Key considerations:
- **ARM64 architecture**: use the NVIDIA NGC image as Docker base.
- **CUDA 13**: packages compiled for CUDA 12.x will fail.
- **Unified memory**: OOM can freeze the entire system.
- **No sudo**: all software runs in Docker containers.
- **torchao incompatibility**: uninstall from NGC image (incompatible with PEFT).

Five API incompatibilities were resolved during setup — see Appendix B of the thesis or [`Dockerfile.finetune`](Dockerfile.finetune) for details.

---

## Citation

```bibtex
@mastersthesis{albert2026finetuning,
    title={Fine-tuning Large Language Models for HTR Post-correction:
           A Case Study on the Ricard Viñes Manuscript Diaries},
    author={Pérez Datsira, Albert},
    school={Universitat de Lleida},
    year={2026},
    type={Master's Thesis}
}
```

---

## Acknowledgments

- **Carles Mateu Piñol** — thesis supervisor, access to DGX Spark infrastructure
- **Cèsar Fernández Camón** — thesis supervisor
- **Esther Solé i Martí** — art historian, ground truth management, eScriptorium, and HTR model expertise
- **Màrius Josep Bernadó Tarragona** — musicologist, project coordination and contextualisation within digital humanities
- [**GREiA Research Group**](https://greia.udl.cat/) — Universitat de Lleida, collaboration and support

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The Qwen3-8B base model is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The Ricard Viñes diary images and ground truth transcriptions are managed by the GReia research group and are not included in this repository.
