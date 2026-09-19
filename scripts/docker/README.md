# Docker Scripts

Wrapper scripts for running Docker commands on the DGX Spark.
All assume that the Docker images (`vines-pipeline`, `vines-kraken`,
`vines-finetune`) are already built.

## Build images

```bash
bash scripts/docker/build.sh
```

## HTR generation

```bash
bash scripts/docker/run_balakirev_all.sh                      # Full corpus
bash scripts/docker/run_balakirev_all.sh RV_1906              # Single year
bash scripts/docker/run_balakirev_page.sh RV_1890 page_00005  # Single page
```

## Evaluation

```bash
bash scripts/docker/run_baseline.sh                   # Baseline CER/WER
bash scripts/docker/run_evaluate.sh                   # Fine-tuned model
bash scripts/docker/run_evaluate_blocks.sh            # Block inference
bash scripts/docker/run_production_cer.sh             # Production CER
```

## Fine-tuning

```bash
bash scripts/docker/run_prepare_finetuning.sh         # Prepare data
bash scripts/docker/run_finetune.sh                   # Train model
bash scripts/docker/run_analyze_outliers.sh           # Analyse outliers
```

## Production

```bash
bash scripts/docker/run_production.sh                           # Correct full corpus
bash scripts/docker/run_test_correction.sh RV_1890 page_00005   # Single page test
```

## Utilities

```bash
bash scripts/docker/run_parser.sh <file.xml>            # Parse ALTO XML
bash scripts/docker/run_pipeline.sh <script.py> [args]  # Run any script
bash scripts/docker/ollama.sh start                     # Manage Ollama
```
