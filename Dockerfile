# ─────────────────────────────────────────────────────────────────────
# Dockerfile — HTR post-correction pipeline
# ─────────────────────────────────────────────────────────────────────
#
# Lightweight image with the pipeline dependencies: ALTO XML parser,
# line image cropper, LLM correctors, evaluation metrics, benchmark
# runner, fine-tuning data preparation, and export tools.
#
# Does NOT include Kraken or PyTorch — this is only the Python pipeline
# code that connects to Ollama (external service) for LLM corrections.
#
# Rebuilds quickly when code changes, since src/ is mounted as a volume
# and does not require a rebuild for each change.
# ─────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

RUN pip install --no-cache-dir \
    lxml \
    Pillow \
    pandas \
    requests \
    pyyaml \
    escriptorium-connector

WORKDIR /app

COPY src/ /app/src/
COPY config.yaml /app/config.yaml

RUN mkdir -p /data /output

ENTRYPOINT ["python"]
