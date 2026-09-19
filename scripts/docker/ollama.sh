#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Manage the Ollama container on the DGX Spark.
#
# Usage:
#   bash scripts/docker/ollama.sh start                    # Start Ollama
#   bash scripts/docker/ollama.sh stop                     # Stop Ollama
#   bash scripts/docker/ollama.sh pull llama3.1:70b        # Download model
#   bash scripts/docker/ollama.sh run llama3.1:70b "hello" # Test model
#   bash scripts/docker/ollama.sh list                     # List models
# ─────────────────────────────────────────────────────────────────────

set -e

ACTION="${1:?Usage: $0 <start|stop|pull|run|list> [args...]}"

case "$ACTION" in
    start)
        if docker ps | grep -q ollama; then
            echo "Ollama is already running"
        else
            echo "Starting Ollama..."
            docker run -d --gpus all \
                -v /tmp/ollama:/root/.ollama \
                -p 11434:11434 \
                --name ollama \
                ollama/ollama
            echo "Ollama started at localhost:11434"
        fi
        ;;
    stop)
        echo "Stopping Ollama..."
        docker stop ollama
        docker rm ollama
        echo "Ollama stopped"
        ;;
    pull)
        MODEL="${2:?Usage: $0 pull <model>}"
        echo "Downloading $MODEL..."
        docker exec ollama ollama pull "$MODEL"
        ;;
    run)
        MODEL="${2:?Usage: $0 run <model> \"prompt\"}"
        PROMPT="${3:?Usage: $0 run <model> \"prompt\"}"
        docker exec ollama ollama run "$MODEL" "$PROMPT"
        ;;
    list)
        docker exec ollama ollama list
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 <start|stop|pull|run|list> [args...]"
        exit 1
        ;;
esac
