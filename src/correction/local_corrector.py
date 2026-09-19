"""
Local LLM correctors via Ollama and vLLM.

Inherits from BaseCorrector and implements _call_llm() for models
running on a local GPU (Llama, Mistral, Qwen, etc.).

Two backends are supported:
    - Ollama: simpler setup, ideal for prototyping and single-page testing.
    - vLLM: faster batch inference, OpenAI-compatible API, ideal for production.

Both support text-only and multimodal (text + image) correction.

Usage:
    corrector = OllamaCorrector(PROMPT_TEXT_ONLY, model="llama3.1:70b")
    result = corrector.correct_page(parsed_page)
"""

import base64
import logging
from pathlib import Path

import requests
import yaml

from src.correction.base_corrector import BaseCorrector

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class OllamaCorrector(BaseCorrector):
    """Corrector using Ollama for local model inference."""

    def __init__(self, prompt_template, model="llama3.1:70b"):
        super().__init__(model_name=model, prompt_template=prompt_template)
        config = load_config()
        self.base_url = config["llm"]["ollama_url"]

    def _call_llm(self, prompt, image_path=None):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        # Attach image for multimodal models (e.g. llama3.2-vision)
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            payload["images"] = [img_b64]

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]


class VLLMCorrector(BaseCorrector):
    """
    Corrector using vLLM for local model inference.

    vLLM exposes an OpenAI-compatible API, so the request format
    matches the OpenAI chat completions endpoint.
    """

    def __init__(self, prompt_template, model="meta-llama/Llama-3.1-70B-Instruct", base_url="http://localhost:8000"):
        super().__init__(model_name=model, prompt_template=prompt_template)
        self.base_url = base_url

    def _call_llm(self, prompt, image_path=None):
        content = []

        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                },
            })

        content.append({"type": "text", "text": prompt})

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"content-type": "application/json"},
            json={
                "model": self.model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
