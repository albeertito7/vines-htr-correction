"""
Cloud API correctors for commercial LLMs (Claude, Gemini, GPT-4o).

Inherits from BaseCorrector and implements _call_llm() for each provider.
All three support both text-only and multimodal (text + image) correction.

Requires the corresponding API keys in config.yaml.

NOTE: These correctors were used during the initial evaluation phase to test
different LLMs before fine-tuning. The production pipeline uses the
fine-tuned local model instead (see scripts/run_production.py).
"""

import base64
import json
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


class ClaudeCorrector(BaseCorrector):
    """Corrector using the Anthropic API (Claude)."""

    def __init__(self, prompt_template, model="claude-sonnet-4-6"):
        super().__init__(model_name=model, prompt_template=prompt_template)
        config = load_config()
        self.api_key = config["llm"]["anthropic_api_key"]
        self.api_url = "https://api.anthropic.com/v1/messages"

    def _call_llm(self, prompt, image_path=None):
        content = []

        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            })

        content.append({"type": "text", "text": prompt})

        response = requests.post(
            self.api_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]


class GeminiCorrector(BaseCorrector):
    """Corrector using the Google Generative AI API (Gemini)."""

    def __init__(self, prompt_template, model="gemini-2.0-flash"):
        super().__init__(model_name=model, prompt_template=prompt_template)
        config = load_config()
        self.api_key = config["llm"]["google_api_key"]

    def _call_llm(self, prompt, image_path=None):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        parts = []

        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": img_b64,
                },
            })

        parts.append({"text": prompt})

        response = requests.post(
            url,
            headers={"content-type": "application/json"},
            json={"contents": [{"parts": parts}]},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OpenAICorrector(BaseCorrector):
    """Corrector using the OpenAI API (GPT-4o)."""

    def __init__(self, prompt_template, model="gpt-4o"):
        super().__init__(model_name=model, prompt_template=prompt_template)
        config = load_config()
        self.api_key = config["llm"]["openai_api_key"]
        self.api_url = "https://api.openai.com/v1/chat/completions"

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
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            json={
                "model": self.model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
