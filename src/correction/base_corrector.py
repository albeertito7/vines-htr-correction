"""
Abstract base class for all HTR post-correction strategies.

Defines the common interface that each corrector must implement.
Subclasses (cloud_corrector, local_corrector) inherit from this class
and implement the _call_llm() method for their specific backend.

The base class handles input preparation, error handling, fallback to
original text on failure, and result formatting. Subclasses only need
to define how to send a prompt to the LLM and receive a response.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCorrector(ABC):
    """
    Base class for LLM-based HTR text correction.

    Each subclass implements _call_llm() for its specific model and
    modality (cloud API, local Ollama/vLLM, multimodal). The logic for
    preparing inputs, handling errors, and formatting results is shared.
    """

    def __init__(self, model_name, prompt_template):
        """
        Args:
            model_name: model identifier (e.g. "claude-sonnet-4-6", "llama3.1:70b").
            prompt_template: prompt string with {text} placeholder
                             (and optionally {errors} for two-pass).
        """
        self.model_name = model_name
        self.prompt_template = prompt_template

    @abstractmethod
    def _call_llm(self, prompt, image_path=None):
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: complete prompt text.
            image_path: path to line image (multimodal only, None for text-only).

        Returns:
            string with the LLM's response.
        """
        pass

    def correct_line(self, line_text, image_path=None):
        """
        Correct a single line of HTR text.

        Args:
            line_text: HTR text of the line.
            image_path: path to the cropped line image (optional, for multimodal).

        Returns:
            string with the corrected text, or the original if correction fails.
        """
        prompt = self.prompt_template.format(text=line_text)

        try:
            response = self._call_llm(prompt, image_path)
            return response.strip()
        except Exception as e:
            logger.error(f"Error correcting line: {e}")
            return line_text  # original on failure

    def correct_block(self, lines, image_paths=None):
        """
        Correct a block of consecutive HTR lines.

        Args:
            lines: list of strings (HTR text per line).
            image_paths: list of image paths (optional).

        Returns:
            list of corrected strings, or originals if correction fails.
        """
        text_block = "\n".join(lines)
        prompt = self.prompt_template.format(text=text_block)

        try:
            response = self._call_llm(prompt)
            corrected_lines = response.strip().split("\n")

            # If the LLM returns a different number of lines, keep the original
            if len(corrected_lines) != len(lines):
                logger.warning(
                    f"LLM returned {len(corrected_lines)} lines "
                    f"instead of {len(lines)}. Keeping original."
                )
                return lines

            return corrected_lines
        except Exception as e:
            logger.error(f"Error correcting block: {e}")
            return lines

    def detect_errors(self, line_text):
        """
        Detect errors in a line (for two-pass strategy, first pass).

        Args:
            line_text: HTR text of the line.

        Returns:
            list of dicts with {original, suggestion, confidence, reason},
            or empty list if no errors found.
        """
        prompt = self.prompt_template.format(text=line_text)

        try:
            response = self._call_llm(prompt)
            # Clean response for JSON parsing
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            errors = json.loads(response)
            return errors if isinstance(errors, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Could not parse detection JSON: {response[:100]}")
            return []
        except Exception as e:
            logger.error(f"Error detecting errors: {e}")
            return []

    def correct_page(self, parsed_page, line_images_dir=None, context_lines=0):
        """
        Correct all lines of a parsed page.

        Args:
            parsed_page: dict with {filename, lines} (ALTO parser output).
            line_images_dir: directory with cropped line images (optional).
            context_lines: if > 0, send blocks of N lines instead of one by one.

        Returns:
            dict with {filename, model, lines: [{line_id, text_original, text_corrected}]}
        """
        lines = parsed_page["lines"]
        results = []

        if context_lines > 0:
            # Strategy E: block correction (context-aware)
            for i in range(0, len(lines), context_lines):
                block = lines[i:i + context_lines]
                texts = [l["text"] for l in block]
                corrected = self.correct_block(texts)

                for line, corrected_text in zip(block, corrected):
                    results.append({
                        "line_id": line["line_id"],
                        "text_original": line["text"],
                        "text_corrected": corrected_text,
                    })
        else:
            # Line-by-line correction
            for line in lines:
                image_path = None
                if line_images_dir:
                    img_path = Path(line_images_dir) / f"{line['line_id']}.png"
                    if img_path.exists():
                        image_path = str(img_path)

                corrected = self.correct_line(line["text"], image_path)
                results.append({
                    "line_id": line["line_id"],
                    "text_original": line["text"],
                    "text_corrected": corrected,
                })

        return {
            "filename": parsed_page["filename"],
            "model": self.model_name,
            "lines": results,
        }
