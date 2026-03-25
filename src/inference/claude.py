"""
VLM inference for CS2 screenshot analysis using Claude (Anthropic API).
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, cast

import anthropic
from anthropic.types import ImageBlockParam, MessageParam, TextBlock, TextBlockParam
from dotenv import load_dotenv

from src.inference.vlm import parse_json_response
from src.prompts import CS2_SYSTEM_PROMPT, build_user_prompt

load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeVLMInference:
    """
    Run inference using Claude via the Anthropic API.

    Mirrors the Qwen3VLInference interface for side-by-side comparison.
    API key is read from ANTHROPIC_API_KEY env var (Anthropic SDK default).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
    ):
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.client: anthropic.Anthropic = anthropic.Anthropic()

    def analyze(
        self,
        image_path: Path | str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a CS2 screenshot via Claude."""
        image_path = Path(image_path)

        # Read and base64-encode the image
        try:
            image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Image not found: {image_path}") from err

        # Determine media type
        suffix = image_path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/jpeg")

        user_text = prompt or build_user_prompt()

        try:
            messages: list[MessageParam] = [
                {
                    "role": "user",
                    "content": [
                        cast(ImageBlockParam, {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        }),
                        TextBlockParam(type="text", text=user_text),
                    ],
                }
            ]
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=CS2_SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {e}") from e

        block = response.content[0]
        assert isinstance(block, TextBlock)
        text = block.text
        return parse_json_response(text)

    def analyze_batch(
        self,
        image_paths: list[Path | str],
        output_dir: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze multiple screenshots sequentially."""
        results = []
        output_dir = Path(output_dir) if output_dir else None

        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            print(f"  [{i + 1}/{len(image_paths)}] {image_path.name}")
            result = self.analyze(image_path)
            result["_source_image"] = str(image_path)
            results.append(result)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                result_path = output_dir / f"{image_path.stem}_analysis.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

        return results
