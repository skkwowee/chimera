"""
Claude-based labeling for CS2 screenshots.
Uses Claude's vision capabilities to extract game state and generate strategic advice.
"""

import base64
import json
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image

from src.prompts import CS2_SYSTEM_PROMPT, CS2_USER_PROMPT


class ClaudeLabeler:
    """Labels CS2 screenshots using Claude's vision API."""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 1024,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = CS2_SYSTEM_PROMPT

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """Encode image to base64 and determine media type."""
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        return image_data, media_type

    def label_screenshot(
        self,
        image_path: Path | str,
        additional_context: Optional[str] = None,
    ) -> dict:
        """
        Label a single CS2 screenshot.

        Args:
            image_path: Path to the screenshot
            additional_context: Optional extra context (e.g., "This is a retake situation")

        Returns:
            Dictionary with game_state, analysis, and advice
        """
        image_path = Path(image_path)
        image_data, media_type = self._encode_image(image_path)

        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": CS2_USER_PROMPT,
            },
        ]

        if additional_context:
            user_content.append({"type": "text", "text": f"Additional context: {additional_context}"})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        response_text = response.content[0].text

        # Parse JSON from response
        try:
            # Try to find JSON in the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return {"raw_response": response_text, "parse_error": "No JSON found"}
        except json.JSONDecodeError as e:
            return {"raw_response": response_text, "parse_error": str(e)}

    def label_batch(
        self,
        image_paths: list[Path | str],
        output_dir: Optional[Path | str] = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        """
        Label multiple screenshots.

        Args:
            image_paths: List of paths to screenshots
            output_dir: If provided, save each label as JSON file
            skip_existing: Skip images that already have labels

        Returns:
            List of label dictionaries
        """
        from tqdm import tqdm

        results = []
        output_dir = Path(output_dir) if output_dir else None

        for image_path in tqdm(image_paths, desc="Labeling screenshots"):
            image_path = Path(image_path)

            # Check for existing label
            if output_dir and skip_existing:
                label_path = output_dir / f"{image_path.stem}.json"
                if label_path.exists():
                    with open(label_path) as f:
                        results.append(json.load(f))
                    continue

            # Generate label
            label = self.label_screenshot(image_path)
            label["_source_image"] = str(image_path)
            results.append(label)

            # Save if output dir specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                label_path = output_dir / f"{image_path.stem}.json"
                with open(label_path, "w") as f:
                    json.dump(label, f, indent=2)

        return results
