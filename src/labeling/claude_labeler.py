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

        self.system_prompt = """You are an expert CS2 analyst. Given a screenshot from Counter-Strike 2,
extract the game state and provide strategic advice.

You must respond with valid JSON in this exact format:
{
    "game_state": {
        "map_name": "string or null",
        "round_phase": "buy|playing|freezetime|post-plant|warmup",
        "player_side": "T|CT",
        "player_health": number,
        "player_armor": number,
        "player_money": number,
        "team_money_total": number or null,
        "weapon_primary": "string or null",
        "weapon_secondary": "string or null",
        "utility": ["list", "of", "grenades"],
        "alive_teammates": number,
        "alive_enemies": number,
        "bomb_status": "carried|planted|dropped|null",
        "site": "A|B|mid|connector|etc or null",
        "visible_enemies": number
    },
    "analysis": {
        "situation_summary": "Brief description of current situation",
        "economy_assessment": "full-buy|half-buy|eco|force-buy|save",
        "round_importance": "low|medium|high|critical",
        "immediate_threats": ["list of threats"],
        "opportunities": ["list of opportunities"]
    },
    "advice": {
        "primary_action": "What to do right now",
        "reasoning": "Why this is the right call",
        "fallback": "What to do if primary fails",
        "callout": "What to communicate to team"
    }
}

Be precise about numbers you can see in the HUD. If you can't determine a value, use null.
For strategic advice, consider economy, positioning, team state, and round context."""

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
                "text": "Analyze this CS2 screenshot. Extract the game state and provide strategic advice.",
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
