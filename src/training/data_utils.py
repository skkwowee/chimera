"""
Data preparation utilities for GRPO training.

Provides data loading and conversion for Unsloth-compatible GRPO training.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from PIL import Image


@dataclass
class GRPODataItem:
    """A single training sample for GRPO."""

    image_path: Path
    prompt: str
    ground_truth: dict
    image: Image.Image | None = field(default=None, repr=False)

    def load_image(self) -> Image.Image:
        """Load the image lazily."""
        if self.image is None:
            self.image = Image.open(self.image_path).convert("RGB")
        return self.image

    def unload_image(self) -> None:
        """Unload image to free memory."""
        self.image = None


# Default prompt template for CS2 analysis
DEFAULT_PROMPT = """Analyze this Counter-Strike 2 screenshot and provide:
1. Game state (health, armor, money, weapons, alive players, etc.)
2. Strategic analysis (economy assessment, round importance, threats, opportunities)
3. Tactical advice (primary action, reasoning, fallback plan, team callout)

Respond in JSON format with keys: game_state, analysis, advice"""


def convert_labeled_to_grpo_format(
    screenshots_dir: Path | str,
    labels_dir: Path | str,
    prompt: str | None = None,
    manifest: dict[str, dict] | None = None,
) -> list[GRPODataItem]:
    """
    Convert existing labeled data to GRPO training format.

    Args:
        screenshots_dir: Directory containing screenshot images
        labels_dir: Directory containing JSON label files
        prompt: Custom prompt to use (defaults to DEFAULT_PROMPT)
        manifest: Optional manifest dict (from load_manifest/filter_manifest)
                  to restrict which samples are included

    Returns:
        List of GRPODataItem instances ready for training
    """
    screenshots_dir = Path(screenshots_dir)
    labels_dir = Path(labels_dir)
    prompt = prompt or DEFAULT_PROMPT

    supported_formats = {".png", ".jpg", ".jpeg", ".webp"}
    items = []

    if not labels_dir.exists():
        return items

    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem

        # Skip if manifest provided and this sample isn't in it
        if manifest is not None and stem not in manifest:
            continue

        # Find matching image
        image_path = None
        for fmt in supported_formats:
            candidate = screenshots_dir / f"{stem}{fmt}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        # Load label
        with open(label_path) as f:
            ground_truth = json.load(f)

        items.append(
            GRPODataItem(
                image_path=image_path,
                prompt=prompt,
                ground_truth=ground_truth,
            )
        )

    return items


def create_grpo_dataset(
    items: list[GRPODataItem],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create Unsloth-compatible dataset from GRPO items.

    Each sample is formatted for Qwen3-VL multimodal input:
    - prompt: List with image and text content
    - ground_truth: The expected JSON output

    Args:
        items: List of GRPODataItem instances
        train_ratio: Ratio of data for training (rest is validation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset) where each dataset
        is a list of dicts with 'prompt' and 'ground_truth' keys
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]

    def format_item(item: GRPODataItem) -> dict:
        """Format item for Unsloth/TRL GRPO training."""
        return {
            "prompt": [
                {"type": "image", "image": str(item.image_path)},
                {"type": "text", "text": item.prompt},
            ],
            "ground_truth": item.ground_truth,
            "image_path": str(item.image_path),
        }

    train_dataset = [format_item(item) for item in train_items]
    val_dataset = [format_item(item) for item in val_items]

    return train_dataset, val_dataset


class GRPODataLoader:
    """
    Lazy-loading data loader for GRPO training.

    Loads images on-demand to minimize memory usage during training.
    """

    def __init__(
        self,
        items: list[GRPODataItem],
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            items: List of GRPODataItem instances
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for shuffling
        """
        self.items = items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.items) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[dict]]:
        """Iterate over batches with lazy image loading."""
        indices = list(range(len(self.items)))

        if self.shuffle:
            self._rng.shuffle(indices)

        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start : batch_start + self.batch_size]
            batch = []

            for idx in batch_indices:
                item = self.items[idx]
                image = item.load_image()

                batch.append(
                    {
                        "image": image,
                        "image_path": str(item.image_path),
                        "prompt": item.prompt,
                        "ground_truth": item.ground_truth,
                    }
                )

            yield batch

            # Unload images after batch is processed
            for idx in batch_indices:
                self.items[idx].unload_image()

    def reset(self, seed: int | None = None) -> None:
        """Reset the random state for a new epoch."""
        if seed is not None:
            self._rng = random.Random(seed)


def prepare_conversation_format(
    image_path: str | Path,
    prompt: str,
    response: str | None = None,
) -> list[dict]:
    """
    Prepare input in Qwen3-VL conversation format.

    This format is compatible with Unsloth's apply_chat_template.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        response: Optional model response (for supervised training)

    Returns:
        List of conversation messages in Qwen format
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if response is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
        )

    return messages


def format_ground_truth_as_json(ground_truth: dict) -> str:
    """
    Format ground truth dict as JSON string for training targets.

    Args:
        ground_truth: Dict containing game_state, analysis, advice

    Returns:
        JSON string formatted for model output
    """
    return json.dumps(ground_truth, indent=2)


def convert_labeled_to_sft_format(
    screenshots_dir: Path | str,
    labels_dir: Path | str,
    prompt: str | None = None,
    manifest: dict[str, dict] | None = None,
) -> list[GRPODataItem]:
    """
    Convert existing labeled data to SFT training format.

    Same matching logic as convert_labeled_to_grpo_format — reuses GRPODataItem
    since the fields are identical.

    Args:
        screenshots_dir: Directory containing screenshot images
        labels_dir: Directory containing JSON label files
        prompt: Custom prompt to use (defaults to DEFAULT_PROMPT)
        manifest: Optional manifest dict (from load_manifest/filter_manifest)
                  to restrict which samples are included

    Returns:
        List of GRPODataItem instances ready for SFT training
    """
    return convert_labeled_to_grpo_format(screenshots_dir, labels_dir, prompt, manifest)


def create_sft_dataset(
    items: list[GRPODataItem],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create SFT dataset from labeled items.

    Each training sample is formatted as a conversation with the ground truth
    as the assistant response. Validation samples also include ground_truth
    as a separate key for use with evaluate().

    Args:
        items: List of GRPODataItem instances
        train_ratio: Ratio of data for training (rest is validation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset) where each entry has 'messages'
        and val entries also have 'ground_truth' and 'image_path'
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]

    def format_train_item(item: GRPODataItem) -> dict:
        """Format item for SFT training with assistant response."""
        messages = prepare_conversation_format(
            image_path=item.image_path,
            prompt=item.prompt,
            response=format_ground_truth_as_json(item.ground_truth),
        )
        return {"messages": messages}

    def format_val_item(item: GRPODataItem) -> dict:
        """Format item for evaluation — user message only + ground_truth for scoring."""
        messages = prepare_conversation_format(
            image_path=item.image_path,
            prompt=item.prompt,
        )
        return {
            "messages": messages,
            "ground_truth": item.ground_truth,
            "image_path": str(item.image_path),
        }

    train_dataset = [format_train_item(item) for item in train_items]
    val_dataset = [format_val_item(item) for item in val_items]

    return train_dataset, val_dataset
