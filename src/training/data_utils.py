"""
Data preparation utilities for GRPO and SFT training.

Provides data loading and conversion for SFT and GRPO training.

Observation model: o_t = (I_{t-k}, ..., I_t, c_t)
  - Prior screenshots provide visual continuity
  - Context string c_t from engine tick data provides round history
  - See D018 in decisions.md
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from PIL import Image

from ..prompts import build_user_prompt


SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class GRPODataItem:
    """A single training sample for GRPO.

    Attributes:
        image_path: Path to the current (most recent) screenshot.
        prior_image_paths: Paths to prior screenshots in the same round,
            ordered oldest to newest. Provides visual continuity.
        context: Round context string c_t generated from tick data.
        ground_truth: Engine-accurate ground truth for reward computation.
        image: Lazily loaded current image.
    """

    image_path: Path
    ground_truth: dict
    prior_image_paths: list[Path] = field(default_factory=list)
    context: str | None = None
    image: Image.Image | None = field(default=None, repr=False)

    def load_image(self) -> Image.Image:
        """Load the current image lazily."""
        if self.image is None:
            self.image = Image.open(self.image_path).convert("RGB")
        return self.image

    def unload_image(self) -> None:
        """Unload image to free memory."""
        self.image = None

    @property
    def prompt(self) -> str:
        """Build user prompt with context."""
        return build_user_prompt(self.context)


def _find_image(directory: Path, stem: str) -> Path | None:
    """Find an image file with the given stem in any supported format."""
    for fmt in SUPPORTED_IMAGE_FORMATS:
        candidate = directory / f"{stem}{fmt}"
        if candidate.exists():
            return candidate
    return None


def convert_labeled_to_grpo_format(
    screenshots_dir: Path | str,
    labels_dir: Path | str,
    manifest: dict[str, dict] | None = None,
) -> list[GRPODataItem]:
    """
    Convert existing labeled data to GRPO training format.

    Labels are expected to contain:
      - game_state: HUD ground truth
      - context: round context string c_t (optional, from generate_sft_labels.py)
      - prior_screenshots: list of screenshot IDs for prior frames (optional)

    Args:
        screenshots_dir: Directory containing screenshot images.
        labels_dir: Directory containing JSON label files.
        manifest: Optional manifest dict to restrict which samples are included.

    Returns:
        List of GRPODataItem instances ready for training.
    """
    screenshots_dir = Path(screenshots_dir)
    labels_dir = Path(labels_dir)

    items = []

    if not labels_dir.exists():
        return items

    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem

        if manifest is not None and stem not in manifest:
            continue

        image_path = _find_image(screenshots_dir, stem)
        if image_path is None:
            continue

        with open(label_path) as f:
            label_data = json.load(f)

        # Extract context and prior screenshot refs from label
        context = label_data.get("context")
        prior_ids = label_data.get("prior_screenshots", [])

        # Resolve prior screenshot IDs to paths
        prior_paths = []
        for prior_id in prior_ids:
            prior_path = _find_image(screenshots_dir, prior_id)
            if prior_path is not None:
                prior_paths.append(prior_path)

        items.append(
            GRPODataItem(
                image_path=image_path,
                ground_truth=label_data,
                prior_image_paths=prior_paths,
                context=context,
            )
        )

    return items


def _build_prompt_content(item: GRPODataItem) -> list[dict]:
    """
    Build Qwen3.5 multimodal content list for a training item.

    Includes prior screenshots (oldest first), then current screenshot,
    then the text prompt with round context.

    Returns:
        List of content dicts for the user message.
    """
    content = []

    # Prior screenshots (oldest first) for visual continuity
    for prior_path in item.prior_image_paths:
        content.append({"type": "image", "image": str(prior_path)})

    # Current screenshot (most recent)
    content.append({"type": "image", "image": str(item.image_path)})

    # Text prompt with context
    content.append({"type": "text", "text": item.prompt})

    return content


def create_grpo_dataset(
    items: list[GRPODataItem],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create dataset from GRPO items.

    Each sample is formatted for Qwen3.5 multimodal input with
    optional prior screenshots and round context.

    Args:
        items: List of GRPODataItem instances.
        train_ratio: Ratio of data for training (rest is validation).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]

    def format_item(item: GRPODataItem) -> dict:
        """Format item for TRL GRPO training."""
        return {
            "prompt": _build_prompt_content(item),
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
        self.items = items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return (len(self.items) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[dict]]:
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

            for idx in batch_indices:
                self.items[idx].unload_image()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)


def prepare_conversation_format(
    image_path: str | Path,
    prompt: str,
    response: str | None = None,
    prior_image_paths: list[str | Path] | None = None,
) -> list[dict]:
    """
    Prepare input in Qwen3.5 conversation format.

    Supports multi-image input: prior screenshots are included before
    the current screenshot.

    Args:
        image_path: Path to the current (most recent) image.
        prompt: Text prompt (should include context via build_user_prompt).
        response: Optional model response (for supervised training).
        prior_image_paths: Optional paths to prior screenshots.

    Returns:
        List of conversation messages in Qwen format.
    """
    content = []

    # Prior images (oldest first)
    if prior_image_paths:
        for prior_path in prior_image_paths:
            content.append({"type": "image", "image": str(prior_path)})

    # Current image
    content.append({"type": "image", "image": str(image_path)})

    # Text prompt with context
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    if response is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        )

    return messages


def format_ground_truth_as_json(ground_truth: dict) -> str:
    """Format ground truth dict as JSON string for training targets.

    Excludes context and metadata — only the fields the model should output.
    """
    output = {}
    for key in ("game_state", "analysis", "advice"):
        if key in ground_truth:
            output[key] = ground_truth[key]
    # If label only has game_state (SFT labels without analysis/advice),
    # return just what's there
    if not output:
        output = ground_truth
    return json.dumps(output, indent=2)


def convert_labeled_to_sft_format(
    screenshots_dir: Path | str,
    labels_dir: Path | str,
    manifest: dict[str, dict] | None = None,
) -> list[GRPODataItem]:
    """
    Convert existing labeled data to SFT training format.

    Same as GRPO format — reuses GRPODataItem since fields are identical.
    """
    return convert_labeled_to_grpo_format(screenshots_dir, labels_dir, manifest)


def create_sft_dataset(
    items: list[GRPODataItem],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create SFT dataset from labeled items.

    Each training sample is a conversation with the ground truth as the
    assistant response. Includes prior screenshots and round context.
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:]

    def format_train_item(item: GRPODataItem) -> dict:
        messages = prepare_conversation_format(
            image_path=item.image_path,
            prompt=item.prompt,
            response=format_ground_truth_as_json(item.ground_truth),
            prior_image_paths=[str(p) for p in item.prior_image_paths],
        )
        return {"messages": messages}

    def format_val_item(item: GRPODataItem) -> dict:
        messages = prepare_conversation_format(
            image_path=item.image_path,
            prompt=item.prompt,
            prior_image_paths=[str(p) for p in item.prior_image_paths],
        )
        return {
            "messages": messages,
            "ground_truth": item.ground_truth,
            "image_path": str(item.image_path),
        }

    train_dataset = [format_train_item(item) for item in train_items]
    val_dataset = [format_val_item(item) for item in val_items]

    return train_dataset, val_dataset
