"""
Data loading utilities for CS2 screenshots and labels.
"""

import json
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image


class ScreenshotDataset:
    """Dataset of CS2 screenshots with optional labels."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(
        self,
        screenshots_dir: Path | str,
        labels_dir: Optional[Path | str] = None,
    ):
        """
        Args:
            screenshots_dir: Directory containing screenshot images
            labels_dir: Optional directory containing JSON label files
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None

        self.image_paths = self._find_images()

    def _find_images(self) -> list[Path]:
        """Find all valid image files in the screenshots directory."""
        if not self.screenshots_dir.exists():
            return []

        images = []
        for path in self.screenshots_dir.iterdir():
            if path.suffix.lower() in self.SUPPORTED_FORMATS:
                images.append(path)

        return sorted(images)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get a screenshot and its label (if available)."""
        image_path = self.image_paths[idx]

        item = {
            "image_path": image_path,
            "image": Image.open(image_path).convert("RGB"),
            "label": None,
        }

        # Load label if available
        if self.labels_dir:
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if label_path.exists():
                with open(label_path) as f:
                    item["label"] = json.load(f)

        return item

    def __iter__(self) -> Iterator[dict]:
        for idx in range(len(self)):
            yield self[idx]

    def unlabeled(self) -> list[Path]:
        """Return paths to screenshots that don't have labels yet."""
        if not self.labels_dir:
            return self.image_paths.copy()

        unlabeled = []
        for image_path in self.image_paths:
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if not label_path.exists():
                unlabeled.append(image_path)

        return unlabeled

    def labeled(self) -> list[Path]:
        """Return paths to screenshots that have labels."""
        if not self.labels_dir:
            return []

        labeled = []
        for image_path in self.image_paths:
            label_path = self.labels_dir / f"{image_path.stem}.json"
            if label_path.exists():
                labeled.append(image_path)

        return labeled

    def stats(self) -> dict:
        """Return dataset statistics."""
        return {
            "total_screenshots": len(self.image_paths),
            "labeled": len(self.labeled()),
            "unlabeled": len(self.unlabeled()),
        }


def load_labeled_data(
    screenshots_dir: Path | str,
    labels_dir: Path | str,
) -> list[dict]:
    """
    Load all labeled screenshots with their labels.

    Returns:
        List of dicts with 'image_path', 'image', and 'label' keys
    """
    dataset = ScreenshotDataset(screenshots_dir, labels_dir)

    labeled_items = []
    for item in dataset:
        if item["label"] is not None:
            labeled_items.append(item)

    return labeled_items


def split_dataset(
    image_paths: list[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split image paths into train and validation sets."""
    import random

    random.seed(seed)
    paths = image_paths.copy()
    random.shuffle(paths)

    split_idx = int(len(paths) * train_ratio)
    return paths[:split_idx], paths[split_idx:]
