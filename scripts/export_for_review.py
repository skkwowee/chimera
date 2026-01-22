#!/usr/bin/env python3
"""
Export screenshots + labels for Label Studio review.

Generates a Label Studio import file that pairs images with their predicted labels.

Usage:
    python scripts/export_for_review.py
    python scripts/export_for_review.py --images data/raw --labels data/labeled --output review/import.json
    python scripts/export_for_review.py --sample 20  # Only export 20 random samples
"""

import argparse
import base64
import json
import random
from pathlib import Path


def image_to_data_uri(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{media_type};base64,{data}"


def find_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching image-label pairs."""
    pairs = []

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        # Look for matching JSON label
        label_path = labels_dir / f"{image_path.stem}.json"
        if label_path.exists():
            pairs.append((image_path, label_path))

    return sorted(pairs, key=lambda x: x[0].name)


def format_json_for_display(label_data: dict) -> str:
    """Format JSON nicely for display in Label Studio."""
    return json.dumps(label_data, indent=2)


def create_label_studio_tasks(
    pairs: list[tuple[Path, Path]],
    images_dir: Path
) -> list[dict]:
    """Create Label Studio task format."""
    tasks = []

    for image_path, label_path in pairs:
        # Load the label
        with open(label_path) as f:
            label_data = json.load(f)

        # Create task with embedded base64 image
        task = {
            "data": {
                "image": image_to_data_uri(image_path),
                "label_json": format_json_for_display(label_data),
                "image_filename": image_path.name
            },
            "predictions": []
        }
        tasks.append(task)

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Export screenshots + labels for Label Studio review"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        default="data/raw",
        help="Directory containing images (default: data/raw)"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="data/labeled",
        help="Directory containing JSON labels (default: data/labeled)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="review/import.json",
        help="Output file for Label Studio import (default: review/import.json)"
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="Randomly sample N pairs (default: all)"
    )

    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_path = Path(args.output)

    # Validate directories
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return 1

    # Find pairs
    pairs = find_pairs(images_dir, labels_dir)
    print(f"Found {len(pairs)} image-label pairs")

    if not pairs:
        print("No matching pairs found!")
        print(f"  Images in {images_dir}: {len(list(images_dir.glob('*')))}")
        print(f"  Labels in {labels_dir}: {len(list(labels_dir.glob('*.json')))}")
        return 1

    # Sample if requested
    if args.sample and args.sample < len(pairs):
        pairs = random.sample(pairs, args.sample)
        print(f"Sampled {len(pairs)} pairs")

    # Create tasks
    tasks = create_label_studio_tasks(pairs, images_dir)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Exported {len(tasks)} tasks to {output_path}")
    print(f"\nNext steps:")
    print(f"1. In Label Studio, go to your project")
    print(f"2. Settings → Cloud Storage → Add Source Storage")
    print(f"3. Storage Type: Local files")
    print(f"4. Absolute local path: {images_dir.absolute()}")
    print(f"5. Import tasks from: {output_path.absolute()}")

    return 0


if __name__ == "__main__":
    exit(main())
