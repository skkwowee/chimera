#!/usr/bin/env python3
"""
Label CS2 screenshots using Claude API.

Usage:
    python scripts/label_screenshots.py
    python scripts/label_screenshots.py --input data/raw --output data/labeled
    python scripts/label_screenshots.py --single path/to/screenshot.png
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling import ClaudeLabeler
from src.data import ScreenshotDataset
from src.utils import load_config


def label_single(image_path: str, output_path: str | None = None):
    """Label a single screenshot and print/save the result."""
    labeler = ClaudeLabeler()
    result = labeler.label_screenshot(image_path)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Label saved to: {output_path}")
    else:
        print(json.dumps(result, indent=2))

    return result


def label_directory(input_dir: str, output_dir: str, skip_existing: bool = True):
    """Label all screenshots in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = ScreenshotDataset(input_path, output_path)

    if skip_existing:
        to_label = dataset.unlabeled()
        print(f"Found {len(to_label)} unlabeled screenshots (skipping {len(dataset.labeled())} existing)")
    else:
        to_label = dataset.image_paths
        print(f"Found {len(to_label)} screenshots to label")

    if not to_label:
        print("Nothing to label!")
        return

    labeler = ClaudeLabeler()
    results = labeler.label_batch(to_label, output_path, skip_existing=skip_existing)

    # Summary
    successful = sum(1 for r in results if "parse_error" not in r)
    print(f"\nLabeling complete: {successful}/{len(results)} successful")

    return results


def main():
    parser = argparse.ArgumentParser(description="Label CS2 screenshots using Claude")
    parser.add_argument("--single", type=str, help="Path to single screenshot to label")
    parser.add_argument("--input", type=str, default="data/raw", help="Input directory")
    parser.add_argument("--output", type=str, default="data/labeled", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output file for single image")
    parser.add_argument("--no-skip", action="store_true", help="Re-label existing screenshots")

    args = parser.parse_args()

    if args.single:
        label_single(args.single, args.output_file)
    else:
        label_directory(args.input, args.output, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
