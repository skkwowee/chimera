#!/usr/bin/env python3
"""
Run inference on CS2 screenshots using local VLM.

Usage:
    python scripts/run_inference.py --single path/to/screenshot.png
    python scripts/run_inference.py --input data/raw --output data/predictions
    python scripts/run_inference.py --model qwen  # Use Qwen2-VL instead
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.vlm import DeepSeekVLInference, Qwen3VLInference, QwenVLInference
from src.data import ScreenshotDataset


def get_model(model_type: str):
    """Get the appropriate model class."""
    if model_type == "deepseek":
        return DeepSeekVLInference()
    elif model_type == "qwen3":
        return Qwen3VLInference()
    elif model_type == "qwen3-moe":
        return Qwen3VLInference(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct")
    elif model_type == "qwen" or model_type == "qwen2":
        return QwenVLInference()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: deepseek, qwen3, qwen3-moe, qwen2")


def analyze_single(image_path: str, model_type: str = "deepseek", output_path: str | None = None):
    """Analyze a single screenshot."""
    model = get_model(model_type)
    model.load_model()

    result = model.analyze(image_path)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved to: {output_path}")
    else:
        print(json.dumps(result, indent=2))

    return result


def analyze_directory(input_dir: str, output_dir: str, model_type: str = "deepseek", labeled_only: bool = False, labels_dir: str = "data/labeled"):
    """Analyze all screenshots in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = ScreenshotDataset(input_path)

    if labeled_only:
        # Only process images that have corresponding labels
        labels_path = Path(labels_dir)
        labeled_stems = {p.stem for p in labels_path.glob("*.json")}
        image_paths = [p for p in dataset.image_paths if p.stem in labeled_stems]
        print(f"Found {len(image_paths)} labeled screenshots to analyze (out of {len(dataset)} total)")
    else:
        image_paths = dataset.image_paths
        print(f"Found {len(dataset)} screenshots to analyze")

    if len(image_paths) == 0:
        print("No screenshots found!")
        return

    model = get_model(model_type)
    results = model.analyze_batch(image_paths, output_path)

    print(f"\nAnalysis complete: {len(results)} screenshots processed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference on CS2 screenshots")
    parser.add_argument("--single", type=str, help="Path to single screenshot")
    parser.add_argument("--input", type=str, default="data/raw", help="Input directory")
    parser.add_argument("--output", type=str, default="data/predictions", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output file for single image")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3",
        choices=["deepseek", "qwen3", "qwen3-moe", "qwen2"],
        help="Model to use (qwen3 recommended)",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Only process images that have corresponding labels",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labeled",
        help="Directory containing labels (for --labeled-only)",
    )

    args = parser.parse_args()

    if args.single:
        analyze_single(args.single, args.model, args.output_file)
    else:
        analyze_directory(args.input, args.output, args.model, args.labeled_only, args.labels_dir)


if __name__ == "__main__":
    main()
