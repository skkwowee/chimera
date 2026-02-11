#!/usr/bin/env python3
"""
Upload datasets and models to Hugging Face Hub.

Usage:
    # Upload labeled dataset
    python scripts/upload_to_hub.py dataset --labels data/labeled --screenshots data/raw

    # Upload trained model (SFT or GRPO)
    python scripts/upload_to_hub.py model --model-path outputs/sft/final_model/merged_16bit

    # Upload both
    python scripts/upload_to_hub.py all --labels data/labeled --screenshots data/raw \
        --model-path outputs/sft/final_model/merged_16bit

    # Custom repo names
    python scripts/upload_to_hub.py dataset --repo skkwowee/chimera-cs2-v2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def upload_dataset(
    labels_dir: str,
    screenshots_dir: str,
    repo_id: str,
    private: bool = False,
):
    """Upload labeled screenshots + JSON labels as a HF dataset."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    labels_dir = Path(labels_dir)
    screenshots_dir = Path(screenshots_dir)

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Create repo
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Uploading dataset to {repo_id}...")

    # Upload labels
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print("Error: No label files found")
        sys.exit(1)

    print(f"  Uploading {len(label_files)} label files...")
    api.upload_folder(
        folder_path=str(labels_dir),
        path_in_repo="labels",
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.json"],
    )

    # Upload matching screenshots
    supported_formats = {".png", ".jpg", ".jpeg", ".webp"}
    label_stems = {f.stem for f in label_files}
    screenshot_count = 0

    if screenshots_dir.exists():
        screenshots_to_upload = []
        for img in screenshots_dir.iterdir():
            if img.suffix.lower() in supported_formats and img.stem in label_stems:
                screenshots_to_upload.append(img.name)
                screenshot_count += 1

        if screenshots_to_upload:
            print(f"  Uploading {screenshot_count} screenshots...")
            api.upload_folder(
                folder_path=str(screenshots_dir),
                path_in_repo="screenshots",
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=screenshots_to_upload,
            )

    # Create dataset card
    card = f"""---
license: mit
task_categories:
  - image-to-text
  - visual-question-answering
tags:
  - gaming
  - cs2
  - vlm
  - game-ai
size_categories:
  - n<1K
---

# Chimera CS2 Dataset

Labeled Counter-Strike 2 screenshots for vision-language model training.

Each sample has:
- A CS2 gameplay screenshot
- Ground truth JSON with `game_state`, `analysis`, and `advice`

## Structure

```
screenshots/    # PNG/JPG images
labels/         # Matching JSON files (same stem name)
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
```

## Stats

- **Samples:** {len(label_files)}
- **Screenshots:** {screenshot_count}

## Paper

*Think Before You See: VLMs as Game Agents Without Reinforcement Learning from Scratch*
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def upload_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
):
    """Upload trained model weights to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    # Create repo
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Uploading model to {repo_id}...")

    # Upload all model files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )

    # Check for training config
    config_candidates = [
        model_path.parent / "training_config.json",
        model_path.parent.parent / "training_config.json",
    ]
    for config_path in config_candidates:
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="training_config.json",
                repo_id=repo_id,
                repo_type="model",
            )
            break

    # Create model card
    card = f"""---
license: mit
library_name: transformers
tags:
  - gaming
  - cs2
  - vlm
  - game-ai
  - qwen
base_model: Qwen/Qwen3-VL-8B-Instruct
pipeline_tag: image-to-text
---

# Chimera — CS2 Game AI (Qwen3-VL-8B)

Fine-tuned Qwen3-VL-8B for Counter-Strike 2 screenshot analysis and strategic advice.

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```

## Paper

*Think Before You See: VLMs as Game Agents Without Reinforcement Learning from Scratch*
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model uploaded: https://huggingface.co//{repo_id}")
    return repo_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload datasets and models to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dataset subcommand
    ds_parser = subparsers.add_parser("dataset", help="Upload labeled dataset")
    ds_parser.add_argument(
        "--labels", type=str, default="data/labeled", help="Labels directory"
    )
    ds_parser.add_argument(
        "--screenshots", type=str, default="data/raw", help="Screenshots directory"
    )
    ds_parser.add_argument(
        "--repo", type=str, default="skkwowee/chimera-cs2-labeled",
        help="HF dataset repo ID",
    )
    ds_parser.add_argument("--private", action="store_true", help="Make repo private")

    # Model subcommand
    model_parser = subparsers.add_parser("model", help="Upload trained model")
    model_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model directory"
    )
    model_parser.add_argument(
        "--repo", type=str, default="skkwowee/chimera-cs2-qwen3-vl-8b",
        help="HF model repo ID",
    )
    model_parser.add_argument("--private", action="store_true", help="Make repo private")

    # All subcommand
    all_parser = subparsers.add_parser("all", help="Upload dataset and model")
    all_parser.add_argument(
        "--labels", type=str, default="data/labeled", help="Labels directory"
    )
    all_parser.add_argument(
        "--screenshots", type=str, default="data/raw", help="Screenshots directory"
    )
    all_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model directory"
    )
    all_parser.add_argument(
        "--dataset-repo", type=str, default="skkwowee/chimera-cs2-labeled",
        help="HF dataset repo ID",
    )
    all_parser.add_argument(
        "--model-repo", type=str, default="skkwowee/chimera-cs2-qwen3-vl-8b",
        help="HF model repo ID",
    )
    all_parser.add_argument("--private", action="store_true", help="Make repos private")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "dataset":
        upload_dataset(args.labels, args.screenshots, args.repo, args.private)

    elif args.command == "model":
        upload_model(args.model_path, args.repo, args.private)

    elif args.command == "all":
        upload_dataset(args.labels, args.screenshots, args.dataset_repo, args.private)
        upload_model(args.model_path, args.model_repo, args.private)


if __name__ == "__main__":
    main()
