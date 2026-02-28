#!/usr/bin/env python3
"""
Pull the pre-quantized Qwen3.5-27B VLM checkpoint from HuggingFace Hub.

The checkpoint is created by cloud_quantize.py on a rented GPU (A100 80GB),
then pulled here for local use on RTX 4090.

Usage:
    uv run python scripts/quantize_base_model.py

See also:
    scripts/cloud_quantize.py          — creates the checkpoint (run on cloud GPU)
    scripts/harness_cloud_quantize.sh  — full cloud setup + quantize + push
"""
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HUB_REPO = "skkwowee/chimera-cs2-qwen3.5"
OUTPUT_DIR = Path("models/Qwen3.5-27B-bnb-4bit")


def main():
    print("=" * 60)
    print("Pull Qwen3.5-27B VLM (BnB NF4) from Hub")
    print("=" * 60)
    print()

    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "quantize_meta.json").exists():
        with open(OUTPUT_DIR / "quantize_meta.json") as f:
            meta = json.load(f)
        print(f"Checkpoint already exists (created {meta['timestamp']})")
        print(f"Source: {meta['source_model']}")
        print(f"Parameters: {meta.get('parameter_count_B', 'unknown')}")
        print(f"Vision encoder: {meta.get('includes_vision_encoder', 'unknown')}")
        print("Delete models/ to re-pull.")
        return

    from huggingface_hub import snapshot_download

    print(f"Downloading from {HUB_REPO}...")
    print("(This is the pre-quantized VLM — includes vision encoder)")
    print()

    snapshot_download(
        repo_id=HUB_REPO,
        local_dir=str(OUTPUT_DIR),
    )

    if (OUTPUT_DIR / "quantize_meta.json").exists():
        with open(OUTPUT_DIR / "quantize_meta.json") as f:
            meta = json.load(f)
        print(f"\nParameters: {meta.get('parameter_count_B', 'unknown')}")
        print(f"Vision encoder: {meta.get('includes_vision_encoder', 'unknown')}")

    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"Checkpoint size: {total_size / 1024**3:.1f}GB")
    print()
    print("=" * 60)
    print("Done! Update config.yaml model paths to:")
    print(f"  models/Qwen3.5-27B-bnb-4bit")
    print("=" * 60)


if __name__ == "__main__":
    main()
