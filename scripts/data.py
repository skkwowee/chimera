#!/usr/bin/env python3
"""
HuggingFace Hub data management for chimera.

Usage:
    python scripts/data.py status              # Show local/remote counts
    python scripts/data.py pull                # Download dataset from Hub
    python scripts/data.py pull --subset 100   # Pull first N samples
    python scripts/data.py push --captures     # Assemble from captures + upload
    python scripts/data.py push --model PATH   # Upload model weights
    python scripts/data.py clean               # Remove local data copies
    python scripts/data.py clean --all         # Remove everything including outputs
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RAW = DATA / "raw"
LABELED = DATA / "labeled"
CAPTURES = DATA / "captures"
HF_CACHE = DATA / ".hf_cache"


def _load_config():
    """Load hub config from config/config.yaml."""
    import yaml

    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _get_repo(args, config, key="dataset_repo"):
    """Resolve repo ID from CLI --repo or config."""
    if getattr(args, "repo", None):
        return args.repo
    hub = config.get("hub", {})
    repo = hub.get(key)
    if not repo:
        print(f"Error: No repo configured. Set hub.{key} in config/config.yaml or pass --repo")
        sys.exit(1)
    return repo


def _count_files(directory, patterns=("*.json", "*.png", "*.jpg", "*.jpeg", "*.webp")):
    """Count files matching patterns in a directory."""
    if not directory.exists():
        return 0
    count = 0
    for pat in patterns:
        count += len(list(directory.glob(pat)))
    return count


def _dir_size_mb(directory):
    """Total size of a directory in MB."""
    if not directory.exists():
        return 0.0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / (1024 * 1024)


# ── status ──────────────────────────────────────────────────────────────


def cmd_status(args):
    config = _load_config()

    print("=== Local data ===")
    raw_count = _count_files(RAW, ("*.png", "*.jpg", "*.jpeg", "*.webp"))
    label_count = _count_files(LABELED, ("*.json",))
    captures_count = 0
    captures_maps = []
    if CAPTURES.exists():
        for map_dir in sorted(CAPTURES.iterdir()):
            if map_dir.is_dir():
                n = _count_files(map_dir / "labels", ("*.json",))
                captures_count += n
                captures_maps.append(f"  {map_dir.name}: {n} labels")

    print(f"  data/raw/       {raw_count} screenshots")
    print(f"  data/labeled/   {label_count} labels")
    print(f"  data/captures/  {captures_count} labels across {len(captures_maps)} maps")
    for line in captures_maps:
        print(line)

    demos_mb = _dir_size_mb(DATA / "demos")
    processed_mb = _dir_size_mb(DATA / "processed")
    print(f"  data/demos/     {demos_mb:.0f} MB" if demos_mb else "  data/demos/     (not present)")
    print(f"  data/processed/ {processed_mb:.0f} MB" if processed_mb else "  data/processed/ (not present)")

    cache_mb = _dir_size_mb(HF_CACHE)
    if HF_CACHE.exists():
        print(f"  data/.hf_cache/ {cache_mb:.1f} MB")

    # Remote
    hub = config.get("hub", {})
    dataset_repo = hub.get("dataset_repo")
    if dataset_repo:
        print(f"\n=== Remote: {dataset_repo} ===")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(dataset_repo, repo_type="dataset")
            remote_labels = [f for f in files if f.startswith("labels/")]
            remote_screenshots = [f for f in files if f.startswith("screenshots/")]
            remote_demos = [f for f in files if f.startswith("demos/")]
            remote_processed = [f for f in files if f.startswith("processed/")]
            remote_captures = [f for f in files if f.startswith("captures/")]
            print(f"  labels/         {len(remote_labels)} files")
            print(f"  screenshots/    {len(remote_screenshots)} files")
            if remote_demos:
                print(f"  demos/          {len(remote_demos)} files")
            if remote_processed:
                print(f"  processed/      {len(remote_processed)} files")
            if remote_captures:
                print(f"  captures/       {len(remote_captures)} files")
        except Exception as e:
            print(f"  (could not query Hub: {e})")
    else:
        print("\n  No hub.dataset_repo configured — skipping remote status")


# ── pull ────────────────────────────────────────────────────────────────


def cmd_pull(args):
    from huggingface_hub import snapshot_download

    config = _load_config()
    repo_id = _get_repo(args, config, "dataset_repo")

    print(f"Pulling dataset from {repo_id}...")

    # Download into cache
    cache_dir = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(HF_CACHE),
    )
    cache_path = Path(cache_dir)

    # Copy labels → data/labeled/
    src_labels = cache_path / "labels"
    if src_labels.exists():
        LABELED.mkdir(parents=True, exist_ok=True)
        label_files = sorted(src_labels.glob("*.json"))
        if args.subset:
            label_files = label_files[: args.subset]
        copied_labels = 0
        for f in label_files:
            shutil.copy2(f, LABELED / f.name)
            copied_labels += 1
        print(f"  Copied {copied_labels} labels → data/labeled/")
    else:
        print("  No labels/ directory in remote dataset")

    # Copy screenshots → data/raw/
    src_screenshots = cache_path / "screenshots"
    if src_screenshots.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        img_patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        img_files = []
        for pat in img_patterns:
            img_files.extend(src_screenshots.glob(pat))
        img_files.sort(key=lambda f: f.name)
        if args.subset:
            img_files = img_files[: args.subset]
        copied_imgs = 0
        for f in img_files:
            shutil.copy2(f, RAW / f.name)
            copied_imgs += 1
        print(f"  Copied {copied_imgs} screenshots → data/raw/")
    else:
        print("  No screenshots/ directory in remote dataset")

    # Copy manifest if present
    src_manifest = cache_path / "manifest.jsonl"
    if src_manifest.exists():
        shutil.copy2(src_manifest, DATA / "manifest.jsonl")
        print("  Copied manifest.jsonl → data/")

    # --all: copy demos, processed, captures
    if args.all:
        for dirname in ("demos", "processed", "captures"):
            src = cache_path / dirname
            dst = DATA / dirname
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                size = _dir_size_mb(dst)
                print(f"  Copied {dirname}/ → data/{dirname}/ ({size:.0f} MB)")

    print("Done.")


# ── push ────────────────────────────────────────────────────────────────


def _assemble_captures():
    """Copy labels and screenshots from data/captures/*/ into data/labeled/ and data/raw/."""
    if not CAPTURES.exists():
        print("  No data/captures/ directory found")
        return 0, 0

    LABELED.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    total_labels = 0
    total_screenshots = 0

    for map_dir in sorted(CAPTURES.iterdir()):
        if not map_dir.is_dir():
            continue

        # Labels
        labels_src = map_dir / "labels"
        if labels_src.exists():
            for f in labels_src.glob("*.json"):
                shutil.copy2(f, LABELED / f.name)
                total_labels += 1

        # Screenshots
        screenshots_src = map_dir / "screenshots"
        if screenshots_src.exists():
            for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for f in screenshots_src.glob(pat):
                    shutil.copy2(f, RAW / f.name)
                    total_screenshots += 1

    print(f"  Assembled {total_labels} labels, {total_screenshots} screenshots from captures")
    return total_labels, total_screenshots


def _generate_dataset_card(repo_id, label_count, screenshot_count):
    """Generate a dataset card README."""
    return f"""---
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
  - {"n<1K" if label_count < 1000 else "1K<n<10K"}
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
manifest.jsonl  # Data provenance tracking
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
```

## Stats

- **Labels:** {label_count}
- **Screenshots:** {screenshot_count}

## Paper

*See, Then Think: Two-Phase VLM Training for Game Understanding*
"""


def cmd_push(args):
    from huggingface_hub import HfApi, create_repo

    config = _load_config()

    # Handle --model separately
    if args.model:
        repo_id = _get_repo(args, config, "model_repo")
        _push_model(args.model, repo_id)
        return

    repo_id = _get_repo(args, config, "dataset_repo")

    # Assemble from captures if requested
    if args.captures:
        _assemble_captures()

    # Check we have something to upload
    label_count = _count_files(LABELED, ("*.json",))
    screenshot_count = _count_files(RAW, ("*.png", "*.jpg", "*.jpeg", "*.webp"))

    if label_count == 0 and screenshot_count == 0:
        print("Error: No data to push. Run with --captures to assemble from data/captures/")
        sys.exit(1)

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"Pushing dataset to {repo_id}...")

    # Upload labels
    if label_count > 0:
        print(f"  Uploading {label_count} labels...")
        api.upload_folder(
            folder_path=str(LABELED),
            path_in_repo="labels",
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.json"],
        )

    # Upload screenshots
    if screenshot_count > 0:
        print(f"  Uploading {screenshot_count} screenshots...")
        api.upload_folder(
            folder_path=str(RAW),
            path_in_repo="screenshots",
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.png", "*.jpg", "*.jpeg", "*.webp"],
        )

    # Upload manifest if present
    manifest = DATA / "manifest.jsonl"
    if manifest.exists():
        print("  Uploading manifest.jsonl...")
        api.upload_file(
            path_or_fileobj=str(manifest),
            path_in_repo="manifest.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # --all: upload demos, processed, captures
    if args.all:
        for dirname, patterns in [
            ("demos", ["*.dem"]),
            ("processed", ["*.parquet", "*.json"]),
            ("captures", ["*.json", "*.png", "*.jpg", "*.jpeg", "*.webp"]),
        ]:
            src = DATA / dirname
            if src.exists():
                file_count = sum(len(list(src.rglob(p))) for p in patterns)
                if file_count > 0:
                    size = _dir_size_mb(src)
                    print(f"  Uploading {dirname}/ ({file_count} files, {size:.0f} MB)...")
                    api.upload_folder(
                        folder_path=str(src),
                        path_in_repo=dirname,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

    # Upload dataset card
    card = _generate_dataset_card(repo_id, label_count, screenshot_count)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")


def _push_model(model_path, repo_id):
    """Upload trained model weights to HF Hub."""
    from huggingface_hub import HfApi, create_repo

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    api = HfApi()
    create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    print(f"Pushing model to {repo_id}...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload training config if found nearby
    for config_path in [
        model_path / "training_config.json",
        model_path.parent / "training_config.json",
        model_path.parent.parent / "training_config.json",
    ]:
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="training_config.json",
                repo_id=repo_id,
                repo_type="model",
            )
            break

    # Model card
    card = f"""---
license: mit
library_name: transformers
tags:
  - gaming
  - cs2
  - vlm
  - game-ai
  - qwen
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: image-to-text
---

# Chimera — CS2 Game AI (Qwen3.5-35B-A3B)

Fine-tuned Qwen3.5-35B-A3B for Counter-Strike 2 screenshot analysis and strategic advice.

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```

## Paper

*See, Then Think: Two-Phase VLM Training for Game Understanding*
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model pushed: https://huggingface.co/{repo_id}")


# ── clean ───────────────────────────────────────────────────────────────


def cmd_clean(args):
    dirs_to_clean = [RAW, LABELED, HF_CACHE]

    if args.all:
        dirs_to_clean.extend([
            DATA / "demos",
            DATA / "processed",
            DATA / "captures",
            DATA / "predictions",
            ROOT / "outputs",
        ])

    total_freed = 0.0
    for d in dirs_to_clean:
        if d.exists():
            size = _dir_size_mb(d)
            shutil.rmtree(d)
            total_freed += size
            print(f"  Removed {d.relative_to(ROOT)}/ ({size:.1f} MB)")
        else:
            print(f"  {d.relative_to(ROOT)}/ (not present)")

    print(f"\nFreed {total_freed:.1f} MB")


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub data management for chimera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show local and remote data counts")

    # pull
    pull_p = sub.add_parser("pull", help="Download dataset from Hub")
    pull_p.add_argument("--repo", help="Override dataset repo ID")
    pull_p.add_argument("--subset", type=int, help="Pull only first N samples")
    pull_p.add_argument("--all", action="store_true", help="Also pull demos/, processed/, captures/")

    # push
    push_p = sub.add_parser("push", help="Upload dataset or model to Hub")
    push_p.add_argument("--repo", help="Override repo ID")
    push_p.add_argument(
        "--captures", action="store_true",
        help="Assemble data from data/captures/*/ before uploading",
    )
    push_p.add_argument(
        "--model", type=str, metavar="PATH",
        help="Upload model weights instead of dataset",
    )
    push_p.add_argument(
        "--all", action="store_true",
        help="Also upload demos/, processed/, captures/",
    )

    # clean
    clean_p = sub.add_parser("clean", help="Remove local data copies")
    clean_p.add_argument(
        "--all", action="store_true",
        help="Also remove processed/, predictions/, outputs/",
    )

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "pull": cmd_pull,
        "push": cmd_push,
        "clean": cmd_clean,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
