#!/usr/bin/env python3
"""Chimera pipeline orchestrator and status dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = PROJECT_ROOT / "data" / "processed" / "demos"
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.json"


def _get_demo_stems() -> list[str]:
    stems: list[str] = []
    for p in sorted(DEMOS_DIR.glob("*_ticks.parquet")):
        stems.append(p.name.replace("_ticks.parquet", ""))
    return stems


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    stems = _get_demo_stems()
    n_demos = len(stems)

    # Rounds
    total_rounds = 0
    bomb_rounds = 0
    total_ticks = 0
    map_counts: dict[str, int] = {}

    for stem in stems:
        rounds_path = DEMOS_DIR / f"{stem}_rounds.json"
        if rounds_path.exists():
            with open(rounds_path) as f:
                rounds = json.load(f)
            total_rounds += len(rounds)
            bomb_rounds += sum(1 for r in rounds if r.get("bomb_plant") is not None)

        parquet = DEMOS_DIR / f"{stem}_ticks.parquet"
        if parquet.exists():
            df = pl.scan_parquet(parquet)
            total_ticks += df.select(pl.len()).collect().item()

        header_path = DEMOS_DIR / f"{stem}_header.json"
        if header_path.exists():
            with open(header_path) as f:
                header = json.load(f)
            map_name = header.get("map_name", "unknown")
            # Friendly name: de_mirage -> Mirage
            friendly = map_name.replace("de_", "").capitalize()
            map_counts[friendly] = map_counts.get(friendly, 0) + 1

    # Captures
    capture_dirs = sorted(d for d in CAPTURES_DIR.iterdir() if d.is_dir()) if CAPTURES_DIR.exists() else []
    maps_with_captures = 0
    total_screenshots = 0
    total_labels = 0
    for cdir in capture_dirs:
        raw_dir = cdir / "raw"
        labels_dir = cdir / "labels"
        n_ss = sum(1 for _ in raw_dir.glob("*.jpg")) if raw_dir.exists() else 0
        n_ss += sum(1 for _ in raw_dir.glob("*.png")) if raw_dir.exists() else 0
        n_lb = sum(1 for _ in labels_dir.glob("*.json")) if labels_dir.exists() else 0
        if n_ss > 0 or n_lb > 0:
            maps_with_captures += 1
        total_screenshots += n_ss
        total_labels += n_lb

    # Training
    training_status = "not started"
    if TRAINING_DIR.exists():
        grpo_dir = TRAINING_DIR / "grpo"
        if grpo_dir.exists():
            jsonl_files = list(grpo_dir.glob("*.jsonl"))
            if jsonl_files:
                n_samples = sum(
                    sum(1 for line in f.open() if line.strip())
                    for f in jsonl_files
                )
                training_status = f"{len(jsonl_files)} file(s), {n_samples} samples"
            else:
                training_status = "grpo/ exists but no .jsonl files"
        else:
            training_status = "directory exists but no grpo/ data"

    # Models
    model_status = "none"
    if OUTPUTS_DIR.exists():
        checkpoints = list(OUTPUTS_DIR.rglob("checkpoint-*"))
        if checkpoints:
            model_status = f"{len(checkpoints)} checkpoint(s)"
        else:
            subdirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()]
            if subdirs:
                model_status = f"{len(subdirs)} output dir(s)"

    # Smoke manifest
    manifest_path = PROJECT_ROOT / "data" / "smoke_manifest.json"
    manifest_info = ""
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if isinstance(manifest, list):
            manifest_info = f"  Samples:    {len(manifest)} (from smoke_manifest.json)"
        elif isinstance(manifest, dict) and "samples" in manifest:
            manifest_info = f"  Samples:    {len(manifest['samples'])} (from smoke_manifest.json)"

    # Print dashboard
    print("=== Chimera Pipeline Status ===\n")
    print(f"Demos:        {n_demos} parsed")
    print(f"Rounds:       {total_rounds} total ({bomb_rounds} with bomb plants)")
    print(f"Captures:     {maps_with_captures}/{n_demos} maps "
          f"({total_screenshots} screenshots, {total_labels} labels)")
    print(f"Training:     {training_status}")
    print(f"Models:       {model_status}")
    print()
    print("Data volume:")
    print(f"  Ticks:      ~{total_ticks:,} (across {n_demos} Parquet files)")
    if manifest_info:
        print(manifest_info)
    print()
    map_str = "  ".join(f"{k}={v}" for k, v in sorted(map_counts.items()))
    print(f"Maps: {map_str}")


# ---------------------------------------------------------------------------
# registry subcommand
# ---------------------------------------------------------------------------

def cmd_registry() -> None:
    stems = _get_demo_stems()
    demos: list[dict] = []

    for stem in stems:
        entry: dict = {"stem": stem}

        # Header / map
        header_path = DEMOS_DIR / f"{stem}_header.json"
        if header_path.exists():
            with open(header_path) as f:
                header = json.load(f)
            entry["map"] = header.get("map_name", "unknown")
        else:
            entry["map"] = "unknown"

        # Rounds
        rounds_path = DEMOS_DIR / f"{stem}_rounds.json"
        if rounds_path.exists():
            with open(rounds_path) as f:
                rounds = json.load(f)
            entry["rounds"] = len(rounds)
        else:
            entry["rounds"] = 0

        # File existence checks
        entry["has_parquet"] = (DEMOS_DIR / f"{stem}_ticks.parquet").exists()
        entry["has_rounds"] = rounds_path.exists()
        entry["has_bomb"] = (DEMOS_DIR / f"{stem}_bomb.json").exists()
        entry["has_kills"] = (DEMOS_DIR / f"{stem}_kills.json").exists()
        entry["has_damages"] = (DEMOS_DIR / f"{stem}_damages.json").exists()
        entry["has_header"] = header_path.exists()

        # Captures
        capture_dir = CAPTURES_DIR / stem
        labels_dir = capture_dir / "labels"
        raw_dir = capture_dir / "raw"

        has_captures = False
        has_labels = False
        label_count = 0
        screenshot_count = 0

        if labels_dir.exists():
            label_count = sum(1 for _ in labels_dir.glob("*.json"))
            has_labels = label_count > 0

        if raw_dir.exists():
            screenshot_count = sum(1 for _ in raw_dir.glob("*.jpg"))
            screenshot_count += sum(1 for _ in raw_dir.glob("*.png"))
            has_captures = screenshot_count > 0

        entry["has_captures"] = has_captures
        entry["has_labels"] = has_labels
        entry["label_count"] = label_count
        entry["screenshot_count"] = screenshot_count

        demos.append(entry)

    registry = {
        "demos": demos,
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Registry written to {REGISTRY_PATH}")
    print(f"  {len(demos)} demos registered")
    for d in demos:
        caps = ""
        if d["has_labels"] or d["has_captures"]:
            caps = f" | {d['screenshot_count']} screenshots, {d['label_count']} labels"
        print(f"  - {d['stem']} ({d['map']}, {d['rounds']} rounds){caps}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Chimera pipeline orchestrator")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Show pipeline health dashboard")
    sub.add_parser("registry", help="Generate/update data/registry.json")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "registry":
        cmd_registry()
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
