#!/usr/bin/env python3
"""
Export downsampled viewer data from parsed parquet files.

Reads *_ticks.parquet + metadata JSONs produced by parse_demos.py and exports
compact per-round JSON files for the React viewer, plus radar map images
and coordinate metadata.

Output:
    site/public/viewer-data/index.json
    site/public/viewer-data/{demo}/meta.json
    site/public/viewer-data/{demo}/round_01.json ...
    site/public/maps/*.png
    site/public/maps/map-data.json

Usage:
    python scripts/export_viewer_data.py
    python scripts/export_viewer_data.py --input data/processed/demos
    python scripts/export_viewer_data.py --skip-every 32   # coarser downsampling
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl

# awpy map data
try:
    from awpy.data.map_data import MAP_DATA
    from awpy.data import MAPS_DIR
except ImportError:
    print("awpy is required: pip install awpy>=2.0.0")
    sys.exit(1)


def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text())


def export_demo(
    stem: str,
    input_dir: Path,
    output_dir: Path,
    skip_every: int,
) -> dict | None:
    """Export one demo's viewer data. Returns index entry or None on error."""
    parquet_path = input_dir / f"{stem}_ticks.parquet"
    header_path = input_dir / f"{stem}_header.json"
    rounds_path = input_dir / f"{stem}_rounds.json"
    kills_path = input_dir / f"{stem}_kills.json"
    bomb_path = input_dir / f"{stem}_bomb.json"

    if not parquet_path.exists():
        print(f"  Skipping {stem}: no parquet file")
        return None

    print(f"  Processing {stem}...")

    # Load data
    ticks_df = pl.read_parquet(parquet_path)
    header = load_json(header_path) if header_path.exists() else {}
    rounds_data = load_json(rounds_path) if rounds_path.exists() else []
    kills_data = load_json(kills_path) if kills_path.exists() else []
    bomb_data = load_json(bomb_path) if bomb_path.exists() else []

    map_name = header.get("map_name", "unknown")

    # Create output dir for this demo
    demo_dir = output_dir / stem
    demo_dir.mkdir(parents=True, exist_ok=True)

    # --- meta.json ---
    meta = {
        "stem": stem,
        "map_name": map_name,
        "header": header,
        "rounds": rounds_data,
        "kills": kills_data,
        "bomb": bomb_data,
    }
    (demo_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    # --- Per-round JSON files ---
    round_count = 0
    for round_info in rounds_data:
        round_num = round_info["round_num"]
        start_tick = round_info.get("start", 0)
        end_tick = round_info.get("end", 0)
        freeze_end = round_info.get("freeze_end", start_tick)
        winner = round_info.get("winner", "unknown")
        reason = round_info.get("reason", "unknown")
        bomb_plant_tick = round_info.get("bomb_plant")
        bomb_site = round_info.get("bomb_site", "not_planted")

        # Filter ticks for this round and downsample
        round_ticks = ticks_df.filter(pl.col("round_num") == round_num)
        if round_ticks.is_empty():
            continue

        # Get unique ticks, sorted
        unique_ticks = round_ticks.select("tick").unique().sort("tick")
        all_tick_vals = unique_ticks["tick"].to_list()

        # Downsample: take every Nth tick
        sampled_ticks = all_tick_vals[::skip_every]
        # Always include first and last tick
        if all_tick_vals and all_tick_vals[-1] not in sampled_ticks:
            sampled_ticks.append(all_tick_vals[-1])

        # Build frames
        frames = []
        sampled_set = set(sampled_ticks)
        sampled_df = round_ticks.filter(pl.col("tick").is_in(sampled_ticks))

        for tick_val, group in sampled_df.group_by("tick", maintain_order=True):
            tick_int = int(tick_val[0]) if isinstance(tick_val, tuple) else int(tick_val)
            players = []
            for row in group.to_dicts():
                side = str(row.get("side", "")).upper()
                if side not in ("T", "CT"):
                    continue
                hp = row.get("health", 0) or 0
                players.append({
                    "name": row.get("name", ""),
                    "side": side,
                    "x": round(float(row.get("X", 0) or 0), 1),
                    "y": round(float(row.get("Y", 0) or 0), 1),
                    "z": round(float(row.get("Z", 0) or 0), 1),
                    "hp": int(hp),
                    "alive": hp > 0,
                })
            frames.append({"tick": tick_int, "players": players})

        # Sort frames by tick
        frames.sort(key=lambda f: f["tick"])

        round_json = {
            "round_num": int(round_num),
            "start_tick": int(start_tick),
            "end_tick": int(end_tick),
            "freeze_end": int(freeze_end),
            "winner": winner,
            "reason": reason,
            "bomb_plant_tick": int(bomb_plant_tick) if bomb_plant_tick else None,
            "bomb_site": bomb_site if bomb_site != "not_planted" else None,
            "frames": frames,
        }

        round_file = demo_dir / f"round_{round_num:02d}.json"
        round_file.write_text(json.dumps(round_json, ensure_ascii=False))
        round_count += 1

    print(f"    {round_count} rounds, {len(frames) if rounds_data else 0} frames in last round")
    print(f"    Map: {map_name}")

    return {
        "stem": stem,
        "map_name": map_name,
        "rounds": len(rounds_data),
    }


def copy_maps(maps_output_dir: Path) -> None:
    """Copy radar images and map-data.json to site/public/maps/."""
    maps_output_dir.mkdir(parents=True, exist_ok=True)

    if not MAPS_DIR.exists():
        print("  WARNING: awpy maps not downloaded. Run: awpy get maps")
        return

    # Copy map-data.json
    map_data_src = MAPS_DIR / "map-data.json"
    if map_data_src.exists():
        shutil.copy2(map_data_src, maps_output_dir / "map-data.json")
        print(f"  Copied map-data.json")

    # Copy only the maps we need (from our demos) + their lower variants
    needed_maps = set()
    for f in maps_output_dir.parent.glob("viewer-data/*/meta.json"):
        meta = load_json(f)
        m = meta.get("map_name", "")
        if m:
            needed_maps.add(m)

    if not needed_maps:
        # Fallback: copy all competitive maps
        needed_maps = {"de_mirage", "de_inferno", "de_nuke", "de_overpass",
                       "de_dust2", "de_ancient", "de_anubis", "de_vertigo", "de_train"}

    copied = 0
    for map_name in sorted(needed_maps):
        for suffix in ["", "_lower"]:
            src = MAPS_DIR / f"{map_name}{suffix}.png"
            if src.exists():
                shutil.copy2(src, maps_output_dir / f"{map_name}{suffix}.png")
                copied += 1

    print(f"  Copied {copied} radar images for {len(needed_maps)} maps")


def main():
    parser = argparse.ArgumentParser(
        description="Export downsampled viewer data from parsed parquet files"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/demos",
        help="Input directory with parquet + JSON files (default: data/processed/demos)",
    )
    parser.add_argument(
        "--output", "-o",
        default="site/public/viewer-data",
        help="Output directory for viewer JSON files (default: site/public/viewer-data)",
    )
    parser.add_argument(
        "--maps-output",
        default="site/public/maps",
        help="Output directory for radar map images (default: site/public/maps)",
    )
    parser.add_argument(
        "--skip-every",
        type=int,
        default=16,
        help="Downsample factor: keep every Nth tick (default: 16, ~4 fps at 64-tick)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    maps_dir = Path(args.maps_output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find all demo stems (from parquet files)
    parquet_files = sorted(input_dir.glob("*_ticks.parquet"))
    if not parquet_files:
        print(f"No *_ticks.parquet files found in {input_dir}")
        sys.exit(1)

    stems = [p.stem.replace("_ticks", "") for p in parquet_files]
    print(f"Found {len(stems)} demo(s) to export")

    # Export each demo
    index_entries = []
    for stem in stems:
        entry = export_demo(stem, input_dir, output_dir, args.skip_every)
        if entry:
            index_entries.append(entry)

    # Write index.json
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index_entries, indent=2, ensure_ascii=False))
    print(f"\nWrote {index_path} ({len(index_entries)} demos)")

    # Copy maps
    print("\nCopying radar maps...")
    copy_maps(maps_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
