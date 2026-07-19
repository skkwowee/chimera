#!/usr/bin/env python3
"""Batch-parse CS2 .dem files into the chimera parquet/JSON schema.

Walks data/demos/*.dem and data/demos_new/*.dem, runs awpy on each in parallel,
emits the same per-demo files the original 4 had:
  - {stem}_ticks.parquet   (per-tick player state — the encoder's input)
  - {stem}_kills.json
  - {stem}_bomb.json
  - {stem}_damages.json
  - {stem}_header.json
  - {stem}_rounds.json

Idempotent: skips any demo whose ticks.parquet already exists.

Usage:
    python scripts/parse_demos.py                          # parse everything new
    python scripts/parse_demos.py --workers 4              # cap parallelism (default: min(cores/2, 4))
    python scripts/parse_demos.py --force                  # re-parse even if outputs exist
    python scripts/parse_demos.py path/to/one.dem          # parse a single file

Per-demo cost on a 13900K: ~3-6 min, ~2-3 GB RAM. The pod parses faster but
this lets us iterate without depending on the pod being up.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMOS_DIRS = [REPO / "data" / "demos", REPO / "data" / "demos_new"]
OUT_DIR = REPO / "data" / "processed" / "demos"

# Player-state columns that match the existing parquet schema (the 17-col
# format the encoder design assumes). yaw/pitch are critical — the encoder
# uses view-angle as a feature.
PLAYER_PROPS = [
    "X", "Y", "Z",
    "health", "armor",
    "has_helmet", "has_defuser",
    "inventory",
    "current_equip_value", "balance",
    "yaw", "pitch",
]


def parse_one(dem_path: Path, force: bool = False) -> tuple[str, bool, str]:
    """Parse a single .dem; return (stem, success, message)."""
    stem = dem_path.stem
    ticks_out = OUT_DIR / f"{stem}_ticks.parquet"
    if ticks_out.exists() and not force:
        return (stem, True, "skip (exists)")

    try:
        from awpy import Demo
        t0 = time.time()
        d = Demo(dem_path, verbose=False)
        d.parse(player_props=PLAYER_PROPS)

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Per-tick parquet (the big one — 50+MB)
        d.ticks.write_parquet(ticks_out)

        # Per-event JSONs (small — used for round/event boundaries)
        for attr, fname in [
            ("kills", f"{stem}_kills.json"),
            ("bomb", f"{stem}_bomb.json"),
            ("damages", f"{stem}_damages.json"),
            ("rounds", f"{stem}_rounds.json"),
        ]:
            df = getattr(d, attr, None)
            if df is None:
                continue
            # awpy returns polars DataFrames; convert via to_dicts for JSON
            (OUT_DIR / fname).write_text(
                json.dumps(df.to_dicts(), default=str, indent=2)
            )

        # Header (dict)
        (OUT_DIR / f"{stem}_header.json").write_text(
            json.dumps(d.header, default=str, indent=2)
        )

        elapsed = time.time() - t0
        rows = d.ticks.height
        return (stem, True, f"OK ({rows:,} ticks, {elapsed:.0f}s)")
    except Exception as e:
        import traceback
        return (stem, False, f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")


def find_demos() -> list[Path]:
    out: list[Path] = []
    for d in DEMOS_DIRS:
        if d.exists():
            out.extend(sorted(d.glob("*.dem")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", type=Path, help="specific .dem files (default: all)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: min(cores/4, 4) — RAM-bounded)")
    ap.add_argument("--force", action="store_true", help="re-parse even if outputs exist")
    args = ap.parse_args()

    demos = args.paths or find_demos()
    if not demos:
        print("No .dem files found. Drop them in data/demos/ or data/demos_new/.")
        sys.exit(1)

    workers = args.workers or max(1, min(mp.cpu_count() // 4, 4))
    print(f"Parsing {len(demos)} demos with {workers} workers")
    print(f"Output: {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_fail = n_skip = 0

    if workers == 1:
        # Serial path — easier to debug
        for p in demos:
            stem, ok, msg = parse_one(p, force=args.force)
            tag = "✓" if ok else "✗"
            print(f"  {tag} {stem}: {msg}")
            if "skip" in msg: n_skip += 1
            elif ok: n_ok += 1
            else: n_fail += 1
    else:
        # Parallel — order is whatever finishes first
        with mp.Pool(workers) as pool:
            results = [pool.apply_async(parse_one, (p, args.force)) for p in demos]
            for r in results:
                stem, ok, msg = r.get()
                tag = "✓" if ok else "✗"
                print(f"  {tag} {stem}: {msg}", flush=True)
                if "skip" in msg: n_skip += 1
                elif ok: n_ok += 1
                else: n_fail += 1

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s — {n_ok} parsed, {n_skip} skipped, {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
