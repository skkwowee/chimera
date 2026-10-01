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

Also preserves smoke/fire lifetimes, shots and footsteps as Parquet tables.
Idempotent: skips only source/version-matched bundles with verified file hashes.

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
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import sys
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMOS_DIRS = [REPO / "data" / "demos", REPO / "data" / "demos_new"]
OUT_DIR = REPO / "data" / "processed" / "demos"

# Archival state is richer than the unchanged model feature projection.
PLAYER_PROPS = [
    "X", "Y", "Z",
    "health", "armor",
    "has_helmet", "has_defuser",
    "inventory",
    "current_equip_value", "balance",
    "yaw", "pitch",
    "flash_duration", "active_weapon_name", "active_weapon_ammo",
    "total_ammo_left", "is_in_reload", "zoom_lvl", "duck_amount",
]

PARSE_VERSION = 2
EVENT_TABLES = ("kills", "bomb", "damages", "rounds")
EXTRA_TABLES = ("smokes", "infernos", "shots", "footsteps")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_complete(marker: Path, identity: dict) -> bool:
    """Only a matching, intact bundle is resumable; legacy ticks alone are not."""
    try:
        record = json.loads(marker.read_text())
        stem = marker.name.removesuffix("_parse.json")
        expected = {f"{stem}_{s}.json" for s in (*EVENT_TABLES, "header")}
        expected |= {f"{stem}_{s}.parquet" for s in ("ticks", *EXTRA_TABLES)}
        return (record["identity"] == identity and set(record["files"]) == expected
                and all(Path(name).name == name
                        and sha256(marker.parent / name) == digest
                        for name, digest in record["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def parse_one(dem_path: Path, force: bool = False) -> tuple[str, bool, str]:
    """Parse a single .dem; return (stem, success, message)."""
    stem = dem_path.stem
    try:
        with dem_path.open("rb") as stream:
            if stream.read(8) != b"PBDEMS2\x00":
                raise ValueError("Not a Source 2 demo")
        identity = {
            "version": PARSE_VERSION,
            "source_sha256": sha256(dem_path),
            "script_sha256": sha256(Path(__file__)),
            "packages": {p: importlib.metadata.version(p)
                         for p in ("awpy", "demoparser2")},
        }
        marker = OUT_DIR / f"{stem}_parse.json"
        if not force and parse_complete(marker, identity):
            return (stem, True, "skip (verified bundle)")
        from awpy import Demo
        t0 = time.time()
        d = Demo(dem_path, verbose=False)
        d.parse(player_props=PLAYER_PROPS)
        missing = set(PLAYER_PROPS) - set(d.ticks.columns)
        if missing:
            raise ValueError(f"Missing requested player properties: {sorted(missing)}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Stage the complete bundle before publishing. The marker is written last.
        with tempfile.TemporaryDirectory(dir=OUT_DIR) as tmp:
            stage = Path(tmp)
            for attr in ("ticks", *EXTRA_TABLES):
                getattr(d, attr).write_parquet(stage / f"{stem}_{attr}.parquet")
            for attr in EVENT_TABLES:
                (stage / f"{stem}_{attr}.json").write_text(
                    json.dumps(getattr(d, attr).to_dicts(), default=str))
            (stage / f"{stem}_header.json").write_text(json.dumps(d.header))
            record = {
                "identity": identity,
                "source_name": dem_path.name,
                "tickrate": d.tickrate,
                "player_props": PLAYER_PROPS,
                "lifetime_fallback_seconds": {
                    "smokes": d.smoke_duration, "infernos": d.inferno_duration},
                "files": {p.name: sha256(p) for p in stage.iterdir()},
            }
            marker.unlink(missing_ok=True)
            for p in stage.iterdir():
                p.replace(OUT_DIR / p.name)
            staged_marker = stage / marker.name
            staged_marker.write_text(json.dumps(record, indent=2))
            staged_marker.replace(marker)

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
