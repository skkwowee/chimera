#!/usr/bin/env python3
"""Backfill (demo_stem, round_num, tick, player_name) metadata for an existing
smoke_test.jsonl that was generated before extract_grpo_samples.py started
emitting a `source` block.

Why we need this without regenerating: training runs (f08v5 etc.) reference
samples by `sample_idx` into smoke_test.jsonl. Regenerating with the new
extract script could change ordering, breaking those audit references.
This script preserves the original ordering and emits a side-car:

  data/training/grpo/smoke_test_source.jsonl
    one line per sample with {idx, demo_stem, round_num, tick, player_name,
    player_side, map_name}

Recovery method: parse "Round N" + "POV: NAME (SIDE)" + "de_MAP" from the
prompt header, match player_health from ground_truth.game_state against the
demo parquet to pin the exact tick.

Usage:
  python scripts/recover_source_metadata.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --demos data/processed/demos \\
      --output data/training/grpo/smoke_test_source.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

POV_RE = re.compile(r"POV: ([^ ]+) \(([TC]T?)\)")
ROUND_RE = re.compile(r"Round (\d+)")
HP_RE = re.compile(r"POV: [^ ]+ \([TC]T?\), (\d+)hp")
MAP_RE = re.compile(r"(de_[a-z0-9]+)")


def parse_prompt(prompt_blocks: list[dict]) -> dict | None:
    """Pull (round_num, player_name, player_side, map_name, hp) from prompt text."""
    text = ""
    for b in prompt_blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            text = b.get("text", "")
            break
    if not text:
        return None
    pov = POV_RE.search(text)
    rnd = ROUND_RE.search(text)
    mp = MAP_RE.search(text)
    hp = HP_RE.search(text)
    if not (pov and rnd and mp and hp):
        return None
    side = pov.group(2)
    if side == "T":
        side = "t"
    elif side == "CT":
        side = "ct"
    return {
        "player_name": pov.group(1),
        "player_side": side,
        "round_num": int(rnd.group(1)),
        "map_name": mp.group(1),
        "player_health": int(hp.group(1)),
    }


def find_tick(parquet_path: Path, round_num: int, player_name: str, hp: int) -> int | None:
    """Scan the demo's parquet for a tick where this player has matching HP."""
    df = pl.scan_parquet(parquet_path)
    rows = (
        df.filter(
            (pl.col("round_num") == round_num)
            & (pl.col("name") == player_name)
            & (pl.col("health") == hp)
        )
        .select("tick")
        .collect()
    )
    if rows.height == 0:
        return None
    # If multiple ticks match (player held this HP for several ticks), take
    # the median — most likely the "active" moment that was sampled.
    ticks = sorted(rows["tick"].to_list())
    return ticks[len(ticks) // 2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--demos", required=True, help="Directory containing *_ticks.parquet")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    demos_dir = Path(args.demos)
    parquets = sorted(demos_dir.glob("*_ticks.parquet"))
    print(f"Found {len(parquets)} demo parquets")

    # Pre-compute (map_name, demo_stem) candidates from headers
    demo_maps: dict[str, list[str]] = {}
    for pq in parquets:
        stem = pq.name.replace("_ticks.parquet", "")
        header_path = demos_dir / f"{stem}_header.json"
        if header_path.exists():
            header = json.loads(header_path.read_text())
            mp = header.get("map_name", "unknown")
            demo_maps.setdefault(mp, []).append(stem)
    print(f"Demos by map: {dict((k, len(v)) for k, v in demo_maps.items())}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_resolved = n_ambiguous = n_failed = 0
    out_records: list[dict] = []

    with open(args.dataset) as f, out_path.open("w") as out_f:
        for idx, line in enumerate(f):
            n_total += 1
            sample = json.loads(line)
            parsed = parse_prompt(sample.get("prompt", []))
            gt_state = sample.get("ground_truth", {}).get("game_state", {})

            if parsed is None:
                n_failed += 1
                continue

            map_name = parsed["map_name"]
            candidates = demo_maps.get(map_name, [])
            if not candidates:
                n_failed += 1
                continue

            # Try each candidate demo on this map; pick the one that yields a tick
            best = None
            for stem in candidates:
                pq = demos_dir / f"{stem}_ticks.parquet"
                tick = find_tick(pq, parsed["round_num"], parsed["player_name"],
                                  parsed["player_health"])
                if tick is not None:
                    if best is None:
                        best = (stem, tick)
                    else:
                        # Multiple demos contain this player+round+HP -- ambiguous
                        n_ambiguous += 1
                        best = None
                        break

            if best is None:
                n_failed += 1
                continue

            stem, tick = best
            rec = {
                "idx": idx,
                "demo_stem": stem,
                "round_num": parsed["round_num"],
                "tick": tick,
                "player_name": parsed["player_name"],
                "player_side": parsed["player_side"],
                "map_name": map_name,
            }
            out_f.write(json.dumps(rec) + "\n")
            n_resolved += 1
            if n_resolved % 200 == 0:
                print(f"  resolved {n_resolved}/{n_total}...")

    print()
    print(f"Total samples:     {n_total}")
    print(f"Resolved:          {n_resolved} ({100 * n_resolved / n_total:.1f}%)")
    print(f"Ambiguous (skip):  {n_ambiguous}")
    print(f"Failed (no match): {n_failed}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
