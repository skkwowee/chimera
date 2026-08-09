#!/usr/bin/env python3
"""Measure P2's tensor-only dropped-C4 inference against local bomb events.

This is a reporting instrument, not a gate: it emits event precision/recall,
tick deltas, and xy error for P2 rounds whose local parquet source key was
verified. No threshold or model/feature decision is encoded here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch

N_PLAYERS = 10
HAS_C4 = 14
G_BOMB_STATE = 15


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return sorted(values)[round(q * (len(values) - 1))]


def _detected_drops(tensor: torch.Tensor, ticks: list[int], ppd: int) -> list[dict]:
    global_start = N_PLAYERS * ppd
    c4 = torch.stack(
        [tensor[:, player * ppd + HAS_C4] for player in range(N_PLAYERS)], dim=1
    ) > 0.5
    any_c4 = c4.any(dim=1)
    bomb_state = tensor[:, global_start + G_BOMB_STATE:global_start + G_BOMB_STATE + 4] > 0.5
    planted = bomb_state[:, 2] | bomb_state[:, 3]
    detections = []
    for frame in range(1, len(ticks)):
        if not bool(any_c4[frame - 1] and not any_c4[frame] and not planted[frame]):
            continue
        slots = c4[frame - 1].nonzero(as_tuple=False).flatten()
        if not slots.numel():
            continue
        carrier = int(slots[0])
        detections.append({
            "tick": int(ticks[frame]),
            "x": float(tensor[frame - 1, carrier * ppd]) * 3000.0,
            "y": float(tensor[frame - 1, carrier * ppd + 1]) * 3000.0,
        })
    return detections


def evaluate(p1: dict, p2: dict, demos_dir: Path, tolerance_ticks: int) -> dict:
    verified_by_stem: dict[str, list[int]] = defaultdict(list)
    for index, meta in enumerate(p2["metas"]):
        if meta.get("p2_sidecar_status") == "verified_local_parquet":
            verified_by_stem[str(meta["demo_stem"])].append(index)

    true_positive = false_positive = false_negative = 0
    tick_deltas: list[float] = []
    xy_errors: list[float] = []
    per_stem = {}
    ppd = int(p1.get("per_player_dim", 56))
    for stem, indices in sorted(verified_by_stem.items()):
        parquet = demos_dir / f"{stem}_ticks.parquet"
        bomb_path = demos_dir / f"{stem}_bomb.json"
        if not parquet.exists() or not bomb_path.exists():
            raise FileNotFoundError(f"verified source files disappeared for {stem}")
        ticks_df = pl.read_parquet(parquet, columns=["tick", "round_num"])
        bomb_events = json.loads(bomb_path.read_text())
        stem_counts = {"rounds": len(indices), "tp": 0, "fp": 0, "fn": 0}
        for index in indices:
            meta = p1["metas"][index]
            round_num = int(meta["round_num"])
            round_ticks = sorted(
                ticks_df.filter(pl.col("round_num") == round_num)["tick"].unique().to_list()
            )[:: int(meta.get("downsample", 8))]
            detections = _detected_drops(p1["tensors"][index], round_ticks, ppd)
            actual = [
                event for event in bomb_events
                if int(event.get("round_num", -1)) == round_num
                and event.get("event") == "drop"
            ]
            used: set[int] = set()
            for detection in detections:
                candidates = [
                    (abs(detection["tick"] - int(event["tick"])), actual_index, event)
                    for actual_index, event in enumerate(actual)
                    if actual_index not in used
                ]
                if not candidates or min(candidates)[0] > tolerance_ticks:
                    false_positive += 1
                    stem_counts["fp"] += 1
                    continue
                delta_abs, actual_index, event = min(candidates)
                del delta_abs
                used.add(actual_index)
                true_positive += 1
                stem_counts["tp"] += 1
                tick_deltas.append(detection["tick"] - int(event["tick"]))
                xy_errors.append(math.hypot(
                    detection["x"] - float(event["X"]),
                    detection["y"] - float(event["Y"]),
                ))
            missed = len(actual) - len(used)
            false_negative += missed
            stem_counts["fn"] += missed
        per_stem[stem] = stem_counts

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    return {
        "source": {
            "verified_stems": len(verified_by_stem),
            "verified_rounds": sum(len(indices) for indices in verified_by_stem.values()),
            "tolerance_raw_ticks": tolerance_ticks,
        },
        "events": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        },
        "tick_delta_raw": {
            "min": min(tick_deltas) if tick_deltas else None,
            "median": statistics.median(tick_deltas) if tick_deltas else None,
            "max": max(tick_deltas) if tick_deltas else None,
        },
        "xy_error_game_units": {
            "n": len(xy_errors),
            "median": statistics.median(xy_errors) if xy_errors else None,
            "p90": _percentile(xy_errors, 0.90),
            "p95": _percentile(xy_errors, 0.95),
            "max": max(xy_errors) if xy_errors else None,
        },
        "per_stem": per_stem,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=Path, default=Path("data/processed/tick_sequences/val_v2m_p1.pt"))
    parser.add_argument("--p2", type=Path, default=Path("data/processed/tick_sequences/val_v2m_p2.pt"))
    parser.add_argument("--demos-dir", type=Path, default=Path("data/processed/demos"))
    parser.add_argument("--tolerance-ticks", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    p1 = torch.load(args.p1, map_location="cpu", weights_only=False, mmap=True)
    p2 = torch.load(args.p2, map_location="cpu", weights_only=False, mmap=True)
    report = evaluate(p1, p2, args.demos_dir, args.tolerance_ticks)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
