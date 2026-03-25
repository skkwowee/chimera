#!/usr/bin/env python3
"""Chimera data quality validation — 5 gates for pipeline health."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = PROJECT_ROOT / "data" / "processed" / "demos"
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"

VALID_MAPS = {
    "de_mirage", "de_inferno", "de_nuke", "de_overpass",
    "de_dust2", "de_vertigo", "de_ancient", "de_anubis",
    "de_train",
}

CRITICAL_PARQUET_COLS = {"health", "side", "name", "tick"}


class Result:
    """A single validation result."""

    def __init__(self, level: str, gate: str, msg: str):
        self.level = level  # PASS / WARN / FAIL
        self.gate = gate
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.level:<5} {self.gate:<10} {self.msg}"


def _get_demo_stems() -> list[str]:
    """Find unique demo stems from parquet files."""
    stems: list[str] = []
    for p in sorted(DEMOS_DIR.glob("*_ticks.parquet")):
        stems.append(p.name.replace("_ticks.parquet", ""))
    return stems


# ---------------------------------------------------------------------------
# Gate 1: Demo parse validation
# ---------------------------------------------------------------------------

def gate_parse(verbose: bool) -> list[Result]:
    results: list[Result] = []
    stems = _get_demo_stems()
    if not stems:
        results.append(Result("WARN", "parse", "No parquet files found in demos/"))
        return results

    for stem in stems:
        parquet = DEMOS_DIR / f"{stem}_ticks.parquet"
        rounds_path = DEMOS_DIR / f"{stem}_rounds.json"
        header_path = DEMOS_DIR / f"{stem}_header.json"
        failed = False

        # Parquet exists and has rows
        try:
            df = pl.read_parquet(parquet)
            n_ticks = len(df)
            if n_ticks == 0:
                results.append(Result("FAIL", "parse", f"{stem} (parquet has 0 rows)"))
                failed = True
                continue
        except Exception as e:
            results.append(Result("FAIL", "parse", f"{stem} (parquet read error: {e})"))
            failed = True
            continue

        # NaN check on critical columns
        for col in CRITICAL_PARQUET_COLS:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    results.append(Result("FAIL", "parse",
                                          f"{stem} ({null_count} NaN in column '{col}')"))
                    failed = True

        # Rounds JSON
        if not rounds_path.exists():
            results.append(Result("FAIL", "parse", f"{stem} (missing _rounds.json)"))
            failed = True
        else:
            with open(rounds_path) as f:
                rounds = json.load(f)
            n_rounds = len(rounds)
            if n_rounds < 10:
                results.append(Result("FAIL", "parse",
                                      f"{stem} (only {n_rounds} rounds, need >= 10)"))
                failed = True

            # freeze_end < end check
            for r in rounds:
                rn = r.get("round_num", "?")
                fe = r.get("freeze_end")
                end = r.get("end")
                if fe is not None and end is not None and fe >= end:
                    results.append(Result("FAIL", "parse",
                                          f"{stem} r{rn} (freeze_end={fe} >= end={end})"))
                    failed = True

        # Header JSON
        if not header_path.exists():
            results.append(Result("FAIL", "parse", f"{stem} (missing _header.json)"))
            failed = True
        else:
            with open(header_path) as f:
                header = json.load(f)
            map_name = header.get("map_name", "")
            if map_name not in VALID_MAPS:
                results.append(Result("WARN", "parse",
                                      f"{stem} (unrecognized map: {map_name})"))

        if not failed:
            if rounds_path.exists():
                with open(rounds_path) as rf:
                    n_rounds = len(json.load(rf))
            else:
                n_rounds = 0
            results.append(Result("PASS", "parse",
                                  f"{stem} ({n_rounds} rounds, {n_ticks} ticks)"))

    return results


# ---------------------------------------------------------------------------
# Gate 2: Label sanity
# ---------------------------------------------------------------------------

def gate_labels(verbose: bool) -> list[Result]:
    results: list[Result] = []
    label_dirs = sorted(CAPTURES_DIR.glob("*/labels"))
    if not label_dirs:
        results.append(Result("WARN", "labels", "No label directories found"))
        return results

    valid_sides = {"T", "CT"}
    valid_phases = {"freezetime", "playing", "post-plant"}
    n_pass = 0
    n_fail = 0

    for ldir in label_dirs:
        map_stem = ldir.parent.name
        for lf in sorted(ldir.glob("*.json")):
            with open(lf) as f:
                label = json.load(f)
            gs = label.get("game_state", {})
            errors: list[str] = []

            hp = gs.get("player_health")
            if hp is not None and not (0 <= hp <= 100):
                errors.append(f"player_health={hp}")

            at = gs.get("alive_teammates")
            if at is not None and not (0 <= at <= 4):
                errors.append(f"alive_teammates={at}")

            ae = gs.get("alive_enemies")
            if ae is not None and not (0 <= ae <= 5):
                errors.append(f"alive_enemies={ae}")

            side = gs.get("player_side")
            if side is not None and side not in valid_sides:
                errors.append(f"player_side={side}")

            phase = gs.get("round_phase")
            if phase is not None and phase not in valid_phases:
                errors.append(f"round_phase={phase}")

            if errors:
                rn = gs.get("round_num", "?")
                results.append(Result("FAIL", "labels",
                                      f"{map_stem} r{rn} ({', '.join(errors)}) [{lf.name}]"))
                n_fail += 1
            else:
                n_pass += 1

    if n_fail == 0:
        results.append(Result("PASS", "labels", f"All {n_pass} labels valid"))
    elif verbose:
        pass  # individual failures already appended
    else:
        results.append(Result("FAIL", "labels", f"{n_fail}/{n_pass + n_fail} labels have errors"))

    return results


# ---------------------------------------------------------------------------
# Gate 3: Training data validation
# ---------------------------------------------------------------------------

def gate_training(verbose: bool) -> list[Result]:
    results: list[Result] = []
    grpo_dir = TRAINING_DIR / "grpo"
    if not grpo_dir.exists():
        results.append(Result("WARN", "training", "No training/grpo/ directory"))
        return results

    jsonl_files = sorted(grpo_dir.glob("*.jsonl"))
    if not jsonl_files:
        results.append(Result("WARN", "training", "No .jsonl files in training/grpo/"))
        return results

    for jf in jsonl_files:
        seen_keys: set[tuple] = set()
        n_samples = 0
        errors: list[str] = []

        for i, line in enumerate(jf.open(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON ({e})")
                continue

            n_samples += 1

            if "prompt" not in sample:
                errors.append(f"line {i}: missing 'prompt'")
            if "ground_truth" not in sample:
                errors.append(f"line {i}: missing 'ground_truth'")
                continue

            gt = sample["ground_truth"]
            for key in ("game_state", "round_won", "player_contribution"):
                if key not in gt:
                    errors.append(f"line {i}: ground_truth missing '{key}'")

            # Duplicate check
            meta = sample.get("metadata", {})
            dup_key = (
                meta.get("demo"),
                meta.get("round"),
                meta.get("tick"),
                meta.get("player"),
            )
            if any(v is not None for v in dup_key):
                if dup_key in seen_keys:
                    errors.append(f"line {i}: duplicate (demo,round,tick,player)")
                seen_keys.add(dup_key)

        if errors:
            for e in errors[:10]:
                results.append(Result("FAIL", "training", f"{jf.name}: {e}"))
            if len(errors) > 10:
                results.append(Result("FAIL", "training",
                                      f"{jf.name}: ... and {len(errors) - 10} more errors"))
        else:
            results.append(Result("PASS", "training", f"{jf.name} ({n_samples} samples)"))

    return results


# ---------------------------------------------------------------------------
# Gate 4: Screenshot-label sync
# ---------------------------------------------------------------------------

def gate_sync(verbose: bool) -> list[Result]:
    results: list[Result] = []
    map_dirs = sorted(d for d in CAPTURES_DIR.iterdir() if d.is_dir())
    if not map_dirs:
        results.append(Result("WARN", "sync", "No capture directories found"))
        return results

    for mdir in map_dirs:
        labels_dir = mdir / "labels"
        raw_dir = mdir / "raw"

        label_stems = {p.stem for p in labels_dir.glob("*.json")} if labels_dir.exists() else set()
        screenshot_stems = set()
        for ext in ("*.jpg", "*.png"):
            if raw_dir.exists():
                screenshot_stems |= {p.stem for p in raw_dir.glob(ext)}

        n_labels = len(label_stems)
        n_screenshots = len(screenshot_stems)

        orphan_labels = label_stems - screenshot_stems
        orphan_screenshots = screenshot_stems - label_stems

        if n_screenshots == 0 and n_labels == 0:
            results.append(Result("WARN", "sync", f"{mdir.name} (no captures yet)"))
        elif orphan_labels or orphan_screenshots:
            msgs: list[str] = []
            if orphan_labels:
                msgs.append(f"{len(orphan_labels)} labels without screenshots")
            if orphan_screenshots:
                msgs.append(f"{len(orphan_screenshots)} screenshots without labels")
            level = "WARN"
            results.append(Result(level, "sync",
                                  f"{mdir.name} ({n_screenshots} screenshots, "
                                  f"{n_labels} labels; {'; '.join(msgs)})"))
        else:
            results.append(Result("PASS", "sync",
                                  f"{mdir.name} ({n_screenshots} screenshots, {n_labels} labels, fully synced)"))

    return results


# ---------------------------------------------------------------------------
# Gate 5: Sparsity check
# ---------------------------------------------------------------------------

def gate_sparsity(verbose: bool) -> list[Result]:
    results: list[Result] = []

    # Check training JSONL files
    grpo_dir = TRAINING_DIR / "grpo"
    if not grpo_dir.exists():
        # Fallback: check labels
        label_count = sum(1 for _ in CAPTURES_DIR.glob("*/labels/*.json"))
        if label_count == 0:
            results.append(Result("WARN", "sparsity", "No training data or labels to check"))
            return results

        results.append(Result("WARN", "sparsity",
                              f"No GRPO training data; {label_count} labels available"))
        return results

    jsonl_files = sorted(grpo_dir.glob("*.jsonl"))
    if not jsonl_files:
        results.append(Result("WARN", "sparsity", "No .jsonl files in training/grpo/"))
        return results

    # Quick L0 coverage: count unique (map, side, phase, weapon_class) tuples
    from src.utils.cs2 import classify_weapon_class

    coverage: set[tuple] = set()
    n_total = 0
    for jf in jsonl_files:
        for line in jf.open():
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            gt = sample.get("ground_truth", {})
            gs = gt.get("game_state", {})
            map_name = gs.get("map_name", "unknown")
            side = gs.get("player_side", "unknown")
            phase = gs.get("round_phase", "unknown")
            inv = []
            wp = gs.get("weapon_primary")
            if wp:
                inv.append(wp)
            wc = classify_weapon_class(inv)
            coverage.add((map_name, side, phase, wc))

    # Expected: ~7 maps * 2 sides * 3 phases * 6 weapon classes = ~252 buckets
    n_buckets = len(coverage)
    max_buckets = 7 * 2 * 3 * 6
    pct = n_buckets / max_buckets * 100

    if pct >= 50:
        results.append(Result("PASS", "sparsity",
                              f"{n_buckets}/{max_buckets} buckets covered ({pct:.0f}%), "
                              f"{n_total} samples"))
    elif pct >= 20:
        results.append(Result("WARN", "sparsity",
                              f"{n_buckets}/{max_buckets} buckets covered ({pct:.0f}%), "
                              f"{n_total} samples"))
    else:
        results.append(Result("FAIL", "sparsity",
                              f"{n_buckets}/{max_buckets} buckets covered ({pct:.0f}%), "
                              f"{n_total} samples — very sparse"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GATES = {
    "parse": gate_parse,
    "labels": gate_labels,
    "training": gate_training,
    "sync": gate_sync,
    "sparsity": gate_sparsity,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Chimera data quality validation")
    parser.add_argument("--gate", choices=list(GATES.keys()),
                        help="Run only a specific gate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-file details")
    args = parser.parse_args()

    gates_to_run = [args.gate] if args.gate else list(GATES.keys())
    all_results: list[Result] = []

    print("=== Chimera Data Validation ===\n")

    for gate_name in gates_to_run:
        results = GATES[gate_name](args.verbose)
        all_results.extend(results)
        for r in results:
            print(r)
        if results:
            print()

    has_fail = any(r.level == "FAIL" for r in all_results)
    n_pass = sum(1 for r in all_results if r.level == "PASS")
    n_warn = sum(1 for r in all_results if r.level == "WARN")
    n_fail = sum(1 for r in all_results if r.level == "FAIL")

    print(f"--- Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL ---")
    return 1 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
