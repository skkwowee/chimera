#!/usr/bin/env python3
"""Offline within-group passer-spread audit on a GRPO training run.

This is the diagnostic that retroactively caught RECALL's collapse in F08v4.
For each training step we have G=4 generated completions in the same state,
each with a reward score. The ONLY useful gradient comes from differences
*among the format-passing completions* — those are the ones the policy can
realistically be pushed toward. If the scoring function gives near-identical
rewards to genuinely different format-passing advices, GRPO has nothing to
learn even though the loss curve looks busy.

Reads `useful_jumps.jsonl` produced by src/training/grpo_trainer.py and reports:

  - The overall reward_std distribution (% of steps with effectively zero
    spread). This is the loud "is the reward signal alive at all" check.
  - The conditional std among format-passing completions in each group,
    split by format_passes count k ∈ {1..G}. The k=2 row is the most
    diagnostic case: two completions passed, what's the median spread
    between them? RECALL on f08v4 showed 0.000 — identical scores for
    different advice in the same state.

Pass thresholds for chimera (binary outcome dataset, p≈0.6):

  k=2 passers, median std ≥ 0.025  → reward signal is alive
  k=2 passers, median std < 0.015  → reward signal is dead (RECALL floor)

  zero-spread fraction at k=2 < 15% → tolerable
  zero-spread fraction at k=2 ≥ 35% → kill the run

Usage:
    python scripts/passer_spread_audit.py outputs/grpo/f08v4/useful_jumps.jsonl
    python scripts/passer_spread_audit.py outputs/grpo/f08v5/ --baseline f08v4

The --baseline flag pretty-prints the f08v4 RECALL numbers next to the run's
own numbers as a quick visual diff. See claude-progress.txt 2026-04-23 entry
for the original analysis this script encodes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# F08v4 (RECALL, no SFT-merge, perception-only) — the floor we measured.
# Sourced from claude-progress.txt 2026-04-23 entry.
F08V4_BASELINE = {
    "n_steps": 202,
    "reward_std_lt_001_frac": 0.337,
    "reward_std_lt_005_frac": 0.881,
    "format_passes_mean": 1.86,
    # median spread among passers, by format_passes count
    "passer_spread_median_by_k": {2: 0.000, 3: 0.0154, 4: 0.0154},
}

PASSER_EPSILON = 1e-6  # reward > epsilon ⇒ format gate passed (gate is
                       # multiplicative; failed gate forces reward=0).


def load_records(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "useful_jumps.jsonl"
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def passer_indices(rewards: list[float]) -> list[int]:
    """Indices of completions that passed the format gate (proxy: reward>eps)."""
    return [i for i, r in enumerate(rewards) if r > PASSER_EPSILON]


def std_of(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def audit(records: list[dict], baseline: dict | None = None) -> dict:
    n = len(records)
    if n == 0:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    overall_stds = [float(r.get("reward_std", 0.0)) for r in records]
    overall_stds_sorted = sorted(overall_stds)

    # Bucket steps by their format_passes count, collect the std-among-passers
    # for each step where ≥2 completions passed (else there is nothing to
    # spread across).
    by_k: dict[int, list[float]] = {}
    fmt_passes_seen = []

    for rec in records:
        rewards = list(rec.get("rewards", []))
        fmt_passes = int(rec.get("format_passes", 0))
        fmt_passes_seen.append(fmt_passes)
        passers = passer_indices(rewards)
        k = len(passers)
        if k < 2:
            continue
        passer_rewards = [rewards[i] for i in passers]
        spread = std_of(passer_rewards)
        by_k.setdefault(k, []).append(spread)

    summary = {
        "n_steps": n,
        "reward_std_lt_001_frac": sum(1 for s in overall_stds if s < 0.01) / n,
        "reward_std_lt_005_frac": sum(1 for s in overall_stds if s < 0.05) / n,
        "reward_std_median": percentile(overall_stds_sorted, 0.50),
        "reward_std_p25": percentile(overall_stds_sorted, 0.25),
        "reward_std_p75": percentile(overall_stds_sorted, 0.75),
        "format_passes_mean": sum(fmt_passes_seen) / len(fmt_passes_seen),
        "passer_spread_by_k": {},
    }
    for k, spreads in sorted(by_k.items()):
        spreads_sorted = sorted(spreads)
        summary["passer_spread_by_k"][k] = {
            "n_steps_at_k": len(spreads),
            "median": percentile(spreads_sorted, 0.50),
            "p25": percentile(spreads_sorted, 0.25),
            "p75": percentile(spreads_sorted, 0.75),
            "zero_spread_frac": sum(1 for s in spreads if s < 1e-4) / len(spreads),
        }

    return summary


def print_report(summary: dict, baseline: dict | None = None, label: str = "run") -> None:
    print(f"\n=== Passer-spread audit: {label} ===\n")
    print(f"steps audited: {summary['n_steps']}")
    print(f"reward_std median: {summary['reward_std_median']:.4f}  "
          f"(p25 {summary['reward_std_p25']:.4f}, p75 {summary['reward_std_p75']:.4f})")
    print(f"reward_std < 0.01: {summary['reward_std_lt_001_frac']:.1%}")
    print(f"reward_std < 0.05: {summary['reward_std_lt_005_frac']:.1%}")
    print(f"format_passes mean: {summary['format_passes_mean']:.2f}")
    print()
    print("Among-passer spread (the live diagnostic):")
    print(f"  {'k':>2}  {'n_steps':>8}  {'median':>8}  {'p25':>8}  {'p75':>8}  "
          f"{'zero%':>6}  {'verdict':<14}", end="")
    if baseline:
        print(f"  {'f08v4_med':>10}")
    else:
        print()
    for k, st in summary["passer_spread_by_k"].items():
        verdict = (
            "ALIVE"     if st["median"] >= 0.025
            else "WEAK"  if st["median"] >= 0.015
            else "DEAD (~RECALL)"
        )
        line = (f"  {k:>2}  {st['n_steps_at_k']:>8}  "
                f"{st['median']:>8.4f}  {st['p25']:>8.4f}  {st['p75']:>8.4f}  "
                f"{st['zero_spread_frac']:>6.1%}  {verdict:<14}")
        if baseline and k in baseline.get("passer_spread_median_by_k", {}):
            line += f"  {baseline['passer_spread_median_by_k'][k]:>10.4f}"
        print(line)
    print()
    if baseline:
        print(f"Baseline (f08v4 RECALL, n={baseline['n_steps']}): "
              f"k=2 median spread {baseline['passer_spread_median_by_k'][2]:.4f}")
        print()
    # Headline pass/fail
    k2 = summary["passer_spread_by_k"].get(2)
    if k2 is not None:
        if k2["median"] >= 0.025 and k2["zero_spread_frac"] < 0.15:
            print("VERDICT: PASS — reward signal is alive at the most common case (k=2).")
        elif k2["median"] < 0.015 or k2["zero_spread_frac"] >= 0.35:
            print("VERDICT: FAIL — reward signal is at or below RECALL's noise floor.")
        else:
            print("VERDICT: MARGINAL — borderline; consider longer audit window or "
                  "stricter scorer before committing pod time.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path,
                    help="useful_jumps.jsonl file or directory containing it")
    ap.add_argument("--baseline", choices=["f08v4", "none"], default="f08v4",
                    help="reference baseline to print alongside")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON summary on stdout instead of human report")
    ap.add_argument("--label", default=None, help="run label for the header")
    args = ap.parse_args()

    records = load_records(args.path)
    summary = audit(records)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    label = args.label or args.path.stem or str(args.path)
    baseline = F08V4_BASELINE if args.baseline == "f08v4" else None
    print_report(summary, baseline=baseline, label=label)


if __name__ == "__main__":
    main()
