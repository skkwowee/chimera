#!/usr/bin/env python3
"""
Data sparsity diagnostic for GRPO training (D023).

Measures state bucket coverage across the dataset to determine whether
contrastive reward pairing is viable at the current data scale.

Reports:
  1. Bucket coverage by hierarchy level (L0, L1, L2)
  2. Within-bucket behavioral variance (wins vs losses)
  3. Simulated R_outcome within-group variance

Usage:
    python scripts/sparsity_diagnostic.py
    python scripts/sparsity_diagnostic.py --demo-data-dir data/processed/demos
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from src.utils.cs2 import (
    ARMOR_COST,
    HELMET_COST,
    WEAPON_VALUES,
    classify_buy,
    classify_weapon_class,
    economy_matchup,
    estimate_team_equip,
)

# Simplified Omega signal matrix (D024): Ω = a·w + (1−a)·(1−w)
# where a = agreement ∈ {0,1}, w = win ∈ {0,1}
OMEGA = {
    ("agree", "win"): 1.0,     # a=1, w=1: 1·1 + 0·0 = 1.0
    ("agree", "lose"): 0.0,    # a=1, w=0: 1·0 + 0·1 = 0.0
    ("deviate", "win"): 0.0,   # a=0, w=1: 0·1 + 1·0 = 0.0
    ("deviate", "lose"): 1.0,  # a=0, w=0: 0·0 + 1·1 = 1.0
}
# Legacy D013 values (phi=1): agree+win=1.0, agree+lose=0.2, deviate+win=0.6, deviate+lose=0.5


def round_time_bucket(tick: int, freeze_end: int, tickrate: int = 64) -> str:
    secs = max(0, (tick - freeze_end) / tickrate)
    if secs < 30:
        return "early"
    elif secs < 60:
        return "mid"
    else:
        return "late"


def advantage_bucket(alive_t: int, alive_ct: int) -> str:
    """Coarse advantage bucket for Level 0."""
    diff = alive_t - alive_ct
    if diff >= 2:
        return "t_advantage"
    elif diff == 1:
        return "t_slight"
    elif diff == 0:
        return "even"
    elif diff == -1:
        return "ct_slight"
    else:
        return "ct_advantage"


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

def extract_samples(
    ticks_df: pl.DataFrame,
    rounds: list[dict],
    bomb_events: list[dict],
    header: dict,
    demo_stem: str,
    sample_interval_ticks: int = 192,  # ~3 seconds at 64 tick
) -> list[dict]:
    """
    Extract planning-frame samples from tick data.

    Samples one snapshot every sample_interval_ticks within each round
    (after freeze time), for each alive player. Each sample gets
    bucketing features at all hierarchy levels.
    """
    map_name = header.get("map_name", "unknown")
    samples = []

    # Build round outcome map
    round_outcomes = {}
    for rnd in rounds:
        round_outcomes[rnd["round_num"]] = rnd.get("winner", "").lower()

    # Build bomb event index by round
    bomb_by_round = defaultdict(list)
    for evt in bomb_events:
        bomb_by_round[evt["round_num"]].append(evt)

    for rnd in rounds:
        rnum = rnd["round_num"]
        freeze_end = rnd["freeze_end"]
        round_end = rnd["end"]
        winner = round_outcomes.get(rnum, "")

        # Get unique ticks in this round after freeze
        round_ticks_available = (
            ticks_df.filter(
                (pl.col("round_num") == rnum) & (pl.col("tick") >= freeze_end)
            )
            .select("tick").unique().sort("tick")
            .to_series().to_list()
        )

        if not round_ticks_available:
            continue

        # Sample at intervals
        first_tick = round_ticks_available[0]
        sampled_ticks = [
            t for t in round_ticks_available
            if (t - first_tick) % sample_interval_ticks < 64  # within 1 tick tolerance
        ]
        # Deduplicate: pick closest tick to each interval
        target_ticks = []
        t = first_tick
        while t <= round_ticks_available[-1]:
            # Find closest available tick
            closest = min(round_ticks_available, key=lambda x: abs(x - t))
            if closest not in target_ticks:
                target_ticks.append(closest)
            t += sample_interval_ticks
        sampled_ticks = target_ticks

        # Get bomb status at round start for economy calculation
        round_start_snap = ticks_df.filter(
            (pl.col("round_num") == rnum) & (pl.col("tick") == round_ticks_available[0])
        ).to_dicts()

        t_equip = estimate_team_equip(round_start_snap, "t")
        ct_equip = estimate_team_equip(round_start_snap, "ct")
        t_buy = classify_buy(t_equip)
        ct_buy = classify_buy(ct_equip)
        econ_matchup = economy_matchup(t_buy, ct_buy)

        for tick in sampled_ticks:
            snap = ticks_df.filter(
                (pl.col("round_num") == rnum) & (pl.col("tick") == tick)
            ).to_dicts()

            if not snap:
                continue

            alive_t = sum(1 for p in snap if p["side"] == "t" and (p.get("health") or 0) > 0)
            alive_ct = sum(1 for p in snap if p["side"] == "ct" and (p.get("health") or 0) > 0)

            # Determine bomb status at this tick
            bomb_status = "not_planted"
            for evt in bomb_by_round.get(rnum, []):
                if evt["tick"] > tick:
                    break
                event_type = str(evt.get("event", "")).lower()
                if event_type in ("plant", "planted", "bomb_planted"):
                    bomb_status = "planted"
                elif event_type in ("drop", "dropped"):
                    bomb_status = "dropped"
                elif event_type in ("pickup", "carried"):
                    bomb_status = "not_planted"

            time_bucket = round_time_bucket(tick, freeze_end)
            # Override time bucket if bomb is planted
            if bomb_status == "planted":
                time_bucket = "post-plant"

            adv = advantage_bucket(alive_t, alive_ct)

            # Create one sample per alive player
            for p in snap:
                if (p.get("health") or 0) == 0:
                    continue

                side = p["side"]
                won = (side == winner)

                health_bucket = "healthy" if p["health"] >= 30 else "critical"
                weapon_class = classify_weapon_class(p.get("inventory"))

                sample = {
                    "demo": demo_stem,
                    "round_num": rnum,
                    "tick": tick,
                    "player": p["name"],
                    "side": side,
                    "won": won,
                    # Level 0 features
                    "advantage_bucket": adv,
                    "bomb_status": bomb_status,
                    # Level 1 features
                    "alive_t": alive_t,
                    "alive_ct": alive_ct,
                    "map": map_name,
                    "economy_matchup": econ_matchup,
                    "round_time_bucket": time_bucket,
                    "health_bucket": health_bucket,
                    "weapon_class": weapon_class,
                }

                # Level 0 key
                sample["L0"] = f"{side}|{adv}|{bomb_status}"
                # Level 1 key (dims 1-10)
                sample["L1"] = (
                    f"{side}|{alive_t}v{alive_ct}|{bomb_status}|{map_name}|"
                    f"{econ_matchup}|{time_bucket}|{health_bucket}|{weapon_class}"
                )

                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Diagnostic analysis
# ---------------------------------------------------------------------------

def analyze_coverage(samples: list[dict], level: str, min_k: int = 5):
    """
    Analyze bucket coverage at a given hierarchy level.

    Returns dict with:
      - total_buckets: number of unique buckets
      - covered_buckets: buckets with both win+loss at >=min_k each
      - partial_buckets: buckets with both outcomes but <min_k for one
      - single_outcome: buckets with only wins or only losses
      - coverage_pct: % of samples in covered buckets
      - bucket_details: list of (bucket, wins, losses, covered)
    """
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0})

    for s in samples:
        key = s[level]
        if s["won"]:
            buckets[key]["wins"] += 1
        else:
            buckets[key]["losses"] += 1

    total = len(buckets)
    covered = 0
    partial = 0
    single = 0
    covered_samples = 0

    details = []
    for bucket, counts in sorted(buckets.items(), key=lambda x: -(x[1]["wins"] + x[1]["losses"])):
        w, l = counts["wins"], counts["losses"]
        if w >= min_k and l >= min_k:
            status = "covered"
            covered += 1
            covered_samples += w + l
        elif w > 0 and l > 0:
            status = "partial"
            partial += 1
        else:
            status = "single_outcome"
            single += 1

        details.append({
            "bucket": bucket,
            "wins": w,
            "losses": l,
            "total": w + l,
            "status": status,
        })

    total_samples = len(samples)
    return {
        "total_buckets": total,
        "covered_buckets": covered,
        "partial_buckets": partial,
        "single_outcome_buckets": single,
        "coverage_pct": (covered_samples / total_samples * 100) if total_samples > 0 else 0,
        "covered_samples": covered_samples,
        "total_samples": total_samples,
        "bucket_details": details,
    }


def simulate_reward_variance(samples: list[dict], level: str):
    """
    Simulate R_outcome variance within GRPO groups.

    For each bucket at the given level, compute the R_outcome scores
    and measure within-bucket variance. Near-zero variance means
    GRPO can't learn from those samples.
    """
    buckets = defaultdict(list)
    for s in samples:
        key = s[level]
        # Simulate: agree with pro (R_decision ~ 0.7 avg), compute R_strategy
        # D024 simplified Ω: agree+win = 1.0, agree+lose = 0.0
        r_outcome = OMEGA[("agree", "win")] if s["won"] else OMEGA[("agree", "lose")]
        buckets[key].append(r_outcome)

    variances = []
    for bucket, scores in buckets.items():
        if len(scores) < 2:
            continue
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        variances.append({
            "bucket": bucket,
            "n": len(scores),
            "mean_reward": mean,
            "variance": var,
            "win_rate": sum(1 for s in scores if s == 1.0) / len(scores),
        })

    variances.sort(key=lambda x: -x["variance"])
    overall_var = sum(v["variance"] * v["n"] for v in variances) / max(1, sum(v["n"] for v in variances))

    return {
        "overall_weighted_variance": overall_var,
        "bucket_variances": variances,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data sparsity diagnostic (D023)")
    parser.add_argument(
        "--demo-data-dir", default="data/processed/demos",
        help="Directory containing parsed demo data",
    )
    parser.add_argument(
        "--min-k", type=int, default=5,
        help="Minimum observations per outcome for a bucket to be 'covered' (default: 5)",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=192,
        help="Tick interval between samples (~3s at 64 tick, default: 192)",
    )
    parser.add_argument(
        "--show-buckets", action="store_true",
        help="Show per-bucket details",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    demo_dir = Path(args.demo_data_dir)
    parquet_files = sorted(demo_dir.glob("*_ticks.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {demo_dir}")
        sys.exit(1)

    # Extract samples from all demos
    all_samples = []

    for pq_path in parquet_files:
        demo_stem = pq_path.stem.replace("_ticks", "")
        print(f"Processing {demo_stem}...")

        ticks_df = pl.read_parquet(pq_path)

        rounds_path = demo_dir / f"{demo_stem}_rounds.json"
        bomb_path = demo_dir / f"{demo_stem}_bomb.json"
        header_path = demo_dir / f"{demo_stem}_header.json"

        rounds = json.loads(rounds_path.read_text())
        bomb_events = json.loads(bomb_path.read_text()) if bomb_path.exists() else []
        header = json.loads(header_path.read_text())

        samples = extract_samples(
            ticks_df, rounds, bomb_events, header, demo_stem,
            sample_interval_ticks=args.sample_interval,
        )
        all_samples.extend(samples)
        print(f"  {len(samples)} samples extracted")

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Demos: {len(parquet_files)}")

    # Unique rounds
    unique_rounds = len(set((s["demo"], s["round_num"]) for s in all_samples))
    wins = sum(1 for s in all_samples if s["won"])
    losses = len(all_samples) - wins
    print(f"Unique rounds: {unique_rounds}")
    print(f"Win samples: {wins}, Loss samples: {losses} ({wins/(wins+losses)*100:.1f}% / {losses/(wins+losses)*100:.1f}%)")

    results = {}

    # Analyze each level
    for level, label in [("L0", "Level 0 (side × advantage × bomb)"),
                          ("L1", "Level 1 (dims 1-10)")]:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        cov = analyze_coverage(all_samples, level, min_k=args.min_k)
        results[level] = cov

        print(f"  Total buckets:    {cov['total_buckets']}")
        print(f"  Covered (≥{args.min_k} each): {cov['covered_buckets']}")
        print(f"  Partial (both, <{args.min_k}): {cov['partial_buckets']}")
        print(f"  Single outcome:   {cov['single_outcome_buckets']}")
        print(f"  Sample coverage:  {cov['coverage_pct']:.1f}% ({cov['covered_samples']}/{cov['total_samples']})")

        if args.show_buckets:
            print(f"\n  {'Bucket':<55} {'Wins':>5} {'Loss':>5} {'Total':>6} {'Status':<15}")
            print(f"  {'-'*55} {'-'*5} {'-'*5} {'-'*6} {'-'*15}")
            for d in cov["bucket_details"]:
                print(f"  {d['bucket']:<55} {d['wins']:>5} {d['losses']:>5} {d['total']:>6} {d['status']:<15}")

        # Reward variance
        rv = simulate_reward_variance(all_samples, level)
        results[f"{level}_reward"] = rv
        print(f"\n  Simulated R_outcome variance:")
        print(f"  Overall weighted variance: {rv['overall_weighted_variance']:.4f}")

        # Show top/bottom variance buckets
        if rv["bucket_variances"]:
            print(f"\n  Top 5 highest-variance buckets (best for learning):")
            for v in rv["bucket_variances"][:5]:
                print(f"    {v['bucket']:<50} var={v['variance']:.4f} n={v['n']:>4} win_rate={v['win_rate']:.2f}")
            print(f"\n  Bottom 5 lowest-variance buckets (worst for learning):")
            for v in rv["bucket_variances"][-5:]:
                print(f"    {v['bucket']:<50} var={v['variance']:.4f} n={v['n']:>4} win_rate={v['win_rate']:.2f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Data scale: {len(parquet_files)} demos, {unique_rounds} rounds, {len(all_samples)} samples")
    print()

    l0 = results["L0"]
    l1 = results["L1"]
    print(f"  Level 0: {l0['covered_buckets']}/{l0['total_buckets']} buckets covered, "
          f"{l0['coverage_pct']:.1f}% samples")
    print(f"  Level 1: {l1['covered_buckets']}/{l1['total_buckets']} buckets covered, "
          f"{l1['coverage_pct']:.1f}% samples")

    l0_var = results["L0_reward"]["overall_weighted_variance"]
    l1_var = results["L1_reward"]["overall_weighted_variance"]
    print(f"\n  R_outcome variance: L0={l0_var:.4f}, L1={l1_var:.4f}")
    print(f"  (Variance > 0.15 needed for meaningful GRPO learning)")

    # Viability assessment
    print()
    if l0["coverage_pct"] > 50 and l0_var > 0.15:
        print("  L0 contrastive pairing: VIABLE")
    elif l0["coverage_pct"] > 20:
        print("  L0 contrastive pairing: MARGINAL")
    else:
        print("  L0 contrastive pairing: NOT VIABLE at current scale")

    if l1["coverage_pct"] > 20 and l1_var > 0.10:
        print("  L1 contrastive pairing: VIABLE")
    elif l1["coverage_pct"] > 5:
        print("  L1 contrastive pairing: MARGINAL")
    else:
        print("  L1 contrastive pairing: NOT VIABLE at current scale")

    if args.json:
        # Write full results to JSON
        output_path = Path("outputs/sparsity_diagnostic.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_results = {
            "demos": len(parquet_files),
            "rounds": unique_rounds,
            "total_samples": len(all_samples),
            "wins": wins,
            "losses": losses,
        }
        for level in ["L0", "L1"]:
            cov = results[level]
            rv = results[f"{level}_reward"]
            json_results[level] = {
                "total_buckets": cov["total_buckets"],
                "covered_buckets": cov["covered_buckets"],
                "partial_buckets": cov["partial_buckets"],
                "single_outcome_buckets": cov["single_outcome_buckets"],
                "coverage_pct": cov["coverage_pct"],
                "reward_variance": rv["overall_weighted_variance"],
            }
        output_path.write_text(json.dumps(json_results, indent=2))
        print(f"\n  Full results written to {output_path}")


if __name__ == "__main__":
    main()
