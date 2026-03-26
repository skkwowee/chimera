#!/usr/bin/env python3
"""
build_sft_dataset.py — Build a curated, coverage-guaranteed SFT dataset from existing labels.

Implements the council's consensus:
  - Downsample redundant 5v5 full-health states
  - Enforce state-space coverage quotas
  - Generate templated analysis/advice (structurally valid; GRPO will improve quality)
  - Output a filtered dataset manifest ready for SFT training
"""

import argparse
import json
import glob
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weapon class buckets
RIFLE_NAMES = {
    "AK-47", "M4A1-S", "M4A4", "Galil AR", "FAMAS", "SG 553",
    "AUG", "SSG 08", "SCAR-20", "G3SG1",
}
SNIPER_NAMES = {"AWP", "SSG 08", "SCAR-20", "G3SG1"}
SMG_NAMES = {"MP7", "MP9", "MAC-10", "UMP-45", "P90", "PP-Bizon", "MP5-SD"}
HEAVY_NAMES = {"Nova", "XM1014", "Sawed-Off", "MAG-7", "M249", "Negev"}
PISTOL_NAMES = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas",
}

SFT_KEEP_FIELDS = {
    "map_name", "round_phase", "player_side", "player_health",
    "player_armor", "player_has_helmet", "player_money",
    "weapon_primary", "weapon_secondary", "utility", "has_defuser",
    "alive_teammates", "alive_enemies", "bomb_status",
    "score_t", "score_ct",
}

SFT_REMOVE_FIELDS = {"site", "visible_enemies", "team_money_total", "round_num", "player_name"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def classify_weapon(weapon_primary: str | None) -> str:
    if not weapon_primary:
        return "pistol_only"
    if weapon_primary in SNIPER_NAMES:
        return "sniper"
    if weapon_primary in RIFLE_NAMES:
        return "rifle"
    if weapon_primary in SMG_NAMES:
        return "smg"
    if weapon_primary in HEAVY_NAMES:
        return "heavy"
    return "other"


def classify_economy(money: int) -> str:
    if money < 2000:
        return "eco"
    if money < 3500:
        return "force-buy"
    if money < 4500:
        return "half-buy"
    return "full-buy"


def round_importance(score_t: int, score_ct: int, round_num: int) -> str:
    total_rounds = score_t + score_ct + 1  # current round
    diff = abs(score_t - score_ct)
    if total_rounds >= 24:  # overtime
        return "critical — overtime round"
    if round_num <= 5:
        return "early — economy establishment phase"
    if diff >= 5:
        leading = "T-side" if score_t > score_ct else "CT-side"
        return f"moderate — {leading} has significant lead ({diff} rounds ahead)"
    if diff == 0:
        return "high — score is tied"
    trailing = "T-side" if score_t < score_ct else "CT-side"
    return f"elevated — {trailing} trails by {diff}"


# ---------------------------------------------------------------------------
# Templated analysis & advice generation
# ---------------------------------------------------------------------------

def build_situation_summary(gs: dict) -> str:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    side = gs["player_side"]
    bomb = gs["bomb_status"]
    phase = gs["round_phase"]

    total_t = at + 1  # include self
    total_e = ae

    if bomb == "planted":
        if side == "T":
            return (
                f"{total_t}v{total_e} post-plant situation, T-side defending planted bomb "
                f"with {at} teammate{'s' if at != 1 else ''} alive"
            )
        else:
            return (
                f"{total_e}v{total_t} post-plant situation, CT-side rushing defuse "
                f"with {at} teammate{'s' if at != 1 else ''} alive"
            )

    if phase == "freezetime":
        return f"Freeze time — round about to begin, {total_t}v{total_e} start"

    if total_t == total_e:
        return (
            f"{total_t}v{total_e} even fight, bomb {bomb}, "
            f"{at} teammate{'s' if at != 1 else ''} supporting"
        )
    elif total_t > total_e:
        advantage = total_t - total_e
        return (
            f"{total_t}v{total_e} man-advantage (+{advantage}), bomb {bomb}, "
            f"opportunity to press"
        )
    else:
        deficit = total_e - total_t
        return (
            f"{total_t}v{total_e} man-disadvantage (-{deficit}), bomb {bomb}, "
            f"must play carefully"
        )


def build_immediate_threats(gs: dict) -> list[str]:
    ae = gs["alive_enemies"]
    bomb = gs["bomb_status"]
    side = gs["player_side"]
    threats = []

    if ae == 0:
        threats.append("No active threats — round likely decided")
        return threats

    if ae >= 4:
        threats.append(f"High enemy count ({ae} active) — multiple angles to watch")
    elif ae >= 2:
        threats.append(f"{ae} enemies still active — coordinated push possible")
    else:
        threats.append("1 enemy remaining — likely playing for information or time")

    if bomb == "planted" and side == "CT":
        threats.append("Bomb planted — time pressure to defuse before detonation")
    elif bomb == "planted" and side == "T":
        threats.append("Bomb planted — enemy will attempt defuse, cover site")

    if gs["player_health"] < 50:
        threats.append(f"Low HP ({gs['player_health']}) — one shot from most weapons is lethal")

    return threats


def build_opportunities(gs: dict) -> list[str]:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    bomb = gs["bomb_status"]
    side = gs["player_side"]
    utility = gs.get("utility") or []
    opps = []

    total_t = at + 1
    if total_t > ae:
        opps.append(f"Numerical advantage ({total_t}v{ae}) — trade efficiently and press")
    elif total_t == ae:
        opps.append("Equal numbers — isolated duels favor better-positioned player")
    else:
        opps.append(f"Outnumbered {total_t}v{ae} — look for off-angles and information plays")

    if utility:
        opps.append(f"Utility available: {', '.join(utility)} — use proactively")
    else:
        opps.append("No utility — rely on positioning and crossfire")

    if bomb == "carried" and side == "T" and ae <= 2:
        opps.append("Low enemy count — safe window to plant bomb and play retake")
    elif bomb == "planted" and side == "T":
        opps.append("Bomb planted — hold passive angles and let clock work")

    return opps


def build_primary_action(gs: dict) -> str:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    bomb = gs["bomb_status"]
    side = gs["player_side"]
    hp = gs["player_health"]
    total_t = at + 1

    if bomb == "planted":
        if side == "T":
            return "Hold passive angle near bomb to deny defuse attempts"
        else:
            if total_t >= ae:
                return "Clear site aggressively and secure defuse with teammate cover"
            else:
                return "Attempt lone-wolf defuse — distract enemies, fake defuse if needed"

    if hp < 50:
        return "Play defensively — get traded out or fall back for recovery"

    if total_t > ae + 1:
        return "Execute coordinated site take with remaining utility"
    elif total_t == ae:
        return "Hold position and play for picks — look for isolated duels"
    else:
        return "Stall for time — play off-angles and avoid direct confrontation"


def build_reasoning(gs: dict) -> str:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    total_t = at + 1
    economy = classify_economy(gs["player_money"])
    weapon_class = classify_weapon(gs.get("weapon_primary"))
    bomb = gs["bomb_status"]

    parts = [
        f"Numbers are {total_t}v{ae}.",
        f"Economy is {economy} ({gs['player_money']} credits).",
        f"Weapon loadout: {weapon_class}.",
    ]
    if bomb == "planted":
        parts.append("With bomb planted, clock is the primary factor.")
    elif total_t > ae:
        parts.append("Numerical advantage allows for aggressive plays.")
    elif total_t < ae:
        parts.append("Numerical disadvantage requires passive, information-based play.")
    return " ".join(parts)


def build_fallback(gs: dict) -> str:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    bomb = gs["bomb_status"]
    side = gs["player_side"]
    total_t = at + 1

    if bomb == "planted" and side == "CT":
        return "If defuse is impossible, trade kills to deny bomb plant next round economy"
    if total_t <= 1 and ae >= 2:
        return "If trade opportunity appears, accept it — dying with value is better than losing 1v2+"
    if total_t > ae:
        return "If execute fails, reset and re-enter from different angle"
    return "Fall back to safer position and wait for teammate assistance"


def build_callout(gs: dict) -> str:
    at = gs["alive_teammates"]
    ae = gs["alive_enemies"]
    bomb = gs["bomb_status"]
    total_t = at + 1

    if at == 0:
        return f"Lone survivor: {ae} remaining — report position via ping"
    if bomb == "planted":
        return f"Bomb planted — {at} with me, {ae} enemies. Hold site exits."
    if total_t > ae:
        return f"{at} with me, {ae} enemies. Push together and trade fast."
    return f"{at} with me, {ae} enemies. Play slow, gather info."


def generate_analysis_advice(gs: dict) -> tuple[dict, dict]:
    """Generate deterministic templated analysis and advice from game_state."""
    round_num = gs.get("round_num", 1)
    score_t = gs.get("score_t", 0)
    score_ct = gs.get("score_ct", 0)

    analysis = {
        "situation_summary": build_situation_summary(gs),
        "economy_assessment": classify_economy(gs["player_money"]),
        "round_importance": round_importance(score_t, score_ct, round_num),
        "immediate_threats": build_immediate_threats(gs),
        "opportunities": build_opportunities(gs),
    }

    advice = {
        "primary_action": build_primary_action(gs),
        "reasoning": build_reasoning(gs),
        "fallback": build_fallback(gs),
        "callout": build_callout(gs),
    }

    return analysis, advice


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(captures_dir: str) -> list[dict[str, Any]]:
    """Load all label JSON files from all capture subdirectories."""
    pattern = os.path.join(captures_dir, "*/labels/*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No label files found matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    records = []
    errors = 0
    for fpath in sorted(files):
        try:
            with open(fpath) as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Skipping {fpath}: {e}", file=sys.stderr)
            errors += 1
            continue

        # Resolve capture dir (grandparent of labels/)
        label_path = Path(fpath)
        capture_dir = label_path.parent.parent
        image_dir = str(capture_dir)

        records.append({
            "raw": data,
            "label_path": str(label_path),
            "image_dir": image_dir,
            "capture_name": capture_dir.name,
        })

    print(f"[INFO] Loaded {len(records)} labels ({errors} skipped) from {captures_dir}")
    return records


# ---------------------------------------------------------------------------
# Classification helpers for filtering
# ---------------------------------------------------------------------------

def is_5v5_full_health(gs: dict) -> bool:
    return (
        gs.get("alive_teammates", 0) == 4
        and gs.get("alive_enemies", 0) == 5
        and gs.get("player_health", 0) >= 90
    )


def is_any_kill(gs: dict) -> bool:
    return (gs.get("alive_teammates", 0) + gs.get("alive_enemies", 0)) < 9


def is_post_plant(gs: dict) -> bool:
    return gs.get("bomb_status") == "planted"


def is_low_health(gs: dict) -> bool:
    return gs.get("player_health", 100) < 80


# ---------------------------------------------------------------------------
# Core curation logic
# ---------------------------------------------------------------------------

def curate_dataset(records: list[dict], max_5v5_full: int) -> list[dict]:
    """
    Apply filtering rules:
    1. Keep ALL samples where kills happened (alive_t + alive_e < 9)
    2. Keep ALL post-plant samples
    3. Keep ALL low-health samples (hp < 80)
    4. From remaining 5v5-full-health pool, keep up to max_5v5_full

    Returns list of kept records.
    """
    always_keep = []
    maybe_keep = []  # 5v5-full-health that don't trigger other rules

    for rec in records:
        gs = rec["raw"].get("game_state", {})
        full_5v5 = is_5v5_full_health(gs)
        any_kill = is_any_kill(gs)
        post_plant = is_post_plant(gs)
        low_hp = is_low_health(gs)

        if any_kill or post_plant or low_hp:
            always_keep.append(rec)
        elif full_5v5:
            maybe_keep.append(rec)
        else:
            # Not 5v5 full health and no special rule — keep it
            always_keep.append(rec)

    # Downsample 5v5 full-health
    random.seed(42)
    kept_5v5 = random.sample(maybe_keep, min(max_5v5_full, len(maybe_keep)))
    print(
        f"[INFO] 5v5 full-health pool: {len(maybe_keep)} → kept {len(kept_5v5)} "
        f"(max {max_5v5_full})"
    )

    curated = always_keep + kept_5v5
    print(
        f"[INFO] Curation: {len(records)} total → {len(curated)} kept "
        f"({len(records) - len(curated)} dropped)"
    )
    return curated


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def build_coverage_stats(records: list[dict]) -> dict:
    """Compute per-(alive_t, alive_e), per-bomb_status, per-weapon_class counts."""
    alive_cell: Counter = Counter()
    bomb_status: Counter = Counter()
    weapon_class: Counter = Counter()
    health_bucket: Counter = Counter()

    for rec in records:
        gs = rec["raw"].get("game_state", {})
        at = gs.get("alive_teammates", 0)
        ae = gs.get("alive_enemies", 0)
        alive_cell[(at, ae)] += 1
        bomb_status[gs.get("bomb_status", "unknown")] += 1
        weapon_class[classify_weapon(gs.get("weapon_primary"))] += 1
        hp = gs.get("player_health", 0)
        if hp >= 90:
            health_bucket[">=90"] += 1
        elif hp >= 80:
            health_bucket["80-89"] += 1
        elif hp >= 60:
            health_bucket["60-79"] += 1
        elif hp >= 40:
            health_bucket["40-59"] += 1
        else:
            health_bucket["<40"] += 1

    # All possible (alive_t, alive_e) cells: 0–4 × 0–5
    all_cells = {(at, ae) for at in range(5) for ae in range(6)}
    present_cells = set(alive_cell.keys())
    gap_cells = sorted(all_cells - present_cells)

    return {
        "alive_cell": dict(alive_cell),
        "bomb_status": dict(bomb_status),
        "weapon_class": dict(weapon_class),
        "health_bucket": dict(health_bucket),
        "gap_cells": gap_cells,
    }


def find_underrepresented(coverage: dict, min_per_cell: int) -> list[tuple]:
    """Return alive cells with count below min_per_cell."""
    under = []
    for cell, count in coverage["alive_cell"].items():
        if count < min_per_cell:
            under.append((cell, count))
    return sorted(under)


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def build_output_record(rec: dict) -> dict:
    raw = rec["raw"]
    gs_raw = raw.get("game_state", {})
    meta = raw.get("metadata", {})

    # Filter game_state to SFT keep fields only
    gs_filtered = {k: v for k, v in gs_raw.items() if k in SFT_KEEP_FIELDS}

    # Generate templated analysis/advice (pass full gs for round_num etc.)
    analysis, advice = generate_analysis_advice(gs_raw)

    screenshot_id = meta.get("screenshot_id") or Path(rec["label_path"]).stem

    return {
        "screenshot_id": screenshot_id,
        "label_path": rec["label_path"],
        "image_dir": rec["image_dir"],
        "game_state": gs_filtered,
        "analysis": analysis,
        "advice": advice,
        "metadata": {
            "round_num": meta.get("round_num") or gs_raw.get("round_num"),
            "demo": meta.get("demo_file", ""),
            "map": gs_raw.get("map_name", ""),
            "alive_t": gs_raw.get("alive_teammates", 0),
            "alive_e": gs_raw.get("alive_enemies", 0),
            "health": gs_raw.get("player_health", 0),
            "weapon_class": classify_weapon(gs_raw.get("weapon_primary")),
        },
    }


def write_coverage_report(
    path: str,
    total_input: int,
    curated: list[dict],
    coverage_before: dict,
    coverage_after: dict,
    underrepresented: list[tuple],
    min_per_cell: int,
) -> None:
    lines = [
        "=" * 72,
        "  Chimera SFT Dataset — Coverage Report",
        "=" * 72,
        "",
        f"Input labels:          {total_input}",
        f"Curated dataset size:  {len(curated)}",
        f"Samples dropped:       {total_input - len(curated)}",
        "",
        "─" * 72,
        "  ALIVE-CELL COVERAGE  (alive_teammates × alive_enemies)",
        "─" * 72,
        "",
        "  BEFORE curation:",
    ]

    # Table header
    lines.append("    at\\ae " + "  ".join(f"{ae:3}" for ae in range(6)))
    lines.append("    " + "─" * 30)
    for at in range(5):
        row = f"    at={at}  "
        for ae in range(6):
            count = coverage_before["alive_cell"].get((at, ae), 0)
            row += f"{count:3}  "
        lines.append(row)

    lines += ["", "  AFTER curation:"]
    lines.append("    at\\ae " + "  ".join(f"{ae:3}" for ae in range(6)))
    lines.append("    " + "─" * 30)
    for at in range(5):
        row = f"    at={at}  "
        for ae in range(6):
            count = coverage_after["alive_cell"].get((at, ae), 0)
            row += f"{count:3}  "
        lines.append(row)

    lines += [""]

    if coverage_after["gap_cells"]:
        lines.append(f"  ZERO-SAMPLE CELLS ({len(coverage_after['gap_cells'])} cells):")
        for cell in coverage_after["gap_cells"]:
            lines.append(f"    at={cell[0]}, ae={cell[1]} → 0 samples")
    else:
        lines.append("  All (alive_t, alive_e) cells have at least 1 sample.")

    if underrepresented:
        lines += [
            "",
            f"  UNDERREPRESENTED CELLS (< {min_per_cell} samples, excluding zero-sample):",
        ]
        for cell, count in underrepresented:
            if count > 0:
                lines.append(f"    at={cell[0]}, ae={cell[1]} → {count} sample(s)")
    else:
        lines.append(f"\n  All cells meet the minimum threshold of {min_per_cell} samples.")

    lines += [
        "",
        "─" * 72,
        "  BOMB STATUS DISTRIBUTION",
        "─" * 72,
    ]
    for status, count in sorted(coverage_after["bomb_status"].items()):
        pct = 100 * count / len(curated) if curated else 0
        lines.append(f"  {status:<12} {count:5}  ({pct:5.1f}%)")

    lines += [
        "",
        "─" * 72,
        "  WEAPON CLASS DISTRIBUTION",
        "─" * 72,
    ]
    for wclass, count in sorted(coverage_after["weapon_class"].items(), key=lambda x: -x[1]):
        pct = 100 * count / len(curated) if curated else 0
        lines.append(f"  {wclass:<12} {count:5}  ({pct:5.1f}%)")

    lines += [
        "",
        "─" * 72,
        "  HEALTH DISTRIBUTION",
        "─" * 72,
    ]
    for bucket in [">=90", "80-89", "60-79", "40-59", "<40"]:
        count = coverage_after["health_bucket"].get(bucket, 0)
        pct = 100 * count / len(curated) if curated else 0
        lines.append(f"  {bucket:<8} {count:5}  ({pct:5.1f}%)")

    lines += ["", "=" * 72]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Coverage report written to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a curated SFT dataset from Chimera label files."
    )
    parser.add_argument(
        "--captures-dir",
        default="data/captures",
        help="Root directory containing capture subdirectories (default: data/captures)",
    )
    parser.add_argument(
        "--output",
        default="data/sft_dataset.json",
        help="Output JSON manifest path (default: data/sft_dataset.json)",
    )
    parser.add_argument(
        "--max-5v5-full-health",
        type=int,
        default=40,
        help="Max samples to keep from 5v5 full-health pool (default: 40)",
    )
    parser.add_argument(
        "--min-per-alive-cell",
        type=int,
        default=3,
        help="Warn if any alive-cell has fewer than this many samples (default: 3)",
    )
    parser.add_argument(
        "--report",
        default="data/sft_coverage_report.txt",
        help="Path for human-readable coverage report (default: data/sft_coverage_report.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to script's parent (chimera root) if not absolute
    script_dir = Path(__file__).parent.parent  # scripts/ → chimera root
    captures_dir = (
        args.captures_dir if os.path.isabs(args.captures_dir)
        else str(script_dir / args.captures_dir)
    )
    output_path = (
        args.output if os.path.isabs(args.output)
        else str(script_dir / args.output)
    )
    report_path = (
        args.report if os.path.isabs(args.report)
        else str(script_dir / args.report)
    )

    print(f"[INFO] Captures dir: {captures_dir}")
    print(f"[INFO] Output:       {output_path}")
    print(f"[INFO] Report:       {report_path}")

    # 1. Load all labels
    records = load_labels(captures_dir)

    # 2. Compute coverage before curation
    coverage_before = build_coverage_stats(records)
    print(f"[INFO] Alive-cell coverage before: {len(coverage_before['alive_cell'])} distinct cells")
    print(f"[INFO] Zero-sample cells before: {len(coverage_before['gap_cells'])}")

    # 3. Curate
    curated_records = curate_dataset(records, args.max_5v5_full_health)

    # 4. Compute coverage after curation
    coverage_after = build_coverage_stats(curated_records)
    underrepresented = find_underrepresented(coverage_after, args.min_per_alive_cell)
    if underrepresented:
        print(f"[WARN] {len(underrepresented)} alive cells below min-per-cell threshold "
              f"({args.min_per_alive_cell}):")
        for cell, count in underrepresented:
            if count > 0:
                print(f"       at={cell[0]}, ae={cell[1]}: {count} sample(s)")
    zero_after = coverage_after["gap_cells"]
    if zero_after:
        print(f"[WARN] {len(zero_after)} alive cells have 0 samples after curation: {zero_after}")
    else:
        print("[INFO] All alive-cell combinations have at least 1 sample after curation.")

    # 5. Build output records (filter game_state + generate analysis/advice)
    print("[INFO] Generating templated analysis/advice for each sample...")
    output_records = [build_output_record(r) for r in curated_records]

    # 6. Write output JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_records, f, indent=2)
    print(f"[INFO] SFT dataset written to {output_path} ({len(output_records)} samples)")

    # 7. Write coverage report
    write_coverage_report(
        path=report_path,
        total_input=len(records),
        curated=curated_records,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        underrepresented=underrepresented,
        min_per_cell=args.min_per_alive_cell,
    )

    # 8. Print summary to stdout
    print()
    print("=" * 50)
    print("  Build complete")
    print("=" * 50)
    print(f"  Input labels:         {len(records)}")
    print(f"  Curated samples:      {len(output_records)}")
    print(f"  Samples dropped:      {len(records) - len(output_records)}")
    print(f"  Bomb-planted:         {coverage_after['bomb_status'].get('planted', 0)}")
    print(f"  Zero alive-cells:     {len(zero_after)}")
    print(f"  Under-rep alive-cells: {sum(1 for _, c in underrepresented if c > 0)}")
    print(f"  Output:               {output_path}")
    print(f"  Report:               {report_path}")


if __name__ == "__main__":
    main()
