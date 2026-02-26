#!/usr/bin/env python3
"""
Generate SFT labels for captured CS2 screenshots.

For each screenshot in a capture plan, builds a game_state JSON label from
the Parquet tick data. The label schema matches src/prompts.py game_state format.

Can run before or after screenshots are captured (it only needs the capture plan
and the parsed demo data, not the actual images).

Usage:
    python scripts/generate_sft_labels.py data/captures/furia-vs-vitality-m1-mirage/capture_plan.json
    python scripts/generate_sft_labels.py data/captures/ --all
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# Weapon classification for the label format
WEAPON_PRIMARY = {
    "AK-47", "M4A4", "M4A1-S", "AWP", "SSG 08", "Scout",
    "Galil AR", "FAMAS", "SG 553", "AUG",
    "MAC-10", "MP9", "MP7", "UMP-45", "PP-Bizon", "P90",
    "Nova", "XM1014", "Sawed-Off", "MAG-7",
    "M249", "Negev",
}

WEAPON_SECONDARY = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Dual Berettas", "Desert Eagle", "R8 Revolver",
}

UTILITY_ITEMS = {
    "Smoke Grenade", "Flashbang", "Molotov", "Incendiary Grenade",
    "HE Grenade", "High Explosive Grenade", "Decoy Grenade",
}

BOMB_ITEM = "C4 Explosive"


def classify_weapon(item: str) -> str:
    """Classify an inventory item as primary/secondary/utility/melee/bomb."""
    if item in WEAPON_PRIMARY:
        return "primary"
    if item in WEAPON_SECONDARY:
        return "secondary"
    if item in UTILITY_ITEMS:
        return "utility"
    if "C4" in item:
        return "bomb"
    if "Knife" in item or "Bayonet" in item or "Dagger" in item:
        return "melee"
    # Unknown — try fuzzy matching
    for w in WEAPON_PRIMARY:
        if w in item or item in w:
            return "primary"
    for w in WEAPON_SECONDARY:
        if w in item or item in w:
            return "secondary"
    for u in UTILITY_ITEMS:
        if u in item or item in u:
            return "utility"
    return "melee"


def parse_inventory(inventory: list | None) -> dict:
    """Parse inventory list into weapon_primary, weapon_secondary, utility."""
    result = {"weapon_primary": None, "weapon_secondary": None, "utility": []}
    if not inventory:
        return result

    for item in inventory:
        item_str = str(item)
        cat = classify_weapon(item_str)
        if cat == "primary" and result["weapon_primary"] is None:
            result["weapon_primary"] = item_str
        elif cat == "secondary" and result["weapon_secondary"] is None:
            result["weapon_secondary"] = item_str
        elif cat == "utility":
            result["utility"].append(item_str)

    return result


def compute_scores(rounds: list[dict], round_num: int) -> tuple[int, int]:
    """Compute T and CT scores at the start of a round.

    Handles side swap at round 13 (halftime).
    """
    score_t = 0
    score_ct = 0
    for rnd in rounds:
        if rnd["round_num"] >= round_num:
            break
        winner = rnd.get("winner", "").lower()
        if winner == "t":
            score_t += 1
        elif winner == "ct":
            score_ct += 1
    return score_t, score_ct


def determine_bomb_status(bomb_events: list[dict], round_num: int, tick: int) -> str | None:
    """Determine bomb status at a given tick."""
    last_event = None
    for evt in bomb_events:
        if evt["round_num"] != round_num:
            continue
        if evt["tick"] > tick:
            break
        last_event = evt

    if last_event is None:
        return "carried"

    event_type = str(last_event.get("event", "")).lower()
    event_map = {
        "plant": "planted", "planted": "planted", "bomb_planted": "planted",
        "defuse": "defused", "defused": "defused", "bomb_defused": "defused",
        "explode": "exploded", "exploded": "exploded", "bomb_exploded": "exploded",
        "drop": "dropped", "dropped": "dropped",
        "pickup": "carried", "carried": "carried",
    }
    return event_map.get(event_type, "carried")


def determine_round_phase(
    tick: int, round_info: dict, bomb_events: list[dict]
) -> str:
    """Determine round phase at a given tick."""
    freeze_end = round_info["freeze_end"]
    if tick < freeze_end:
        return "freezetime"

    # Check if bomb is planted
    for evt in bomb_events:
        if evt["round_num"] != round_info["round_num"]:
            continue
        if evt.get("event", "").lower() in ("plant", "planted", "bomb_planted"):
            if evt["tick"] <= tick:
                return "post-plant"

    return "playing"


def generate_label(
    cap: dict,
    ticks_df: pl.DataFrame,
    rounds: list[dict],
    bomb_events: list[dict],
    header: dict,
) -> dict:
    """Generate a game_state label for a single capture."""
    tick = cap["tick"]
    round_num = cap["round_num"]
    pov_name = cap["player_name"]
    pov_side = cap["player_side"]

    # Get all players at this tick
    snap = ticks_df.filter(
        (pl.col("tick") == tick) & (pl.col("round_num") == round_num)
    )
    if snap.is_empty():
        # Fall back to nearest tick
        round_ticks = ticks_df.filter(pl.col("round_num") == round_num)
        available = round_ticks.select("tick").unique().sort("tick")
        closest = available.filter(pl.col("tick") <= tick).tail(1)
        if closest.is_empty():
            closest = available.head(1)
        actual_tick = closest.item(0, 0)
        snap = ticks_df.filter(
            (pl.col("tick") == actual_tick) & (pl.col("round_num") == round_num)
        )

    players = snap.to_dicts()

    # Find the POV player
    pov = None
    for p in players:
        if p["name"] == pov_name:
            pov = p
            break

    if pov is None:
        # Player not found at this tick — skip
        return None

    # Parse POV player inventory
    inv = parse_inventory(pov.get("inventory"))

    # Count alive players
    alive_same = 0
    alive_other = 0
    teammates = []
    for p in players:
        is_alive = (p.get("health") or 0) > 0
        p_side = (p.get("side") or "").lower()
        if p_side == pov_side:
            if is_alive:
                alive_same += 1
            if p["name"] != pov_name:
                teammates.append({
                    "name": p["name"],
                    "alive": is_alive,
                    "health": p.get("health", 0),
                })
        else:
            if is_alive:
                alive_other += 1

    # Round info
    round_info = None
    for rnd in rounds:
        if rnd["round_num"] == round_num:
            round_info = rnd
            break

    # Scores
    score_t, score_ct = compute_scores(rounds, round_num)

    # Bomb status and round phase
    bomb_status = determine_bomb_status(bomb_events, round_num, tick)
    phase = determine_round_phase(tick, round_info, bomb_events) if round_info else "playing"

    map_name = header.get("map_name", "unknown")

    label = {
        "game_state": {
            "map_name": map_name,
            "round_num": round_num,
            "round_phase": phase,
            "player_name": pov_name,
            "player_side": pov_side.upper(),
            "player_health": pov.get("health", 0),
            "player_armor": pov.get("armor", 0),
            "player_has_helmet": bool(pov.get("has_helmet", False)),
            "player_money": pov.get("balance", 0),
            "weapon_primary": inv["weapon_primary"],
            "weapon_secondary": inv["weapon_secondary"],
            "utility": inv["utility"],
            "has_defuser": bool(pov.get("has_defuser", False)),
            "alive_teammates": alive_same - 1,  # exclude self
            "alive_enemies": alive_other,
            "bomb_status": bomb_status,
            "score_t": score_t,
            "score_ct": score_ct,
        },
        "teammates": teammates,
        "metadata": {
            "demo_file": f"{cap.get('demo_stem', '')}.dem",
            "tick": tick,
            "round_num": round_num,
            "screenshot_id": cap["screenshot_id"],
            "pov_player": pov_name,
            "pov_side": pov_side,
            "capture_reason": cap.get("reason", "sample"),
        },
    }

    return label


def process_plan(plan_path: Path, demo_data_dir: Path, output_dir: Path, dry_run: bool):
    """Generate labels for all captures in a plan."""
    plan = json.loads(plan_path.read_text())
    demo_stem = plan["demo_stem"]

    # Load demo data
    ticks_df = pl.read_parquet(demo_data_dir / f"{demo_stem}_ticks.parquet")
    rounds = json.loads((demo_data_dir / f"{demo_stem}_rounds.json").read_text())

    bomb_path = demo_data_dir / f"{demo_stem}_bomb.json"
    bomb_events = json.loads(bomb_path.read_text()) if bomb_path.exists() else []

    header = json.loads((demo_data_dir / f"{demo_stem}_header.json").read_text())

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for cap in plan["captures"]:
        # Inject demo_stem into capture for label metadata
        cap["demo_stem"] = demo_stem

        label = generate_label(cap, ticks_df, rounds, bomb_events, header)
        if label is None:
            skipped += 1
            continue

        if not dry_run:
            label_path = labels_dir / f"{cap['screenshot_id']}.json"
            label_path.write_text(json.dumps(label, indent=2, ensure_ascii=False))

        generated += 1

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT labels for captured CS2 screenshots"
    )
    parser.add_argument(
        "input",
        help="Path to capture_plan.json, or captures directory with --all",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all capture plans in the directory",
    )
    parser.add_argument(
        "--demo-data-dir",
        default="data/processed/demos",
        help="Directory containing parsed demo data (default: data/processed/demos)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate labels in memory but don't write files",
    )

    args = parser.parse_args()
    demo_data_dir = Path(args.demo_data_dir)
    input_path = Path(args.input)

    if args.all:
        plan_paths = sorted(input_path.glob("*/capture_plan.json"))
    elif input_path.is_file():
        plan_paths = [input_path]
    else:
        plan_paths = sorted(input_path.glob("capture_plan.json"))
        if not plan_paths:
            plan_paths = sorted(input_path.glob("*/capture_plan.json"))

    if not plan_paths:
        print(f"No capture plans found at {input_path}")
        sys.exit(1)

    total_generated = 0
    total_skipped = 0

    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text())
        demo_stem = plan["demo_stem"]
        output_dir = plan_path.parent

        print(f"Generating labels for {demo_stem}...")
        generated, skipped = process_plan(plan_path, demo_data_dir, output_dir, args.dry_run)
        print(f"  {generated} labels generated, {skipped} skipped")

        total_generated += generated
        total_skipped += skipped

    print(f"\nTotal: {total_generated} labels generated, {total_skipped} skipped")
    if not args.dry_run:
        print(f"Labels written to: {input_path}/*/labels/")


if __name__ == "__main__":
    main()
