#!/usr/bin/env python3
"""
Generate SFT labels for captured CS2 screenshots.

For each screenshot in a capture plan, builds:
  1. game_state JSON label from Parquet tick data
  2. Round context string c_t summarizing all events up to that tick
  3. Prior screenshot references for multi-image input

The context string c_t provides the model with full round history:
economy, kills, utility usage, bomb events, and current state.
See D018 in decisions.md for design rationale.

Usage:
    python scripts/generate_sft_labels.py data/captures/furia-vs-vitality-m1-mirage/capture_plan.json
    python scripts/generate_sft_labels.py data/captures/ --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from src.utils.cs2 import (
    UTILITY_ITEMS,
    classify_team_economy,
    parse_inventory,
)

# ---------------------------------------------------------------------------
# Score and phase helpers
# ---------------------------------------------------------------------------

def compute_scores(rounds: list[dict[str, Any]], round_num: int) -> tuple[int, int]:
    """Compute T and CT scores at the start of a round."""
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


def determine_bomb_status(bomb_events: list[dict[str, Any]], round_num: int, tick: int) -> str | None:
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
    tick: int, round_info: dict[str, Any], bomb_events: list[dict[str, Any]]
) -> str:
    """Determine round phase at a given tick."""
    freeze_end = round_info["freeze_end"]
    if tick < freeze_end:
        return "freezetime"

    for evt in bomb_events:
        if evt["round_num"] != round_info["round_num"]:
            continue
        if evt.get("event", "").lower() in ("plant", "planted", "bomb_planted") and evt["tick"] <= tick:
                return "post-plant"

    return "playing"


# ---------------------------------------------------------------------------
# Round event detection from tick data
# ---------------------------------------------------------------------------

def detect_round_events(
    ticks_df: pl.DataFrame,
    round_num: int,
    up_to_tick: int,
    freeze_end: int,
    bomb_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect game events within a round up to a given tick.

    Extracts:
      - Deaths: player health transitions from >0 to 0
      - Utility usage: grenade items disappearing from inventory
      - Bomb events: from the bomb_events list

    Returns list of events sorted by tick, each with:
      {tick, time_s, type, description}
    """
    round_ticks = ticks_df.filter(pl.col("round_num") == round_num)
    available_ticks = (
        round_ticks.select("tick").unique().sort("tick")
        .filter(pl.col("tick") <= up_to_tick)
        .to_series().to_list()
    )

    if len(available_ticks) < 2:
        return []

    events = []

    # --- Detect deaths by health transitions ---
    for i in range(1, len(available_ticks)):
        prev_tick = available_ticks[i - 1]
        curr_tick = available_ticks[i]

        prev_snap = round_ticks.filter(pl.col("tick") == prev_tick)
        curr_snap = round_ticks.filter(pl.col("tick") == curr_tick)

        prev_players = {p["name"]: p for p in prev_snap.to_dicts()}
        curr_players = {p["name"]: p for p in curr_snap.to_dicts()}

        for name, curr_p in curr_players.items():
            prev_p = prev_players.get(name)
            if prev_p is None:
                continue

            prev_health = prev_p.get("health") or 0
            curr_health = curr_p.get("health") or 0

            # Death detected
            if prev_health > 0 and curr_health == 0:
                side = (curr_p.get("side") or "").upper()
                time_s = _tick_to_time(curr_tick, freeze_end)
                events.append({
                    "tick": curr_tick,
                    "time_s": time_s,
                    "type": "death",
                    "player": name,
                    "side": side,
                    "description": f"{name} ({side}) eliminated",
                })

        # --- Detect utility usage by inventory changes ---
        for name, curr_p in curr_players.items():
            prev_p = prev_players.get(name)
            if prev_p is None:
                continue
            if (prev_p.get("health") or 0) == 0:
                continue  # dead players don't use utility

            prev_inv = set(str(x) for x in (prev_p.get("inventory") or []))
            curr_inv = set(str(x) for x in (curr_p.get("inventory") or []))

            lost_items = prev_inv - curr_inv
            for item in lost_items:
                if item in UTILITY_ITEMS:
                    side = (curr_p.get("side") or "").upper()
                    time_s = _tick_to_time(curr_tick, freeze_end)
                    events.append({
                        "tick": curr_tick,
                        "time_s": time_s,
                        "type": "utility",
                        "player": name,
                        "side": side,
                        "item": item,
                        "description": f"{name} ({side}) used {item}",
                    })

    # --- Bomb events from the dedicated event list ---
    for evt in bomb_events:
        if evt["round_num"] != round_num:
            continue
        if evt["tick"] > up_to_tick:
            continue
        event_type = str(evt.get("event", "")).lower()
        time_s = _tick_to_time(evt["tick"], freeze_end)

        if event_type in ("plant", "planted", "bomb_planted"):
            events.append({
                "tick": evt["tick"],
                "time_s": time_s,
                "type": "bomb_planted",
                "description": "Bomb planted",
            })
        elif event_type in ("defuse", "defused", "bomb_defused"):
            events.append({
                "tick": evt["tick"],
                "time_s": time_s,
                "type": "bomb_defused",
                "description": "Bomb defused",
            })
        elif event_type in ("drop", "dropped"):
            events.append({
                "tick": evt["tick"],
                "time_s": time_s,
                "type": "bomb_dropped",
                "description": "Bomb dropped",
            })

    events.sort(key=lambda e: e["tick"])
    return events


def _tick_to_time(tick: int, freeze_end: int, tickrate: int = 64) -> float:
    """Convert tick to seconds since freeze end (round start)."""
    return max(0.0, (tick - freeze_end) / tickrate)


# ---------------------------------------------------------------------------
# Context string generation (c_t)
# ---------------------------------------------------------------------------

def generate_round_context(
    tick: int,
    round_num: int,
    pov_name: str,
    pov_side: str,
    ticks_df: pl.DataFrame,
    rounds: list[dict[str, Any]],
    bomb_events: list[dict[str, Any]],
    header: dict[str, Any],
    strip_current_state: bool = False,
) -> str:
    """
    Generate the round context string c_t for a decision point.

    Summarizes everything that has happened in the round up to tick t:
    round/score info, economy, chronological events (kills, utility, bomb),
    and current player states.

    Args:
        tick: Current tick (decision point).
        round_num: Current round number.
        pov_name: Name of the POV player.
        pov_side: Side of the POV player ("t" or "ct").
        ticks_df: Full ticks DataFrame for the demo.
        rounds: Rounds metadata list.
        bomb_events: Bomb event list.
        header: Demo header dict (contains map_name).
        strip_current_state: When True, omit the CURRENT STATE section (alive
            counts, bomb status, POV player state, teammate states). Use this
            for GRPO training where current state must come from the screenshot.

    Returns:
        Formatted context string.
    """
    map_name = header.get("map_name", "unknown")
    score_t, score_ct = compute_scores(rounds, round_num)

    # Find round info
    round_info = None
    for rnd in rounds:
        if rnd["round_num"] == round_num:
            round_info = rnd
            break

    freeze_end = round_info["freeze_end"] if round_info else tick

    # Get players at freeze_end (start of round) for economy assessment
    freeze_snap = ticks_df.filter(
        pl.col("round_num") == round_num
    )
    available = freeze_snap.select("tick").unique().sort("tick")
    # Get first tick at or after freeze_end
    start_ticks = available.filter(pl.col("tick") >= freeze_end)
    if not start_ticks.is_empty():
        start_tick = start_ticks.item(0, 0)
    elif not available.is_empty():
        start_tick = available.item(-1, 0)
    else:
        start_tick = tick

    start_snap = ticks_df.filter(
        (pl.col("tick") == start_tick) & (pl.col("round_num") == round_num)
    ).to_dicts()

    # Economy at round start
    t_buy, t_equip = classify_team_economy(start_snap, "t")
    ct_buy, ct_equip = classify_team_economy(start_snap, "ct")

    # Get players at current tick for current state (only needed when not stripped)
    current_players = []
    if not strip_current_state:
        curr_snap = ticks_df.filter(
            pl.col("round_num") == round_num
        )
        curr_available = curr_snap.select("tick").unique().sort("tick")
        curr_ticks = curr_available.filter(pl.col("tick") <= tick)
        if not curr_ticks.is_empty():
            actual_tick = curr_ticks.item(-1, 0)
        elif not curr_available.is_empty():
            actual_tick = curr_available.item(0, 0)
        else:
            actual_tick = tick

        current_players = ticks_df.filter(
            (pl.col("tick") == actual_tick) & (pl.col("round_num") == round_num)
        ).to_dicts()

    # Detect events up to this tick
    events = detect_round_events(
        ticks_df, round_num, tick, freeze_end, bomb_events,
    )

    # --- Build context string ---
    lines = []

    # Header
    lines.append(f"Round {round_num} | T {score_t} - {score_ct} CT | {map_name}")
    time_s = _tick_to_time(tick, freeze_end)
    lines.append(f"Round time: {time_s:.0f}s")
    lines.append("")

    # Economy
    lines.append("ECONOMY (round start):")
    lines.append(f"  T-side: {t_buy} (~${t_equip:,})")
    lines.append(f"  CT-side: {ct_buy} (~${ct_equip:,})")
    lines.append("")

    # Events
    if events:
        lines.append("ROUND EVENTS:")
        for evt in events:
            lines.append(f"  {evt['time_s']:5.1f}s — {evt['description']}")
        lines.append("")

    # Current state (omitted when strip_current_state=True for GRPO training)
    if not strip_current_state:
        alive_t = sum(
            1 for p in current_players
            if (p.get("side") or "").lower() == "t" and (p.get("health") or 0) > 0
        )
        alive_ct = sum(
            1 for p in current_players
            if (p.get("side") or "").lower() == "ct" and (p.get("health") or 0) > 0
        )

        bomb_status = determine_bomb_status(bomb_events, round_num, tick)

        pov = None
        for p in current_players:
            if p["name"] == pov_name:
                pov = p
                break

        lines.append("CURRENT STATE:")
        lines.append(f"  Alive: {alive_t}T vs {alive_ct}CT")
        lines.append(f"  Bomb: {bomb_status}")

        if pov:
            inv = parse_inventory(pov.get("inventory"))
            weapon = inv["weapon_primary"] or inv["weapon_secondary"] or "knife"
            util_str = ", ".join(inv["utility"]) if inv["utility"] else "none"
            lines.append(
                f"  POV: {pov_name} ({pov_side.upper()}), "
                + f"{pov.get('health', 0)}hp/{pov.get('armor', 0)}armor, "
                + f"{weapon}, utility: {util_str}"
            )

        # Teammate states
        teammates = [
            p for p in current_players
            if (p.get("side") or "").lower() == pov_side.lower()
            and p["name"] != pov_name
        ]
        if teammates:
            alive_mates = [
                t for t in teammates if (t.get("health") or 0) > 0
            ]
            dead_mates = [
                t for t in teammates if (t.get("health") or 0) == 0
            ]
            if alive_mates:
                mate_strs = []
                for t in alive_mates:
                    t_inv = parse_inventory(t.get("inventory"))
                    t_weapon = t_inv["weapon_primary"] or t_inv["weapon_secondary"] or "knife"
                    mate_strs.append(f"{t['name']} ({t.get('health', 0)}hp, {t_weapon})")
                lines.append(f"  Teammates alive: {', '.join(mate_strs)}")
            if dead_mates:
                lines.append(
                    f"  Teammates dead: {', '.join(t['name'] for t in dead_mates)}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prior screenshot references
# ---------------------------------------------------------------------------

def find_prior_screenshots(
    current_cap: dict[str, Any],
    all_captures: list[dict[str, Any]],
    max_prior: int = 2,
) -> list[str]:
    """
    Find screenshot IDs for prior decision points in the same round
    with the same POV player.

    Returns up to max_prior screenshot IDs, ordered oldest to newest.
    """
    round_num = current_cap["round_num"]
    pov_name = current_cap["player_name"]
    current_tick = current_cap["tick"]

    # Same round, same POV, earlier tick
    prior = [
        c for c in all_captures
        if c["round_num"] == round_num
        and c["player_name"] == pov_name
        and c["tick"] < current_tick
    ]
    prior.sort(key=lambda c: c["tick"])

    # Take the most recent max_prior
    return [c["screenshot_id"] for c in prior[-max_prior:]]


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_label(
    cap: dict[str, Any],
    ticks_df: pl.DataFrame,
    rounds: list[dict[str, Any]],
    bomb_events: list[dict[str, Any]],
    header: dict[str, Any],
    all_captures: list[dict[str, Any]] | None = None,
    strip_current_state: bool = False,
) -> dict[str, Any] | None:
    """Generate a game_state label + context for a single capture.

    Args:
        cap: Capture metadata dict.
        ticks_df: Full ticks DataFrame for the demo.
        rounds: Rounds metadata list.
        bomb_events: Bomb event list.
        header: Demo header dict.
        all_captures: All captures in the plan (for prior screenshot lookup).
        strip_current_state: When True, omit CURRENT STATE from the context
            string c_t. Use for GRPO training where current state must come
            from the screenshot, not the context.
    """
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

    # Generate round context string
    context = generate_round_context(
        tick, round_num, pov_name, pov_side,
        ticks_df, rounds, bomb_events, header,
        strip_current_state=strip_current_state,
    )

    # Find prior screenshots for multi-image input
    prior_screenshots = []
    if all_captures is not None:
        prior_screenshots = find_prior_screenshots(cap, all_captures)

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
        "context": context,
        "prior_screenshots": prior_screenshots,
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

    # Inject demo_stem into all captures for metadata
    all_captures = plan["captures"]
    for cap in all_captures:
        cap["demo_stem"] = demo_stem

    generated = 0
    skipped = 0

    for cap in all_captures:
        label = generate_label(
            cap, ticks_df, rounds, bomb_events, header,
            all_captures=all_captures,
        )
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
