#!/usr/bin/env python3
"""
Extract GRPO training samples from parsed CS2 demo data.

For each alive player at sampled post-plant ticks, builds:
  - A text prompt (context string c_t) from generate_sft_labels
  - Ground truth with game state, behavioral features, outcome, and contribution

This is the F05b smoke test data extraction script.

Usage:
    python scripts/extract_grpo_samples.py
    python scripts/extract_grpo_samples.py --demo-data-dir data/processed/demos
    python scripts/extract_grpo_samples.py --output data/training/grpo/smoke_test.jsonl
    python scripts/extract_grpo_samples.py --delta 900 --sample-interval 192
    python scripts/extract_grpo_samples.py --all-rounds
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.generate_sft_labels import (
    _tick_to_time,
    compute_scores,
    determine_bomb_status,
)
from src.utils.cs2 import (
    UTILITY_ITEMS,
    classify_team_economy,
    parse_inventory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKRATE = 64
DEFAULT_DELTA = 900  # ~14s lookahead at 64Hz
DEFAULT_SAMPLE_INTERVAL = 192  # ~3s at 64Hz
MOVEMENT_THRESHOLD = 100  # units for advance/retreat/toward/away


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_nearest_tick(available_ticks: list[int], target: int) -> int:
    """Find the available tick closest to target via binary search."""
    if not available_ticks:
        return target
    lo, hi = 0, len(available_ticks) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if available_ticks[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best = available_ticks[lo]
    if lo > 0 and abs(available_ticks[lo - 1] - target) < abs(best - target):
        best = available_ticks[lo - 1]
    return best


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_snap_dicts(ticks_df: pl.DataFrame, round_num: int, tick: int) -> list[dict[str, Any]]:
    """Get player snapshots at a specific tick."""
    return ticks_df.filter(
        (pl.col("round_num") == round_num) & (pl.col("tick") == tick)
    ).to_dicts()


def get_bomb_plant_pos(bomb_events: list[dict[str, Any]], round_num: int) -> tuple[float, float, float] | None:
    """Get bomb plant XYZ from bomb events for this round."""
    for evt in bomb_events:
        if evt["round_num"] != round_num:
            continue
        if str(evt.get("event", "")).lower() in ("plant", "planted", "bomb_planted"):
            return (evt["X"], evt["Y"], evt["Z"])
    return None


# ---------------------------------------------------------------------------
# Fast round event precomputation (avoids O(ticks^2) per sample)
# ---------------------------------------------------------------------------

def precompute_round_events(
    kills: list[dict[str, Any]],
    bomb_events: list[dict[str, Any]],
    round_num: int,
    freeze_end: int,
) -> list[dict[str, Any]]:
    """
    Precompute round events from kills and bomb events JSON (fast path).

    Instead of scanning tick-by-tick health transitions, we use the kills
    and bomb_events JSON which are already parsed. This is O(events)
    rather than O(ticks^2).
    """
    events = []

    # Deaths from kills JSON
    for k in kills:
        if k["round_num"] != round_num:
            continue
        # Skip warmup self-kills
        if k.get("weapon") == "world" and k["attacker_name"] == k["victim_name"]:
            continue
        time_s = _tick_to_time(k["tick"], freeze_end)
        side = (k.get("victim_side") or "").upper()
        events.append({
            "tick": k["tick"],
            "time_s": time_s,
            "type": "death",
            "player": k["victim_name"],
            "side": side,
            "description": f"{k['victim_name']} ({side}) eliminated",
        })

    # Bomb events
    for evt in bomb_events:
        if evt["round_num"] != round_num:
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

    events.sort(key=lambda e: e["tick"])
    return events


def build_context_fast(
    tick: int,
    round_num: int,
    pov_name: str,
    pov_side: str,
    current_players: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    bomb_events: list[dict[str, Any]],
    header: dict[str, Any],
    round_info: dict[str, Any],
    precomputed_events: list[dict[str, Any]],
    start_snap: list[dict[str, Any]],
) -> str:
    """
    Fast version of generate_round_context that avoids re-scanning tick data.

    Uses precomputed events and pre-fetched snapshots.
    """
    map_name = header.get("map_name", "unknown")
    score_t, score_ct = compute_scores(rounds, round_num)
    freeze_end = round_info["freeze_end"]

    # Economy at round start
    t_buy, t_equip = classify_team_economy(start_snap, "t")
    ct_buy, ct_equip = classify_team_economy(start_snap, "ct")

    # Filter events up to this tick
    events = [e for e in precomputed_events if e["tick"] <= tick]

    # Build context string
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

    # Current state
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
        alive_mates = [t for t in teammates if (t.get("health") or 0) > 0]
        dead_mates = [t for t in teammates if (t.get("health") or 0) == 0]
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
# Behavioral feature extraction
# ---------------------------------------------------------------------------

def compute_movement_direction(
    player_t: dict[str, Any],
    player_t_delta: dict[str, Any] | None,
    enemies_t: list[dict[str, Any]],
) -> int:
    """Compute d_move: -1=retreat, 0=hold, 1=advance relative to enemies."""
    if not enemies_t or player_t_delta is None:
        return 0

    alive_enemies = [e for e in enemies_t if (e.get("health") or 0) > 0]
    if not alive_enemies:
        return 0

    mean_ex = sum(e["X"] for e in alive_enemies) / len(alive_enemies)
    mean_ey = sum(e["Y"] for e in alive_enemies) / len(alive_enemies)

    dist_t = distance_2d(player_t["X"], player_t["Y"], mean_ex, mean_ey)
    dist_td = distance_2d(player_t_delta["X"], player_t_delta["Y"], mean_ex, mean_ey)

    diff = dist_t - dist_td  # positive = got closer = advance
    if diff > MOVEMENT_THRESHOLD:
        return 1
    elif diff < -MOVEMENT_THRESHOLD:
        return -1
    return 0


def compute_objective_direction(
    player_t: dict[str, Any],
    player_t_delta: dict[str, Any] | None,
    bomb_pos: tuple[float, float, float] | None,
) -> int:
    """Compute d_obj: -1=away, 0=neutral, 1=toward bomb."""
    if bomb_pos is None or player_t_delta is None:
        return 0

    bx, by, _ = bomb_pos
    dist_t = distance_2d(player_t["X"], player_t["Y"], bx, by)
    dist_td = distance_2d(player_t_delta["X"], player_t_delta["Y"], bx, by)

    diff = dist_t - dist_td  # positive = got closer = toward
    if diff > MOVEMENT_THRESHOLD:
        return 1
    elif diff < -MOVEMENT_THRESHOLD:
        return -1
    return 0


def compute_utility_used(player_t: dict[str, Any], player_t_delta: dict[str, Any] | None) -> list[str]:
    """Detect grenades that disappeared from inventory between t and t+delta."""
    if player_t_delta is None:
        return []

    inv_t = set(str(x) for x in (player_t.get("inventory") or []))
    inv_td = set(str(x) for x in (player_t_delta.get("inventory") or []))

    lost = inv_t - inv_td
    return sorted(item for item in lost if item in UTILITY_ITEMS)


def compute_engagement_features(
    player_name: str,
    tick: int,
    delta: int,
    round_damages: list[dict[str, Any]],
) -> tuple[bool, float]:
    """
    Compute initiated_engagement and engagement_delay.

    Returns (initiated_engagement, engagement_delay).
    """
    t_end = tick + delta

    first_dealt = None
    first_received = None

    for dmg in round_damages:
        dtick = dmg["tick"]
        if dtick < tick or dtick > t_end:
            continue
        if dmg["attacker_name"] == dmg["victim_name"]:
            continue

        if dmg["attacker_name"] == player_name and first_dealt is None:
            first_dealt = dtick
        if dmg["victim_name"] == player_name and first_received is None:
            first_received = dtick

        if first_dealt is not None and first_received is not None:
            break

    if first_dealt is not None and first_received is not None:
        initiated = first_dealt <= first_received
    elif first_dealt is not None:
        initiated = True
    else:
        initiated = False

    first_any = None
    if first_dealt is not None and first_received is not None:
        first_any = min(first_dealt, first_received)
    elif first_dealt is not None:
        first_any = first_dealt
    elif first_received is not None:
        first_any = first_received

    if first_any is not None:
        delay = (first_any - tick) / delta if delta > 0 else 0.0
        delay = max(0.0, min(1.0, delay))
    else:
        delay = 1.0

    return initiated, delay


# ---------------------------------------------------------------------------
# Action description and categories
# ---------------------------------------------------------------------------

def describe_action(
    movement_dir: int,
    objective_dir: int,
    utility_used: list[str],
    initiated: bool,
    player_side: str,
) -> tuple[list[str], str]:
    """Generate coarse categories and text description of the pro's play."""
    categories = []
    desc_parts = []

    if movement_dir == 1:
        categories.append("aggressive")
        desc_parts.append("pushed toward enemies")
    elif movement_dir == -1:
        categories.append("fall_back")
        desc_parts.append("fell back from enemies")
    else:
        categories.append("hold")
        desc_parts.append("held position")

    if objective_dir == 1:
        if player_side == "ct":
            desc_parts.append("moved toward bomb")
            categories.append("rotate")
        else:
            desc_parts.append("stayed near bomb")
    elif objective_dir == -1:
        if player_side == "ct":
            desc_parts.append("moved away from bomb")
        else:
            desc_parts.append("moved away from planted bomb")
            categories.append("rotate")

    if utility_used:
        categories.append("utility")
        desc_parts.append(f"used {', '.join(utility_used)}")

    if initiated:
        categories.append("engage")
        desc_parts.append("initiated engagement")

    description = "; ".join(desc_parts) if desc_parts else "minimal action"
    return categories, description


# ---------------------------------------------------------------------------
# Player contribution (precomputed per round)
# ---------------------------------------------------------------------------

def precompute_round_contributions(
    round_num: int,
    damages: list[dict[str, Any]],
    bomb_events: list[dict[str, Any]],
    kills: list[dict[str, Any]],
    freeze_end: int,
    round_end: int,
    winner: str,
    end_snap: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, float]:
    """Precompute player contribution for all players in a round."""

    round_damages = [
        d for d in damages
        if d["round_num"] == round_num
        and d["attacker_name"] != d["victim_name"]
    ]

    per_player_damage = defaultdict(float)
    for d in round_damages:
        per_player_damage[d["attacker_name"]] += d.get("dmg_health_real", 0) or 0
    max_round_damage = max(per_player_damage.values()) if per_player_damage else 1.0

    round_duration = max(1, round_end - freeze_end)

    death_ticks = {}
    for k in kills:
        if k["round_num"] != round_num:
            continue
        if k["victim_name"] not in death_ticks:
            death_ticks[k["victim_name"]] = k["tick"]

    bomb_actors = set()
    for evt in bomb_events:
        if evt["round_num"] != round_num:
            continue
        event_type = str(evt.get("event", "")).lower()
        if event_type in ("plant", "planted", "bomb_planted", "defuse", "defused", "bomb_defused") and evt.get("name"):
                bomb_actors.add(evt["name"])

    last_alive_name = None
    if end_snap:
        alive_winners = [
            p for p in end_snap
            if (p.get("side") or "").lower() == winner
            and (p.get("health") or 0) > 0
        ]
        if len(alive_winners) == 1:
            last_alive_name = alive_winners[0]["name"]

    # Collect all player names
    all_players = set()
    for d in round_damages:
        all_players.add(d["attacker_name"])
        all_players.add(d["victim_name"])
    for k in kills:
        if k["round_num"] == round_num:
            all_players.add(k["victim_name"])
            all_players.add(k["attacker_name"])
    # Also add players from end snap (may not have damage/kill events)
    for p in end_snap:
        all_players.add(p["name"])

    contributions = {}
    for name in all_players:
        damage_dealt = per_player_damage.get(name, 0.0)

        dt = death_ticks.get(name)
        survival_time = max(0, dt - freeze_end) if dt is not None else round_duration

        objective_action = name in bomb_actors or name == last_alive_name

        contributions[name] = {
            "damage_dealt": float(damage_dealt),
            "max_round_damage": float(max_round_damage),
            "survival_time": float(survival_time),
            "round_duration": float(round_duration),
            "objective_action": objective_action,
        }

    return contributions, max_round_damage, round_duration


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_grpo_samples(
    demo_data_dir: Path,
    delta: int = DEFAULT_DELTA,
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL,
    all_rounds: bool = False,
) -> list[dict[str, Any]]:
    """Extract GRPO samples from all demos in demo_data_dir."""

    parquet_files = sorted(demo_data_dir.glob("*_ticks.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {demo_data_dir}")
        return []

    all_samples = []

    for pq_path in parquet_files:
        demo_stem = pq_path.stem.replace("_ticks", "")
        print(f"Processing {demo_stem}...")

        ticks_df = pl.read_parquet(pq_path)

        rounds_path = demo_data_dir / f"{demo_stem}_rounds.json"
        bomb_path = demo_data_dir / f"{demo_stem}_bomb.json"
        header_path = demo_data_dir / f"{demo_stem}_header.json"
        damages_path = demo_data_dir / f"{demo_stem}_damages.json"
        kills_path = demo_data_dir / f"{demo_stem}_kills.json"

        rounds = json.loads(rounds_path.read_text())
        bomb_events = json.loads(bomb_path.read_text()) if bomb_path.exists() else []
        header = json.loads(header_path.read_text())
        damages = json.loads(damages_path.read_text()) if damages_path.exists() else []
        kills = json.loads(kills_path.read_text()) if kills_path.exists() else []

        map_name = header.get("map_name", "unknown")

        # Index by round for efficiency
        round_outcomes = {r["round_num"]: r.get("winner", "").lower() for r in rounds}
        damages_by_round = defaultdict(list)
        for d in damages:
            damages_by_round[d["round_num"]].append(d)

        demo_sample_count = 0

        for rnd in rounds:
            rnum = rnd["round_num"]
            freeze_end = rnd["freeze_end"]
            round_end = rnd["end"]
            winner = round_outcomes.get(rnum, "")
            bomb_plant_tick = rnd.get("bomb_plant")

            if not all_rounds and bomb_plant_tick is None:
                continue

            # Get all available ticks for this round
            round_available = (
                ticks_df.filter(pl.col("round_num") == rnum)
                .select("tick").unique().sort("tick")
                .to_series().to_list()
            )
            if not round_available:
                continue

            # Determine sampling start tick
            start_from = freeze_end if all_rounds else bomb_plant_tick
            post_ticks = [t for t in round_available if t >= start_from]
            if not post_ticks:
                continue

            # Sample at intervals
            sampled_ticks = []
            target = post_ticks[0]
            while target <= post_ticks[-1]:
                nearest = find_nearest_tick(post_ticks, target)
                if nearest not in sampled_ticks:
                    sampled_ticks.append(nearest)
                target += sample_interval

            # --- Precompute expensive things per round (not per sample) ---

            # Round events for context string
            round_events = precompute_round_events(kills, bomb_events, rnum, freeze_end)

            # Start-of-round snapshot for economy
            post_freeze = [t for t in round_available if t >= freeze_end]
            start_tick = post_freeze[0] if post_freeze else round_available[-1]
            start_snap = get_snap_dicts(ticks_df, rnum, start_tick)

            # End-of-round snapshot for last-alive check
            end_tick = find_nearest_tick(round_available, round_end)
            end_snap = get_snap_dicts(ticks_df, rnum, end_tick)

            # Player contributions for the full round
            contributions, max_round_damage, round_duration = precompute_round_contributions(
                rnum, damages, bomb_events, kills, freeze_end, round_end, winner, end_snap,
            )

            # Bomb plant position
            bomb_pos = get_bomb_plant_pos(bomb_events, rnum)

            round_damages = damages_by_round.get(rnum, [])

            for tick in sampled_ticks:
                snap = get_snap_dicts(ticks_df, rnum, tick)
                if not snap:
                    continue

                # Get snap at t+delta (clamped to round_end)
                future_tick_target = min(tick + delta, round_end)
                future_tick = find_nearest_tick(round_available, future_tick_target)
                future_snap = get_snap_dicts(ticks_df, rnum, future_tick)
                future_by_name = {p["name"]: p for p in future_snap}

                for player in snap:
                    if (player.get("health") or 0) == 0:
                        continue

                    p_name = player["name"]
                    p_side = (player.get("side") or "").lower()

                    # Enemies at tick t
                    enemies_t = [
                        p for p in snap
                        if (p.get("side") or "").lower() != p_side
                        and (p.get("health") or 0) > 0
                    ]

                    player_future = future_by_name.get(p_name)

                    # Behavioral features
                    movement_dir = compute_movement_direction(player, player_future, enemies_t)
                    objective_dir = compute_objective_direction(player, player_future, bomb_pos)
                    utility_used = compute_utility_used(player, player_future)
                    initiated, eng_delay = compute_engagement_features(
                        p_name, tick, delta, round_damages,
                    )

                    categories, description = describe_action(
                        movement_dir, objective_dir, utility_used, initiated, p_side,
                    )

                    # Game state
                    inv = parse_inventory(player.get("inventory"))
                    alive_same = sum(
                        1 for p in snap
                        if (p.get("side") or "").lower() == p_side
                        and (p.get("health") or 0) > 0
                    )
                    alive_other = sum(
                        1 for p in snap
                        if (p.get("side") or "").lower() != p_side
                        and (p.get("health") or 0) > 0
                    )

                    is_post_plant = bomb_plant_tick is not None and tick >= bomb_plant_tick
                    game_state = {
                        "map_name": map_name,
                        "round_phase": "post-plant" if is_post_plant else "playing",
                        "player_side": p_side.upper(),
                        "player_health": player.get("health", 0),
                        "player_armor": player.get("armor", 0),
                        "player_money": player.get("balance", 0),
                        "weapon_primary": inv["weapon_primary"],
                        "weapon_secondary": inv["weapon_secondary"],
                        "utility": inv["utility"],
                        "alive_teammates": alive_same - 1,
                        "alive_enemies": alive_other,
                        "bomb_status": "planted" if is_post_plant else determine_bomb_status(bomb_events, rnum, tick),
                        "visible_enemies": 0,
                    }

                    # Player contribution (precomputed)
                    contrib = contributions.get(p_name, {
                        "damage_dealt": 0.0,
                        "max_round_damage": float(max_round_damage),
                        "survival_time": float(round_duration),
                        "round_duration": float(round_duration),
                        "objective_action": False,
                    })

                    ground_truth = {
                        "game_state": game_state,
                        "pro_action": {
                            "behavior": {
                                "movement_direction": movement_dir,
                                "objective_direction": objective_dir,
                                "utility_used": utility_used,
                                "initiated_engagement": initiated,
                                "engagement_delay": round(eng_delay, 4),
                            },
                            "categories": categories,
                            "description": description,
                        },
                        "round_won": p_side == winner,
                        "player_contribution": contrib,
                    }

                    # Context string (fast path)
                    context = build_context_fast(
                        tick, rnum, p_name, p_side,
                        snap, rounds, bomb_events, header,
                        rnd, round_events, start_snap,
                    )

                    sample = {
                        "prompt": [{"type": "text", "text": context}],
                        "ground_truth": ground_truth,
                    }
                    all_samples.append(sample)
                    demo_sample_count += 1

        print(f"  {demo_sample_count} samples from {demo_stem}")

    return all_samples


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(samples: list[dict[str, Any]], num_demos: int) -> dict[str, Any]:
    """Build manifest summary."""
    t_count = ct_count = t_wins = ct_wins = 0

    for s in samples:
        gt = s["ground_truth"]
        side = gt["game_state"]["player_side"].lower()
        if side == "t":
            t_count += 1
            if gt["round_won"]:
                t_wins += 1
        else:
            ct_count += 1
            if gt["round_won"]:
                ct_wins += 1

    # Count distinct rounds from prompt headers
    round_ids = set()
    for s in samples:
        first_line = s["prompt"][0]["text"].split("\n")[0]
        round_ids.add(first_line)

    return {
        "created": datetime.now(UTC).isoformat(),
        "demos": num_demos,
        "rounds": len(round_ids),
        "samples": len(samples),
        "samples_per_side": {"t": t_count, "ct": ct_count},
        "win_rate": {
            "t": round(t_wins / max(t_count, 1), 4),
            "ct": round(ct_wins / max(ct_count, 1), 4),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract GRPO training samples from parsed CS2 demo data"
    )
    parser.add_argument(
        "--demo-data-dir", default="data/processed/demos",
        help="Directory containing parsed demo data (default: data/processed/demos)",
    )
    parser.add_argument(
        "--output", default="data/training/grpo/smoke_test.jsonl",
        help="Output JSONL path (default: data/training/grpo/smoke_test.jsonl)",
    )
    parser.add_argument(
        "--delta", type=int, default=DEFAULT_DELTA,
        help=f"Lookahead window in ticks (default: {DEFAULT_DELTA})",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=DEFAULT_SAMPLE_INTERVAL,
        help=f"Sampling interval in ticks (default: {DEFAULT_SAMPLE_INTERVAL})",
    )
    parser.add_argument(
        "--all-rounds", action="store_true",
        help="Extract from all rounds, not just post-plant",
    )
    args = parser.parse_args()

    demo_data_dir = Path(args.demo_data_dir)
    output_path = Path(args.output)

    samples = extract_grpo_samples(
        demo_data_dir,
        delta=args.delta,
        sample_interval=args.sample_interval,
        all_rounds=args.all_rounds,
    )

    if not samples:
        print("No samples extracted!")
        sys.exit(1)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Manifest
    num_demos = len(list(demo_data_dir.glob("*_ticks.parquet")))
    manifest = build_manifest(samples, num_demos)

    manifest_path = output_path.parent / "smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Print stats
    print(f"\n{'='*50}")
    print("GRPO Smoke Test Extraction Complete")
    print(f"{'='*50}")
    print(f"Output:         {output_path}")
    print(f"Manifest:       {manifest_path}")
    print(f"Demos:          {manifest['demos']}")
    print(f"Rounds:         {manifest['rounds']}")
    print(f"Total samples:  {manifest['samples']}")
    print(f"  T-side:       {manifest['samples_per_side']['t']}")
    print(f"  CT-side:      {manifest['samples_per_side']['ct']}")
    print(f"T win rate:     {manifest['win_rate']['t']:.2%}")
    print(f"CT win rate:    {manifest['win_rate']['ct']:.2%}")


if __name__ == "__main__":
    main()
