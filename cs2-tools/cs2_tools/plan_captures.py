#!/usr/bin/env python3
"""
Plan screenshot captures from parsed CS2 demo data.

Reads Parquet tick data + round/kill/bomb JSONs and produces a capture plan:
which ticks to visit, which players to spectate.

Samples every ~3 seconds during active rounds (freeze_end -> end), plus
event ticks (first kill, bomb plant). For each tick, selects 1 T + 1 CT
player to spectate.

Usage:
    python -m cs2_tools.plan_captures data/processed/demos/furia-vs-vitality-m1-mirage
    python -m cs2_tools.plan_captures data/processed/demos/  --all
    python -m cs2_tools.plan_captures data/processed/demos/furia-vs-vitality-m1-mirage --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# Tickrate and sampling
DEFAULT_TICKRATE = 64  # SourceTV tick rate
DEFAULT_INTERVAL_S = 3.0  # seconds between samples
BOMB_POST_PLANT_S = 5.0  # seconds after plant to add a sample

# Map short names for screenshot IDs
MAP_SHORT = {
    "de_mirage": "mirage",
    "de_inferno": "inferno",
    "de_nuke": "nuke",
    "de_overpass": "overpass",
    "de_dust2": "dust2",
    "de_anubis": "anubis",
    "de_ancient": "ancient",
    "de_vertigo": "vertigo",
}


def load_demo_data(stem_path: str) -> tuple[pl.DataFrame, list, list, list, dict]:
    """Load all parsed demo data for a given stem path.

    stem_path: path like 'data/processed/demos/furia-vs-vitality-m1-mirage'
               (without file extension -- we append _ticks.parquet, _rounds.json, etc.)
    """
    stem = Path(stem_path)
    parent = stem.parent
    name = stem.name

    ticks_df = pl.read_parquet(parent / f"{name}_ticks.parquet")
    rounds = json.loads((parent / f"{name}_rounds.json").read_text())
    kills = json.loads((parent / f"{name}_kills.json").read_text())

    bomb_path = parent / f"{name}_bomb.json"
    bomb = json.loads(bomb_path.read_text()) if bomb_path.exists() else []

    header = json.loads((parent / f"{name}_header.json").read_text())

    return ticks_df, rounds, kills, bomb, header


def get_first_kill_tick(kills: list[dict], round_num: int) -> int | None:
    """Get tick of first real kill in a round (skip world/self kills)."""
    for k in kills:
        if k["round_num"] != round_num:
            continue
        if k.get("weapon") == "world":
            continue
        if k.get("attacker_name") == k.get("victim_name"):
            continue
        return k["tick"]
    return None


def get_alive_players(ticks_df: pl.DataFrame, tick: int, round_num: int) -> dict[str, list[str]]:
    """Get alive player names by side at a given tick.

    Returns {"t": ["player1", ...], "ct": ["player2", ...]}.
    Falls back to nearest tick if exact tick not found.
    """
    snap = ticks_df.filter(
        (pl.col("tick") == tick) & (pl.col("round_num") == round_num)
    )
    if snap.is_empty():
        round_ticks = ticks_df.filter(pl.col("round_num") == round_num)
        if round_ticks.is_empty():
            return {"t": [], "ct": []}
        available = round_ticks.select("tick").unique().sort("tick")
        closest = available.filter(pl.col("tick") <= tick).tail(1)
        if closest.is_empty():
            closest = available.head(1)
        tick = closest.item(0, 0)
        snap = ticks_df.filter(
            (pl.col("tick") == tick) & (pl.col("round_num") == round_num)
        )

    alive = {"t": [], "ct": []}
    for row in snap.to_dicts():
        if (row.get("health") or 0) > 0:
            side = row.get("side", "").lower()
            if side in ("t", "ct"):
                alive[side].append(row["name"])
    return alive


def get_bomb_carrier(ticks_df: pl.DataFrame, tick: int, round_num: int) -> str | None:
    """Find the bomb carrier at a given tick (player with C4 in inventory)."""
    snap = ticks_df.filter(
        (pl.col("tick") == tick) & (pl.col("round_num") == round_num)
        & (pl.col("side") == "t") & (pl.col("health") > 0)
    )
    for row in snap.to_dicts():
        inv = row.get("inventory", [])
        if inv and any("C4" in str(item) for item in inv):
            return row["name"]
    return None


def select_pov_players(
    ticks_df: pl.DataFrame, tick: int, round_num: int, bomb_events: list[dict]
) -> list[dict]:
    """Select 1 T + 1 CT player to spectate at a given tick.

    Priority: bomb carrier (T), then first alive player per side.
    """
    alive = get_alive_players(ticks_df, tick, round_num)
    selections = []

    # T-side: prefer bomb carrier
    t_players = alive["t"]
    if t_players:
        carrier = get_bomb_carrier(ticks_df, tick, round_num)
        if carrier and carrier in t_players:
            t_pick = carrier
        else:
            t_pick = t_players[0]
        selections.append({"name": t_pick, "side": "t"})

    # CT-side: first alive
    ct_players = alive["ct"]
    if ct_players:
        selections.append({"name": ct_players[0], "side": "ct"})

    return selections


def plan_demo_captures(
    stem_path: str,
    interval_s: float = DEFAULT_INTERVAL_S,
    tickrate: int = DEFAULT_TICKRATE,
) -> dict:
    """Generate a capture plan for a single demo."""
    ticks_df, rounds, kills, bomb, header = load_demo_data(stem_path)

    stem = Path(stem_path).name
    map_name = header.get("map_name", "unknown")
    map_short = MAP_SHORT.get(map_name, map_name.replace("de_", ""))
    interval_ticks = int(interval_s * tickrate)
    post_plant_ticks = int(BOMB_POST_PLANT_S * tickrate)

    captures = []

    for rnd in rounds:
        round_num = rnd["round_num"]
        freeze_end = rnd["freeze_end"]
        end = rnd["end"]

        # Collect target ticks for this round
        target_ticks = set()

        # Regular interval samples during active play
        tick = freeze_end
        while tick <= end:
            target_ticks.add(tick)
            tick += interval_ticks

        # Event ticks: first kill
        first_kill = get_first_kill_tick(kills, round_num)
        if first_kill and freeze_end <= first_kill <= end:
            target_ticks.add(first_kill)

        # Event ticks: bomb plant + post-plant
        plant_tick = rnd.get("bomb_plant")
        if plant_tick is not None:
            plant_tick = int(plant_tick)
            if freeze_end <= plant_tick <= end:
                target_ticks.add(plant_tick)
                post = plant_tick + post_plant_ticks
                if post <= end:
                    target_ticks.add(post)

        # Sort and generate captures
        for t in sorted(target_ticks):
            pov_players = select_pov_players(ticks_df, t, round_num, bomb)
            for p in pov_players:
                ss_id = f"{map_short}_r{round_num:02d}_t{t:06d}_{p['name']}"
                reason = "sample"
                if first_kill and t == first_kill:
                    reason = "first_kill"
                elif plant_tick and t == plant_tick:
                    reason = "bomb_plant"
                elif plant_tick and t == plant_tick + post_plant_ticks:
                    reason = "post_plant"

                captures.append({
                    "tick": t,
                    "round_num": round_num,
                    "player_name": p["name"],
                    "player_side": p["side"],
                    "reason": reason,
                    "screenshot_id": ss_id,
                })

    plan = {
        "demo_stem": stem,
        "demo_file": f"{stem}.dem",
        "map_name": map_name,
        "map_short": map_short,
        "tickrate": tickrate,
        "interval_s": interval_s,
        "interval_ticks": interval_ticks,
        "total_rounds": len(rounds),
        "total_captures": len(captures),
        "captures": captures,
    }

    return plan


def main():
    parser = argparse.ArgumentParser(
        description="Plan screenshot captures from parsed CS2 demo data"
    )
    parser.add_argument(
        "input",
        help="Path stem (e.g. data/processed/demos/furia-vs-vitality-m1-mirage) "
             "or directory with --all",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all demos in the directory",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S,
        help=f"Seconds between samples (default: {DEFAULT_INTERVAL_S})",
    )
    parser.add_argument(
        "--tickrate", type=int, default=DEFAULT_TICKRATE,
        help=f"Demo tickrate in Hz (default: {DEFAULT_TICKRATE})",
    )
    parser.add_argument(
        "--output", "-o", default="data/captures",
        help="Output directory (default: data/captures)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing files",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    # Collect demo stems to process
    input_path = Path(args.input)
    if args.all and input_path.is_dir():
        parquets = sorted(input_path.glob("*_ticks.parquet"))
        stems = [str(p).replace("_ticks.parquet", "") for p in parquets]
    elif input_path.is_dir():
        print("Use --all to process all demos in a directory, or provide a specific stem path.")
        sys.exit(1)
    else:
        stem_str = str(input_path)
        for suffix in ("_ticks.parquet", "_rounds.json", "_header.json", ".dem"):
            stem_str = stem_str.replace(suffix, "")
        stems = [stem_str]

    total_captures = 0
    for stem in stems:
        name = Path(stem).name
        print(f"Planning captures for {name}...")

        plan = plan_demo_captures(stem, args.interval, args.tickrate)

        print(f"  {plan['map_name']}: {plan['total_rounds']} rounds, "
              f"{plan['total_captures']} captures")

        # Breakdown by reason
        reasons = {}
        for cap in plan["captures"]:
            r = cap["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        for r, count in sorted(reasons.items()):
            print(f"    {r}: {count}")

        total_captures += plan["total_captures"]

        if not args.dry_run:
            plan_dir = output_dir / name
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_path = plan_dir / "capture_plan.json"
            plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
            print(f"  Wrote {plan_path}")

    print(f"\nTotal: {total_captures} captures across {len(stems)} demo(s)")


if __name__ == "__main__":
    main()
