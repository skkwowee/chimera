#!/usr/bin/env python3
"""
Parse CS2 .dem files into Parquet tick data + metadata JSONs.

Default mode saves full tick-level data as Parquet plus per-demo JSON files
for rounds, kills, bomb events, and header info.

Legacy --snapshots flag generates the old 3-moment JSON snapshots.

Usage:
    python scripts/parse_demos.py data/demos/                          # full parse → parquet
    python scripts/parse_demos.py data/demos/match.dem --dry-run       # just print stats
    python scripts/parse_demos.py data/demos/ --output data/processed/demos
    python scripts/parse_demos.py data/demos/ --snapshots              # legacy JSON snapshots
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.manifest import append_to_manifest

VALID_ROUND_PHASES = {"buy", "playing", "freezetime", "post-plant", "warmup"}
VALID_BOMB_STATUSES = {"carried", "planted", "dropped", "defused", "exploded", None}

try:
    from awpy import Demo
except ImportError:
    print("awpy is required: pip install awpy>=2.0.0")
    sys.exit(1)

VALID_MOMENTS = {"pre_round", "first_contact", "post_plant"}

PLAYER_PROPS = [
    "X", "Y", "Z",
    "health", "armor_value",
    "has_helmet", "has_defuser",
    "inventory",
    "current_equip_value",
    "balance",
]

# Columns to keep in parquet (rename armor_value -> armor at write time)
TICK_COLUMNS = [
    "tick", "round_num", "name", "steamid", "side",
    "X", "Y", "Z", "health", "armor",
    "has_helmet", "has_defuser", "inventory",
    "current_equip_value", "balance",
]


# ---------------------------------------------------------------------------
# Parquet mode (default)
# ---------------------------------------------------------------------------

def parse_demo_parquet(dem_path: Path, output_dir: Path, dry_run: bool) -> dict:
    """Parse a demo file and save as parquet + metadata JSONs.

    Returns a stats dict for reporting.
    """
    stem = dem_path.stem
    print(f"  Parsing {dem_path.name}...")

    dem = Demo(str(dem_path))
    dem.parse(player_props=PLAYER_PROPS)

    header = dem.header if isinstance(dem.header, dict) else {}
    map_name = header.get("map_name", "unknown")
    ticks_df = dem.ticks
    rounds_df = dem.rounds
    kills_df = dem.kills
    bomb_df = dem.bomb

    tick_count = ticks_df.shape[0] if ticks_df is not None else 0
    round_count = rounds_df.shape[0] if rounds_df is not None else 0
    kill_count = kills_df.shape[0] if kills_df is not None else 0
    bomb_count = bomb_df.shape[0] if bomb_df is not None else 0

    stats = {
        "demo": dem_path.name,
        "map": map_name,
        "ticks": tick_count,
        "rounds": round_count,
        "kills": kill_count,
        "bomb_events": bomb_count,
    }

    print(f"    {map_name}: {tick_count:,} ticks, {round_count} rounds, "
          f"{kill_count} kills, {bomb_count} bomb events")

    if dry_run:
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Ticks parquet ---
    if ticks_df is not None and not ticks_df.is_empty():
        # awpy returns 'armor' (from armor_value prop) — keep as-is
        # Select and order columns that exist
        available = set(ticks_df.columns)
        select_cols = [c for c in TICK_COLUMNS if c in available]
        out_df = ticks_df.select(select_cols)
        parquet_path = output_dir / f"{stem}_ticks.parquet"
        out_df.write_parquet(parquet_path)
        print(f"    Wrote {parquet_path.name} ({out_df.shape[0]:,} rows)")

    # --- Rounds JSON ---
    if rounds_df is not None and not rounds_df.is_empty():
        rounds_path = output_dir / f"{stem}_rounds.json"
        # Convert polars df to list of dicts, handling special types
        rounds_data = json.loads(rounds_df.write_json())
        rounds_path.write_text(json.dumps(rounds_data, indent=2, ensure_ascii=False))
        print(f"    Wrote {rounds_path.name} ({round_count} rounds)")

    # --- Kills JSON ---
    if kills_df is not None and not kills_df.is_empty():
        # Keep only the most useful kill columns
        kill_cols = [
            "tick", "round_num", "attacker_name", "attacker_steamid",
            "attacker_side", "attacker_X", "attacker_Y", "attacker_Z",
            "victim_name", "victim_steamid", "victim_side",
            "victim_X", "victim_Y", "victim_Z",
            "weapon", "headshot", "penetrated", "assistedflash",
            "assister_name", "assister_side",
        ]
        available = set(kills_df.columns)
        kill_select = [c for c in kill_cols if c in available]
        kills_slim = kills_df.select(kill_select)
        kills_path = output_dir / f"{stem}_kills.json"
        kills_data = json.loads(kills_slim.write_json())
        kills_path.write_text(json.dumps(kills_data, indent=2, ensure_ascii=False))
        print(f"    Wrote {kills_path.name} ({kill_count} kills)")

    # --- Bomb JSON ---
    if bomb_df is not None and not bomb_df.is_empty():
        bomb_path = output_dir / f"{stem}_bomb.json"
        bomb_data = json.loads(bomb_df.write_json())
        bomb_path.write_text(json.dumps(bomb_data, indent=2, ensure_ascii=False))
        print(f"    Wrote {bomb_path.name} ({bomb_count} events)")

    # --- Header JSON ---
    header_path = output_dir / f"{stem}_header.json"
    header_path.write_text(json.dumps(header, indent=2, ensure_ascii=False))
    print(f"    Wrote {header_path.name}")

    return stats


# ---------------------------------------------------------------------------
# Legacy snapshot mode (--snapshots)
# ---------------------------------------------------------------------------

def classify_inventory(inventory: list | None) -> list[str]:
    if not inventory or not isinstance(inventory, list):
        return []
    return [str(item) for item in inventory if item]


def extract_player_state(row: dict) -> dict:
    inventory = classify_inventory(row.get("inventory"))
    return {
        "name": row.get("name", "unknown"),
        "health": row.get("health", 0),
        "armor": row.get("armor_value", row.get("armor", 0)),
        "has_helmet": bool(row.get("has_helmet", False)),
        "has_defuser": bool(row.get("has_defuser", False)),
        "is_alive": (row.get("health", 0) or 0) > 0,
        "position": {
            "x": round(float(row.get("X", 0) or 0), 1),
            "y": round(float(row.get("Y", 0) or 0), 1),
            "z": round(float(row.get("Z", 0) or 0), 1),
        },
        "inventory": inventory,
        "equip_value": int(row.get("current_equip_value", 0) or 0),
        "cash_spent": 0,
    }


def get_players_at_tick(ticks_df: pl.DataFrame, tick: int, round_num: int) -> dict:
    snap = ticks_df.filter(
        (pl.col("tick") == tick) & (pl.col("round_num") == round_num)
    )
    if snap.is_empty():
        round_ticks = ticks_df.filter(pl.col("round_num") == round_num)
        if round_ticks.is_empty():
            return {"T": {"players": []}, "CT": {"players": []}}
        available_ticks = round_ticks.select("tick").unique().sort("tick")
        closest = available_ticks.filter(pl.col("tick") <= tick).tail(1)
        if closest.is_empty():
            closest = available_ticks.head(1)
        tick = closest.item(0, 0)
        snap = round_ticks.filter(pl.col("tick") == tick)

    teams = {"T": {"players": []}, "CT": {"players": []}}
    for row in snap.to_dicts():
        side = row.get("side", "").upper()
        if side in ("T", "CT"):
            teams[side]["players"].append(extract_player_state(row))
    return teams


def determine_bomb_status(bomb_df: pl.DataFrame, round_num: int, tick: int) -> str:
    if bomb_df is None or bomb_df.is_empty():
        return "carried"
    round_bomb = bomb_df.filter(
        (pl.col("round_num") == round_num) & (pl.col("tick") <= tick)
    )
    if round_bomb.is_empty():
        return "carried"
    last_event = round_bomb.sort("tick").tail(1).to_dicts()[0]
    event = str(last_event.get("event", last_event.get("status", ""))).lower()
    event_map = {
        "plant": "planted", "planted": "planted", "bomb_planted": "planted",
        "defuse": "defused", "defused": "defused", "bomb_defused": "defused",
        "explode": "exploded", "exploded": "exploded", "bomb_exploded": "exploded",
        "drop": "dropped", "dropped": "dropped",
        "pickup": "carried", "carried": "carried",
    }
    status = event_map.get(event, "carried")
    return status if status in VALID_BOMB_STATUSES else "carried"


def determine_round_phase(moment_type: str) -> str:
    phase_map = {
        "pre_round": "freezetime",
        "first_contact": "playing",
        "post_plant": "post-plant",
    }
    phase = phase_map.get(moment_type, "playing")
    assert phase in VALID_ROUND_PHASES, f"Invalid phase: {phase}"
    return phase


def normalize_bomb_site(raw_site: str | None) -> str | None:
    if not raw_site or raw_site == "not_planted":
        return None
    site = str(raw_site).upper()
    if "A" in site:
        return "A"
    if "B" in site:
        return "B"
    return site


def normalize_win_reason(reason: str | None) -> str:
    if not reason:
        return "Unknown"
    reason_map = {
        "bomb_exploded": "BombExploded",
        "bomb_defused": "BombDefused",
        "ct_killed": "TKilledAll",
        "t_killed": "CTKilledAll",
        "ct_win": "CTWin",
        "t_win": "TWin",
        "target_saved": "TargetSaved",
    }
    return reason_map.get(reason.lower().strip(), reason)


def build_snapshot(demo_file, map_name, round_info, moment_type, tick, teams, bomb_df):
    round_num = round_info["round_num"]
    snapshot_id = f"{Path(demo_file).stem}_r{round_num:02d}_{moment_type}"
    bomb_planted = round_info.get("bomb_plant") is not None and round_info.get(
        "bomb_site", "not_planted"
    ) != "not_planted"
    alive_t = sum(1 for p in teams.get("T", {}).get("players", []) if p["is_alive"])
    alive_ct = sum(1 for p in teams.get("CT", {}).get("players", []) if p["is_alive"])
    return {
        "metadata": {
            "demo_file": Path(demo_file).name,
            "map_name": map_name,
            "round_num": int(round_num),
            "moment_type": moment_type,
            "tick": int(tick),
            "snapshot_id": snapshot_id,
        },
        "round_outcome": {
            "winner": round_info.get("winner", "Unknown").upper(),
            "reason": normalize_win_reason(round_info.get("reason")),
            "bomb_planted": bomb_planted,
            "bomb_site": normalize_bomb_site(round_info.get("bomb_site")),
        },
        "teams": teams,
        "context": {
            "alive_T": alive_t,
            "alive_CT": alive_ct,
            "bomb_status": determine_bomb_status(bomb_df, round_num, tick),
            "round_phase": determine_round_phase(moment_type),
        },
    }


def get_first_kill_tick(kills_df: pl.DataFrame, round_num: int) -> int | None:
    if kills_df is None or kills_df.is_empty():
        return None
    round_kills = kills_df.filter(pl.col("round_num") == round_num)
    if round_kills.is_empty():
        return None
    return int(round_kills.sort("tick").head(1).item(0, "tick"))


def get_bomb_plant_tick(round_info: dict) -> int | None:
    plant_tick = round_info.get("bomb_plant")
    if plant_tick is None:
        return None
    try:
        tick = int(plant_tick)
        return tick if tick > 0 else None
    except (TypeError, ValueError):
        return None


def parse_demo_snapshots(dem_path: Path, moments: set[str]) -> list[dict]:
    """Parse a single .dem file and extract snapshots (legacy mode)."""
    print(f"  Parsing {dem_path.name} (snapshot mode)...")
    dem = Demo(str(dem_path))
    dem.parse(player_props=PLAYER_PROPS)

    header = dem.header if isinstance(dem.header, dict) else {}
    map_name = header.get("map_name", "unknown")
    rounds_df, kills_df, ticks_df, bomb_df = dem.rounds, dem.kills, dem.ticks, dem.bomb

    if rounds_df is None or rounds_df.is_empty():
        print(f"    No rounds found in {dem_path.name}")
        return []
    if ticks_df is None or ticks_df.is_empty():
        print(f"    No tick data found in {dem_path.name}")
        return []

    snapshots = []
    for round_info in rounds_df.to_dicts():
        round_num = round_info["round_num"]
        if "pre_round" in moments:
            freeze_end = round_info.get("freeze_end")
            if freeze_end is not None:
                tick = int(freeze_end)
                teams = get_players_at_tick(ticks_df, tick, round_num)
                snapshots.append(build_snapshot(
                    str(dem_path), map_name, round_info, "pre_round", tick, teams, bomb_df))
        if "first_contact" in moments:
            first_kill = get_first_kill_tick(kills_df, round_num)
            if first_kill is not None:
                teams = get_players_at_tick(ticks_df, first_kill, round_num)
                snapshots.append(build_snapshot(
                    str(dem_path), map_name, round_info, "first_contact", first_kill, teams, bomb_df))
        if "post_plant" in moments:
            plant_tick = get_bomb_plant_tick(round_info)
            if plant_tick is not None:
                teams = get_players_at_tick(ticks_df, plant_tick, round_num)
                snapshots.append(build_snapshot(
                    str(dem_path), map_name, round_info, "post_plant", plant_tick, teams, bomb_df))

    print(f"    Extracted {len(snapshots)} snapshots from {rounds_df.shape[0]} rounds")
    return snapshots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse CS2 .dem files into Parquet tick data + metadata JSONs"
    )
    parser.add_argument(
        "input",
        help="Path to a .dem file or directory of .dem files",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/demos",
        help="Output directory (default: data/processed/demos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing files",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Legacy mode: generate 3-moment JSON snapshots instead of parquet",
    )
    parser.add_argument(
        "--moments", "-m",
        default="pre_round,first_contact,post_plant",
        help="(Snapshot mode only) Comma-separated moment types",
    )

    args = parser.parse_args()

    # Collect .dem files
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix == ".dem":
        dem_files = [input_path]
    elif input_path.is_dir():
        dem_files = sorted(input_path.glob("*.dem"))
    else:
        print(f"Input must be a .dem file or directory: {args.input}")
        sys.exit(1)

    if not dem_files:
        print(f"No .dem files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(dem_files)} demo file(s)")
    output_dir = Path(args.output)

    # --- Legacy snapshot mode ---
    if args.snapshots:
        moments = {m.strip() for m in args.moments.split(",")}
        invalid = moments - VALID_MOMENTS
        if invalid:
            print(f"Invalid moment types: {invalid}")
            sys.exit(1)
        print(f"Moments: {', '.join(sorted(moments))}")

        all_snapshots = []
        for dem_path in dem_files:
            try:
                all_snapshots.extend(parse_demo_snapshots(dem_path, moments))
            except Exception as e:
                print(f"  ERROR parsing {dem_path.name}: {e}")

        print(f"\nTotal: {len(all_snapshots)} snapshots")
        if args.dry_run:
            by_moment = {}
            for snap in all_snapshots:
                mt = snap["metadata"]["moment_type"]
                by_moment[mt] = by_moment.get(mt, 0) + 1
            print("\nBreakdown by moment type:")
            for mt, count in sorted(by_moment.items()):
                print(f"  {mt}: {count}")
            if all_snapshots:
                print("\nSample snapshot:")
                print(json.dumps(all_snapshots[0], indent=2))
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = Path("data/manifest.jsonl")
        written = 0
        for snap in all_snapshots:
            snapshot_id = snap["metadata"]["snapshot_id"]
            out_path = output_dir / f"{snapshot_id}.json"
            out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False))
            append_to_manifest(manifest_path, {
                "id": snapshot_id, "source": "demo",
                "demo_file": snap["metadata"]["demo_file"],
                "map_name": snap["metadata"]["map_name"],
                "round_num": snap["metadata"]["round_num"],
                "moment_type": snap["metadata"]["moment_type"],
                "tick": snap["metadata"]["tick"],
                "round_winner": snap["round_outcome"]["winner"],
            })
            written += 1
        print(f"\nWrote {written} snapshot files to {output_dir}/")
        return

    # --- Default parquet mode ---
    all_stats = []
    for dem_path in dem_files:
        try:
            stats = parse_demo_parquet(dem_path, output_dir, args.dry_run)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR parsing {dem_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {len(all_stats)} demo(s) processed")
    total_ticks = sum(s["ticks"] for s in all_stats)
    total_rounds = sum(s["rounds"] for s in all_stats)
    total_kills = sum(s["kills"] for s in all_stats)
    print(f"  Total ticks:  {total_ticks:,}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Total kills:  {total_kills}")
    if not args.dry_run:
        print(f"  Output dir:   {output_dir}/")


if __name__ == "__main__":
    main()
