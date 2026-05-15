#!/usr/bin/env python3
"""Build per-tick feature tensors for the Level-2 round encoder.

Reads parsed demos (ticks.parquet + bomb/kills/rounds/header JSONs) and emits
one record per round with:

  features      : Tensor [L, F]   downsampled per-tick feature vectors
  tick_indices  : Tensor [L]      original tick numbers (for joining w/ events)
  tick_seconds  : Tensor [L]      round-relative seconds (for time pos-enc)
  events        : list of {tick, type}  for time-to-next-event + next-event-type targets
  round_won     : bool            T-side won? (DIAGNOSTIC PROBE ONLY — never an objective)
  metadata      : {demo_stem, round_num, map_name}

Splits demos at the demo level (never within a demo — F1 sibling-tick leakage).

Per-tick feature schema (~320 dim total, see feature_schema_v1.json output):
  - 10 player slots, each ~29 dim (pos, view, velocity, HP/armor, money, eq value,
    weapon class one-hot, util counts, helmet/defuser bits, alive mask)
  - global block: bomb status, bomb pos, bomb timer, round timer, score, round#,
    map one-hot, phase one-hot, sinusoidal time encoding
  - player slot assignment is positional (sorted by side then steamid within round) —
    no player identity embedding; the encoder learns role patterns

Usage:
    python scripts/build_tick_sequences.py
    python scripts/build_tick_sequences.py --target-hz 8 --val-demos 1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

# ---------------------------------------------------------------------------
# Feature schema (kept in code so build + train + downstream can all import)
# ---------------------------------------------------------------------------

# Weapon class buckets — `inventory` from the parquet is a list of weapon
# strings. We map each one to a class. "primary" wins if a primary-class
# weapon is present; otherwise "pistol_only"; otherwise "no_primary".
RIFLES = {"ak47", "m4a4", "m4a1_silencer", "aug", "sg556", "galilar", "famas"}
SNIPERS = {"awp", "ssg08", "scar20", "g3sg1"}
SMGS = {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"}
SHOTGUNS = {"nova", "xm1014", "mag7", "sawedoff"}
LMGS = {"m249", "negev"}
PISTOLS = {
    "deagle", "revolver", "hkp2000", "usp_silencer", "glock", "p250",
    "fiveseven", "tec9", "cz75a", "elite",
}
UTIL = {
    "smokegrenade": "smoke",
    "hegrenade": "he",
    "flashbang": "flash",
    "decoy": "decoy",
    "molotov": "molotov",
    "incgrenade": "molotov",  # incendiary = molotov
}
WEAPON_CLASSES = ["no_primary", "rifle", "sniper", "smg", "shotgun", "lmg", "pistol_only", "unknown"]
WEAPON_CLASS_IDX = {c: i for i, c in enumerate(WEAPON_CLASSES)}

MAPS = ["de_mirage", "de_inferno", "de_nuke", "de_overpass"]
MAP_IDX = {m: i for i, m in enumerate(MAPS)}

EVENT_TYPES = ["kill_T", "kill_CT", "plant", "defuse", "freeze_end", "round_end"]
EVENT_TYPE_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}

# CS2 map coordinates roughly span ±4096 game units; normalize positions to
# [-1, 1]. Bomb timer post-plant = 40s; round time live phase ≤ 115s.
COORD_NORM = 4096.0
BOMB_TIMER_MAX = 40.0
ROUND_TIMER_MAX = 115.0
EQUIP_MAX = 16000.0
MONEY_MAX_LOG = math.log(16001.0)
PHASES = ["freeze", "live", "post-plant"]
N_PLAYER_SLOTS = 10  # 5 T + 5 CT


def _classify_weapon(inv: list[str]) -> tuple[int, dict[str, int]]:
    """Return (weapon_class_idx, util_counts).

    Picks the highest-priority class present (primary > pistol > none).
    util_counts has keys smoke/he/flash/decoy/molotov.
    """
    util_counts = {"smoke": 0, "he": 0, "flash": 0, "decoy": 0, "molotov": 0}
    has_primary_class = None
    has_pistol = False
    for w in inv:
        w_lower = w.lower() if isinstance(w, str) else ""
        if w_lower in RIFLES:
            has_primary_class = "rifle"
        elif w_lower in SNIPERS:
            has_primary_class = has_primary_class or "sniper"
        elif w_lower in SMGS:
            has_primary_class = has_primary_class or "smg"
        elif w_lower in SHOTGUNS:
            has_primary_class = has_primary_class or "shotgun"
        elif w_lower in LMGS:
            has_primary_class = has_primary_class or "lmg"
        elif w_lower in PISTOLS:
            has_pistol = True
        elif w_lower in UTIL:
            util_counts[UTIL[w_lower]] += 1
    if has_primary_class is not None:
        return WEAPON_CLASS_IDX[has_primary_class], util_counts
    if has_pistol:
        return WEAPON_CLASS_IDX["pistol_only"], util_counts
    return WEAPON_CLASS_IDX["no_primary"], util_counts


def _player_slot_features(row: dict, prev_pos: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-player feature vector + new position for velocity tracking.

    Returns (features [29], current_position [3]).
    """
    x, y, z = row["X"], row["Y"], row["Z"]
    pos = np.array([x, y, z], dtype=np.float32) / COORD_NORM
    if prev_pos is None:
        vel = np.zeros(3, dtype=np.float32)
    else:
        vel = pos - prev_pos  # tick-to-tick delta in normalized units

    yaw = float(row["yaw"])
    pitch = float(row["pitch"])
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    view = np.array(
        [math.sin(yaw_rad), math.cos(yaw_rad), math.sin(pitch_rad), math.cos(pitch_rad)],
        dtype=np.float32,
    )

    hp = float(row["health"]) / 100.0
    armor = float(row["armor"]) / 100.0
    helmet = 1.0 if row["has_helmet"] else 0.0
    defuser = 1.0 if row["has_defuser"] else 0.0
    money = math.log(1.0 + float(row["balance"])) / MONEY_MAX_LOG
    equip = float(row["current_equip_value"]) / EQUIP_MAX

    inv = row["inventory"] if isinstance(row["inventory"], list) else []
    weapon_class, util = _classify_weapon(inv)
    weapon_oh = np.zeros(len(WEAPON_CLASSES), dtype=np.float32)
    weapon_oh[weapon_class] = 1.0
    util_vec = np.array(
        [util["smoke"], util["molotov"], util["flash"], util["he"], util["decoy"]],
        dtype=np.float32,
    ) / 4.0  # normalize counts (max ~4 of any single util)

    alive = 1.0 if hp > 0 else 0.0

    feats = np.concatenate([
        pos,                                   # 3
        view,                                  # 4
        vel,                                   # 3
        np.array([hp, armor, helmet, defuser, money, equip], dtype=np.float32),  # 6
        weapon_oh,                             # 8
        util_vec,                              # 5
        np.array([alive], dtype=np.float32),   # 1
    ])
    assert feats.shape == (30,), feats.shape
    return feats, pos


def _global_features(
    tick: int,
    round_info: dict,
    bomb_state: dict,
    map_name: str,
    score_t: int,
    score_ct: int,
    round_num: int,
) -> np.ndarray:
    """Global per-tick features (~30 dim)."""
    # Bomb status
    bomb_status = np.zeros(4, dtype=np.float32)  # [not_planted, planted_A, planted_B, defused_or_exploded]
    bomb_pos = np.zeros(2, dtype=np.float32)
    bomb_timer = 0.0
    if bomb_state.get("plant_tick") is not None and tick >= bomb_state["plant_tick"]:
        if bomb_state.get("end_tick") is not None and tick >= bomb_state["end_tick"]:
            bomb_status[3] = 1.0  # defused/exploded
        else:
            site = bomb_state.get("site", "a")
            bomb_status[1 if site == "a" else 2] = 1.0
            if bomb_state.get("plant_pos") is not None:
                bomb_pos[0] = bomb_state["plant_pos"][0] / COORD_NORM
                bomb_pos[1] = bomb_state["plant_pos"][1] / COORD_NORM
            elapsed_ticks = tick - bomb_state["plant_tick"]
            bomb_timer = (elapsed_ticks / 64.0) / BOMB_TIMER_MAX
    else:
        bomb_status[0] = 1.0

    # Round timer (seconds since freeze_end, clamped)
    freeze_end = round_info.get("freeze_end") or round_info["start"]
    if tick < freeze_end:
        round_timer = 0.0
    else:
        round_timer = ((tick - freeze_end) / 64.0) / ROUND_TIMER_MAX
        round_timer = min(round_timer, 1.0)

    # Phase
    phase = np.zeros(3, dtype=np.float32)  # [freeze, live, post-plant]
    if tick < freeze_end:
        phase[0] = 1.0
    elif bomb_state.get("plant_tick") is not None and tick >= bomb_state["plant_tick"]:
        phase[2] = 1.0
    else:
        phase[1] = 1.0

    # Map identity is now handled via a learned per-map nn.Embedding in the
    # encoder (see train_round_encoder.py); the per-tick global block no
    # longer carries a map one-hot. Each round-record carries `map_id` (int).

    # Sinusoidal time encoding (6 sin/cos pairs at different frequencies)
    secs = (tick - round_info["start"]) / 64.0
    time_enc = []
    for k in range(6):
        period = 2.0 * (2 ** k)  # 2, 4, 8, 16, 32, 64 second periods
        time_enc.append(math.sin(2 * math.pi * secs / period))
        time_enc.append(math.cos(2 * math.pi * secs / period))
    time_enc = np.array(time_enc, dtype=np.float32)  # 12

    score = np.array([score_t / 16.0, score_ct / 16.0, round_num / 30.0], dtype=np.float32)

    feats = np.concatenate([
        bomb_status,    # 4
        bomb_pos,       # 2
        np.array([bomb_timer, round_timer], dtype=np.float32),  # 2
        score,          # 3
        phase,          # 3
        time_enc,       # 12
    ])
    assert feats.shape == (26,), feats.shape
    return feats


def _build_round_features(
    round_ticks: pl.DataFrame,
    round_info: dict,
    bomb_events: list[dict],
    map_name: str,
    score_t: int,
    score_ct: int,
    target_hz: int,
    tickrate: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build per-tick feature tensor for one round.

    Returns (features [L, F], tick_indices [L], tick_seconds [L]).
    Returns None if the round is malformed (no players, etc.).
    """
    # Downsample ticks
    stride = tickrate // target_hz  # 64/8 = 8
    all_ticks = round_ticks["tick"].unique().sort().to_list()
    sampled_ticks = all_ticks[::stride]
    if not sampled_ticks:
        return None

    # Build bomb_state: when does the bomb get planted, where, and end?
    bomb_state: dict = {"plant_tick": None, "site": None, "plant_pos": None, "end_tick": None}
    if round_info.get("bomb_plant") is not None:
        bomb_state["plant_tick"] = int(round_info["bomb_plant"])
        bomb_state["site"] = round_info.get("bomb_site", "a") or "a"
        for ev in bomb_events:
            if ev.get("event") == "plant" and ev.get("round_num") == round_info["round_num"]:
                bomb_state["plant_pos"] = (float(ev["X"]), float(ev["Y"]))
            if ev.get("event") in ("defuse", "exploded") and ev.get("round_num") == round_info["round_num"]:
                bomb_state["end_tick"] = int(ev["tick"])

    # Group ticks rows by tick number
    by_tick = round_ticks.partition_by("tick", as_dict=True)
    # Player slot assignment: stable per round, sorted by (side, steamid)
    # Use the FIRST tick to establish slot order
    first_tick_rows = by_tick[(sampled_ticks[0],)].sort(["side", "steamid"]).to_dicts()
    slot_steamids: list[int] = []
    for row in first_tick_rows:
        slot_steamids.append(row["steamid"])
    if len(slot_steamids) == 0:
        return None
    # If <10 players (disconnect), pad with sentinel that we'll mask
    while len(slot_steamids) < N_PLAYER_SLOTS:
        slot_steamids.append(-1)

    # Per-slot previous position (for velocity)
    prev_pos: list[np.ndarray | None] = [None] * N_PLAYER_SLOTS

    feature_rows: list[np.ndarray] = []
    for t in sampled_ticks:
        if (t,) not in by_tick:
            continue
        rows_by_steamid = {r["steamid"]: r for r in by_tick[(t,)].to_dicts()}
        slot_feats: list[np.ndarray] = []
        for slot_idx, sid in enumerate(slot_steamids):
            if sid in rows_by_steamid:
                pf, new_pos = _player_slot_features(rows_by_steamid[sid], prev_pos[slot_idx])
                slot_feats.append(pf)
                prev_pos[slot_idx] = new_pos
            else:
                # Player missing this tick (disconnect or pre-spawn) — zeros
                slot_feats.append(np.zeros(30, dtype=np.float32))
        per_player = np.concatenate(slot_feats)  # 30 * 10 = 300
        glob = _global_features(t, round_info, bomb_state, map_name, score_t, score_ct, round_info["round_num"])
        feature_rows.append(np.concatenate([per_player, glob]))

    if not feature_rows:
        return None

    features = np.stack(feature_rows).astype(np.float32)
    tick_indices = np.array(sampled_ticks[: len(feature_rows)], dtype=np.int64)
    tick_seconds = (tick_indices - round_info["start"]) / float(tickrate)
    return features, tick_indices, tick_seconds.astype(np.float32)


def _round_events(
    round_num: int,
    round_info: dict,
    kills: list[dict],
    bomb_events: list[dict],
) -> list[dict]:
    """Collect typed events for a round, sorted by tick."""
    evs: list[dict] = []
    if round_info.get("freeze_end") is not None:
        evs.append({"tick": int(round_info["freeze_end"]), "type": "freeze_end"})
    for k in kills:
        if k.get("round_num") != round_num:
            continue
        side = k.get("attacker_side", "")
        if side == "t":
            evs.append({"tick": int(k["tick"]), "type": "kill_T"})
        elif side == "ct":
            evs.append({"tick": int(k["tick"]), "type": "kill_CT"})
    for b in bomb_events:
        if b.get("round_num") != round_num:
            continue
        if b.get("event") == "plant":
            evs.append({"tick": int(b["tick"]), "type": "plant"})
        elif b.get("event") == "defuse":
            evs.append({"tick": int(b["tick"]), "type": "defuse"})
    if round_info.get("end") is not None:
        evs.append({"tick": int(round_info["end"]), "type": "round_end"})
    evs.sort(key=lambda e: e["tick"])
    return evs


def process_demo(demo_stem: str, demos_dir: Path, target_hz: int) -> list[dict]:
    """Build all round records for one demo. Returns list of round-dicts."""
    ticks_path = demos_dir / f"{demo_stem}_ticks.parquet"
    rounds_path = demos_dir / f"{demo_stem}_rounds.json"
    bomb_path = demos_dir / f"{demo_stem}_bomb.json"
    kills_path = demos_dir / f"{demo_stem}_kills.json"
    header_path = demos_dir / f"{demo_stem}_header.json"

    for p in (ticks_path, rounds_path, bomb_path, kills_path, header_path):
        if not p.exists():
            print(f"  skip {demo_stem}: missing {p.name}")
            return []

    ticks_df = pl.read_parquet(ticks_path)
    rounds = json.loads(rounds_path.read_text())
    bomb_events = json.loads(bomb_path.read_text())
    kills = json.loads(kills_path.read_text())
    header = json.loads(header_path.read_text())
    map_name = header.get("map_name", "unknown")

    out: list[dict] = []
    score_t = 0
    score_ct = 0
    for round_info in rounds:
        rn = int(round_info["round_num"])
        round_ticks = ticks_df.filter(pl.col("round_num") == rn)
        if round_ticks.height == 0:
            continue
        built = _build_round_features(
            round_ticks=round_ticks,
            round_info=round_info,
            bomb_events=bomb_events,
            map_name=map_name,
            score_t=score_t,
            score_ct=score_ct,
            target_hz=target_hz,
        )
        if built is None:
            continue
        features, tick_indices, tick_seconds = built
        events = _round_events(rn, round_info, kills, bomb_events)
        winner = round_info.get("winner", "")
        round_won_t = (winner == "t")
        out.append({
            "demo_stem": demo_stem,
            "round_num": rn,
            "map_name": map_name,
            "map_id": int(MAP_IDX.get(map_name, 0)),  # for the learned map embedding
            "features": torch.from_numpy(features),
            "tick_indices": torch.from_numpy(tick_indices),
            "tick_seconds": torch.from_numpy(tick_seconds),
            "events": events,
            "round_won_t": round_won_t,
        })
        # Update score for next round
        if winner == "t":
            score_t += 1
        elif winner == "ct":
            score_ct += 1

    print(f"  {demo_stem}: {len(out)} rounds, ~{out[0]['features'].shape[1] if out else 0}-dim per tick")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demos-dir", type=Path, default=Path("data/processed/demos"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/tick_sequences"))
    ap.add_argument("--target-hz", type=int, default=8, help="downsample to N Hz (from 64 Hz tickrate)")
    ap.add_argument("--val-demos", type=int, default=1, help="number of demos to hold out for val (demo-level split)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover demo stems from *_ticks.parquet
    stems = sorted({p.stem.removesuffix("_ticks") for p in args.demos_dir.glob("*_ticks.parquet")})
    if not stems:
        print(f"No *_ticks.parquet found in {args.demos_dir}")
        return
    print(f"Found {len(stems)} demos: {stems}")

    train_stems = stems[: -args.val_demos] if args.val_demos > 0 else stems
    val_stems = stems[-args.val_demos:] if args.val_demos > 0 else []
    print(f"Split (demo-level): train={len(train_stems)}, val={len(val_stems)}")
    print()

    print("Building train rounds...")
    train_rounds: list[dict] = []
    for s in train_stems:
        train_rounds.extend(process_demo(s, args.demos_dir, args.target_hz))

    print()
    print("Building val rounds...")
    val_rounds: list[dict] = []
    for s in val_stems:
        val_rounds.extend(process_demo(s, args.demos_dir, args.target_hz))

    train_path = args.output_dir / "train.pt"
    val_path = args.output_dir / "val.pt"
    torch.save(train_rounds, train_path)
    torch.save(val_rounds, val_path)

    feature_dim = train_rounds[0]["features"].shape[1] if train_rounds else 0
    schema = {
        "version": 2,
        "feature_dim": feature_dim,
        "target_hz": args.target_hz,
        "n_player_slots": N_PLAYER_SLOTS,
        "per_player_dim": 30,
        "global_dim": 26,
        "n_maps": len(MAPS),
        "map_embed_dim_default": 8,
        "weapon_classes": WEAPON_CLASSES,
        "maps": MAPS,
        "event_types": EVENT_TYPES,
        "phases": PHASES,
        "notes": (
            "Per tick: [10 players × 30 dim concatenated] + [26 global dim]. "
            "Player slots are positional (sorted by side then steamid within round); "
            "missing-player slots are zero-padded. Velocity is tick-to-tick position "
            "delta in normalized units. round_won_t is per-round, NEVER an objective — "
            "diagnostic probe only. v2: replaced 4-dim map one-hot with a per-round "
            "map_id (int) consumed by the encoder's learned nn.Embedding (default 8 dim) — "
            "lets the model learn map similarity, not just identity."
        ),
    }
    schema_path = args.output_dir / "feature_schema_v1.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    print()
    print(f"Wrote {len(train_rounds)} train rounds → {train_path}")
    print(f"Wrote {len(val_rounds)} val rounds   → {val_path}")
    print(f"Feature dim per tick: {feature_dim}")
    print(f"Schema: {schema_path}")


if __name__ == "__main__":
    main()
