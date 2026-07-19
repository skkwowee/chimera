#!/usr/bin/env python3
"""Build tick-sequence tensors for the Level-2 round encoder.

Reads data/processed/demos/*_ticks.parquet (per-tick player state, long format)
plus the matching *_rounds.json (round boundaries) and *_header.json (map name),
then emits per-round tensors that the round encoder consumes.

Per docs/round-encoder-design.md §2:
  - Downsample 64 Hz -> 8 Hz (every 8th tick)
  - Per-tick feature vector ~= per-player block × 10 players + global state
  - Player slots ordered T1..T5, CT1..CT5 (positional, not identity-based)
  - Output one (T, F) tensor per round; T variable per round, F fixed

Output:
  data/processed/tick_sequences/train.pt          dict[str, list[tensor]]
  data/processed/tick_sequences/val.pt
  data/processed/tick_sequences/feature_schema.json  (+ legacy-name copy
                                feature_schema_v1.json for one transition)
  data/processed/tick_sequences/manifest.json     per-round metadata for joins

Train/val split is DEMO-LEVEL (per the design, avoids round-leakage). Default:
20 demos train, 4 val (use --val-demos to override).

Usage:
    python scripts/build_tick_sequences.py
    python scripts/build_tick_sequences.py --downsample 8 --val-demos 4
    python scripts/build_tick_sequences.py --limit 2     # smoke test on 2 demos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO / "data" / "processed" / "demos"
OUT_DIR = REPO / "data" / "processed" / "tick_sequences"

# Vocabularies — derived once on first run, then frozen in feature_schema_v1.json
# so encoder + every downstream consumer reads the same indexing.
MAP_VOCAB = ["de_ancient", "de_dust2", "de_inferno", "de_mirage",
             "de_nuke", "de_overpass", "de_train"]

# Item categories so we don't need a 100-dim weapon one-hot. Buckets cover the
# tactical roles the encoder cares about. Anything unknown -> "other".
WEAPON_CATEGORIES = {
    "knife": {"Butterfly Knife", "Karambit", "M9 Bayonet", "Bayonet",
              "Bowie Knife", "Falchion Knife", "Flip Knife", "Gut Knife",
              "Huntsman Knife", "Shadow Daggers", "Stiletto Knife",
              "Talon Knife", "Ursus Knife", "Navaja Knife", "Skeleton Knife",
              "Classic Knife", "Nomad Knife", "Paracord Knife", "Survival Knife",
              "Knife"},
    "pistol_low": {"Glock-18", "USP-S", "P2000", "P250"},
    "pistol_high": {"Five-SeveN", "Tec-9", "CZ75 Auto", "Desert Eagle",
                    "R8 Revolver", "Dual Berettas"},
    "smg": {"MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon"},
    "shotgun": {"MAG-7", "Nova", "Sawed-Off", "XM1014"},
    "rifle_t": {"AK-47", "Galil AR", "SG 553"},
    "rifle_ct": {"M4A4", "M4A1-S", "AUG", "FAMAS"},
    "awp": {"AWP"},
    "scout": {"SSG 08"},
    "auto_sniper": {"G3SG1", "SCAR-20"},
    "lmg": {"M249", "Negev"},
    "smoke": {"Smoke Grenade"},
    "flash": {"Flashbang"},
    "molly": {"Molotov", "Incendiary Grenade"},
    "he": {"High Explosive Grenade"},
    "decoy": {"Decoy Grenade"},
    "c4": {"C4 Explosive"},
}
WEAPON_CAT_LIST = list(WEAPON_CATEGORIES.keys()) + ["other"]
WEAPON_CAT_IDX = {c: i for i, c in enumerate(WEAPON_CAT_LIST)}

# Round phases — derived from rounds.json fields
PHASE_VOCAB = ["freeze", "live", "post_plant", "end"]

# Per-tick event labels (axis-1 SSL, augments forward-prediction with event prediction).
# Index 6 ("none") means no listed event happens within EVENT_HORIZON_TICKS.
EVENT_VOCAB = [
    "kill_t",        # T player killed a CT player
    "kill_ct",       # CT player killed a T player
    "bomb_planted",
    "bomb_defused",
    "bomb_exploded",
    "round_end",
    "none",
]
EVENT_IDX = {e: i for i, e in enumerate(EVENT_VOCAB)}
NONE_EVENT_IDX = EVENT_IDX["none"]
# In 64Hz raw ticks: 256 = 4.0s. Event labels look at the next 256 raw ticks.
# Short enough that "next event" is temporally precise (vs aggregating over a
# whole round); long enough that "none" is meaningfully informative during
# slow phases instead of being the trivial answer everywhere.
EVENT_HORIZON_TICKS = 256

# Bumped v2 -> v2.1 when the site-from-plant-position fix landed (datasheet
# D-defect) + bomb_exploded labels from round-end reason. Dim-preserving:
# feature_dim is unchanged, but planted_a/planted_b semantics differ, so
# pre-fix and post-fix blobs must never be merged silently.
SCHEMA_VERSION = "feature_schema_v2.1"

# Site-from-plant-position (datasheet D-defect): the awpy round-level
# `bomb_site` label is broken corpus-wide (873/879 labeled B while plant
# positions cluster at BOTH sites on every map). Site MUST be derived from
# the plant position. Centroids below were measured from the local corpus's
# plant events (whose per-event `bombsite` labels cluster tightly, unlike
# the round-level label). de_nuke's sites are vertically stacked, so xy is
# ambiguous there — use the Z split instead (A upper z~-406, B lower z~-767).
BOMBSITE_CENTROIDS = {
    #              A site (x, y)         B site (x, y)
    "de_ancient":  ((-1324.0,   775.0), (  868.0,    23.0)),
    "de_anubis":   (( 1267.0,  1969.0), (-1060.0,   642.0)),
    "de_dust2":    (( 1089.0,  2521.0), (-1537.0,  2572.0)),
    "de_inferno":  (( 1952.0,   428.0), (  491.0,  2864.0)),
    "de_mirage":   (( -411.0, -2074.0), (-1981.0,   391.0)),
    "de_nuke":     ((  674.0,  -757.0), (  535.0, -1007.0)),
    "de_overpass": ((-2476.0,   648.0), (-1143.0,   -22.0)),
    "de_train":    ((  417.0,   -45.0), ( -121.0, -1287.0)),
}
NUKE_Z_SPLIT = -600.0  # z >= split -> A (upper); z < split -> B (lower/ramp)

# Cross-check counters (per-process): derived site vs the plant event's own
# `bombsite` string and vs the known-broken awpy round label. process_demo
# prints the per-demo deltas so disagreement rates are visible in bake logs.
SITE_XCHECK = {"plants": 0, "event_label_disagree": 0, "awpy_label_disagree": 0}


def derive_site_from_plant(map_name: str, x: float, y: float,
                           z: float | None,
                           event_bombsite: str | None = None) -> str:
    """Classify a plant as site 'a' or 'b' from its position.

    Nearest bombsite centroid in xy for all maps; Z-split override on
    de_nuke (vertically stacked sites). Unknown maps fall back to the plant
    event's own `bombsite` string (e.g. 'BombsiteA'), then 'a'.
    """
    if map_name == "de_nuke" and z is not None:
        return "a" if z >= NUKE_Z_SPLIT else "b"
    cents = BOMBSITE_CENTROIDS.get(map_name)
    if cents is None:
        if event_bombsite and event_bombsite.strip().lower().endswith("b"):
            return "b"
        return "a"
    (ax, ay), (bx, by) = cents
    da = (x - ax) ** 2 + (y - ay) ** 2
    db = (x - bx) ** 2 + (y - by) ** 2
    return "a" if da <= db else "b"


DOWNSAMPLE_DEFAULT = 8  # 64Hz -> 8Hz
PLAYERS_PER_SIDE = 5
N_PLAYERS = 2 * PLAYERS_PER_SIDE


def categorize_weapon(name: str) -> int:
    """Map a weapon/util name to its category index. Unknown -> 'other'."""
    for cat, items in WEAPON_CATEGORIES.items():
        if name in items:
            return WEAPON_CAT_IDX[cat]
    return WEAPON_CAT_IDX["other"]


def inventory_to_categorical(inv: list[str] | None) -> dict:
    """Reduce a player's inventory list into:
      - primary_cat (int): the most-impactful weapon present
      - secondary_cat (int): second weapon (typically pistol)
      - util_bits (np.ndarray of shape (5,)): smoke/flash/molly/he/decoy held (multi-hot)
      - has_c4 (int): 1 if carrying C4
    """
    if not inv:
        return {"primary": WEAPON_CAT_IDX["other"],
                "secondary": WEAPON_CAT_IDX["other"],
                "util_bits": np.zeros(5, dtype=np.float32),
                "has_c4": 0.0}

    # Order of "primaryness" — first hit wins
    primary_order = ["awp", "scout", "auto_sniper", "rifle_t", "rifle_ct",
                     "lmg", "shotgun", "smg"]
    secondary_order = ["pistol_high", "pistol_low"]

    cats = [categorize_weapon(item) for item in inv]
    cat_names = [WEAPON_CAT_LIST[c] for c in cats]

    primary = WEAPON_CAT_IDX["other"]
    for cat in primary_order:
        if cat in cat_names:
            primary = WEAPON_CAT_IDX[cat]
            break

    secondary = WEAPON_CAT_IDX["other"]
    for cat in secondary_order:
        if cat in cat_names:
            secondary = WEAPON_CAT_IDX[cat]
            break

    util_bits = np.zeros(5, dtype=np.float32)
    for ui, ucat in enumerate(["smoke", "flash", "molly", "he", "decoy"]):
        if ucat in cat_names:
            util_bits[ui] = 1.0

    has_c4 = 1.0 if "c4" in cat_names else 0.0

    return {"primary": primary, "secondary": secondary,
            "util_bits": util_bits, "has_c4": has_c4}


# Per-player feature dim:
#   pos (3) + sin/cos(yaw,pitch) (4) + hp (1) + armor (1) + helmet (1) + defuser (1)
#   + balance (1) + equip_value (1) + alive (1) + has_c4 (1)
#   + primary_onehot (len(WEAPON_CAT_LIST)) + secondary_onehot (len(WEAPON_CAT_LIST))
#   + util_bits (5)
PER_PLAYER_DIM = 3 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + len(WEAPON_CAT_LIST) * 2 + 5
HAS_C4_IDX = 14  # index of has_c4 within a per-player block (pos 3 + view 4
                 # + hp/armor/helmet/defuser 4 + balance/equip/alive 3)
# Global feature dim (v2 schema, bumped from v1):
#   v1 components (22 dims):
#     map_onehot (7) + phase_onehot (4) + score (2) + round_num (1) + round_time (1)
#     + bomb_state_onehot (4) + bomb_pos (2) + bomb_age (1)
#   v2 additions (15 dims) — team aggregates and tempo, requested 2026-05-16:
#     money_diff_norm (1)     — (T_money_total - CT_money_total) / 5e4
#     equip_diff_norm (1)     — (T_equip_total - CT_equip_total) / 5e4
#     alive_t_norm (1)        — T_alive / 5
#     alive_ct_norm (1)       — CT_alive / 5
#     alive_diff_norm (1)     — (T_alive - CT_alive) / 5
#     ticks_since_last_kill (1)   — capped, normalized to [0, 1] over 256 ticks
#     ticks_since_last_event (1)  — same, includes plant/defuse/explode too
#     round_time_sin_cos (8)  — multi-scale sinusoidal at periods 5s/20s/115s
#                              (was only 1 raw normalized dim before)
GLOBAL_DIM = (len(MAP_VOCAB) + len(PHASE_VOCAB) + 2 + 1 + 1 + 4 + 2 + 1
              + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 8)
TOTAL_DIM = N_PLAYERS * PER_PLAYER_DIM + GLOBAL_DIM


def encode_player_row(row: dict) -> np.ndarray:
    """Encode one player at one tick into a per-player feature vector."""
    out = np.zeros(PER_PLAYER_DIM, dtype=np.float32)
    if row is None:
        return out  # all zeros — slot empty

    i = 0
    # Position (normalized roughly to [-1, 1] — map coords are typically in [-3000, 3000])
    out[i] = (row.get("X") or 0.0) / 3000.0; i += 1
    out[i] = (row.get("Y") or 0.0) / 3000.0; i += 1
    out[i] = (row.get("Z") or 0.0) / 500.0; i += 1
    # View angles -> sin/cos
    yaw = (row.get("yaw") or 0.0) * np.pi / 180.0
    pitch = (row.get("pitch") or 0.0) * np.pi / 180.0
    out[i] = np.sin(yaw); i += 1
    out[i] = np.cos(yaw); i += 1
    out[i] = np.sin(pitch); i += 1
    out[i] = np.cos(pitch); i += 1
    # HP / armor / helmet / defuser
    hp = max(0, row.get("health") or 0)
    out[i] = hp / 100.0; i += 1
    out[i] = (row.get("armor") or 0) / 100.0; i += 1
    out[i] = 1.0 if row.get("has_helmet") else 0.0; i += 1
    out[i] = 1.0 if row.get("has_defuser") else 0.0; i += 1
    # Money (log-normalized)
    bal = max(0, row.get("balance") or 0)
    out[i] = np.log1p(bal) / 10.0; i += 1
    # Equip value (log-normalized)
    eq = max(0, row.get("current_equip_value") or 0)
    out[i] = np.log1p(eq) / 10.0; i += 1
    # Alive
    out[i] = 1.0 if hp > 0 else 0.0; i += 1
    # Inventory features
    inv = inventory_to_categorical(row.get("inventory"))
    out[i] = inv["has_c4"]; i += 1
    # primary one-hot
    out[i + inv["primary"]] = 1.0; i += len(WEAPON_CAT_LIST)
    # secondary one-hot
    out[i + inv["secondary"]] = 1.0; i += len(WEAPON_CAT_LIST)
    # util bits
    out[i:i + 5] = inv["util_bits"]; i += 5

    assert i == PER_PLAYER_DIM, f"player encoding mismatch: {i} vs {PER_PLAYER_DIM}"
    return out


def assign_player_slots(round_df: pl.DataFrame) -> dict[int, tuple[str, int]]:
    """Map steamid -> (slot_role, slot_idx) where role in {'t', 'ct'}, idx in 0..4.

    Slots are assigned by order of appearance per side at the round's first tick,
    keeping the encoder positional (not identity-based — same slot can be filled
    by different players in different rounds).
    """
    first_tick = round_df["tick"].min()
    first = round_df.filter(pl.col("tick") == first_tick).sort("steamid")
    t_ids = first.filter(pl.col("side") == "t")["steamid"].to_list()[:5]
    ct_ids = first.filter(pl.col("side") == "ct")["steamid"].to_list()[:5]
    assignment: dict[int, tuple[str, int]] = {}
    for i, sid in enumerate(t_ids):
        assignment[sid] = ("t", i)
    for i, sid in enumerate(ct_ids):
        assignment[sid] = ("ct", i)
    return assignment


def encode_global(map_name: str, phase: str, score_t: int, score_ct: int,
                  round_num: int, round_time_s: float,
                  bomb_state: str, bomb_x: float, bomb_y: float,
                  bomb_age_s: float) -> np.ndarray:
    out = np.zeros(GLOBAL_DIM, dtype=np.float32)
    i = 0
    # map one-hot
    if map_name in MAP_VOCAB:
        out[i + MAP_VOCAB.index(map_name)] = 1.0
    i += len(MAP_VOCAB)
    # phase one-hot
    if phase in PHASE_VOCAB:
        out[i + PHASE_VOCAB.index(phase)] = 1.0
    i += len(PHASE_VOCAB)
    # score (normalized roughly)
    out[i] = score_t / 16.0; i += 1
    out[i] = score_ct / 16.0; i += 1
    # round_num (normalized)
    out[i] = round_num / 30.0; i += 1
    # round_time (seconds since round start, normalized to ~115s max)
    out[i] = round_time_s / 115.0; i += 1
    # bomb state one-hot: {none, carried, planted_a, planted_b}
    bomb_states = ["none", "carried", "planted_a", "planted_b"]
    if bomb_state in bomb_states:
        out[i + bomb_states.index(bomb_state)] = 1.0
    i += 4
    # bomb position
    out[i] = bomb_x / 3000.0; i += 1
    out[i] = bomb_y / 3000.0; i += 1
    # bomb age (seconds since plant; 0 if not planted; ~40s max)
    out[i] = bomb_age_s / 40.0; i += 1
    assert i == GLOBAL_DIM
    return out


def _encode_player_block_vectorized(
    df: pl.DataFrame, kept_ticks: np.ndarray
) -> np.ndarray:
    """Encode one player slot's per-tick state into shape (T, PER_PLAYER_DIM).

    df is the player's rows for this round (already filtered to this slot's
    steamid). kept_ticks is the downsampled tick list. Missing ticks (player
    didn't have a row at that tick) become all-zero rows.
    """
    T = len(kept_ticks)
    out = np.zeros((T, PER_PLAYER_DIM), dtype=np.float32)
    if df.height == 0:
        return out

    # Reindex onto kept_ticks via a left-join on tick
    tick_lookup = pl.DataFrame({"tick": kept_ticks.astype(np.int32)})
    aligned = tick_lookup.join(df, on="tick", how="left")

    # Vectorized normalizations on the aligned columns
    x = (aligned["X"].fill_null(0.0).to_numpy() / 3000.0).astype(np.float32)
    y = (aligned["Y"].fill_null(0.0).to_numpy() / 3000.0).astype(np.float32)
    z = (aligned["Z"].fill_null(0.0).to_numpy() / 500.0).astype(np.float32)
    yaw = (aligned["yaw"].fill_null(0.0).to_numpy() * np.pi / 180.0).astype(np.float32)
    pitch = (aligned["pitch"].fill_null(0.0).to_numpy() * np.pi / 180.0).astype(np.float32)
    hp = np.clip(aligned["health"].fill_null(0).to_numpy(), 0, None).astype(np.float32)
    armor = np.clip(aligned["armor"].fill_null(0).to_numpy(), 0, None).astype(np.float32)
    helmet = aligned["has_helmet"].fill_null(False).to_numpy().astype(np.float32)
    defuser = aligned["has_defuser"].fill_null(False).to_numpy().astype(np.float32)
    bal = np.clip(aligned["balance"].fill_null(0).to_numpy(), 0, None).astype(np.float32)
    eq = np.clip(aligned["current_equip_value"].fill_null(0).to_numpy(),
                 0, None).astype(np.float32)
    inv_lists = aligned["inventory"].to_list()  # list of lists or None per tick

    i = 0
    out[:, i] = x; i += 1
    out[:, i] = y; i += 1
    out[:, i] = z; i += 1
    out[:, i] = np.sin(yaw); i += 1
    out[:, i] = np.cos(yaw); i += 1
    out[:, i] = np.sin(pitch); i += 1
    out[:, i] = np.cos(pitch); i += 1
    out[:, i] = hp / 100.0; i += 1
    out[:, i] = armor / 100.0; i += 1
    out[:, i] = helmet; i += 1
    out[:, i] = defuser; i += 1
    out[:, i] = np.log1p(bal) / 10.0; i += 1
    out[:, i] = np.log1p(eq) / 10.0; i += 1
    out[:, i] = (hp > 0).astype(np.float32); i += 1

    # Inventory features (still per-tick Python — there's no vectorized way to
    # categorize variable-length string lists). But polars already returned a
    # plain Python list, so this is just the list iteration cost, not the heavy
    # row-by-row dict access.
    has_c4 = np.zeros(T, dtype=np.float32)
    primary_idx = np.full(T, WEAPON_CAT_IDX["other"], dtype=np.int32)
    secondary_idx = np.full(T, WEAPON_CAT_IDX["other"], dtype=np.int32)
    util_bits = np.zeros((T, 5), dtype=np.float32)
    for t_idx, inv in enumerate(inv_lists):
        if not inv:
            continue
        info = inventory_to_categorical(inv)
        has_c4[t_idx] = info["has_c4"]
        primary_idx[t_idx] = info["primary"]
        secondary_idx[t_idx] = info["secondary"]
        util_bits[t_idx] = info["util_bits"]

    out[:, i] = has_c4; i += 1
    # primary one-hot
    primary_block = np.zeros((T, len(WEAPON_CAT_LIST)), dtype=np.float32)
    primary_block[np.arange(T), primary_idx] = 1.0
    out[:, i:i + len(WEAPON_CAT_LIST)] = primary_block; i += len(WEAPON_CAT_LIST)
    # secondary one-hot
    secondary_block = np.zeros((T, len(WEAPON_CAT_LIST)), dtype=np.float32)
    secondary_block[np.arange(T), secondary_idx] = 1.0
    out[:, i:i + len(WEAPON_CAT_LIST)] = secondary_block; i += len(WEAPON_CAT_LIST)
    # util bits
    out[:, i:i + 5] = util_bits; i += 5

    assert i == PER_PLAYER_DIM, f"player encoding mismatch: {i} vs {PER_PLAYER_DIM}"
    return out


def compute_event_labels(
    kept_ticks: np.ndarray,
    round_meta: dict,
    kills: list[dict],
    bomb_events: list[dict],
    horizon_ticks: int = EVENT_HORIZON_TICKS,
) -> tuple[np.ndarray, np.ndarray]:
    """For each kept tick, find the next significant event within `horizon_ticks`
    raw ticks. Returns (labels (T,) int8, time_to_event (T,) int16).

    `time_to_event` is in raw ticks (not downsampled). When no event lands in
    the window, label is `none` and time = horizon_ticks (so the regression
    target is bounded and the loss can be masked or weighted).
    """
    rn = round_meta["round_num"]
    end_tick = round_meta.get("end") or round_meta.get("official_end")

    # Gather (tick, event_idx) pairs for THIS round, sorted by tick.
    events: list[tuple[int, int]] = []
    for k in kills:
        if k.get("round_num") != rn:
            continue
        # `attacker_side` ∈ {t, ct, ""}; sometimes the kill has no attacker
        # (suicide / world damage). Ignore those — they're noisy labels.
        side = (k.get("attacker_side") or "").lower()
        if side == "t":
            events.append((int(k["tick"]), EVENT_IDX["kill_t"]))
        elif side == "ct":
            events.append((int(k["tick"]), EVENT_IDX["kill_ct"]))
    for be in bomb_events:
        if be.get("round_num") != rn:
            continue
        ev = (be.get("event") or "").lower()
        # awpy emits "bomb_planted" / "bomb_defused" / "bomb_exploded" — match prefix
        if "plant" in ev:
            events.append((int(be["tick"]), EVENT_IDX["bomb_planted"]))
        elif "defus" in ev:
            events.append((int(be["tick"]), EVENT_IDX["bomb_defused"]))
        elif "explod" in ev or "detonat" in ev:
            events.append((int(be["tick"]), EVENT_IDX["bomb_exploded"]))
    if end_tick is not None:
        # awpy's bomb.json has NO explode rows (events are only
        # {plant, drop, defuse, pickup}) — the 'explod'/'detonat' branch
        # above never fires. Detonations are recoverable ONLY from the
        # round-end reason; the explosion coincides with the round end tick.
        if "explod" in (round_meta.get("reason") or "").lower():
            events.append((int(end_tick), EVENT_IDX["bomb_exploded"]))
        events.append((int(end_tick), EVENT_IDX["round_end"]))

    events.sort()
    if not events:
        T = len(kept_ticks)
        return (np.full(T, NONE_EVENT_IDX, dtype=np.int8),
                np.full(T, horizon_ticks, dtype=np.int16))

    ev_ticks = np.array([e[0] for e in events], dtype=np.int64)
    ev_types = np.array([e[1] for e in events], dtype=np.int8)

    # For each kept tick, np.searchsorted finds the index of the first event
    # at or after that tick. Then check if it's within horizon.
    idxs = np.searchsorted(ev_ticks, kept_ticks, side="left")
    T = len(kept_ticks)
    labels = np.full(T, NONE_EVENT_IDX, dtype=np.int8)
    times = np.full(T, horizon_ticks, dtype=np.int16)
    in_range = idxs < len(ev_ticks)
    if in_range.any():
        next_ticks = ev_ticks[idxs[in_range]]
        delta = next_ticks - kept_ticks[in_range]
        within_horizon = delta <= horizon_ticks
        target_positions = np.flatnonzero(in_range)[within_horizon]
        labels[target_positions] = ev_types[idxs[target_positions]]
        times[target_positions] = delta[within_horizon].astype(np.int16)
    return labels, times


def build_round_tensor(
    round_df: pl.DataFrame,
    round_meta: dict,
    map_name: str,
    score_so_far: tuple[int, int],
    bomb_events: list[dict],
    kills: list[dict],
    downsample: int,
) -> tuple[torch.Tensor, dict, np.ndarray, np.ndarray]:
    """Build the (T, F) tensor for one round at the given downsample rate.

    Returns:
      tensor (T, F) float32, meta dict, event_labels (T,) int8, event_times (T,) int16
    """
    ticks_all = round_df["tick"].unique().sort().to_numpy()
    if len(ticks_all) == 0:
        return (torch.empty(0, TOTAL_DIM), {},
                np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int16))

    freeze_end = round_meta["freeze_end"]
    end_tick = round_meta["end"] or round_meta["official_end"] or int(ticks_all[-1])
    plant_tick = round_meta.get("bomb_plant")
    bomb_site = round_meta.get("bomb_site")

    plant_pos = None
    plant_z = None
    plant_event_site = None
    for be in bomb_events:
        if be.get("event") == "plant" and be.get("round_num") == round_meta["round_num"]:
            plant_pos = (be.get("X") or 0.0, be.get("Y") or 0.0)
            plant_z = be.get("Z")
            plant_event_site = be.get("bombsite")   # e.g. 'BombsiteA'
            break

    slot_map = assign_player_slots(round_df)
    kept_ticks = ticks_all[::downsample]
    T = len(kept_ticks)
    out = np.zeros((T, TOTAL_DIM), dtype=np.float32)

    # Per-player blocks — one vectorized pass per slot
    offset = 0
    # Build sid -> rows lookup once
    by_sid = {sid: round_df.filter(pl.col("steamid") == sid)
              for sid, _ in slot_map.items()}
    # T0..T4 then CT0..CT4
    for side in ("t", "ct"):
        for slot_i in range(5):
            sid = next((s for s, r in slot_map.items() if r == (side, slot_i)), None)
            df = by_sid.get(sid) if sid is not None else pl.DataFrame()
            out[:, offset:offset + PER_PLAYER_DIM] = _encode_player_block_vectorized(
                df if df is not None else pl.DataFrame(), kept_ticks,
            )
            offset += PER_PLAYER_DIM

    # Global features — vectorized over kept_ticks
    # Phase
    phases = np.empty(T, dtype=object)
    phases[:] = "live"
    phases[kept_ticks < freeze_end] = "freeze"
    if plant_tick:
        phases[kept_ticks >= plant_tick] = "post_plant"
    phases[kept_ticks >= end_tick] = "end"

    # Bomb state
    bomb_states = np.empty(T, dtype=object)
    bomb_states[:] = "none"
    # 'carried': some player has the C4 in inventory this frame. The per-player
    # blocks are already baked into `out` above, so read the has_c4 bits back
    # rather than re-deriving from inventory. Post-plant frames get overridden
    # to planted_a/b below (the C4 also leaves inventories at plant).
    c4_cols = [p * PER_PLAYER_DIM + HAS_C4_IDX for p in range(N_PLAYERS)]
    bomb_states[out[:, c4_cols].max(axis=1) > 0.5] = "carried"
    bomb_x = np.zeros(T, dtype=np.float32)
    bomb_y = np.zeros(T, dtype=np.float32)
    bomb_age_s = np.zeros(T, dtype=np.float32)
    if plant_tick and plant_pos is not None:
        planted = kept_ticks >= plant_tick
        # Datasheet D-defect fix: derive the site from the plant POSITION.
        # The awpy round-level `bomb_site` label is broken corpus-wide
        # (873/879 labeled B) and is kept only as a logged cross-check.
        site = derive_site_from_plant(map_name, plant_pos[0], plant_pos[1],
                                      plant_z, plant_event_site)
        site_str = f"planted_{site}"
        SITE_XCHECK["plants"] += 1
        if plant_event_site:
            ev_site = ("b" if plant_event_site.strip().lower().endswith("b")
                       else "a")
            if ev_site != site:
                SITE_XCHECK["event_label_disagree"] += 1
                print(f"    WARN r{round_meta['round_num']}: derived site "
                      f"'{site}' != plant-event bombsite '{plant_event_site}' "
                      f"at ({plant_pos[0]:.0f},{plant_pos[1]:.0f},"
                      f"{plant_z if plant_z is not None else 0:.0f}) on {map_name}")
        if bomb_site and bomb_site != "not_planted":
            awpy_site = "b" if bomb_site.lower().endswith("b") else "a"
            if awpy_site != site:
                SITE_XCHECK["awpy_label_disagree"] += 1  # known-broken label
        bomb_states[planted] = site_str
        bomb_x[planted] = plant_pos[0]
        bomb_y[planted] = plant_pos[1]
        bomb_age_s[planted] = (kept_ticks[planted] - plant_tick) / 64.0

    # Anchored at freeze_end, clamped at 0: round_meta["start"] is the previous
    # round's official_end, so anchoring there bakes halftime/timeout pauses
    # into the clock. Live play starts at ~0 every round; freeze frames are 0.
    round_time_s = np.maximum(0, (kept_ticks - freeze_end) / 64.0).astype(np.float32)

    # v2: team aggregates (money, equipment, alive count) per kept tick.
    # Vectorized via polars group_by(tick) sum filtered by side.
    agg = (round_df
           .group_by("tick")
           .agg([
               pl.when(pl.col("side") == "t").then(pl.col("balance")).otherwise(0).sum().alias("t_money"),
               pl.when(pl.col("side") == "ct").then(pl.col("balance")).otherwise(0).sum().alias("ct_money"),
               pl.when(pl.col("side") == "t").then(pl.col("current_equip_value")).otherwise(0).sum().alias("t_equip"),
               pl.when(pl.col("side") == "ct").then(pl.col("current_equip_value")).otherwise(0).sum().alias("ct_equip"),
               pl.when((pl.col("side") == "t") & (pl.col("health") > 0)).then(1).otherwise(0).sum().alias("t_alive"),
               pl.when((pl.col("side") == "ct") & (pl.col("health") > 0)).then(1).otherwise(0).sum().alias("ct_alive"),
           ]))
    lookup = pl.DataFrame({"tick": kept_ticks.astype(np.int64)})
    aligned = lookup.join(agg, on="tick", how="left").fill_null(0)
    t_money = aligned["t_money"].to_numpy().astype(np.float32)
    ct_money = aligned["ct_money"].to_numpy().astype(np.float32)
    t_equip = aligned["t_equip"].to_numpy().astype(np.float32)
    ct_equip = aligned["ct_equip"].to_numpy().astype(np.float32)
    t_alive = aligned["t_alive"].to_numpy().astype(np.float32)
    ct_alive = aligned["ct_alive"].to_numpy().astype(np.float32)
    # Normalizations: $5e4 is a typical full-team fresh-buy total.
    money_diff_n = (t_money - ct_money) / 5.0e4
    equip_diff_n = (t_equip - ct_equip) / 5.0e4
    alive_t_n = t_alive / 5.0
    alive_ct_n = ct_alive / 5.0
    alive_diff_n = (t_alive - ct_alive) / 5.0

    # v2: tempo features — ticks since last KILL and since last ANY event
    # (kill/plant/defuse/explode). Capped at 256 ticks (4s) then normalized.
    rn = round_meta["round_num"]
    kill_ticks = sorted(int(k["tick"]) for k in kills if k.get("round_num") == rn)
    # bomb.json has no explode rows — detonation comes from the round-end
    # reason (at end_tick), matching compute_event_labels.
    event_only_ticks = sorted(set(kill_ticks) | set(
        int(b["tick"]) for b in bomb_events
        if b.get("round_num") == rn
        and (b.get("event") or "").lower() in {"plant", "defuse"}
    ) | ({int(end_tick)}
         if "explod" in (round_meta.get("reason") or "").lower() else set()))
    cap = 256  # 4s at 64Hz; same horizon as EVENT_HORIZON_TICKS so labels and
               # tempo features share a time scale.

    def ticks_since(prior_ticks: list[int], at_ticks: np.ndarray) -> np.ndarray:
        if not prior_ticks:
            return np.full(len(at_ticks), cap, dtype=np.float32)
        arr = np.array(prior_ticks, dtype=np.int64)
        # For each at_tick, find the largest tick in arr that is <= at_tick.
        idx = np.searchsorted(arr, at_ticks, side="right") - 1
        out = np.full(len(at_ticks), cap, dtype=np.float32)
        mask = idx >= 0
        if mask.any():
            out[mask] = np.minimum(at_ticks[mask] - arr[idx[mask]], cap)
        return out

    tsl_kill = ticks_since(kill_ticks, kept_ticks) / float(cap)
    tsl_event = ticks_since(event_only_ticks, kept_ticks) / float(cap)

    # v2: multi-scale round-time sinusoidal at 4 periods (2s / 5s / 20s / 115s)
    # Periods in seconds — gives the model "where are we in the round" at
    # multiple time scales. 2s/5s capture micro-tempo (post-kill window,
    # plant-timing); 20s/115s capture macro (mid-round, end-of-round).
    periods_s = np.array([2.0, 5.0, 20.0, 115.0], dtype=np.float32)
    theta = 2 * np.pi * round_time_s[:, None] / periods_s[None, :]  # (T, 4)
    ts_sin_cos = np.concatenate([np.sin(theta), np.cos(theta)], axis=1)  # (T, 8)

    # Build the global block per-tick (still Python loop but only T iterations)
    bomb_state_names = ["none", "carried", "planted_a", "planted_b"]
    map_idx = MAP_VOCAB.index(map_name) if map_name in MAP_VOCAB else -1
    for t_idx in range(T):
        g = np.zeros(GLOBAL_DIM, dtype=np.float32)
        gi = 0
        if map_idx >= 0:
            g[gi + map_idx] = 1.0
        gi += len(MAP_VOCAB)
        ph = phases[t_idx]
        if ph in PHASE_VOCAB:
            g[gi + PHASE_VOCAB.index(ph)] = 1.0
        gi += len(PHASE_VOCAB)
        g[gi] = score_so_far[0] / 16.0; gi += 1
        g[gi] = score_so_far[1] / 16.0; gi += 1
        g[gi] = round_meta["round_num"] / 30.0; gi += 1
        g[gi] = round_time_s[t_idx] / 115.0; gi += 1
        bs = bomb_states[t_idx]
        if bs in bomb_state_names:
            g[gi + bomb_state_names.index(bs)] = 1.0
        gi += 4
        g[gi] = bomb_x[t_idx] / 3000.0; gi += 1
        g[gi] = bomb_y[t_idx] / 3000.0; gi += 1
        g[gi] = bomb_age_s[t_idx] / 40.0; gi += 1
        # v2 additions
        g[gi] = money_diff_n[t_idx]; gi += 1
        g[gi] = equip_diff_n[t_idx]; gi += 1
        g[gi] = alive_t_n[t_idx]; gi += 1
        g[gi] = alive_ct_n[t_idx]; gi += 1
        g[gi] = alive_diff_n[t_idx]; gi += 1
        g[gi] = tsl_kill[t_idx]; gi += 1
        g[gi] = tsl_event[t_idx]; gi += 1
        g[gi:gi + 8] = ts_sin_cos[t_idx]; gi += 8
        out[t_idx, offset:offset + GLOBAL_DIM] = g

    event_labels, event_times = compute_event_labels(
        kept_ticks, round_meta, kills, bomb_events,
    )

    meta = {
        "round_num": round_meta["round_num"],
        "n_ticks": T,
        "first_tick": int(kept_ticks[0]),
        "last_tick": int(kept_ticks[-1]),
        "downsample": downsample,
        "winner": round_meta.get("winner"),
        "reason": round_meta.get("reason"),
    }
    return torch.from_numpy(out), meta, event_labels, event_times


def process_demo(
    parq: Path, downsample: int,
) -> tuple[list[torch.Tensor], list[dict], list[torch.Tensor], list[torch.Tensor], dict]:
    """Process one demo into per-round (tensors, metas, event_labels, event_times)
    plus a demo-level summary."""
    stem = parq.stem.replace("_ticks", "")
    base = parq.parent

    rounds = json.loads((base / f"{stem}_rounds.json").read_text())
    bomb = json.loads((base / f"{stem}_bomb.json").read_text())
    kills = json.loads((base / f"{stem}_kills.json").read_text())
    header = json.loads((base / f"{stem}_header.json").read_text())
    map_name = header.get("map_name", "unknown")

    df = pl.read_parquet(parq)

    xc0 = dict(SITE_XCHECK)   # per-demo cross-check deltas printed below

    # Running score — winner of each round so far
    score_t = score_ct = 0
    tensors = []
    metas = []
    label_seqs = []
    time_seqs = []
    for r in rounds:
        round_df = df.filter(pl.col("round_num") == r["round_num"])
        if round_df.height == 0:
            continue
        ten, m, ev_lbl, ev_time = build_round_tensor(
            round_df, r, map_name,
            (score_t, score_ct), bomb, kills, downsample,
        )
        if ten.numel() == 0:
            continue
        m["map_name"] = map_name
        m["demo_stem"] = stem
        tensors.append(ten)
        metas.append(m)
        label_seqs.append(torch.from_numpy(ev_lbl).long())
        time_seqs.append(torch.from_numpy(ev_time).float())
        # update score for NEXT round
        if r.get("winner") == "t":
            score_t += 1
        elif r.get("winner") == "ct":
            score_ct += 1

    d_plants = SITE_XCHECK["plants"] - xc0["plants"]
    if d_plants:
        print(f"    site cross-check [{stem}]: {d_plants} plants, "
              f"{SITE_XCHECK['event_label_disagree'] - xc0['event_label_disagree']} "
              f"disagree vs plant-event label, "
              f"{SITE_XCHECK['awpy_label_disagree'] - xc0['awpy_label_disagree']} "
              f"disagree vs awpy round label (known broken)")

    summary = {
        "demo_stem": stem,
        "map_name": map_name,
        "n_rounds": len(tensors),
        "total_ticks": sum(t.shape[0] for t in tensors),
        "feature_dim": TOTAL_DIM,
    }
    return tensors, metas, label_seqs, time_seqs, summary


def builder_provenance() -> dict:
    """Best-effort git provenance stamped into blobs + manifests at bake time.

    The demo pipeline runs a COPY of this script outside any git repo, so
    commit/dirty are None there — the pipeline records the hash itself from
    the source checkout before copying (see chimera-demo-pipeline process.py).
    """
    import subprocess
    commit, dirty = None, None
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            commit = r.stdout.strip()
            s = subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain", "--",
                 "scripts/build_tick_sequences.py", "scripts/parse_demos.py"],
                capture_output=True, text=True, timeout=10)
            dirty = bool(s.stdout.strip())
    except Exception:
        pass
    return {"schema_version": SCHEMA_VERSION, "builder_commit": commit,
            "builder_dirty": dirty, "baked_at_unix": int(time.time())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE_DEFAULT,
                    help=f"keep every Nth tick (64Hz / N) — default {DOWNSAMPLE_DEFAULT} (8Hz)")
    ap.add_argument("--val-demos", type=int, default=12,
                    help="how many demos to hold out for validation (default 12, ~15%)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total demos processed (for smoke testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parqs = sorted(DEMOS_DIR.glob("*_ticks.parquet"))
    if args.limit:
        parqs = parqs[:args.limit]
    if not parqs:
        print(f"No parquets in {DEMOS_DIR}. Run scripts/parse_demos.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Demo-level shuffle then split — deterministic via seed
    rng = np.random.default_rng(args.seed)
    order = list(range(len(parqs)))
    rng.shuffle(order)
    parqs = [parqs[i] for i in order]
    val_n = min(args.val_demos, len(parqs) - 1)
    train_parqs = parqs[val_n:]
    val_parqs = parqs[:val_n]

    prov = builder_provenance()

    print(f"Total demos: {len(parqs)} (train: {len(train_parqs)}, val: {len(val_parqs)})")
    print(f"Schema: {SCHEMA_VERSION} (builder {prov['builder_commit'] or 'unknown'}"
          f"{'+dirty' if prov['builder_dirty'] else ''})")
    print(f"Downsample: {args.downsample}x (8Hz from 64Hz)")
    print(f"Feature dim: {TOTAL_DIM} ({N_PLAYERS}×{PER_PLAYER_DIM} player + {GLOBAL_DIM} global)")
    print(f"Output dir: {OUT_DIR}")
    print()

    def process_split(parqs_sub: list[Path], split_name: str) -> dict:
        all_tensors = []
        all_metas = []
        all_event_labels = []
        all_event_times = []
        summaries = []
        t0 = time.time()
        for i, p in enumerate(parqs_sub):
            t1 = time.time()
            try:
                tensors, metas, ev_lbls, ev_times, summary = process_demo(
                    p, args.downsample,
                )
            except Exception as e:
                print(f"  [{split_name} {i+1}/{len(parqs_sub)}] FAIL {p.stem}: "
                      f"{type(e).__name__}: {e}")
                continue
            all_tensors.extend(tensors)
            all_metas.extend(metas)
            all_event_labels.extend(ev_lbls)
            all_event_times.extend(ev_times)
            summaries.append(summary)
            elapsed = time.time() - t1
            print(f"  [{split_name} {i+1}/{len(parqs_sub)}] {summary['demo_stem']}: "
                  f"{summary['n_rounds']} rounds, "
                  f"{summary['total_ticks']:,} encoded ticks, "
                  f"{elapsed:.1f}s")

        out_path = OUT_DIR / f"{split_name}.pt"
        torch.save({
            "tensors": all_tensors,
            "metas": all_metas,
            "event_labels": all_event_labels,
            "event_times": all_event_times,
            "event_vocab": EVENT_VOCAB,
            "event_horizon_ticks": EVENT_HORIZON_TICKS,
            "summaries": summaries,
            "feature_dim": TOTAL_DIM,
            "downsample": args.downsample,
            **prov,   # schema_version, builder_commit, builder_dirty, baked_at_unix
        }, out_path)
        total_ticks = sum(t.shape[0] for t in all_tensors)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  -> {out_path.name}: {len(all_tensors)} rounds, "
              f"{total_ticks:,} ticks, {size_mb:.1f} MB, "
              f"{time.time()-t0:.0f}s total")
        return {
            "split": split_name,
            "n_demos": len(summaries),
            "n_rounds": len(all_tensors),
            "total_ticks": total_ticks,
            "size_mb": size_mb,
        }

    print("=== train split ===")
    train_summary = process_split(train_parqs, "train")
    print()
    print("=== val split ===")
    val_summary = process_split(val_parqs, "val")
    print()

    # Schema doc — every downstream consumer reads this
    schema = {
        "version": SCHEMA_VERSION,
        "builder_commit": prov["builder_commit"],
        "builder_dirty": prov["builder_dirty"],
        "downsample": args.downsample,
        "tickrate_hz": 8,
        "feature_dim": TOTAL_DIM,
        "n_players": N_PLAYERS,
        "per_player_dim": PER_PLAYER_DIM,
        "global_dim": GLOBAL_DIM,
        "weapon_categories": WEAPON_CAT_LIST,
        "map_vocab": MAP_VOCAB,
        "phase_vocab": PHASE_VOCAB,
        "player_slot_layout": "T1, T2, T3, T4, T5, CT1, CT2, CT3, CT4, CT5",
        "per_player_layout": [
            "x", "y", "z",
            "sin_yaw", "cos_yaw", "sin_pitch", "cos_pitch",
            "hp", "armor", "has_helmet", "has_defuser",
            "log_balance", "log_equip_value", "alive", "has_c4",
            f"primary_weapon_onehot({len(WEAPON_CAT_LIST)})",
            f"secondary_weapon_onehot({len(WEAPON_CAT_LIST)})",
            "util_smoke", "util_flash", "util_molly", "util_he", "util_decoy",
        ],
        "global_layout": [
            f"map_onehot({len(MAP_VOCAB)})",
            f"phase_onehot({len(PHASE_VOCAB)})",
            "score_t_norm", "score_ct_norm",
            "round_num_norm", "round_time_s_norm",
            "bomb_state_onehot(4)", "bomb_x", "bomb_y", "bomb_age_s_norm",
            # v2 additions
            "money_diff_norm", "equip_diff_norm",
            "alive_t_norm", "alive_ct_norm", "alive_diff_norm",
            "ticks_since_last_kill_norm", "ticks_since_last_event_norm",
            "round_time_multi_scale_sinusoidal(8)",
        ],
        "event_vocab": EVENT_VOCAB,
        "event_horizon_ticks": EVENT_HORIZON_TICKS,
        "event_label_dtypes": {
            "event_labels": "int64 (T,) per round, 0..6 indexing event_vocab",
            "event_times": "float32 (T,) per round, ticks until next event "
                           "(capped at event_horizon_ticks; raw 64Hz ticks)",
        },
    }
    # Canonical name is now feature_schema.json (the content self-versions via
    # the "version" key — the old feature_schema_v1.json NAME contradicted its
    # own "feature_schema_v2" content). Keep writing the old name as a copy for
    # one transition release so existing consumers don't break.
    schema_body = json.dumps(schema, indent=2)
    (OUT_DIR / "feature_schema.json").write_text(schema_body)
    (OUT_DIR / "feature_schema_v1.json").write_text(schema_body)

    manifest = {
        "train": train_summary,
        "val": val_summary,
        "feature_schema": "feature_schema.json",
        **prov,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("=== summary ===")
    print(json.dumps(manifest, indent=2))
    print()
    print(f"Schema: {OUT_DIR / 'feature_schema.json'} ({SCHEMA_VERSION})")
    print("Done.")


if __name__ == "__main__":
    main()
