"""Regression cases for missing observations, clocks, bombs and side swaps."""
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from build_tick_sequences import IncompleteRound, assign_player_slots, build_round_tensor, process_demo
from build_v3_features import compute_derived


def frame(ticks, rn=1, swapped=False):
    return pl.DataFrame([
        {"tick": tick, "round_num": rn, "steamid": sid,
         "side": "t" if (sid <= 5) != swapped else "ct",
         "X": 100.0, "Y": 200.0, "Z": 0.0, "yaw": 0.0, "pitch": 0.0,
         "health": 100, "armor": 50, "has_helmet": True, "has_defuser": False,
         "balance": 1000, "current_equip_value": 3000, "inventory": ["AK-47"]}
        for tick in ticks for sid in range(1, 11)
    ])


def round_meta(rn=1, **overrides):
    return {"round_num": rn, "freeze_end": 100, "end": 140, "official_end": 148,
            "winner": "t", "bomb_plant": None, **overrides}


def test_uniform_ticks_and_missing_player_are_not_dead():
    df = frame([100, 101, 108, 116])
    _, meta, _, _ = build_round_tensor(df, round_meta(), "de_mirage", (0, 0), [], [], 8)
    assert meta["raw_ticks"] == [100, 108, 116]
    assert meta["player_steamids"] == list(range(1, 11))
    missing = df.filter(~((pl.col("tick") == 100) & (pl.col("steamid") == 1)))
    assert len(assign_player_slots(missing)) == 10
    for incomplete in (missing, df.filter(pl.col("tick") != 108),
                       df.with_columns(pl.lit(None).cast(pl.Int64).alias("health"))):
        with pytest.raises(IncompleteRound, match="Missing player state"):
            build_round_tensor(incomplete, round_meta(), "de_mirage", (0, 0), [], [], 8)

    paused = frame([100, 101, 124, 132, 140])
    args = (round_meta(freeze_end=132), "de_mirage", (0, 0), [], [], 8)
    tensor, meta, labels, times = build_round_tensor(paused, *args)
    assert meta["raw_ticks"] == [124, 132, 140]
    assert meta["pre_live_trim_ticks"] == 24
    full, _, full_labels, full_times = build_round_tensor(frame(range(100, 141, 8)), *args)
    np.testing.assert_array_equal(tensor, full[-3:])
    np.testing.assert_array_equal(labels, full_labels[-3:])
    np.testing.assert_array_equal(times, full_times[-3:])
    # The same gap at/after live start is not a recoverable freeze prefix.
    with pytest.raises(IncompleteRound, match="Missing player state"):
        build_round_tensor(paused, round_meta(freeze_end=108), "de_mirage", (0, 0), [], [], 8)


def test_dropped_bomb_is_not_planted_and_age_stops():
    df = frame(range(100, 141, 8)).with_columns(
        pl.when((pl.col("tick") == 116) & (pl.col("steamid") == 1))
        .then(pl.lit(["C4 Explosive"])).otherwise(pl.col("inventory")).alias("inventory"))
    bomb = [{"round_num": 1, "tick": tick, "event": event, "X": x, "Y": y}
            for tick, event, x, y in [(100, "drop", 600, 600), (110, "pickup", 600, 600),
                                     (124, "plant", -411, -2074), (132, "defuse", -411, -2074)]]
    t, _, _, _ = build_round_tensor(df, round_meta(bomb_plant=124), "de_mirage", (0, 0), bomb, [], 8)
    g = t.numpy()[:, 560:]
    assert g[:, 15:19].argmax(axis=1).tolist() == [0, 0, 1, 2, 2, 2]
    np.testing.assert_allclose(g[:3, 19], [0.2, 0.2, 0])
    assert g[-1, 21] == g[-2, 21] == (132 - 124) / 64 / 40
    derived = compute_derived(t.numpy(), SimpleNamespace(is_visible=lambda *_: False)).reshape(-1, 10, 9)
    assert (derived[:3, :, 7] == 1).all()
    assert (derived[3:, :, 7] < 1).all()


def test_scores_follow_rosters_even_when_a_round_is_rejected(tmp_path):
    rounds, frames = [], []
    for rn, swapped, winner in [(1, False, "t"), (2, False, "t"), (3, True, "ct"),
                                 (4, True, "t"), (5, False, "ct")]:
        ticks = [rn * 100, rn * 100 + 8]
        df = frame(ticks, rn, swapped)
        if rn == 2:
            df = df.filter(~((pl.col("tick") == ticks[0]) & (pl.col("steamid") == 1)))
        frames.append(df)
        rounds.append(round_meta(rn, freeze_end=ticks[0], end=ticks[-1], winner=winner))
    parq = tmp_path / "match_ticks.parquet"
    pl.concat(frames).write_parquet(parq)
    for name, value in {"rounds": rounds, "bomb": [], "kills": [],
                        "header": {"map_name": "de_mirage"}}.items():
        (tmp_path / f"match_{name}.json").write_text(json.dumps(value))
    tensors, metas, _, _, summary = process_demo(parq, 8)
    assert [m["round_num"] for m in metas] == [1, 3, 4, 5]
    assert [r["round_num"] for r in summary["rejected_rounds"]] == [2]
    assert [(t[0, 571:573] * 16).tolist() for t in tensors] == [[0, 0], [0, 2], [0, 3], [3, 1]]
