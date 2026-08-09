"""Fixture certification for the dimension-preserving P2 corpus patch."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from patch_corpus_p2 import (
    G_BOMB,
    G_BOMB_AGE,
    G_BX,
    G_BY,
    G_PHASE,
    HAS_C4,
    MAX_END_FRAMES,
    MISSING_PLACE,
    N_PLAYERS,
    RAW_PPD,
    attach_local_sidecars,
    check_p2_blob,
    p2_keep_length,
    patch_round_p2,
    update_meta_after_crop,
)

GLOBAL_DIM = 37
FEATURE_DIM = N_PLAYERS * RAW_PPD + GLOBAL_DIM


def _round(n: int = 12) -> torch.Tensor:
    tensor = torch.zeros(n, FEATURE_DIM)
    gs = N_PLAYERS * RAW_PPD
    tensor[:, gs + G_PHASE + 1] = 1.0  # live
    tensor[:, gs + G_BOMB] = 1.0       # none
    return tensor


def test_dropped_bomb_position_persists_until_pickup():
    tensor = _round(8)
    gs = N_PLAYERS * RAW_PPD
    carrier = 2
    tensor[:3, carrier * RAW_PPD + HAS_C4] = 1.0
    tensor[:3, gs + G_BOMB] = 0.0
    tensor[:3, gs + G_BOMB + 1] = 1.0
    tensor[2, carrier * RAW_PPD] = 0.25
    tensor[2, carrier * RAW_PPD + 1] = -0.40
    # Pickup at frame 5. P1 coordinates are zero again while carried.
    tensor[5:, 4 * RAW_PPD + HAS_C4] = 1.0
    tensor[5:, gs + G_BOMB] = 0.0
    tensor[5:, gs + G_BOMB + 1] = 1.0

    patched, stats = patch_round_p2(tensor, RAW_PPD)

    assert torch.allclose(patched[3:5, gs + G_BX], torch.tensor([0.25, 0.25]))
    assert torch.allclose(patched[3:5, gs + G_BY], torch.tensor([-0.40, -0.40]))
    assert torch.equal(patched[3:5, gs + G_BOMB], torch.ones(2))
    assert patched[5:, gs + G_BX].eq(0).all()
    assert stats["drop_events"] == 1
    assert stats["dropped_frames"] == 2


def test_plant_position_wins_over_prior_drop():
    tensor = _round(7)
    gs = N_PLAYERS * RAW_PPD
    tensor[:2, HAS_C4] = 1.0
    tensor[:2, gs + G_BOMB] = 0.0
    tensor[:2, gs + G_BOMB + 1] = 1.0
    tensor[1, 0] = 0.1
    tensor[1, 1] = 0.2
    tensor[4:, gs + G_BOMB] = 0.0
    tensor[4:, gs + G_BOMB + 2] = 1.0
    tensor[4:, gs + G_BX] = 0.8
    tensor[4:, gs + G_BY] = -0.7
    tensor[4:, gs + G_PHASE + 1] = 0.0
    tensor[4:, gs + G_PHASE + 2] = 1.0

    patched, _ = patch_round_p2(tensor, RAW_PPD)

    assert torch.allclose(patched[2:4, gs + G_BX], torch.tensor([0.1, 0.1]))
    assert patched[4:, gs + G_BX].eq(0.8).all()
    assert patched[4:, gs + G_BY].eq(-0.7).all()


def test_end_tail_crop_and_bomb_age_clamp():
    tensor = _round(90)
    gs = N_PLAYERS * RAW_PPD
    first_end = 10
    tensor[first_end:, gs + G_PHASE + 1] = 0.0
    tensor[first_end:, gs + G_PHASE + 3] = 1.0
    tensor[:, gs + G_BOMB_AGE] = torch.linspace(-0.2, 2.2, len(tensor))

    assert p2_keep_length(tensor, RAW_PPD) == first_end + MAX_END_FRAMES
    patched, stats = patch_round_p2(tensor, RAW_PPD)

    assert patched.shape[0] == first_end + MAX_END_FRAMES
    assert patched[:, gs + G_BOMB_AGE].min() == 0
    assert patched[:, gs + G_BOMB_AGE].max() == 1
    assert stats["frames_cropped"] == 90 - len(patched)
    assert stats["rounds_cropped"] == 1


def _write_demo_parquet(path: Path) -> None:
    rows = []
    for tick, places in ((100, ("TSpawn", "CTSpawn")), (108, ("Mid", "A Site"))):
        for side, base_sid, place in (("t", 10, places[0]), ("ct", 20, places[1])):
            for player in range(5):
                rows.append({
                    "tick": tick,
                    "round_num": 1,
                    "steamid": base_sid + player,
                    "name": f"{side}{player}",
                    "side": side,
                    "place": place,
                })
    pl.DataFrame(rows).write_parquet(path)


def test_sidecar_join_is_source_keyed_and_slot_ordered(tmp_path):
    _write_demo_parquet(tmp_path / "demo_ticks.parquet")
    tensor = _round(2)
    blob = {
        "tensors": [tensor],
        "metas": [{
            "demo_stem": "demo",
            "round_num": 1,
            "n_ticks": 2,
            "first_tick": 100,
            "last_tick": 108,
            "downsample": 1,
        }],
    }
    vocab = [MISSING_PLACE, "A Site", "CTSpawn", "Mid", "TSpawn"]

    places, report, exact = attach_local_sidecars(blob, tmp_path, vocab, [2])

    assert report == {
        "verified_rounds": 1,
        "source_unavailable_rounds": 0,
        "source_key_mismatch_rounds": 0,
        "verified_stems": 1,
    }
    assert blob["metas"][0]["slot_steamids"] == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert blob["metas"][0]["slot_names"] == [
        "t0", "t1", "t2", "t3", "t4", "ct0", "ct1", "ct2", "ct3", "ct4"
    ]
    assert places[0][0, :5].eq(vocab.index("TSpawn")).all()
    assert places[0][1, :5].eq(vocab.index("Mid")).all()
    assert places[0][0, 5:].eq(vocab.index("CTSpawn")).all()
    assert places[0][1, 5:].eq(vocab.index("A Site")).all()
    assert exact == {0: 108}


def test_sidecar_rejects_same_stem_with_different_source_key(tmp_path):
    _write_demo_parquet(tmp_path / "collision_ticks.parquet")
    blob = {
        "tensors": [_round(2)],
        "metas": [{
            "demo_stem": "collision",
            "round_num": 1,
            "n_ticks": 2,
            "first_tick": 999,
            "last_tick": 1007,
            "downsample": 1,
        }],
    }

    places, report, exact = attach_local_sidecars(blob, tmp_path, [MISSING_PLACE], [2])

    assert places[0].eq(0).all()
    assert report["source_key_mismatch_rounds"] == 1
    assert exact == {}
    assert "slot_steamids" not in blob["metas"][0]


def test_p2_structural_check_and_crop_metadata():
    tensor = _round(5)
    meta = {"n_ticks": 5, "first_tick": 100, "last_tick": 132, "downsample": 8}
    update_meta_after_crop(meta, old_length=5, new_length=3, exact_last_tick=116)
    blob = {
        "tensors": [tensor[:3]],
        "metas": [meta],
        "place_ids": [torch.zeros(3, 10, dtype=torch.int16)],
        "place_vocab": [MISSING_PLACE],
        "per_player_dim": RAW_PPD,
    }

    assert check_p2_blob(blob) == []
    assert meta["n_ticks"] == 3
    assert meta["last_tick"] == 116
    assert meta["p2_original_last_tick"] == 132

    blob["place_ids"][0][0, 0] = 4
    assert any("outside vocabulary" in message for message in check_p2_blob(blob))
