"""Unit test for scripts/_corpus.clean_blob on a synthetic blob (ported from
the inline unit check noted in the _corpus.py commit, 13ee681): parallel
per-round lists are filtered in LOCKSTEP, scalars and non-parallel lists are
untouched, and EXCLUDED_MAPS carries exactly the D1/D2 maps."""
from __future__ import annotations

import torch
from _corpus import EXCLUDED_MAPS, clean_blob


def _synthetic_blob():
    maps = ["de_mirage", "de_anubis", "de_nuke", "de_train", "de_dust2"]
    return {
        "tensors": [torch.full((4, 3), float(i)) for i in range(5)],
        "metas": [{"round_num": i, "map_name": m, "match_id": str(100 + i)}
                  for i, m in enumerate(maps)],
        "event_labels": [torch.full((4,), i, dtype=torch.int64) for i in range(5)],
        "event_times": [torch.full((4,), 10.0 * i) for i in range(5)],
        # scalars must survive untouched
        "feature_dim": 597,
        "downsample": 8,
        "schema_version": "feature_schema_v2",
        # a list that is NOT round-parallel (different length) must be untouched
        "notes": ["a", "b"],
    }


def test_excluded_maps_content():
    assert frozenset({"de_anubis", "de_train"}) == EXCLUDED_MAPS


def test_clean_blob_lockstep_filtering():
    blob = _synthetic_blob()
    kept = clean_blob(blob, verbose=False)

    assert kept == 3
    kept_idx = [0, 2, 4]  # mirage, nuke, dust2
    assert [m["round_num"] for m in blob["metas"]] == kept_idx
    assert all(m["map_name"] not in EXCLUDED_MAPS for m in blob["metas"])
    # every parallel list filtered by the SAME index set (indices stay aligned)
    assert [float(t[0, 0]) for t in blob["tensors"]] == [float(i) for i in kept_idx]
    assert [int(e[0]) for e in blob["event_labels"]] == kept_idx
    assert [float(e[0]) for e in blob["event_times"]] == [10.0 * i for i in kept_idx]
    assert len(blob["tensors"]) == len(blob["metas"]) == len(blob["event_labels"]) == 3


def test_clean_blob_scalars_and_nonparallel_untouched():
    blob = _synthetic_blob()
    clean_blob(blob, verbose=False)
    assert blob["feature_dim"] == 597
    assert blob["downsample"] == 8
    assert blob["schema_version"] == "feature_schema_v2"
    assert blob["notes"] == ["a", "b"]


def test_clean_blob_noop_when_all_kept():
    blob = _synthetic_blob()
    for m, name in zip(blob["metas"], ["de_mirage"] * 5):
        m["map_name"] = name
    kept = clean_blob(blob, verbose=False)
    assert kept == 5
    assert len(blob["tensors"]) == 5
