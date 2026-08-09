from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fit_dist_edges import (
    CANONICAL_MAPS,
    EXPECTED_ROUNDS,
    TRAIN_PT,
    parse_args,
    validate_fit_scope,
)


def test_canonical_fit_arguments_are_guarded():
    args = parse_args([])
    assert args.train_pt == TRAIN_PT
    assert args.maps.split(",") == list(CANONICAL_MAPS)
    assert "de_overpass" not in args.maps
    assert args.expected_rounds == EXPECTED_ROUNDS == 3573


def test_fit_arguments_keep_positional_blob_compatibility():
    args = parse_args(["custom.pt", "--expected-rounds", "0"])
    assert args.train_pt == "custom.pt"
    assert args.expected_rounds == 0


def test_fit_scope_guards_round_count_and_ood_holdout():
    maps = ",".join(CANONICAL_MAPS)
    metas = [{"map_name": "de_mirage"}, {"map_name": "de_nuke"}]
    assert validate_fit_scope(metas, maps, expected_rounds=2) == {
        "de_mirage",
        "de_nuke",
    }

    with pytest.raises(AssertionError, match="expected 3"):
        validate_fit_scope(metas, maps, expected_rounds=3)
    with pytest.raises(AssertionError, match="OOD holdout"):
        validate_fit_scope([{"map_name": "de_overpass"}], maps + ",de_overpass", 0)
