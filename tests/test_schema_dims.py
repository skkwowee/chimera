"""Schema-vs-code dimension consistency (the test that would have caught
D1/anubis): parse feature_schema_v1.json and assert the layout arithmetic and
the code constants in build_tick_sequences.py agree."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SCHEMA_PATH = (Path(__file__).resolve().parent.parent
               / "data" / "processed" / "tick_sequences" / "feature_schema_v1.json")

pytestmark = pytest.mark.skipif(
    not SCHEMA_PATH.exists(),
    reason=f"schema json absent (CI has no data dir): {SCHEMA_PATH}")


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


def _layout_width(entries: list[str]) -> int:
    """Sum layout entries, expanding 'name(N)' one-hot/multi-dim entries."""
    total = 0
    for e in entries:
        m = re.search(r"\((\d+)\)", e)
        total += int(m.group(1)) if m else 1
    return total


def test_feature_dim_arithmetic(schema):
    assert schema["feature_dim"] == 597
    assert schema["n_players"] == 10
    assert schema["per_player_dim"] == 56
    assert schema["global_dim"] == 37
    assert 597 == 10 * 56 + 37
    assert schema["feature_dim"] == (schema["n_players"] * schema["per_player_dim"]
                                     + schema["global_dim"])


def test_per_player_layout_width(schema):
    # 15 scalars + 2x onehot(18) + 5 util = 56
    assert _layout_width(schema["per_player_layout"]) == schema["per_player_dim"] == 56
    onehots = [e for e in schema["per_player_layout"] if "(18)" in e]
    assert len(onehots) == 2, "expected primary+secondary weapon onehot(18)"
    assert len(schema["weapon_categories"]) == 18


def test_global_layout_width(schema):
    assert _layout_width(schema["global_layout"]) == schema["global_dim"] == 37


def test_map_vocab_matches_code(schema):
    from build_tick_sequences import MAP_VOCAB
    assert len(schema["map_vocab"]) == 7
    assert len(MAP_VOCAB) == 7, "code MAP_VOCAB drifted from schema (D1 class bug)"
    assert schema["map_vocab"] == MAP_VOCAB


def test_phase_vocab_order(schema):
    from build_tick_sequences import PHASE_VOCAB
    assert schema["phase_vocab"] == ["freeze", "live", "post_plant", "end"]
    assert PHASE_VOCAB == ["freeze", "live", "post_plant", "end"]
