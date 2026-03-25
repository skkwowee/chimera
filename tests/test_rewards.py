"""Unit tests for CS2 reward functions."""

import json
import pytest

from src.training.rewards import (
    format_gate_reward,
    perceptual_accuracy_reward,
    decision_alignment_reward,
    outcome_reward,
    compute_player_contribution,
    compute_outcome_modulation,
    _classify_movement,
    _classify_objective_direction,
    _extract_utility_types,
    _classify_engagement_timing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_json(**overrides) -> str:
    """Build a valid model response JSON string."""
    obj = {
        "game_state": {"player_health": 100},
        "analysis": {"situation": "ok"},
        "advice": {"primary_action": "hold angle", "fallback": "fall back", "reasoning": "safe"},
    }
    obj.update(overrides)
    return json.dumps(obj)


def _wrap_code_block(json_str: str) -> str:
    return f"```json\n{json_str}\n```"


# ===================================================================
# 1. format_gate_reward
# ===================================================================

class TestFormatGateReward:

    def test_valid_json_returns_1(self):
        resp = _make_valid_json()
        assert format_gate_reward(resp) == 1.0

    def test_invalid_json_returns_0(self):
        assert format_gate_reward("this is not json at all") == 0.0

    def test_missing_required_keys(self):
        # Has game_state but missing analysis and advice
        resp = json.dumps({"game_state": {"hp": 100}})
        assert format_gate_reward(resp) == 0.0

    def test_json_in_markdown_code_block(self):
        resp = _wrap_code_block(_make_valid_json())
        assert format_gate_reward(resp) == 1.0

    def test_empty_string(self):
        assert format_gate_reward("") == 0.0

    def test_keys_not_dicts(self):
        resp = json.dumps({"game_state": "string", "analysis": {}, "advice": {}})
        assert format_gate_reward(resp) == 0.0


# ===================================================================
# 2. perceptual_accuracy_reward
# ===================================================================

class TestPerceptualAccuracyReward:

    def _gt(self, **fields):
        return {"game_state": fields}

    def test_perfect_match(self):
        gt = self._gt(player_health=100, player_armor=100, alive_teammates=3)
        resp = json.dumps({
            "game_state": {"player_health": 100, "player_armor": 100, "alive_teammates": 3},
            "analysis": {}, "advice": {},
        })
        assert perceptual_accuracy_reward(resp, ground_truth=gt) == 1.0

    def test_partial_match(self):
        gt = self._gt(player_health=100, player_armor=50)
        resp = json.dumps({
            "game_state": {"player_health": 100, "player_armor": 0},
            "analysis": {}, "advice": {},
        })
        score = perceptual_accuracy_reward(resp, ground_truth=gt)
        # health matches (1.0), armor is way off (0.0) → average ~0.5
        assert 0.4 <= score <= 0.6

    def test_numeric_within_tolerance(self):
        gt = self._gt(player_health=100)
        resp = json.dumps({
            "game_state": {"player_health": 95},
            "analysis": {}, "advice": {},
        })
        assert perceptual_accuracy_reward(resp, ground_truth=gt) == 1.0

    def test_numeric_out_of_tolerance(self):
        gt = self._gt(player_health=100)
        resp = json.dumps({
            "game_state": {"player_health": 50},
            "analysis": {}, "advice": {},
        })
        assert perceptual_accuracy_reward(resp, ground_truth=gt) < 1.0

    def test_string_field_exact_match(self):
        gt = self._gt(weapon_primary="AK-47")
        resp = json.dumps({
            "game_state": {"weapon_primary": "ak-47"},
            "analysis": {}, "advice": {},
        })
        # case-insensitive
        assert perceptual_accuracy_reward(resp, ground_truth=gt) == 1.0

    def test_list_field_jaccard(self):
        gt = self._gt(utility=["smoke", "flash", "he"])
        resp = json.dumps({
            "game_state": {"utility": ["smoke", "flash"]},
            "analysis": {}, "advice": {},
        })
        score = perceptual_accuracy_reward(resp, ground_truth=gt)
        # Jaccard = 2/3 ≈ 0.667
        assert 0.6 <= score <= 0.7

    def test_missing_ground_truth(self):
        resp = _make_valid_json()
        assert perceptual_accuracy_reward(resp, ground_truth=None) == 0.0

    def test_invalid_response(self):
        gt = self._gt(player_health=100)
        assert perceptual_accuracy_reward("not json", ground_truth=gt) == 0.0


# ===================================================================
# 3. decision_alignment_reward
# ===================================================================

class TestDecisionAlignmentReward:

    def test_behavioral_matching_vectors(self):
        """Model and pro have identical behavioral features → 1.0."""
        gt = {
            "pro_action": {
                "behavior": {
                    "movement_direction": 1,
                    "objective_direction": 1,
                    "utility_used": ["smoke"],
                    "initiated_engagement": True,
                    "engagement_delay": 0.2,
                }
            }
        }
        # Advice text that maps to: advance(push site), toward-obj(site),
        # smoke, initiate(peek/fight), delay≈0.2
        resp = json.dumps({
            "game_state": {},
            "analysis": {},
            "advice": {
                "primary_action": "push site and peek",
                "fallback": "throw a smoke",
                "reasoning": "take fight",
            },
        })
        score = decision_alignment_reward(resp, ground_truth=gt)
        assert score == 1.0

    def test_behavioral_opposite_vectors(self):
        """Model advises opposite of pro behavior → 0.0."""
        gt = {
            "pro_action": {
                "behavior": {
                    "movement_direction": -1,  # retreat
                    "objective_direction": -1,  # away
                    "utility_used": [],
                    "initiated_engagement": False,
                    "engagement_delay": 0.8,
                }
            }
        }
        # Model advises aggressive advance
        resp = json.dumps({
            "game_state": {},
            "analysis": {},
            "advice": {
                "primary_action": "push site and peek and swing",
                "fallback": "entry rush",
                "reasoning": "take fight challenge",
            },
        })
        score = decision_alignment_reward(resp, ground_truth=gt)
        assert score < 0.3

    def test_fallback_jaccard(self):
        """No behavioral features → uses Jaccard over action categories."""
        gt = {
            "pro_action": {
                "categories": ["aggressive", "utility"],
            }
        }
        resp = json.dumps({
            "game_state": {},
            "analysis": {},
            "advice": {
                "primary_action": "push and throw smoke",
                "fallback": "",
            },
        })
        score = decision_alignment_reward(resp, ground_truth=gt)
        assert score > 0.0

    def test_no_features_no_categories(self):
        gt = {"pro_action": {}}
        resp = _make_valid_json()
        assert decision_alignment_reward(resp, ground_truth=gt) == 0.0

    def test_no_ground_truth(self):
        assert decision_alignment_reward(_make_valid_json(), ground_truth=None) == 0.0


# ===================================================================
# 4. outcome_reward
# ===================================================================

class TestOutcomeReward:

    def test_with_round_won_and_contribution(self):
        gt = {
            "pro_action": {
                "categories": ["aggressive"],
            },
            "round_won": True,
            "player_contribution": {
                "damage_dealt": 200,
                "max_round_damage": 200,
                "survival_time": 60,
                "round_duration": 60,
                "objective_action": True,
            },
        }
        resp = json.dumps({
            "game_state": {},
            "analysis": {},
            "advice": {"primary_action": "push and entry", "fallback": ""},
        })
        score = outcome_reward(resp, ground_truth=gt)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_no_ground_truth(self):
        assert outcome_reward(_make_valid_json(), ground_truth=None) == 0.0

    def test_no_round_won_falls_back_to_alignment(self):
        gt = {
            "pro_action": {"categories": ["hold"]},
        }
        resp = json.dumps({
            "game_state": {},
            "analysis": {},
            "advice": {"primary_action": "hold angle and wait", "fallback": ""},
        })
        score = outcome_reward(resp, ground_truth=gt)
        alignment = decision_alignment_reward(resp, ground_truth=gt)
        assert score == alignment


# ===================================================================
# 5. compute_player_contribution
# ===================================================================

class TestComputePlayerContribution:

    def test_full_contribution(self):
        c = {
            "damage_dealt": 300,
            "max_round_damage": 300,
            "survival_time": 90,
            "round_duration": 90,
            "objective_action": True,
        }
        assert compute_player_contribution(c) == 1.0

    def test_zero_contribution(self):
        c = {
            "damage_dealt": 0,
            "max_round_damage": 300,
            "survival_time": 0,
            "round_duration": 90,
            "objective_action": False,
        }
        assert compute_player_contribution(c) == 0.0

    def test_empty_dict_returns_0_5(self):
        # Empty dict is falsy → early return 0.5
        assert compute_player_contribution({}) == 0.5

    def test_non_dict_list_returns_0_5(self):
        assert compute_player_contribution([]) == 0.5

    def test_none_input_returns_0_5(self):
        assert compute_player_contribution(None) == 0.5

    def test_non_dict_returns_0_5(self):
        assert compute_player_contribution("invalid") == 0.5

    def test_partial_contribution(self):
        c = {
            "damage_dealt": 150,
            "max_round_damage": 300,
            "survival_time": 45,
            "round_duration": 90,
            "objective_action": False,
        }
        # 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0 = 0.35
        assert compute_player_contribution(c) == pytest.approx(0.35)


# ===================================================================
# 6. compute_outcome_modulation
# ===================================================================

class TestComputeOutcomeModulation:

    def test_agree_win(self):
        # W=1, a=1, phi=1 → 1*(0.5+0.5) = 1.0
        assert compute_outcome_modulation(True, 1.0, 1.0) == pytest.approx(1.0)

    def test_agree_lose(self):
        # W=0, a=1, phi=1 → 1*(0.5-0.3) = 0.2
        assert compute_outcome_modulation(False, 1.0, 1.0) == pytest.approx(0.2)

    def test_deviate_win(self):
        # W=1, a=0, phi=1 → 1*(0.5+0) = 0.5
        assert compute_outcome_modulation(True, 1.0, 0.0) == pytest.approx(0.5)

    def test_deviate_lose(self):
        # W=0, a=0, phi=1 → 1*(0.5-0) = 0.5
        assert compute_outcome_modulation(False, 1.0, 0.0) == pytest.approx(0.5)

    def test_phi_zero(self):
        # phi=0 → always 0 regardless of other inputs
        assert compute_outcome_modulation(True, 0.0, 1.0) == 0.0
        assert compute_outcome_modulation(False, 0.0, 1.0) == 0.0
        assert compute_outcome_modulation(True, 0.0, 0.0) == 0.0
        assert compute_outcome_modulation(False, 0.0, 0.0) == 0.0

    def test_boundary_phi1_a05(self):
        # W=1, phi=1, a=0.5 → 1*(0.5+0.25) = 0.75
        assert compute_outcome_modulation(True, 1.0, 0.5) == pytest.approx(0.75)
        # W=0, phi=1, a=0.5 → 1*(0.5-0.15) = 0.35
        assert compute_outcome_modulation(False, 1.0, 0.5) == pytest.approx(0.35)


# ===================================================================
# 7. Behavioral feature extraction
# ===================================================================

class TestBehavioralExtraction:

    def test_classify_movement_advance(self):
        assert _classify_movement("push site and take fight") == 1

    def test_classify_movement_retreat(self):
        assert _classify_movement("fall back and save") == -1

    def test_classify_movement_hold(self):
        assert _classify_movement("hold angle and wait") == 0

    def test_classify_movement_empty(self):
        assert _classify_movement("") is None

    def test_classify_movement_no_keywords(self):
        assert _classify_movement("do something random") is None

    def test_extract_utility_types(self):
        result = _extract_utility_types("throw a smoke and flash")
        assert result == {"smoke", "flash"}

    def test_extract_utility_types_empty(self):
        assert _extract_utility_types("") == set()

    def test_classify_engagement_initiate(self):
        assert _classify_engagement_timing("peek and take the fight") == 1

    def test_classify_engagement_wait(self):
        assert _classify_engagement_timing("wait and hold for them") == 0

    def test_classify_engagement_empty(self):
        assert _classify_engagement_timing("") is None

    def test_classify_objective_toward(self):
        assert _classify_objective_direction("push site and plant") == 1

    def test_classify_objective_away(self):
        assert _classify_objective_direction("rotate and flank") == -1
