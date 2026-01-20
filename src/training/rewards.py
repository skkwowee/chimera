"""
CS2-specific reward functions for GRPO training.

Provides reward signals for:
- JSON format validity
- Game state extraction accuracy
- Reasoning and advice quality
"""

import json
import re
from typing import Any


# Valid enum values for categorical fields
VALID_ROUND_PHASES = {"buy", "playing", "freezetime", "post-plant", "warmup"}
VALID_PLAYER_SIDES = {"T", "CT"}
VALID_BOMB_STATUSES = {"carried", "planted", "dropped", "defused", "exploded", None}
VALID_ECONOMY_ASSESSMENTS = {"full-buy", "half-buy", "eco", "force-buy", "save"}
VALID_ROUND_IMPORTANCE = {"low", "medium", "high", "critical"}

# Numeric fields and their reasonable ranges for CS2
NUMERIC_FIELDS = {
    "player_health": (0, 100),
    "player_armor": (0, 100),
    "player_money": (0, 16000),
    "team_money_total": (0, 80000),
    "alive_teammates": (0, 4),
    "alive_enemies": (0, 5),
    "visible_enemies": (0, 5),
}


def _extract_json_from_response(response: str) -> dict | None:
    """
    Extract JSON object from model response.

    Handles responses with markdown code blocks or raw JSON.
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_pattern = r"\{[\s\S]*\}"
    match = re.search(json_pattern, response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def json_format_reward(response: str) -> float:
    """
    Reward for valid JSON structure with expected fields.

    Scoring:
    - +0.5 for valid JSON
    - +0.2 for having game_state field
    - +0.15 for having analysis field
    - +0.15 for having advice field

    Args:
        response: Raw model output string

    Returns:
        Reward score between 0.0 and 1.0
    """
    reward = 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    # Valid JSON structure
    reward += 0.5

    # Check for expected top-level fields
    if "game_state" in parsed and isinstance(parsed["game_state"], dict):
        reward += 0.2

    if "analysis" in parsed and isinstance(parsed["analysis"], dict):
        reward += 0.15

    if "advice" in parsed and isinstance(parsed["advice"], dict):
        reward += 0.15

    return reward


def game_state_accuracy_reward(
    response: str,
    ground_truth: dict[str, Any],
    numeric_tolerance: float = 0.1,
) -> float:
    """
    Reward for game state field accuracy compared to ground truth.

    Scoring per field:
    - Numeric fields: 1.0 if within tolerance, partial credit otherwise
    - Categorical fields: 1.0 for exact match, 0.0 otherwise
    - List fields (utility): Jaccard similarity
    - String fields: 1.0 for exact match (case-insensitive)

    Args:
        response: Raw model output string
        ground_truth: Dict with ground truth game_state values
        numeric_tolerance: Relative tolerance for numeric fields (0.1 = 10%)

    Returns:
        Reward score between 0.0 and 1.0
    """
    parsed = _extract_json_from_response(response)
    if parsed is None or "game_state" not in parsed:
        return 0.0

    predicted = parsed["game_state"]
    gt_state = ground_truth.get("game_state", ground_truth)

    if not isinstance(predicted, dict) or not isinstance(gt_state, dict):
        return 0.0

    field_scores = []

    for field, (min_val, max_val) in NUMERIC_FIELDS.items():
        if field not in gt_state:
            continue

        gt_val = gt_state[field]
        pred_val = predicted.get(field)

        if gt_val is None:
            # Ground truth is null, reward if prediction is also null
            field_scores.append(1.0 if pred_val is None else 0.0)
            continue

        if pred_val is None:
            field_scores.append(0.0)
            continue

        try:
            gt_val = float(gt_val)
            pred_val = float(pred_val)
        except (TypeError, ValueError):
            field_scores.append(0.0)
            continue

        # Calculate relative error
        if gt_val == 0:
            score = 1.0 if pred_val == 0 else 0.0
        else:
            rel_error = abs(pred_val - gt_val) / abs(gt_val)
            if rel_error <= numeric_tolerance:
                score = 1.0
            else:
                # Partial credit: linear decay from tolerance to 2x tolerance
                score = max(0.0, 1.0 - (rel_error - numeric_tolerance) / numeric_tolerance)

        field_scores.append(score)

    # Categorical fields
    categorical_fields = [
        ("round_phase", VALID_ROUND_PHASES),
        ("player_side", VALID_PLAYER_SIDES),
        ("bomb_status", VALID_BOMB_STATUSES),
    ]

    for field, valid_values in categorical_fields:
        if field not in gt_state:
            continue

        gt_val = gt_state[field]
        pred_val = predicted.get(field)

        # Case-insensitive comparison for non-null values
        if gt_val is not None and pred_val is not None:
            gt_val_lower = str(gt_val).lower() if gt_val else None
            pred_val_lower = str(pred_val).lower() if pred_val else None
            score = 1.0 if gt_val_lower == pred_val_lower else 0.0
        else:
            score = 1.0 if gt_val == pred_val else 0.0

        field_scores.append(score)

    # String fields (map_name, site, weapons)
    string_fields = ["map_name", "site", "weapon_primary", "weapon_secondary"]

    for field in string_fields:
        if field not in gt_state:
            continue

        gt_val = gt_state[field]
        pred_val = predicted.get(field)

        if gt_val is None:
            score = 1.0 if pred_val is None else 0.0
        elif pred_val is None:
            score = 0.0
        else:
            # Case-insensitive comparison
            gt_lower = str(gt_val).lower().strip()
            pred_lower = str(pred_val).lower().strip()
            score = 1.0 if gt_lower == pred_lower else 0.0

        field_scores.append(score)

    # List field (utility)
    if "utility" in gt_state:
        gt_utility = gt_state["utility"]
        pred_utility = predicted.get("utility", [])

        if not isinstance(gt_utility, list):
            gt_utility = []
        if not isinstance(pred_utility, list):
            pred_utility = []

        # Jaccard similarity
        gt_set = set(str(x).lower() for x in gt_utility)
        pred_set = set(str(x).lower() for x in pred_utility)

        if len(gt_set) == 0 and len(pred_set) == 0:
            score = 1.0
        elif len(gt_set) == 0 or len(pred_set) == 0:
            score = 0.0
        else:
            intersection = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            score = intersection / union

        field_scores.append(score)

    if not field_scores:
        return 0.0

    return sum(field_scores) / len(field_scores)


def reasoning_quality_reward(response: str) -> float:
    """
    Reward for quality of analysis and advice reasoning.

    Scoring components:
    - Analysis quality (0.5 max):
      - Has situation_summary with reasonable length: +0.15
      - Has valid economy_assessment enum: +0.15
      - Has valid round_importance enum: +0.1
      - Has immediate_threats list: +0.05
      - Has opportunities list: +0.05

    - Advice quality (0.5 max):
      - Has primary_action with reasonable length: +0.2
      - Has reasoning with reasonable length: +0.15
      - Has fallback: +0.1
      - Has callout: +0.05

    Args:
        response: Raw model output string

    Returns:
        Reward score between 0.0 and 1.0
    """
    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    reward = 0.0

    # Analysis quality
    analysis = parsed.get("analysis", {})
    if isinstance(analysis, dict):
        # Situation summary
        summary = analysis.get("situation_summary", "")
        if isinstance(summary, str) and 10 <= len(summary) <= 500:
            reward += 0.15

        # Economy assessment
        economy = analysis.get("economy_assessment")
        if economy in VALID_ECONOMY_ASSESSMENTS:
            reward += 0.15

        # Round importance
        importance = analysis.get("round_importance")
        if importance in VALID_ROUND_IMPORTANCE:
            reward += 0.1

        # Threats list
        threats = analysis.get("immediate_threats", [])
        if isinstance(threats, list) and len(threats) > 0:
            reward += 0.05

        # Opportunities list
        opportunities = analysis.get("opportunities", [])
        if isinstance(opportunities, list) and len(opportunities) > 0:
            reward += 0.05

    # Advice quality
    advice = parsed.get("advice", {})
    if isinstance(advice, dict):
        # Primary action
        action = advice.get("primary_action", "")
        if isinstance(action, str) and 5 <= len(action) <= 300:
            reward += 0.2

        # Reasoning
        reasoning = advice.get("reasoning", "")
        if isinstance(reasoning, str) and 10 <= len(reasoning) <= 500:
            reward += 0.15

        # Fallback
        fallback = advice.get("fallback", "")
        if isinstance(fallback, str) and len(fallback) > 0:
            reward += 0.1

        # Callout
        callout = advice.get("callout", "")
        if isinstance(callout, str) and len(callout) > 0:
            reward += 0.05

    return reward


def combined_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    format_weight: float = 0.3,
    accuracy_weight: float = 0.5,
    reasoning_weight: float = 0.2,
) -> float:
    """
    Weighted combination of all reward components.

    Default weights emphasize accuracy (50%), with format (30%) and
    reasoning quality (20%) as secondary signals.

    Args:
        response: Raw model output string
        ground_truth: Dict with ground truth values (required for accuracy)
        format_weight: Weight for JSON format reward
        accuracy_weight: Weight for game state accuracy reward
        reasoning_weight: Weight for reasoning quality reward

    Returns:
        Combined reward score between 0.0 and 1.0
    """
    # Normalize weights
    total_weight = format_weight + accuracy_weight + reasoning_weight
    format_weight /= total_weight
    accuracy_weight /= total_weight
    reasoning_weight /= total_weight

    format_score = json_format_reward(response)
    reasoning_score = reasoning_quality_reward(response)

    # Accuracy requires ground truth
    if ground_truth is not None:
        accuracy_score = game_state_accuracy_reward(response, ground_truth)
    else:
        # If no ground truth, redistribute weight to other components
        accuracy_score = 0.0
        format_weight += accuracy_weight / 2
        reasoning_weight += accuracy_weight / 2
        accuracy_weight = 0.0

    return (
        format_weight * format_score
        + accuracy_weight * accuracy_score
        + reasoning_weight * reasoning_score
    )
