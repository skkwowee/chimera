"""
CS2 reward functions for GRPO training.

Two-stage training: SFT teaches visual grounding (reading the HUD), then GRPO
teaches strategic reasoning using pro demo outcomes as the gradient signal.

Rewards split into two groups:

  Vision (prevent SFT regression during RL):
    1. Format gate        — binary 0/1, invalid JSON gets nothing
    2. Hard field accuracy — verifiable HUD-readable fields (health, armor, weapons)
    3. Soft field accuracy — inferential fields (money, bomb status, map)

  Reasoning (the RL training signal):
    4. Decision alignment — model's advice vs what the pro actually did
    5. Outcome reward     — alignment modulated by round result
    6. Consistency        — does reasoning follow from perceived game state
    7. Reasoning quality  — structural quality of analysis and advice

Decision reward ground truth schema (from demo data):
    pro_action: {
        "categories": list[str]   — action types from ACTION_TAXONOMY
        "description": str        — what the pro actually did
    }
    round_won: bool               — did the pro's team win this round?

Additional criteria TBD — extend REWARD_FUNCTIONS and adjust weights.
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

# Hard fields: directly readable from the HUD, high signal
# Values are (min, max) ranges for CS2
HARD_FIELDS = {
    "player_health": (0, 100),
    "player_armor": (0, 100),
    "alive_teammates": (0, 4),
    "alive_enemies": (0, 5),
    "visible_enemies": (0, 5),
}

HARD_STRING_FIELDS = ["weapon_primary", "weapon_secondary"]
HARD_CATEGORICAL_FIELDS = [
    ("player_side", VALID_PLAYER_SIDES),
    ("round_phase", VALID_ROUND_PHASES),
]

# Soft fields: require inference or reading small text
SOFT_FIELDS = {
    "player_money": (0, 16000),
    "team_money_total": (0, 80000),
}

SOFT_STRING_FIELDS = ["map_name", "site"]
SOFT_CATEGORICAL_FIELDS = [
    ("bomb_status", VALID_BOMB_STATUSES),
]


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


# ---------------------------------------------------------------------------
# Reward component helpers
# ---------------------------------------------------------------------------

def _score_numeric_field(
    predicted: dict,
    gt_state: dict,
    field: str,
    tolerance: float = 0.1,
) -> float | None:
    """Score a single numeric field. Returns None if field missing from ground truth."""
    if field not in gt_state:
        return None

    gt_val = gt_state[field]
    pred_val = predicted.get(field)

    if gt_val is None:
        return 1.0 if pred_val is None else 0.0
    if pred_val is None:
        return 0.0

    try:
        gt_val = float(gt_val)
        pred_val = float(pred_val)
    except (TypeError, ValueError):
        return 0.0

    if gt_val == 0:
        return 1.0 if pred_val == 0 else 0.0

    rel_error = abs(pred_val - gt_val) / abs(gt_val)
    if rel_error <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (rel_error - tolerance) / tolerance)


def _score_string_field(predicted: dict, gt_state: dict, field: str) -> float | None:
    """Score a string field (case-insensitive). Returns None if missing from GT."""
    if field not in gt_state:
        return None

    gt_val = gt_state[field]
    pred_val = predicted.get(field)

    if gt_val is None:
        return 1.0 if pred_val is None else 0.0
    if pred_val is None:
        return 0.0

    return 1.0 if str(gt_val).lower().strip() == str(pred_val).lower().strip() else 0.0


def _score_categorical_field(
    predicted: dict,
    gt_state: dict,
    field: str,
) -> float | None:
    """Score a categorical field. Returns None if missing from GT."""
    if field not in gt_state:
        return None

    gt_val = gt_state[field]
    pred_val = predicted.get(field)

    if gt_val is not None and pred_val is not None:
        return 1.0 if str(gt_val).lower() == str(pred_val).lower() else 0.0
    return 1.0 if gt_val == pred_val else 0.0


def _score_list_field(predicted: dict, gt_state: dict, field: str) -> float | None:
    """Score a list field via Jaccard similarity. Returns None if missing from GT."""
    if field not in gt_state:
        return None

    gt_list = gt_state[field]
    pred_list = predicted.get(field, [])

    if not isinstance(gt_list, list):
        gt_list = []
    if not isinstance(pred_list, list):
        pred_list = []

    gt_set = {str(x).lower() for x in gt_list}
    pred_set = {str(x).lower() for x in pred_list}

    if len(gt_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(gt_set) == 0 or len(pred_set) == 0:
        return 0.0

    intersection = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    return intersection / union


# ---------------------------------------------------------------------------
# Action taxonomy — categorizes both model output and pro actions
# ---------------------------------------------------------------------------

ACTION_TAXONOMY = {
    "aggressive": [
        "push", "entry", "rush", "execute", "take", "aggress", "peek",
        "wide peek", "dry peek", "swing", "w key",
    ],
    "hold": [
        "hold", "anchor", "wait", "passive", "camp", "stay", "setup",
        "play time", "hold angle", "default",
    ],
    "rotate": [
        "rotate", "flank", "lurk", "reposition", "move to", "shift",
        "wrap", "backstab",
    ],
    "fall_back": [
        "fall back", "retreat", "save", "disengage", "safe", "avoid",
        "give up", "play retake", "pull back",
    ],
    "utility": [
        "smoke", "flash", "molly", "molotov", "incendiary", "nade",
        "grenade", "he grenade", "utility", "throw",
    ],
    "engage": [
        "fight", "duel", "trade", "engage", "shoot", "fire", "contact",
        "take fight", "challenge",
    ],
}


def _categorize_action(text: str) -> set[str]:
    """Map free-text action description to taxonomy categories."""
    if not text:
        return set()
    text_lower = text.lower()
    categories = set()
    for category, keywords in ACTION_TAXONOMY.items():
        if any(kw in text_lower for kw in keywords):
            categories.add(category)
    return categories


def _action_similarity(model_cats: set[str], pro_cats: set[str]) -> float:
    """Jaccard similarity between two sets of action categories."""
    if not model_cats and not pro_cats:
        return 1.0
    if not model_cats or not pro_cats:
        return 0.0
    return len(model_cats & pro_cats) / len(model_cats | pro_cats)


def _get_model_action_text(parsed: dict) -> str:
    """Extract combined action text from parsed model output."""
    advice = parsed.get("advice", {})
    if not isinstance(advice, dict):
        return ""
    return f"{advice.get('primary_action', '')} {advice.get('fallback', '')}"


def _get_pro_categories(ground_truth: dict) -> set[str] | None:
    """Extract pro action categories from ground truth."""
    pro_action = ground_truth.get("pro_action", {})
    if not isinstance(pro_action, dict):
        return None
    cats = set(pro_action.get("categories", []))
    if not cats:
        cats = _categorize_action(pro_action.get("description", ""))
    return cats if cats else None


# ---------------------------------------------------------------------------
# 1. Format gate reward
# ---------------------------------------------------------------------------

def format_gate_reward(response: str, **kwargs) -> float:
    """
    Binary gate: 1.0 if response contains valid JSON with expected top-level
    keys, 0.0 otherwise. All other rewards are meaningless without valid JSON,
    so this acts as a hard filter.
    """
    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    # Must have all three top-level sections
    has_game_state = isinstance(parsed.get("game_state"), dict)
    has_analysis = isinstance(parsed.get("analysis"), dict)
    has_advice = isinstance(parsed.get("advice"), dict)

    return 1.0 if (has_game_state and has_analysis and has_advice) else 0.0


# ---------------------------------------------------------------------------
# 2. Hard field accuracy reward (HUD-readable, verifiable)
# ---------------------------------------------------------------------------

def hard_field_accuracy_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """
    Accuracy on fields directly readable from the HUD.

    These are high-signal, verifiable fields: health, armor, weapons, player
    counts, side, round phase. Getting these wrong means the model can't see
    the screen — this is the most important reward signal.
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None or "game_state" not in parsed:
        return 0.0

    predicted = parsed["game_state"]
    gt_state = ground_truth.get("game_state", ground_truth)

    if not isinstance(predicted, dict) or not isinstance(gt_state, dict):
        return 0.0

    scores = []

    # Numeric HUD fields
    for field in HARD_FIELDS:
        score = _score_numeric_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # String fields (weapon names)
    for field in HARD_STRING_FIELDS:
        score = _score_string_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Categorical fields
    for field, _ in HARD_CATEGORICAL_FIELDS:
        score = _score_categorical_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Utility list
    score = _score_list_field(predicted, gt_state, "utility")
    if score is not None:
        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 3. Soft field accuracy reward (inferential)
# ---------------------------------------------------------------------------

def soft_field_accuracy_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """
    Accuracy on fields that require inference or reading small/contextual text.

    Money values, bomb status, map name, site — these are harder to read and
    more forgivable to get wrong.
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None or "game_state" not in parsed:
        return 0.0

    predicted = parsed["game_state"]
    gt_state = ground_truth.get("game_state", ground_truth)

    if not isinstance(predicted, dict) or not isinstance(gt_state, dict):
        return 0.0

    scores = []

    # Numeric soft fields
    for field in SOFT_FIELDS:
        score = _score_numeric_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # String soft fields
    for field in SOFT_STRING_FIELDS:
        score = _score_string_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Categorical soft fields
    for field, _ in SOFT_CATEGORICAL_FIELDS:
        score = _score_categorical_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 4. Consistency reward (perception → reasoning coherence)
# ---------------------------------------------------------------------------

def consistency_reward(response: str, **kwargs) -> float:
    """
    Checks whether analysis and advice are coherent with the extracted game state.

    This is the reward that teaches actual reasoning — not "did you fill out
    the fields" but "does your output form a coherent chain from
    perception → analysis → action."
    """
    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    gs = parsed.get("game_state", {})
    analysis = parsed.get("analysis", {})
    advice = parsed.get("advice", {})

    if not all(isinstance(x, dict) for x in (gs, analysis, advice)):
        return 0.0

    checks_passed = 0
    checks_total = 0

    # --- Economy assessment should match money ---
    money = gs.get("player_money")
    econ = analysis.get("economy_assessment", "")
    if money is not None and econ:
        checks_total += 1
        try:
            money = float(money)
            if money >= 4000 and econ in ("full-buy", "force-buy"):
                checks_passed += 1
            elif 2000 <= money < 4000 and econ in ("half-buy", "force-buy", "eco"):
                checks_passed += 1
            elif money < 2000 and econ in ("eco", "save"):
                checks_passed += 1
        except (TypeError, ValueError):
            pass

    # --- Round importance should escalate when few players alive ---
    alive_t = gs.get("alive_teammates")
    alive_e = gs.get("alive_enemies")
    importance = analysis.get("round_importance", "")
    if alive_t is not None and alive_e is not None and importance:
        checks_total += 1
        try:
            alive_t = int(alive_t)
            alive_e = int(alive_e)
            total_alive = alive_t + alive_e
            if total_alive <= 3 and importance in ("high", "critical"):
                checks_passed += 1
            elif total_alive >= 8 and importance in ("low", "medium"):
                checks_passed += 1
            elif importance in VALID_ROUND_IMPORTANCE:
                checks_passed += 0.5  # valid but can't verify
        except (TypeError, ValueError):
            pass

    # --- Low health → advice should reflect caution ---
    health = gs.get("player_health")
    action = advice.get("primary_action", "")
    fallback = advice.get("fallback", "")
    action_lower = (action + " " + fallback).lower()
    if health is not None and action:
        checks_total += 1
        try:
            health = float(health)
            if health < 30:
                caution_words = ("fall back", "retreat", "safe", "avoid", "passive",
                                 "rotate", "disengage", "careful", "low health")
                if any(w in action_lower for w in caution_words):
                    checks_passed += 1
                else:
                    checks_passed += 0.25  # gave advice but didn't account for health
            else:
                checks_passed += 1  # healthy — any action is reasonable
        except (TypeError, ValueError):
            pass

    # --- Visible enemies → advice should acknowledge engagement ---
    visible = gs.get("visible_enemies")
    if visible is not None and action:
        checks_total += 1
        try:
            visible = int(visible)
            if visible > 0:
                engage_words = ("enem", "fight", "peek", "shoot", "engage", "trade",
                                "duel", "contact", "spotted", "push")
                if any(w in action_lower for w in engage_words):
                    checks_passed += 1
                else:
                    checks_passed += 0.25
            else:
                checks_passed += 1  # no enemies visible — any positioning is fine
        except (TypeError, ValueError):
            pass

    # --- Bomb status → advice should reflect it ---
    bomb = gs.get("bomb_status")
    if bomb and isinstance(bomb, str) and action:
        checks_total += 1
        bomb_lower = bomb.lower()
        if bomb_lower == "planted":
            defuse_words = ("defuse", "retake", "bomb", "site", "rotate", "plant")
            if any(w in action_lower for w in defuse_words):
                checks_passed += 1
            else:
                checks_passed += 0.25
        elif bomb_lower == "carried":
            # If you're carrying the bomb (T side), advice might mention planting
            side = gs.get("player_side", "")
            if isinstance(side, str) and side.upper() == "T":
                carry_words = ("plant", "bomb", "site", "execute", "entry")
                if any(w in action_lower for w in carry_words):
                    checks_passed += 1
                else:
                    checks_passed += 0.5
            else:
                checks_passed += 1  # CT side, bomb carried by enemy
        else:
            checks_passed += 1  # other statuses — less constrained

    # --- Threats mentioned should be grounded in game state ---
    threats = analysis.get("immediate_threats", [])
    if isinstance(threats, list) and len(threats) > 0:
        checks_total += 1
        # At least one threat should reference something from game state
        threat_text = " ".join(str(t).lower() for t in threats)
        grounded = False
        if visible is not None and int(visible) > 0 and "enem" in threat_text:
            grounded = True
        if bomb and bomb.lower() == "planted" and "bomb" in threat_text:
            grounded = True
        if health is not None and float(health) < 50 and "health" in threat_text:
            grounded = True
        # If we can't verify, give partial credit for having threats at all
        checks_passed += 1.0 if grounded else 0.5

    if checks_total == 0:
        return 0.0

    return checks_passed / checks_total


# ---------------------------------------------------------------------------
# 5. Reasoning quality reward
# ---------------------------------------------------------------------------

def reasoning_quality_reward(response: str, **kwargs) -> float:
    """
    Structural quality of analysis and advice fields.

    Checks that the model produces substantive reasoning with valid enum values,
    not just placeholder text. Focuses on form rather than correctness (that's
    what consistency_reward is for).
    """
    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    reward = 0.0

    # Analysis quality (0.5 max)
    analysis = parsed.get("analysis", {})
    if isinstance(analysis, dict):
        summary = analysis.get("situation_summary", "")
        if isinstance(summary, str) and 10 <= len(summary) <= 500:
            reward += 0.15

        economy = analysis.get("economy_assessment")
        if economy in VALID_ECONOMY_ASSESSMENTS:
            reward += 0.15

        importance = analysis.get("round_importance")
        if importance in VALID_ROUND_IMPORTANCE:
            reward += 0.1

        threats = analysis.get("immediate_threats", [])
        if isinstance(threats, list) and len(threats) > 0:
            reward += 0.05

        opportunities = analysis.get("opportunities", [])
        if isinstance(opportunities, list) and len(opportunities) > 0:
            reward += 0.05

    # Advice quality (0.5 max)
    advice = parsed.get("advice", {})
    if isinstance(advice, dict):
        action = advice.get("primary_action", "")
        if isinstance(action, str) and 5 <= len(action) <= 300:
            reward += 0.2

        reasoning = advice.get("reasoning", "")
        if isinstance(reasoning, str) and 10 <= len(reasoning) <= 500:
            reward += 0.15

        fallback = advice.get("fallback", "")
        if isinstance(fallback, str) and len(fallback) > 0:
            reward += 0.1

        callout = advice.get("callout", "")
        if isinstance(callout, str) and len(callout) > 0:
            reward += 0.05

    return reward


# ---------------------------------------------------------------------------
# 6. Decision alignment — model advice vs pro action
# ---------------------------------------------------------------------------

def decision_alignment_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """
    How well the model's advised action aligns with what the pro actually did.

    Compares action categories from the model's advice.primary_action against
    the pro's categorized action from demo data. Pure alignment — no outcome
    weighting (that's outcome_reward's job).
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    model_text = _get_model_action_text(parsed)
    model_cats = _categorize_action(model_text)

    pro_cats = _get_pro_categories(ground_truth)
    if pro_cats is None:
        return 0.0

    return _action_similarity(model_cats, pro_cats)


# ---------------------------------------------------------------------------
# 7. Outcome-weighted reward
# ---------------------------------------------------------------------------

def outcome_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    """
    Decision alignment modulated by round outcome.

    Signal matrix:
        Model agrees + pro wins  → 1.0  (learn from winning play)
        Model agrees + pro loses → 0.2  (endorsed a losing play)
        Model deviates + pro wins → 0.4 (alternative might also work)
        Model deviates + pro loses → 0.6 (model may have seen something)

    The asymmetry is deliberate: deviating from a winning play isn't punished
    hard (the model's plan might work too), but endorsing a losing play is
    mildly penalized.
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    model_text = _get_model_action_text(parsed)
    model_cats = _categorize_action(model_text)

    pro_cats = _get_pro_categories(ground_truth)
    if pro_cats is None:
        return 0.0

    alignment = _action_similarity(model_cats, pro_cats)
    round_won = ground_truth.get("round_won")

    if round_won is None:
        return alignment

    if round_won:
        # Pro won. Agreeing is good. Deviating isn't penalized.
        return 0.4 + 0.6 * alignment
    else:
        # Pro lost. Agreeing is mildly bad. Deviating might be good.
        return 0.6 - 0.4 * alignment


# ---------------------------------------------------------------------------
# All reward functions in order (for trainer consumption)
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS = [
    # Vision (prevent SFT regression)
    format_gate_reward,
    hard_field_accuracy_reward,
    soft_field_accuracy_reward,
    # Reasoning (RL training signal)
    decision_alignment_reward,
    outcome_reward,
    consistency_reward,
    reasoning_quality_reward,
]

# Weights: format, hard_acc, soft_acc, decision, outcome, consistency, reasoning
DEFAULT_REWARD_WEIGHTS = [0.05, 0.15, 0.05, 0.15, 0.30, 0.20, 0.10]
