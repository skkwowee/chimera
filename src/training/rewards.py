r"""
CS2 reward functions for GRPO training — revised formulation.

=============================================================================
Mathematical Formulation (for NeurIPS submission)
=============================================================================

Problem setup
-------------
At each decision point t within a round, the model receives an observation
o_t = (I_{t-k}, ..., I_t, c_t) where I_{t-j} are screenshots (current +
up to k prior frames for visual continuity) and c_t is a structured context
string generated from engine tick data summarizing round progression:
economy, kills, utility usage, bomb events, and current player states.
The model's output y is structured JSON containing game state extraction,
analysis, and advice. See D018 in decisions.md.

The reward function has access to ground truth from demo data that the model
never sees: the pro player's actual state s_t, their subsequent behavior
τ_{t:t+Δ}^{pro} over the next Δ ticks, round outcome W ∈ {0,1}, and
per-player contribution metrics φ_i.

Reward function
---------------
The total reward decomposes into three weighted signals gated by a
multiplicative format constraint:

    r(y, o_t) = 𝟙[valid_json(y)] · (α·R_percept + β·R_decision + γ·R_outcome)

where α = 0.20, β = 0.30, γ = 0.50, and 𝟙[·] is a hard format gate — invalid
JSON receives zero total reward regardless of other signal quality. This is
multiplicative, not additive: the gate enforces structure as a prerequisite,
not a soft nudge.

Component 1: Perceptual accuracy R_percept (weight α = 0.20)
-------------------------------------------------------------
Prevents SFT regression by comparing extracted game state fields against
engine-accurate demo ground truth:

    R_percept(y, s_t) = (1/|F|) Σ_{f∈F} match(y_f, s_{t,f})

where F is the union of hard fields (health, armor, weapons, player counts —
directly HUD-readable) and soft fields (money, map, bomb status — require
inference). match(·) is exact for categoricals, tolerance-based for numerics
(±10% relative error), and Jaccard for lists (utility inventory).

Component 2: Decision alignment R_decision (weight β = 0.30)
-------------------------------------------------------------
Compares the model's advised action against the pro's actual behavioral
signature extracted from the next Δ ≈ 10–15 seconds of tick data
(~900–1350 ticks at 90Hz).

From tick data we extract a behavioral feature vector:

    b_t^{pro} = (d_move, d_obj, u_type, e_timing, δ_engage)

where:
  • d_move ∈ {-1, 0, 1}: net movement direction relative to enemies
    (retreat, hold, advance), computed from position delta over Δ ticks
    and mean enemy position
  • d_obj ∈ {-1, 0, 1}: movement relative to bomb/site
    (away, neutral, toward)
  • u_type ∈ {0,1}^k: binary vector of utility types used in the window
    (smoke, flash, HE, molotov)
  • e_timing ∈ {0, 1}: whether the player initiated engagement (fired first)
  • δ_engage ∈ [0, 1]: normalized time until first damage event
    (0 = immediate, 1 = no engagement in window)

The model's text output is mapped to the same feature space via keyword
extraction (Approach A) from the advice section. The alignment score is:

    R_decision(y, τ^{pro}) = (1/|b|) Σ_j 𝟙[b_{t,j}^{model} = b_{t,j}^{pro}]

For binary/categorical features this is exact match; for δ_engage we use a
tolerance window of ±0.2. When behavioral features are unavailable (pre-F05),
falls back to Jaccard similarity over coarse action categories.

Component 3: Outcome-modulated decision reward R_outcome (weight γ = 0.50)
---------------------------------------------------------------------------
The core learning signal. Modulates decision alignment by round outcome,
weighted by how much the spectated player actually influenced the result:

    R_outcome(y, τ^{pro}, W, φ) = R_decision(y, τ^{pro}) · Ω(W, φ, a)

where a = R_decision (the alignment score) and Ω is the outcome modulation:

    Ω(W, φ, a) = W·φ·(0.5 + 0.5·a) + (1−W)·φ·(0.5 − 0.3·a)

The player contribution φ ∈ [0,1] weights the outcome by causal relevance:

    φ = 0.4·(damage_dealt / max_damage_in_round)
      + 0.3·(survival_time / round_duration)
      + 0.3·𝟙[objective_action]

where objective_action ∈ {plant, defuse, last_alive}.

Interpretation of Ω:
  • φ ≈ 0 (player died early, no impact): outcome signal ≈ 0 regardless of
    W. The round result tells us nothing about advice quality at this moment.
  • W=1, high a (agree with winning play): Ω ≈ φ·1.0 — strong positive signal.
  • W=1, low a (deviate from winning play): Ω ≈ φ·0.5 — moderate, the model's
    alternative might also work.
  • W=0, high a (agree with losing play): Ω ≈ φ·0.2 — penalized for endorsing
    a losing strategy.
  • W=0, low a (deviate from losing play): Ω ≈ φ·0.5 — moderate positive, the
    model may have identified something better.

This asymmetry allows the model to learn beyond imitation: deviation from
losing play is mildly rewarded, enabling strategy discovery.

GRPO dynamics
-------------
In Group Relative Policy Optimization (Shao et al., 2024), for each prompt x
we sample G completions {y_1, ..., y_G} and normalize advantages:

    Â_i = (r_i − mean({r_j})) / (std({r_j}) + ε)

GRPO only requires correct relative ordering within the group, not calibrated
absolute rewards. The φ weighting is crucial for stability: without it, noisy
outcome labels on low-contribution snapshots inject high-variance gradients.

KL regularization (λ_KL = 0.02) prevents mode collapse onto narrow "safe"
advice that scores well across diverse game states.

=============================================================================

Ground truth schema (populated by F05 from Parquet tick data):

    ground_truth = {
        "game_state": { ... },           # engine-accurate HUD fields
        "pro_action": {
            "categories": list[str],      # coarse action taxonomy (fallback)
            "description": str,           # text description of pro's play
            "behavior": {                 # behavioral feature vector (Δ ticks)
                "movement_direction": int,     # -1=retreat, 0=hold, 1=advance
                "objective_direction": int,    # -1=away, 0=neutral, 1=toward
                "utility_used": list[str],     # e.g. ["smoke", "flash"]
                "initiated_engagement": bool,  # fired first?
                "engagement_delay": float,     # normalized [0,1]
            }
        },
        "round_won": bool,
        "player_contribution": {
            "damage_dealt": float,
            "max_round_damage": float,
            "survival_time": float,
            "round_duration": float,
            "objective_action": bool,     # planted/defused/last_alive
        }
    }
"""

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Valid enum values for categorical fields
# ---------------------------------------------------------------------------

VALID_ROUND_PHASES = {"buy", "playing", "freezetime", "post-plant", "warmup"}
VALID_PLAYER_SIDES = {"T", "CT"}
VALID_BOMB_STATUSES = {"carried", "planted", "dropped", "defused", "exploded", None}

# Hard fields: directly readable from the HUD
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


# ---------------------------------------------------------------------------
# Action taxonomy — fallback when behavioral features unavailable
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


# ---------------------------------------------------------------------------
# Behavioral feature extraction — keyword mapping for model output
# ---------------------------------------------------------------------------

# Movement direction keywords: model text → d_move ∈ {-1, 0, 1}
_ADVANCE_KEYWORDS = [
    "push", "entry", "rush", "execute", "aggress", "swing", "peek",
    "advance", "move forward", "press", "take ground", "go in",
    "attack", "charge", "approach",
]
_RETREAT_KEYWORDS = [
    "fall back", "retreat", "pull back", "disengage", "back off",
    "give up", "save", "run away", "escape", "withdraw", "back out",
]
_HOLD_KEYWORDS = [
    "hold", "anchor", "wait", "passive", "stay", "camp", "setup",
    "default", "play time", "hold angle", "maintain", "keep position",
]

# Objective direction keywords: model text → d_obj ∈ {-1, 0, 1}
_TOWARD_OBJ_KEYWORDS = [
    "site", "bomb", "plant", "defuse", "retake", "execute",
    "take site", "go to site", "push site", "go a", "go b",
]
_AWAY_OBJ_KEYWORDS = [
    "rotate", "flank", "lurk", "away from site", "leave site",
    "reposition", "wrap", "backstab",
]

# Utility keywords: model text → u_type components
_UTILITY_MAP = {
    "smoke": ["smoke", "smok"],
    "flash": ["flash", "flashbang", "blind"],
    "he": ["he grenade", "he nade", "frag", "nade"],
    "molotov": ["molly", "molotov", "incendiary", "fire"],
}

# Engagement timing keywords: model text → e_timing ∈ {0, 1}
_INITIATE_KEYWORDS = [
    "peek", "swing", "entry", "push", "challenge", "fight",
    "take fight", "engage", "contact", "dry peek", "wide peek",
]
_WAIT_KEYWORDS = [
    "wait", "hold", "bait", "let them come", "play passive",
    "crossfire", "setup", "trap", "patience",
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
# Field scoring helpers
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
# Behavioral feature extraction from model text
# ---------------------------------------------------------------------------

def _classify_movement(text: str) -> int | None:
    """Classify model's advised movement direction: -1=retreat, 0=hold, 1=advance."""
    if not text:
        return None
    text_lower = text.lower()

    advance_score = sum(1 for kw in _ADVANCE_KEYWORDS if kw in text_lower)
    retreat_score = sum(1 for kw in _RETREAT_KEYWORDS if kw in text_lower)
    hold_score = sum(1 for kw in _HOLD_KEYWORDS if kw in text_lower)

    scores = {1: advance_score, -1: retreat_score, 0: hold_score}
    max_score = max(scores.values())
    if max_score == 0:
        return None
    # Return the direction with the highest keyword count
    return max(scores, key=scores.get)


def _classify_objective_direction(text: str) -> int | None:
    """Classify movement relative to objective: -1=away, 0=neutral, 1=toward."""
    if not text:
        return None
    text_lower = text.lower()

    toward_score = sum(1 for kw in _TOWARD_OBJ_KEYWORDS if kw in text_lower)
    away_score = sum(1 for kw in _AWAY_OBJ_KEYWORDS if kw in text_lower)

    if toward_score > away_score:
        return 1
    if away_score > toward_score:
        return -1
    if toward_score > 0:
        return 0  # ambiguous, both mentioned
    return None


def _extract_utility_types(text: str) -> set[str]:
    """Extract utility types mentioned in model advice."""
    if not text:
        return set()
    text_lower = text.lower()
    used = set()
    for util_type, keywords in _UTILITY_MAP.items():
        if any(kw in text_lower for kw in keywords):
            used.add(util_type)
    return used


def _classify_engagement_timing(text: str) -> int | None:
    """Classify whether model advises initiating (1) or waiting (0)."""
    if not text:
        return None
    text_lower = text.lower()

    initiate_score = sum(1 for kw in _INITIATE_KEYWORDS if kw in text_lower)
    wait_score = sum(1 for kw in _WAIT_KEYWORDS if kw in text_lower)

    if initiate_score > wait_score:
        return 1
    if wait_score > initiate_score:
        return 0
    return None


def _extract_model_behavior(parsed: dict) -> dict:
    """
    Extract behavioral feature vector from model's advice text.

    Maps text → b^{model} = (d_move, d_obj, u_type, e_timing, δ_engage).
    δ_engage is not extractable from text (it's a continuous value from tick
    data), so we infer it from engagement timing: initiate → low delay (0.2),
    wait → high delay (0.8).
    """
    advice = parsed.get("advice", {})
    if not isinstance(advice, dict):
        return {}

    action = advice.get("primary_action", "")
    fallback = advice.get("fallback", "")
    reasoning = advice.get("reasoning", "")
    combined = f"{action} {fallback} {reasoning}"

    behavior = {}

    d_move = _classify_movement(combined)
    if d_move is not None:
        behavior["movement_direction"] = d_move

    d_obj = _classify_objective_direction(combined)
    if d_obj is not None:
        behavior["objective_direction"] = d_obj

    utility = _extract_utility_types(combined)
    behavior["utility_used"] = sorted(utility)

    e_timing = _classify_engagement_timing(combined)
    if e_timing is not None:
        behavior["initiated_engagement"] = bool(e_timing)
        # Infer engagement delay from timing classification
        behavior["engagement_delay"] = 0.2 if e_timing == 1 else 0.8

    return behavior


# ---------------------------------------------------------------------------
# Legacy: coarse action taxonomy (fallback)
# ---------------------------------------------------------------------------

def _categorize_action(text: str) -> set[str]:
    """Map free-text action description to taxonomy categories (fallback)."""
    if not text:
        return set()
    text_lower = text.lower()
    categories = set()
    for category, keywords in ACTION_TAXONOMY.items():
        if any(kw in text_lower for kw in keywords):
            categories.add(category)
    return categories


def _action_similarity(model_cats: set[str], pro_cats: set[str]) -> float:
    """Jaccard similarity between two sets of action categories (fallback)."""
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
    """Extract pro action categories from ground truth (fallback path)."""
    pro_action = ground_truth.get("pro_action", {})
    if not isinstance(pro_action, dict):
        return None
    cats = set(pro_action.get("categories", []))
    if not cats:
        cats = _categorize_action(pro_action.get("description", ""))
    return cats if cats else None


# ---------------------------------------------------------------------------
# Behavioral feature alignment scoring
# ---------------------------------------------------------------------------

def _score_behavioral_alignment(
    model_behavior: dict,
    pro_behavior: dict,
) -> float:
    r"""
    Score alignment between model and pro behavioral feature vectors.

    R_decision = (1/|b|) Σ_j 𝟙[b_{t,j}^{model} = b_{t,j}^{pro}]

    For categorical features: exact match.
    For engagement_delay: tolerance of ±0.2.
    For utility_used: Jaccard similarity over utility types.
    """
    if not pro_behavior or not model_behavior:
        return 0.0

    scores = []

    # d_move: movement direction (-1, 0, 1) — exact match
    if "movement_direction" in pro_behavior and "movement_direction" in model_behavior:
        scores.append(
            1.0 if model_behavior["movement_direction"] == pro_behavior["movement_direction"]
            else 0.0
        )

    # d_obj: objective direction (-1, 0, 1) — exact match
    if "objective_direction" in pro_behavior and "objective_direction" in model_behavior:
        scores.append(
            1.0 if model_behavior["objective_direction"] == pro_behavior["objective_direction"]
            else 0.0
        )

    # u_type: utility usage — Jaccard similarity
    if "utility_used" in pro_behavior:
        pro_util = set(pro_behavior["utility_used"])
        model_util = set(model_behavior.get("utility_used", []))
        if not pro_util and not model_util:
            scores.append(1.0)
        elif not pro_util or not model_util:
            scores.append(0.0)
        else:
            intersection = len(pro_util & model_util)
            union = len(pro_util | model_util)
            scores.append(intersection / union)

    # e_timing: engagement initiation — exact match
    if "initiated_engagement" in pro_behavior and "initiated_engagement" in model_behavior:
        scores.append(
            1.0 if model_behavior["initiated_engagement"] == pro_behavior["initiated_engagement"]
            else 0.0
        )

    # δ_engage: engagement delay — tolerance ±0.2
    if "engagement_delay" in pro_behavior and "engagement_delay" in model_behavior:
        pro_delay = float(pro_behavior["engagement_delay"])
        model_delay = float(model_behavior["engagement_delay"])
        if abs(pro_delay - model_delay) <= 0.2:
            scores.append(1.0)
        else:
            # Linear decay beyond tolerance
            scores.append(max(0.0, 1.0 - (abs(pro_delay - model_delay) - 0.2) / 0.3))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Player contribution φ
# ---------------------------------------------------------------------------

def compute_player_contribution(contribution: dict) -> float:
    r"""
    Compute player contribution metric φ ∈ [0, 1].

    φ = 0.4 · (damage_dealt / max_round_damage)
      + 0.3 · (survival_time / round_duration)
      + 0.3 · 𝟙[objective_action]

    Weights the outcome signal by how much the spectated player influenced
    the round result. If they died in 5 seconds without dealing damage,
    φ ≈ 0 and the outcome signal is nearly zeroed out.

    Args:
        contribution: Dict with damage_dealt, max_round_damage, survival_time,
                      round_duration, objective_action fields.

    Returns:
        φ ∈ [0, 1]
    """
    if not contribution or not isinstance(contribution, dict):
        return 1.0  # no contribution data → assume full relevance (conservative)

    damage = float(contribution.get("damage_dealt", 0))
    max_damage = float(contribution.get("max_round_damage", 1))
    survival = float(contribution.get("survival_time", 0))
    duration = float(contribution.get("round_duration", 1))
    objective = bool(contribution.get("objective_action", False))

    # Avoid division by zero
    damage_ratio = damage / max(max_damage, 1.0)
    survival_ratio = survival / max(duration, 1.0)

    phi = (
        0.4 * min(damage_ratio, 1.0)
        + 0.3 * min(survival_ratio, 1.0)
        + 0.3 * (1.0 if objective else 0.0)
    )

    return max(0.0, min(1.0, phi))


# ---------------------------------------------------------------------------
# Outcome modulation Ω
# ---------------------------------------------------------------------------

def compute_outcome_modulation(
    round_won: bool,
    phi: float,
    alignment: float,
) -> float:
    r"""
    Compute outcome modulation function Ω(W, φ, a).

    Ω(W, φ, a) = W·φ·(0.5 + 0.5·a) + (1−W)·φ·(0.5 − 0.3·a)

    Interpretation:
      W=1, high a → Ω ≈ φ·1.0  (agree with winning play: strong positive)
      W=1, low a  → Ω ≈ φ·0.5  (deviate from win: moderate, alternative may work)
      W=0, high a → Ω ≈ φ·0.2  (agree with losing play: penalized)
      W=0, low a  → Ω ≈ φ·0.5  (deviate from loss: moderate positive)

    Signal matrix at φ=1:
        |              | Pro wins | Pro loses |
        |--------------|----------|-----------|
        | Model agrees |   1.0    |    0.2    |
        | Model deviates|  0.5    |    0.5    |

    Args:
        round_won: Whether the pro's team won the round.
        phi: Player contribution ∈ [0, 1].
        alignment: Decision alignment score a ∈ [0, 1].

    Returns:
        Ω ∈ [0, 1]
    """
    w = 1.0 if round_won else 0.0
    a = max(0.0, min(1.0, alignment))
    phi = max(0.0, min(1.0, phi))

    omega = (
        w * phi * (0.5 + 0.5 * a)
        + (1.0 - w) * phi * (0.5 - 0.3 * a)
    )

    return max(0.0, min(1.0, omega))


# ===================================================================
# Reward functions
# ===================================================================


# ---------------------------------------------------------------------------
# Format gate — multiplicative mask, NOT a weighted signal
# ---------------------------------------------------------------------------

def format_gate_reward(response: str, **kwargs) -> float:
    """
    Binary gate: 1.0 if response contains valid JSON with expected top-level
    keys, 0.0 otherwise.

    This is applied as a multiplicative mask on the total reward:
        total = format_gate * (α·R_percept + β·R_decision + γ·R_outcome)

    Invalid JSON → zero total reward. The gate is a hard constraint.
    """
    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    has_game_state = isinstance(parsed.get("game_state"), dict)
    has_analysis = isinstance(parsed.get("analysis"), dict)
    has_advice = isinstance(parsed.get("advice"), dict)

    return 1.0 if (has_game_state and has_analysis and has_advice) else 0.0


# ---------------------------------------------------------------------------
# R_percept: Perceptual accuracy (merged hard + soft field accuracy)
# ---------------------------------------------------------------------------

def perceptual_accuracy_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    r"""
    Merged hard and soft field accuracy.

    R_percept(y, s_t) = (1/|F|) Σ_{f∈F} match(y_f, s_{t,f})

    Combines all HUD-readable (hard) and inferential (soft) fields into a
    single perceptual accuracy score. This prevents SFT regression during
    GRPO — the model must maintain accurate visual grounding.
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

    # Hard numeric fields
    for field in HARD_FIELDS:
        score = _score_numeric_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Hard string fields (weapon names)
    for field in HARD_STRING_FIELDS:
        score = _score_string_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Hard categorical fields
    for field, _ in HARD_CATEGORICAL_FIELDS:
        score = _score_categorical_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Utility list (hard — visible on HUD)
    score = _score_list_field(predicted, gt_state, "utility")
    if score is not None:
        scores.append(score)

    # Soft numeric fields
    for field in SOFT_FIELDS:
        score = _score_numeric_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Soft string fields
    for field in SOFT_STRING_FIELDS:
        score = _score_string_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    # Soft categorical fields
    for field, _ in SOFT_CATEGORICAL_FIELDS:
        score = _score_categorical_field(predicted, gt_state, field)
        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# R_decision: Decision alignment on behavioral features
# ---------------------------------------------------------------------------

def decision_alignment_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    r"""
    Behavioral decision alignment.

    R_decision(y, τ^{pro}) = (1/|b|) Σ_j 𝟙[b_{t,j}^{model} = b_{t,j}^{pro}]

    When behavioral features are available (from tick data via F05), scores
    the model's advice against the pro's actual behavioral signature across
    5 dimensions: movement direction, objective direction, utility usage,
    engagement timing, and engagement delay.

    Falls back to Jaccard similarity over coarse action categories when
    behavioral features are not available in ground_truth.
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    # Check for behavioral features (rich path)
    pro_action = ground_truth.get("pro_action", {})
    if isinstance(pro_action, dict):
        pro_behavior = pro_action.get("behavior")
        if pro_behavior and isinstance(pro_behavior, dict):
            model_behavior = _extract_model_behavior(parsed)
            if model_behavior:
                return _score_behavioral_alignment(model_behavior, pro_behavior)

    # Fallback: coarse action taxonomy (Jaccard)
    model_text = _get_model_action_text(parsed)
    model_cats = _categorize_action(model_text)

    pro_cats = _get_pro_categories(ground_truth)
    if pro_cats is None:
        return 0.0

    return _action_similarity(model_cats, pro_cats)


# ---------------------------------------------------------------------------
# R_outcome: Outcome-modulated decision reward
# ---------------------------------------------------------------------------

def outcome_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    r"""
    Decision alignment modulated by round outcome and player contribution.

    R_outcome = R_decision(y, τ^{pro}) · Ω(W, φ, R_decision)

    where Ω(W, φ, a) = W·φ·(0.5 + 0.5·a) + (1−W)·φ·(0.5 − 0.3·a)

    and φ = 0.4·(dmg/max_dmg) + 0.3·(surv/dur) + 0.3·𝟙[obj]

    The φ weighting addresses the credit assignment problem: if the spectated
    player died early without impact, the round outcome tells us almost nothing
    about advice quality at that moment, so the signal is dampened.

    Falls back to the original asymmetric signal matrix when player_contribution
    data is not available.
    """
    if ground_truth is None:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    # Compute decision alignment (reuses the same logic)
    alignment = decision_alignment_reward(response, ground_truth=ground_truth)

    round_won = ground_truth.get("round_won")
    if round_won is None:
        return alignment

    # Compute player contribution φ
    contribution = ground_truth.get("player_contribution")
    phi = compute_player_contribution(contribution)

    # Compute outcome modulation Ω
    omega = compute_outcome_modulation(round_won, phi, alignment)

    # R_outcome = R_decision · Ω
    return alignment * omega


# ===================================================================
# Reward function registry and weights
# ===================================================================

# The three reward signals (format gate is separate, multiplicative)
REWARD_FUNCTIONS = [
    perceptual_accuracy_reward,   # R_percept  (α = 0.20)
    decision_alignment_reward,    # R_decision (β = 0.30)
    outcome_reward,               # R_outcome  (γ = 0.50)
]

# Weights for the 3 reward signals (format gate is multiplicative, not here)
DEFAULT_REWARD_WEIGHTS = [0.20, 0.30, 0.50]

