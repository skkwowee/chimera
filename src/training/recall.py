r"""
RECALL: Retrieval-Enhanced Counterfactual Advantage from Learned Lookups.

=============================================================================
Mathematical Formulation
=============================================================================

Overview
--------
RECALL provides a non-parametric reward signal for GRPO by estimating
state-action values from a kNN index of historical pro-play outcomes.
Instead of learning a value function, we retrieve similar game states from
a pre-built index and estimate advantage directly from observed win rates.

Definitions
-----------
Let S be the space of tactical game states and A the space of behavioral
actions. Given a dataset D = {(s_i, a_i, w_i)}_{i=1}^N of pro-play
samples where w_i ∈ {0,1} is the round outcome:

Embedding functions:
    τ: S → ℝ^d_s     (tactical state embedding)
    α: A → ℝ^d_a     (action embedding)

For a query state s and action a:

1. Retrieve K nearest neighbors by state embedding:
       N_K(s) = {i : τ(s_i) ∈ kNN(τ(s), K)}

2. State value estimate (baseline):
       V̂(s) = (1/K) Σ_{i ∈ N_K(s)} w_i

3. Filter neighbors by action similarity:
       N_A(s, a) = {i ∈ N_K(s) : sim(α(a_i), α(a)) > θ}
   where sim is cosine similarity and θ is a threshold.

4. State-action value estimate:
       Q̂(s, a) = (1/|N_A|) Σ_{i ∈ N_A(s,a)} w_i

5. RECALL advantage:
       A(s, a) = Q̂(s, a) − V̂(s)   if |N_A| ≥ k_min
                 0                    otherwise

The advantage is positive when the chosen action's historical win rate
exceeds the baseline for that state, and negative when it underperforms.
When insufficient action-matched neighbors exist, we return 0 (uncertain).

=============================================================================
"""

import json
import logging
import re
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore[reportMissingImports]
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

# Map names to numeric IDs for embedding
_MAP_IDS = {
    "de_dust2": 0, "de_mirage": 1, "de_inferno": 2, "de_nuke": 3,
    "de_overpass": 4, "de_ancient": 5, "de_anubis": 6, "de_vertigo": 7,
}
_NUM_MAPS = len(_MAP_IDS)

# Action similarity threshold (cosine similarity)
_ACTION_SIM_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers (shared with rewards.py)
# ---------------------------------------------------------------------------

def _extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON object from model response."""
    code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    json_pattern = r"\{[\s\S]*\}"
    match = re.search(json_pattern, response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

def tactical_embedding(game_state: dict[str, Any]) -> np.ndarray:
    """
    Convert a game state dict to a flat tactical embedding vector.

    Features (~20 dims):
        [0]     side (T=0, CT=1)
        [1]     round_phase (playing=0, post-plant=1)
        [2-9]   map_name one-hot (8 maps)
        [10]    man_advantage (alive_teammates - alive_enemies), normalized
        [11]    has_primary_weapon (0 or 1)
        [12]    health_normalized (0-1)
        [13]    armor_normalized (0-1)
        [14]    utility_count (normalized by max 4)
        [15]    bomb_status (carried=0, planted=1, dropped=2), normalized
        [16]    alive_teammates (normalized by 4)
        [17]    alive_enemies (normalized by 5)
        [18]    visible_enemies (normalized by 5)

    Args:
        game_state: Dict with game state fields from ground_truth.

    Returns:
        np.ndarray of shape (19,) with float32 values.
    """
    vec = np.zeros(19, dtype=np.float32)

    # Side
    side = str(game_state.get("player_side", "")).upper()
    vec[0] = 1.0 if side == "CT" else 0.0

    # Round phase
    phase = str(game_state.get("round_phase", "")).lower()
    vec[1] = 1.0 if phase == "post-plant" else 0.0

    # Map one-hot (indices 2-9)
    map_name = str(game_state.get("map_name", "")).lower()
    map_id = _MAP_IDS.get(map_name)
    if map_id is not None and map_id < _NUM_MAPS:
        vec[2 + map_id] = 1.0

    # Man advantage
    alive_t = int(game_state.get("alive_teammates", 0))
    alive_e = int(game_state.get("alive_enemies", 0))
    vec[10] = (alive_t - alive_e) / 5.0  # normalize to roughly [-1, 1]

    # Has primary weapon
    weapon = game_state.get("weapon_primary")
    vec[11] = 1.0 if weapon else 0.0

    # Health and armor
    vec[12] = float(game_state.get("player_health", 0)) / 100.0
    vec[13] = float(game_state.get("player_armor", 0)) / 100.0

    # Utility count
    utility = game_state.get("utility", [])
    if isinstance(utility, list):
        vec[14] = min(len(utility), 4) / 4.0
    else:
        vec[14] = 0.0

    # Bomb status
    bomb = str(game_state.get("bomb_status", "")).lower()
    if bomb == "planted":
        vec[15] = 0.5
    elif bomb == "dropped":
        vec[15] = 1.0
    else:
        vec[15] = 0.0  # carried or unknown

    # Alive counts (normalized)
    vec[16] = min(alive_t, 4) / 4.0
    vec[17] = min(alive_e, 5) / 5.0

    # Visible enemies
    vec[18] = min(int(game_state.get("visible_enemies", 0)), 5) / 5.0

    return vec


def action_embedding(behavior: dict[str, Any]) -> np.ndarray:
    """
    Convert behavioral features to a flat action embedding vector.

    Features (5 dims):
        [0]  movement_direction: -1, 0, 1 (normalized to -1..1)
        [1]  objective_direction: -1, 0, 1 (normalized to -1..1)
        [2]  utility_count (normalized by 4)
        [3]  initiated_engagement: 0 or 1
        [4]  engagement_delay: float 0-1

    Args:
        behavior: Dict with behavioral feature fields.

    Returns:
        np.ndarray of shape (5,) with float32 values.
    """
    vec = np.zeros(5, dtype=np.float32)

    vec[0] = float(behavior.get("movement_direction", 0))
    vec[1] = float(behavior.get("objective_direction", 0))

    utility = behavior.get("utility_used", [])
    if isinstance(utility, list):
        vec[2] = min(len(utility), 4) / 4.0
    elif isinstance(utility, (int, float)):
        vec[2] = min(float(utility), 4) / 4.0
    else:
        vec[2] = 0.0

    initiated = behavior.get("initiated_engagement", False)
    vec[3] = 1.0 if initiated else 0.0

    vec[4] = float(behavior.get("engagement_delay", 1.0))

    return vec


# ---------------------------------------------------------------------------
# RECALL Index
# ---------------------------------------------------------------------------

class RECALLIndex:
    """
    kNN index for retrieval-enhanced advantage estimation.

    Stores tactical state embeddings in a FAISS IndexFlatL2 and action
    embeddings + outcomes in parallel arrays. Supports querying for
    state-action advantage estimates.
    """

    def __init__(self):
        self._state_index: Any = None  # faiss.IndexFlatL2
        self._action_embeddings: np.ndarray | None = None  # np.ndarray (N, d_a)
        self._outcomes: np.ndarray | None = None  # np.ndarray (N,) float32
        self._n: int = 0

    @property
    def size(self) -> int:
        """Number of samples in the index."""
        return self._n

    def build_from_samples(self, samples: list[dict[str, Any]]) -> None:
        """
        Build the index from GRPO sample dicts.

        Each sample should have ground_truth with:
            - game_state: dict
            - pro_action.behavior: dict
            - round_won: bool

        Args:
            samples: List of GRPO sample dicts.
        """
        if faiss is None:
            raise ImportError(
                "faiss is required for RECALLIndex. Install with: pip install faiss-cpu"
            )

        state_vecs = []
        action_vecs = []
        outcomes = []

        for sample in samples:
            gt = sample.get("ground_truth", {})
            gs = gt.get("game_state")
            pa = gt.get("pro_action", {})
            beh = pa.get("behavior") if isinstance(pa, dict) else None
            won = gt.get("round_won")

            if gs is None or beh is None or won is None:
                continue

            state_vecs.append(tactical_embedding(gs))
            action_vecs.append(action_embedding(beh))
            outcomes.append(1.0 if won else 0.0)

        if not state_vecs:
            logger.warning("No valid samples for RECALL index")
            self._n = 0
            return

        state_mat = np.stack(state_vecs).astype(np.float32)
        self._action_embeddings = np.stack(action_vecs).astype(np.float32)
        self._outcomes = np.array(outcomes, dtype=np.float32)
        self._n = len(outcomes)

        d = state_mat.shape[1]
        self._state_index = faiss.IndexFlatL2(d)
        self._state_index.add(state_mat)

        logger.info("RECALL index built with %d samples (dim=%d)", self._n, d)

    def query(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        k: int = 32,
        k_min: int = 5,
    ) -> tuple[float, float, bool]:
        """
        Query the index for advantage estimation.

        Args:
            state: Game state dict.
            action: Behavioral action dict.
            k: Number of nearest neighbors to retrieve.
            k_min: Minimum action-matched neighbors for confidence.

        Returns:
            (Q_hat, V_hat, confident):
                Q_hat: estimated state-action value (mean win rate of
                       action-matched neighbors).
                V_hat: estimated state value (mean win rate of all K neighbors).
                confident: True if >= k_min action-matched neighbors found.
        """
        if self._n == 0 or self._state_index is None or self._outcomes is None or self._action_embeddings is None:
            return 0.5, 0.5, False

        # Clamp k to index size
        k_actual = min(k, self._n)

        state_vec = tactical_embedding(state).reshape(1, -1)
        _distances, indices = self._state_index.search(state_vec, k_actual)
        indices = indices[0]

        # Filter out -1 indices (FAISS returns -1 if fewer than k results)
        valid = indices[indices >= 0]
        if len(valid) == 0:
            return 0.5, 0.5, False

        # V_hat: baseline win rate of all K neighbors
        neighbor_outcomes = self._outcomes[valid]
        v_hat = float(np.mean(neighbor_outcomes))

        # Action filtering: cosine similarity
        query_action = action_embedding(action).reshape(1, -1)
        neighbor_actions = self._action_embeddings[valid]

        # Cosine similarity
        query_norm = np.linalg.norm(query_action)
        neighbor_norms = np.linalg.norm(neighbor_actions, axis=1)

        # Avoid division by zero
        if query_norm < 1e-8:
            # Zero action vector — fall back to movement_direction match
            q_move = float(action.get("movement_direction", 0))
            matched_mask = np.abs(neighbor_actions[:, 0] - q_move) < 0.5
        else:
            # Safe division: zero-norm neighbors get 0 similarity
            safe_norms = np.maximum(neighbor_norms, 1e-8)
            cos_sim = (neighbor_actions @ query_action.T).flatten() / (
                safe_norms * query_norm
            )
            matched_mask = cos_sim > _ACTION_SIM_THRESHOLD

        matched_indices = np.where(matched_mask)[0]

        if len(matched_indices) == 0:
            return v_hat, v_hat, False

        matched_outcomes = neighbor_outcomes[matched_indices]
        q_hat = float(np.mean(matched_outcomes))
        confident = len(matched_indices) >= k_min

        return q_hat, v_hat, confident

    def recall_advantage(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        k: int = 32,
        k_min: int = 5,
    ) -> float:
        """
        Compute RECALL advantage: A = Q̂(s,a) − V̂(s).

        Returns 0.0 if not confident (fewer than k_min action-matched
        neighbors).

        Args:
            state: Game state dict.
            action: Behavioral action dict.
            k: Number of nearest neighbors.
            k_min: Minimum action-matched neighbors for confidence.

        Returns:
            Advantage ∈ [-1, 1], or 0.0 if uncertain.
        """
        q_hat, v_hat, confident = self.query(state, action, k=k, k_min=k_min)
        if not confident:
            return 0.0
        return q_hat - v_hat


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def recall_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    recall_index: RECALLIndex | None = None,
    **kwargs: Any,
) -> float:
    r"""
    RECALL advantage reward.

    Computes the advantage of the model's advised action over the baseline
    state value using kNN retrieval from historical pro-play data.

        r_RECALL(y, s) = Q̂(s, a^{model}) − V̂(s)

    Returns 0.0 if the index is unavailable, the response can't be parsed,
    or there are insufficient action-matched neighbors for confidence.

    Args:
        response: Model's text response (JSON with advice section).
        ground_truth: Dict with game_state and pro_action fields.
        recall_index: Pre-built RECALLIndex instance.

    Returns:
        Advantage ∈ [-1, 1], or 0.0 if uncertain.
    """
    if ground_truth is None or recall_index is None:
        return 0.0

    if recall_index.size == 0:
        return 0.0

    parsed = _extract_json_from_response(response)
    if parsed is None:
        return 0.0

    # Extract game state from ground truth
    game_state = ground_truth.get("game_state")
    if not isinstance(game_state, dict):
        return 0.0

    # Extract action from model's advice
    advice = parsed.get("advice", {})
    if not isinstance(advice, dict):
        return 0.0

    # Build action dict from model's text using keyword classification
    action_text = f"{advice.get('primary_action', '')} {advice.get('fallback', '')} {advice.get('reasoning', '')}"
    model_action = _extract_action_from_text(action_text)

    return recall_index.recall_advantage(game_state, model_action)


def _extract_action_from_text(text: str) -> dict[str, Any]:
    """
    Extract behavioral action features from model advice text.

    Lightweight keyword classifier — mirrors the approach in rewards.py
    (_extract_model_behavior) but returns a dict suitable for
    action_embedding().
    """
    if not text:
        return {}

    text_lower = text.lower()

    # Movement direction
    advance_kws = [
        "push", "entry", "rush", "execute", "aggress", "swing", "peek",
        "advance", "move forward", "press", "go in", "attack", "charge",
    ]
    retreat_kws = [
        "fall back", "retreat", "pull back", "disengage", "back off",
        "save", "run away", "escape", "withdraw",
    ]
    hold_kws = [
        "hold", "anchor", "wait", "passive", "stay", "camp", "setup",
        "default", "hold angle", "maintain",
    ]

    adv = sum(1 for kw in advance_kws if kw in text_lower)
    ret = sum(1 for kw in retreat_kws if kw in text_lower)
    hld = sum(1 for kw in hold_kws if kw in text_lower)
    scores = {1: adv, -1: ret, 0: hld}
    max_score = max(scores.values())
    movement = max(scores, key=lambda k: scores[k]) if max_score > 0 else 0

    # Objective direction
    toward_kws = ["site", "bomb", "plant", "defuse", "retake", "execute"]
    away_kws = ["rotate", "flank", "lurk", "away from site", "reposition"]
    toward = sum(1 for kw in toward_kws if kw in text_lower)
    away = sum(1 for kw in away_kws if kw in text_lower)
    if toward > away:
        obj_dir = 1
    elif away > toward:
        obj_dir = -1
    else:
        obj_dir = 0

    # Utility count
    util_kws = ["smoke", "flash", "molly", "molotov", "incendiary", "nade", "grenade"]
    util_count = sum(1 for kw in util_kws if kw in text_lower)

    # Engagement
    init_kws = ["peek", "swing", "entry", "push", "challenge", "fight", "engage"]
    wait_kws = ["wait", "hold", "bait", "passive", "crossfire", "setup", "patience"]
    init_score = sum(1 for kw in init_kws if kw in text_lower)
    wait_score = sum(1 for kw in wait_kws if kw in text_lower)
    initiated = init_score > wait_score
    delay = 0.2 if initiated else 0.8

    return {
        "movement_direction": movement,
        "objective_direction": obj_dir,
        "utility_used": [f"util_{i}" for i in range(min(util_count, 4))],
        "initiated_engagement": initiated,
        "engagement_delay": delay,
    }


def simplified_outcome_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    r"""
    Simplified outcome reward for ablation baseline.

    A clean, minimal outcome signal without φ weighting or Ω modulation:

        r(y, gt) = a · w + (1 − a) · (1 − w)

    where:
        a = decision alignment score ∈ [0, 1]
        w = round_won ∈ {0, 1}

    Signal matrix:
        |                | w=1 (win)  | w=0 (loss) |
        |----------------|------------|------------|
        | a=1 (agree)    |    1.0     |    0.0     |
        | a=0 (disagree) |    0.0     |    1.0     |

    This reward simply asks: did the model agree with the winning play
    and disagree with the losing play? No contribution weighting, no
    asymmetric modulation.

    Args:
        response: Model's text response.
        ground_truth: Dict with pro_action and round_won fields.

    Returns:
        Reward ∈ [0, 1].
    """
    if ground_truth is None:
        return 0.0

    # Import here to avoid circular dependency
    from .rewards import decision_alignment_reward

    a = decision_alignment_reward(response, ground_truth=ground_truth)

    round_won = ground_truth.get("round_won")
    if round_won is None:
        return a  # no outcome info, just return alignment

    w = 1.0 if round_won else 0.0
    return a * w + (1.0 - a) * (1.0 - w)
