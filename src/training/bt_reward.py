"""Bradley-Terry head reward for GRPO.

A small neural reward head trained on expert human pairwise preferences over
CS2 strategic-advice completions. Unlike judge_reward (which queries Claude
once per training step and ranks the full group of G sibling completions),
bt_reward scores ONE (state, completion) pair at a time with a local model
checkpoint -- no API calls, no per-step network round-trip, no group context
needed.

Why this exists: judge_reward gives strong gradients but costs $1-2 per 100
GRPO steps in API fees and adds ~1-3s of latency to every step. Once we have
~5k human preference pairs collected via the labeling UI, training a Bradley-
Terry head on those pairs yields a learned reward that approximates the human
judge for free at inference time. This becomes the primary GRPO signal once
the head is trained; judge_reward stays available as a fallback / sanity
check.

Differences from judge_reward at a glance:
  - judge_reward:   Claude API, group-aware (needs `siblings`), $-cost, slow.
  - bt_reward:      local neural head, per-completion only, free, fast.

Interface contract: same signature as judge_reward (response: str,
ground_truth: dict | None, **kwargs) so it slots into the trainer's reward
loop with no plumbing changes.

Failure mode: if CHIMERA_BT_HEAD_PATH is unset, the directory is missing, the
checkpoint won't load, or the forward pass raises, returns neutral 0.5 and
logs the failure ONCE per process. Mirrors judge_reward's fail-soft pattern
so a misconfigured run degrades to "no gradient signal" instead of crashing.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

_NEUTRAL_SCORE = 0.5  # returned when the head can't be loaded / called
_ENV_PATH = "CHIMERA_BT_HEAD_PATH"

# One-shot warn flags so a broken config doesn't spam logs every step.
_warned_no_path = False
_warned_load_fail = False
_warned_score_fail = False


def _warn_once(flag_name: str, msg: str) -> None:
    """Print msg once per process for the named flag, then mute it."""
    global _warned_no_path, _warned_load_fail, _warned_score_fail
    if flag_name == "no_path" and not _warned_no_path:
        print(msg, flush=True)
        _warned_no_path = True
    elif flag_name == "load_fail" and not _warned_load_fail:
        print(msg, flush=True)
        _warned_load_fail = True
    elif flag_name == "score_fail" and not _warned_score_fail:
        print(msg, flush=True)
        _warned_score_fail = True


def _ensure_scripts_on_path() -> None:
    """Make `scripts/` importable so `from scripts.train_bt_head import ...` works.

    The repo isn't installed as a package; trainer scripts run from repo root
    so `src.*` resolves, but `scripts.*` does not unless we add it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if scripts_dir.is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@functools.cache
def _load_head() -> Any:
    """Lazily load the BT head from CHIMERA_BT_HEAD_PATH. Cached for the process.

    Returns an opaque "loaded head" object (the tuple expected by the
    train_bt_head helper, or None if loading failed). Heavy deps (torch,
    sentence-transformers) are imported INSIDE this function so a plain
    `import src.training` stays cheap.
    """
    head_path = os.environ.get(_ENV_PATH)
    if not head_path:
        _warn_once(
            "no_path",
            f"  [bt_reward] {_ENV_PATH} not set; returning neutral "
            f"{_NEUTRAL_SCORE} for all completions.",
        )
        return None
    if not Path(head_path).exists():
        _warn_once(
            "load_fail",
            f"  [bt_reward] head path {head_path!r} does not exist; "
            f"returning neutral {_NEUTRAL_SCORE} for all completions.",
        )
        return None

    try:
        _ensure_scripts_on_path()
        # Lazy heavy-dep imports.
        import torch  # noqa: F401  (used by BTHead internals)
        from scripts.train_bt_head import load_bt_head  # type: ignore

        model = load_bt_head(head_path)
        if hasattr(model, "eval"):
            model.eval()
        return model
    except Exception as e:
        _warn_once(
            "load_fail",
            f"  [bt_reward] failed to load head from {head_path!r} "
            f"({type(e).__name__}: {str(e)[:160]}); returning neutral "
            f"{_NEUTRAL_SCORE} for all completions.",
        )
        return None


def _to_unit_interval(x: float) -> float:
    """Squash a possibly-unbounded raw score into [0, 1].

    train_bt_head may return either a sigmoided probability or a raw logit;
    we sigmoid + clip here so the reward is well-behaved either way. Values
    already in [0, 1] are essentially unchanged after one extra sigmoid pass
    only if they're near 0.5; to stay safe we ONLY sigmoid when the value
    falls outside [0, 1], otherwise we just clip.
    """
    if 0.0 <= x <= 1.0:
        return float(x)
    # Logit -> probability.
    import math
    return 1.0 / (1.0 + math.exp(-float(x)))


def bt_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """GRPO reward for one completion, scored by a local Bradley-Terry head.

    Args:
        response: The completion text generated by the policy.
        ground_truth: Dict with at least `game_state`. Other keys are unused.
        **kwargs: Ignored (siblings, recall_index, etc. -- kept for compat
                  with the trainer's uniform reward-fn calling convention).

    Returns:
        Score in [0, 1]. Neutral 0.5 on any error (missing env var, missing
        checkpoint, load failure, forward-pass exception).
    """
    if ground_truth is None or not response:
        return _NEUTRAL_SCORE

    model = _load_head()
    if model is None:
        return _NEUTRAL_SCORE

    state = ground_truth.get("game_state", {}) or {}

    try:
        from scripts.train_bt_head import bt_head_score  # type: ignore
        raw = bt_head_score(model, state, response)
        return _to_unit_interval(float(raw))
    except Exception as e:
        _warn_once(
            "score_fail",
            f"  [bt_reward] head forward failed "
            f"({type(e).__name__}: {str(e)[:160]}); returning neutral "
            f"{_NEUTRAL_SCORE} for this and subsequent failures (silenced).",
        )
        return _NEUTRAL_SCORE
