"""Claude-judge reward for GRPO.

Replaces RECALL's kNN-over-embeddings with a single Claude call per training
step that scores all G generated completions against each other on a CS2
strategic-advice rubric.

Why this exists: empirical analysis of 402 training steps from f08v4 +
f08v4_resumed showed that when 2+ completions passed the format gate at the
same step, RECALL's median spread among the passers was 0.000 -- it gave
identical or near-identical scores to different strategic advices. The 19-dim
state embedding was dominated by surface features (map, side, HP) so kNN
retrieved mostly same-round neighbors with identical round_won outcomes,
collapsing V_hat == Q_hat and zeroing the advantage. The Claude judge
bypasses state-equivalence entirely by reading the actual advice text.

See claude-progress.txt 2026-04-23 entry for full diagnosis.

Cost: ~1 Anthropic API call per GRPO training step (not per completion --
results are cached). At Opus 4.7 prices, ~$0.005-0.02 per step, so ~$1-2
for a 100-step run. Trivial vs pod cost.

Failure mode: if the judge call errors (network, JSON parse, rate limit),
returns neutral 0.5 for every completion. That step contributes no gradient
signal but does not crash training.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import anthropic

_DEFAULT_MODEL = os.environ.get("CHIMERA_JUDGE_MODEL", "claude-opus-4-7")
_NEUTRAL_SCORE = 0.5  # returned when the judge call fails for any reason
_MAX_TOKENS = 512

_RUBRIC = """\
You are evaluating tactical CS2 advice for the spectated player at a specific
moment in a pro match. You are given the GROUND-TRUTH game state, what the
pro player ACTUALLY did, and whether their team won the round. Your job: rate
each piece of candidate advice from 0.0 (terrible) to 1.0 (excellent) on
whether it would lead to a winning outcome in this state.

Score each candidate on:
  - Strategic soundness: respects economy, numbers, time, bomb status
  - Action match with the pro: did this advice point in the same direction as
    what the pro actually chose? Anchor on this.
  - Specificity: concrete and actionable, not vague
  - Coherence with the actual game state, not generic patter

Outcome anchor: if the pro WON the round, the pro's action is a positive
example -- advice matching it should score higher. If the pro LOST the round,
their action is a negative example -- advice matching it can be wrong, and an
alternative might be better.

Be discriminating. If candidates differ in quality, give them clearly
different scores -- don't bunch them all near 0.5. Ties are allowed only when
candidates are genuinely indistinguishable.

Return STRICT JSON only, no prose around it. Format exactly:
{
  "scores": [<float for c1>, <float for c2>, ...],
  "best_idx": <int 0-based>,
  "rationale": "<1-2 sentence explanation focused on WHY the best is best>"
}
"""


def _strip_code_fence(text: str) -> str:
    """Pull the JSON object out of a possibly-fenced response."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for p in parts:
        p = p.strip()
        if p.startswith("json"):
            p = p[4:].strip()
        if p.startswith("{"):
            return p
    return text


@functools.lru_cache(maxsize=128)
def _judge_call_cached(
    state_json: str,
    gt_json: str,
    completions_tuple: tuple[str, ...],
    model: str,
) -> tuple[float, ...]:
    """Cached single judge call. Returns scores in completion order.

    Cache key includes the full completions_tuple, so the same group of
    completions only triggers one API call across the per-completion calls
    in the trainer's reward loop.
    """
    n = len(completions_tuple)
    if n == 0:
        return ()

    client = anthropic.Anthropic()

    comp_block = "\n\n".join(
        f"=== Candidate {i + 1} ===\n{c}" for i, c in enumerate(completions_tuple)
    )

    prompt = (
        f"{_RUBRIC}\n\n"
        f"GAME STATE (ground truth):\n{state_json}\n\n"
        f"PRO ACTION + ROUND OUTCOME:\n{gt_json}\n\n"
        f"CANDIDATE ADVICES ({n} total):\n{comp_block}\n\n"
        "Return JSON now."
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        text = _strip_code_fence(text)
        result = json.loads(text)
        scores = result.get("scores", [])
        if not isinstance(scores, list) or len(scores) != n:
            raise ValueError(f"expected {n} scores, got {scores!r}")
        clamped = tuple(float(min(1.0, max(0.0, float(s)))) for s in scores)
        return clamped
    except Exception as e:
        # Soft-fail: this step gets no gradient. Better than crashing training.
        print(
            f"  [judge_reward] call failed ({type(e).__name__}: {str(e)[:100]}), "
            f"returning neutral {_NEUTRAL_SCORE} for all {n} completions"
        )
        return tuple([_NEUTRAL_SCORE] * n)


def judge_reward(
    response: str,
    ground_truth: dict[str, Any] | None = None,
    siblings: list[str] | None = None,
    judge_model: str | None = None,
    **kwargs: Any,
) -> float:
    """GRPO reward for one completion, scored by Claude judge across the group.

    The judge sees ALL G sibling completions at once and ranks them. Score for
    this specific completion is returned. The trainer's per-completion reward
    loop calls this G times; the cache means only the first call hits the API.

    Args:
        response: This completion's text.
        ground_truth: Dict with 'game_state', 'pro_action', 'round_won'.
        siblings: ALL G completions in the order they were generated. MUST
                  include `response` itself. The trainer is responsible for
                  passing this through.
        judge_model: Override the default judge model.
        **kwargs: Ignored (compatibility with other reward fns).

    Returns:
        Score in [0, 1] for this completion. Neutral 0.5 on any error or when
        the trainer didn't pass `siblings`.
    """
    if ground_truth is None or siblings is None:
        return _NEUTRAL_SCORE
    if response not in siblings:
        # Mismatch suggests the trainer wasn't wired correctly. Don't crash.
        return _NEUTRAL_SCORE

    state = ground_truth.get("game_state", {})
    gt_summary = {
        "pro_action": ground_truth.get("pro_action", {}),
        "round_won": ground_truth.get("round_won"),
    }

    state_json = json.dumps(state, sort_keys=True)
    gt_json = json.dumps(gt_summary, sort_keys=True)
    completions_tuple = tuple(siblings)
    model = judge_model or _DEFAULT_MODEL

    scores = _judge_call_cached(state_json, gt_json, completions_tuple, model)

    if len(scores) != len(siblings):
        return _NEUTRAL_SCORE
    idx = siblings.index(response)
    return scores[idx]
