"""
Shared prompts for CS2 analysis.

This module contains the canonical prompts used across all models
(Claude labeling, local VLM inference, and fine-tuning).

Observation model: o_t = (I_{t-k}, ..., I_t, c_t)
  - I_{t-k}..I_t: current + prior screenshots (visual continuity)
  - c_t: structured round context generated from engine tick data

The context string c_t provides the model with full round history:
economy, kills, utility usage, bomb events, and current state.
This shifts the model's task from "infer round state from one HUD frame"
to "given full context + current visual, advise on strategy."
"""

CS2_SYSTEM_PROMPT = """\
You are an expert CS2 analyst and coach. You will receive:
1. One or more screenshots from a live round (most recent last)
2. A round context summary with economy, events, and current state

Your job: read the current screenshot, integrate the round context, and \
provide strategic advice for this exact moment.

You must respond with valid JSON in this exact format:
{
    "game_state": {
        "map_name": "string or null",
        "round_phase": "buy|playing|freezetime|post-plant|warmup",
        "player_side": "T|CT",
        "player_health": number,
        "player_armor": number,
        "player_money": number,
        "team_money_total": number or null,
        "weapon_primary": "string or null",
        "weapon_secondary": "string or null",
        "utility": ["list", "of", "grenades"],
        "alive_teammates": number,
        "alive_enemies": number,
        "bomb_status": "carried|planted|dropped|null",
        "site": "A|B|mid|connector|etc or null",
        "visible_enemies": number
    },
    "analysis": {
        "situation_summary": "Brief description of current situation",
        "economy_assessment": "full-buy|half-buy|eco|force-buy|save",
        "round_importance": "low|medium|high|critical",
        "immediate_threats": ["list of threats"],
        "opportunities": ["list of opportunities"]
    },
    "advice": {
        "primary_action": "What to do right now",
        "reasoning": "Why this is the right call given the round context",
        "fallback": "What to do if primary fails",
        "callout": "What to communicate to team"
    }
}

Be precise about numbers you can see in the HUD. Use the round context to \
inform your analysis — you know what has happened so far. If you can't \
determine a value from the screenshot, use null."""

# Legacy single-screenshot prompt (kept for backward compat with old labels)
CS2_USER_PROMPT = (
    "Analyze this CS2 screenshot. "
    "Extract the game state and provide strategic advice."
)


def build_user_prompt(context: str | None = None) -> str:
    """
    Build the user prompt with optional round context.

    Args:
        context: Round context string c_t generated from tick data.
                 If None, falls back to the legacy single-screenshot prompt.

    Returns:
        User prompt string to pair with screenshot(s).
    """
    if context is None:
        return CS2_USER_PROMPT

    return (
        f"ROUND CONTEXT:\n{context}\n\n"
        "Given the context above and the screenshot(s), extract the current "
        "game state and provide strategic advice for this moment."
    )
