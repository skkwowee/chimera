#!/usr/bin/env python3
"""Generate natural-language tactical captions for CS2 game states via Claude.

Data-generation half of the "verbalization discriminative check" (see
claude-progress.txt). Research question: does a natural-language description
of a game state retain the tactical signal that predicts the round outcome?
If a text classifier trained on these captions predicts round_won as well as
one trained on the raw structured features, language is a near-sufficient
statistic for tactical state — the core premise of the "See, Then Think" L3
verbalization layer.

We render each EGOCENTRIC per-player game_state into an analyst briefing and
ask Claude for a 1-2 sentence tactical read. The prompt FORBIDS outcome
prediction so the caption can't smuggle in "they'll win" (which would make
the downstream round_won probe dishonestly easy). The caption describes the
*situation*; the discriminative check then measures whether situation-language
alone carries outcome signal.

Single-tick caveat: one tick under-specifies intent (execute vs. fake). That
caps how much ANY method extracts here; the structured-feature ceiling in
discriminative_check.py controls for it — both classifiers face the same
single-tick noise, so the COMPARISON stays valid even if absolute accuracy
is capped.

Schema note: game_state lives in smoke_test.jsonl under
ground_truth.game_state (egocentric: player_health, weapon_primary,
alive_teammates, alive_enemies, bomb_status, ...). smoke_test_source.jsonl
is line-aligned and supplies demo_stem/round_num/tick.

Output:
    data/captions/captions.jsonl
      {demo_stem, round_num, tick, caption, round_won}

Usage:
    python scripts/generate_captions.py                 # all rows
    python scripts/generate_captions.py --limit 200     # smoke test
    python scripts/generate_captions.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "training" / "grpo" / "smoke_test_source.jsonl"
DATA = REPO / "data" / "training" / "grpo" / "smoke_test.jsonl"
OUT_DIR = REPO / "data" / "captions"
OUT_PATH = OUT_DIR / "captions.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-6"  # captioning is easy; opus is overkill

SYSTEM = """You are a professional Counter-Strike 2 analyst providing concise \
tactical reads of mid-round situations, in the style of a pro caster or coach.

You are given a snapshot from ONE player's point of view. Write ONE OR TWO \
sentences (≤55 words) describing that player's TACTICAL SITUATION: their \
role and likely positioning given side and phase, the numbers and economy \
picture, the utility they hold, and the immediate pressure or decision they \
face.

Hard rules:
- Describe the SITUATION ONLY. Do NOT predict, hint at, or imply who will win \
the round. No "they should win this", no "advantage", no outcome language.
- Be specific and concrete (phase, numbers, weapon, utility, bomb state).
- Professional register. No filler, no preamble. Output the read directly."""


def render_state(gs: dict) -> str:
    """Render the egocentric per-player game_state into an analyst briefing."""
    util = gs.get("utility") or []
    util_s = ", ".join(util) if util else "none"
    return "\n".join([
        f"Map: {gs.get('map_name','?').replace('de_','')}",
        f"Phase: {gs.get('round_phase','?')}",
        f"Your side: {gs.get('player_side','?')}",
        f"Your health: {gs.get('player_health','?')} | armor: {gs.get('player_armor','?')}",
        f"Your money: ${gs.get('player_money','?')}",
        f"Your weapons: {gs.get('weapon_primary','?')} / {gs.get('weapon_secondary','?')}",
        f"Your utility: {util_s}",
        f"Teammates alive (incl. you): {gs.get('alive_teammates','?')}",
        f"Enemies alive: {gs.get('alive_enemies','?')}",
        f"Bomb: {gs.get('bomb_status','?')}",
        f"Enemies currently visible to you: {gs.get('visible_enemies','?')}",
    ])


def caption_one(client: Anthropic, model: str, prompt: str,
                max_retries: int = 4) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=120,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [error] {type(e).__name__}: {e}", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of rows (cheap smoke test)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    sources = [json.loads(l) for l in SOURCE.open() if l.strip()]
    data = [json.loads(l) for l in DATA.open() if l.strip()]
    assert len(sources) == len(data), f"{len(sources)} != {len(data)}"
    rows = list(zip(sources, data))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Captioning {len(rows)} game states with {args.model} "
          f"({args.workers} workers)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = Anthropic()

    # Resume: skip rows already captioned (keyed by demo/round/tick)
    done: set[tuple] = set()
    if args.out.exists():
        for l in args.out.open():
            if l.strip():
                r = json.loads(l)
                done.add((r["demo_stem"], r["round_num"], r["tick"]))
        print(f"  resuming: {len(done)} already captioned", flush=True)

    def task(idx_row):
        _, (s, d) = idx_row
        key = (s["demo_stem"], s["round_num"], s["tick"])
        if key in done:
            return None
        gt = d.get("ground_truth", {}) or {}
        gs = gt.get("game_state")
        rw = gt.get("round_won")
        if gs is None or rw is None:
            return None
        cap = caption_one(client, args.model, render_state(gs))
        if cap is None:
            return None
        return {
            "demo_stem": s["demo_stem"],
            "round_num": s["round_num"],
            "tick": s["tick"],
            "caption": cap,
            "round_won": bool(rw),
        }

    n_written = 0
    t0 = time.time()
    with open(args.out, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(task, ir) for ir in enumerate(rows)]
            for j, fut in enumerate(as_completed(futures)):
                res = fut.result()
                if res is not None:
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                    n_written += 1
                if (j + 1) % 100 == 0:
                    dt = time.time() - t0
                    print(f"  {j+1}/{len(futures)} done, {n_written} written, "
                          f"{dt:.0f}s ({(j+1)/dt:.1f}/s)", flush=True)

    print(f"\nWrote {n_written} captions to {args.out} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
