#!/usr/bin/env python3
"""One-shot: read pseudo_gold_stub.jsonl, fill in A_correct / B_anti_pro / C_generic
based on pro_action and game_state. Leaves D_plausible_wrong as TODO.

Run from chimera repo root."""

import json
from pathlib import Path

PATH = Path("data/eval/pseudo_gold_stub.jsonl")
GENERIC = (
    "Stay focused and play the round out. Communicate with teammates, "
    "watch your common angles, and don't take unnecessary fights. "
    "Trust your training and the standard CT/T responses for this kind of situation."
)


def numbers_phrase(gs):
    me = 1  # self alive (we're the POV)
    t = gs.get("alive_teammates", 0) + me
    e = gs.get("alive_enemies", 0)
    diff = t - e
    if diff > 0:
        return f"You're up {t}v{e}"
    if diff < 0:
        return f"You're down {t}v{e}"
    return f"It's a {t}v{e}"


def time_phrase(gs):
    if gs.get("bomb_status") == "planted":
        return "the bomb is planted, so time pressure is on CT to retake"
    if gs.get("round_phase") == "post-plant" and gs.get("player_side", "").lower() == "t":
        return "you've planted, so time is on your side"
    return None


def util_phrase(behavior):
    used = behavior.get("utility_used", []) or []
    if used:
        joined = ", ".join(used).lower()
        return f"use your {joined} to support the play"
    return None


def a_correct(record):
    gs = record["game_state"]
    pa = record["pro_action"]
    behavior = pa.get("behavior", {})
    cats = [c.lower() for c in pa.get("categories", [])]
    cats_str = " ".join(cats)

    move = behavior.get("movement_direction", 0)
    obj = behavior.get("objective_direction", 0)
    initiated = behavior.get("initiated_engagement", False)
    side = gs.get("player_side", "").upper()
    bomb = gs.get("bomb_status")

    # Decide the headline action
    if "execute" in cats_str or "push" in cats_str or move == 1:
        action = "Push and look for entry"
    elif "rotate" in cats_str and obj > 0:
        action = "Rotate toward the bomb"
    elif "rotate" in cats_str and obj < 0:
        action = "Rotate away from the bomb to find a flank angle"
    elif "hold" in cats_str and initiated:
        action = "Hold your angle but be ready to take a fight if they push"
    elif "hold" in cats_str:
        action = "Hold your post-plant angle and don't reposition unnecessarily"
    elif "retreat" in cats_str or "save" in cats_str or move == -1:
        action = "Disengage and save for the next round"
    elif "engage" in cats_str and initiated:
        action = "Take the first shot — initiate the engagement"
    elif "engage" in cats_str:
        action = "Be patient but don't refuse a clean trade"
    else:
        action = "Play this slow and follow your teammate's lead"

    # State-anchored reasoning
    parts = [numbers_phrase(gs)]
    tp = time_phrase(gs)
    if tp:
        parts.append(tp)
    up = util_phrase(behavior)
    if up:
        parts.append(up)
    reasoning = "; ".join(parts) + "."
    return f"{action}. {reasoning}"


def b_anti_pro(record):
    gs = record["game_state"]
    pa = record["pro_action"]
    behavior = pa.get("behavior", {})
    cats = [c.lower() for c in pa.get("categories", [])]
    cats_str = " ".join(cats)

    move = behavior.get("movement_direction", 0)
    obj = behavior.get("objective_direction", 0)
    initiated = behavior.get("initiated_engagement", False)
    bomb = gs.get("bomb_status")

    # Invert
    if "execute" in cats_str or "push" in cats_str or move == 1:
        action = "Stop pushing and pull back to spawn"
    elif "hold" in cats_str:
        action = "Don't sit here — rush forward and try to get a pick before they set up"
    elif "rotate" in cats_str and obj > 0:
        action = "Stay where you are; don't rotate toward the bomb, you'll get traded"
    elif "rotate" in cats_str and obj < 0:
        action = "Push directly onto the bomb and look for the defuse"
    elif "retreat" in cats_str or "save" in cats_str:
        action = "Don't save — push out and contest with what you have"
    elif "engage" in cats_str and initiated:
        action = "Don't take a fight here; disengage and play passive"
    else:
        action = "Stop playing passive and force the action"

    parts = [numbers_phrase(gs)]
    parts.append("you have to make something happen, time isn't on your side")
    return f"{action}. {'; '.join(parts)}."


def fill(record):
    cands = record.setdefault("candidates", {})
    cands["A_correct"] = a_correct(record)
    cands["B_anti_pro"] = b_anti_pro(record)
    cands["C_generic"] = GENERIC
    # D_plausible_wrong stays as the original TODO
    return record


def main():
    rows = [json.loads(line) for line in PATH.read_text().splitlines() if line.strip()]
    out = [fill(r) for r in rows]
    with open(PATH, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"Filled A/B/C for {len(out)} records. D_plausible_wrong still TODO.")


if __name__ == "__main__":
    main()
