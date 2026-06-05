#!/usr/bin/env python3
"""Interactive walker for the D_plausible_wrong slot in pseudo_gold_stub.jsonl.

The pseudo-gold offline benchmark needs four advice candidates per state:
A_correct (parrots pro), B_anti_pro (inverted), C_generic (vague), and
D_plausible_wrong — confident, polished, uses CS2 vocabulary, but tactically
wrong for THIS state. A, B, C are mechanical; D needs domain knowledge.

This script walks you through every record that still has a TODO D, displays
the state and the existing A_correct (so you can see what "right" looks
like), and prompts you for a D. Autosaves after each entry. You can quit
(empty line or Ctrl-C) and re-run to pick up where you left off.

Usage:
    python scripts/label_d_advices.py
    python scripts/label_d_advices.py --path data/eval/pseudo_gold_stub.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/eval/pseudo_gold_stub.jsonl")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def is_todo(value: str) -> bool:
    return str(value).startswith("TODO")


def render_record(idx: int, total: int, rec: dict) -> str:
    cands = rec.get("candidates", {})
    pa = rec.get("pro_action", {})
    gs = rec.get("game_state", {})
    lines = [
        "=" * 72,
        f"  Record {idx + 1}/{total}  |  id={rec.get('id')}  |  "
        f"{gs.get('map_name')} {gs.get('player_side')} {gs.get('round_phase')}  |  "
        f"round_won={rec.get('round_won')}",
        "=" * 72,
        "",
        "STATE SUMMARY:",
        rec.get("state_summary", "")[:1200],
        "",
        f"PRO ACTION: {pa.get('description', '')}",
        f"  categories: {pa.get('categories', [])}",
        f"  behavior: {pa.get('behavior', {})}",
        "",
        f"A_correct  (HIGH): {cands.get('A_correct', '<missing>')}",
        f"B_anti_pro (LOW):  {cands.get('B_anti_pro', '<missing>')}",
        f"C_generic  (MID):  {cands.get('C_generic', '<missing>')}",
        "",
        "D_plausible_wrong should:",
        "  - sound confident, use CS2 vocabulary",
        "  - be specific (not generic like C)",
        "  - be TACTICALLY WRONG for THIS state (e.g., recommend the wrong",
        "    site, the wrong angle, ignore the bomb timer, push into a 1v3)",
        "  - NOT just parrot the inverted action (that's B)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} not found. Run scripts/build_pseudo_gold.py first.")
        sys.exit(1)

    records = load(args.path)
    pending = [i for i, r in enumerate(records)
               if is_todo(r.get("candidates", {}).get("D_plausible_wrong", "TODO"))]

    if not pending:
        print(f"All {len(records)} records already have D_plausible_wrong filled in.")
        print(f"Next step: python scripts/eval_scorer.py {args.path}")
        return

    print(f"{len(pending)} of {len(records)} records still need D_plausible_wrong.")
    print("Type your D advice, press Enter. Empty line or Ctrl-C to quit (progress saves).")
    print()

    try:
        for n, idx in enumerate(pending):
            rec = records[idx]
            print(render_record(n, len(pending), rec))
            try:
                d = input("D_plausible_wrong > ").strip()
            except EOFError:
                d = ""
            if not d:
                print(f"\nStopping. {len(pending) - n} records still pending.")
                break
            rec["candidates"]["D_plausible_wrong"] = d
            save(args.path, records)
            print(f"  saved.\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
        return

    remaining = sum(1 for r in records
                    if is_todo(r.get("candidates", {}).get("D_plausible_wrong", "TODO")))
    print()
    if remaining == 0:
        print(f"Done. All {len(records)} records have D_plausible_wrong filled.")
        print(f"Next step: python scripts/eval_scorer.py {args.path}")
    else:
        print(f"{remaining} records still pending. Re-run to continue.")


if __name__ == "__main__":
    main()
