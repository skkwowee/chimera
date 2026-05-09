#!/usr/bin/env python3
"""Build the pseudo-gold offline benchmark stub.

This is the gating offline test for any GRPO strategy-reward candidate. The
council's recommendation: do not spend pod time on a candidate that cannot
distinguish constructed-correct advice from constructed-wrong advice on cases
we built ourselves.

This script produces a JSONL stub: 30 (default) diverse states sampled from
data/training/grpo/smoke_test.jsonl, stratified across (map_name × side ×
round_phase) and de-duplicated to one sample per (demo, round). Each record
has empty slots for four hand-authored advices labeled by construction:

    A_correct          — parrots the pro_action with reasoning. Should score HIGH.
    B_anti_pro         — opposite of pro_action. Should score LOW (when pro won).
    C_generic          — vague platitudes ("play smart", "communicate"). Should
                         score MEDIUM-LOW. Used to detect raters that reward
                         confidence without specificity.
    D_plausible_wrong  — confident, polished, uses pro CS2 vocabulary, but
                         tactically wrong for THIS state (e.g. recommend a
                         losing play). The trickiest case — separates
                         "rates real strategy" from "rates polished writing."

The output JSONL is the input file you fill in by hand. After authoring, the
unbuilt `scripts/eval_scorer.py` consumes the same file and computes pairwise
AUC against construction labels for each candidate scorer (RECALL+mask, judge,
BT-head). The scorer with the highest AUC is the next pod-bet.

See docs/methodology.md and docs/reward-candidates.md for the full protocol.

Usage:
    python scripts/build_pseudo_gold.py
    python scripts/build_pseudo_gold.py --n 50 --output data/eval/pseudo_gold_50.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DEFAULT_DATA = Path("data/training/grpo/smoke_test.jsonl")
DEFAULT_SOURCE = Path("data/training/grpo/smoke_test_source.jsonl")
DEFAULT_OUTPUT = Path("data/eval/pseudo_gold_stub.jsonl")
DEFAULT_N = 30
DEFAULT_SEED = 42

CONSTRUCTION_LABELS = ["A_correct", "B_anti_pro", "C_generic", "D_plausible_wrong"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratify_key(sample: dict, source: dict | None) -> tuple[str, str, str]:
    gs = sample.get("ground_truth", {}).get("game_state", {})
    map_name = gs.get("map_name") or (source.get("map_name") if source else "unknown")
    side = gs.get("player_side") or (source.get("player_side") if source else "unknown")
    phase = gs.get("round_phase", "unknown")
    return (str(map_name), str(side), str(phase))


def round_key(source: dict) -> tuple[str, int]:
    return (str(source.get("demo_stem")), int(source.get("round_num", -1)))


def state_summary(sample: dict, max_len: int = 600) -> str:
    """Pull a readable header from the prompt for the human author's reference."""
    prompt = sample.get("prompt", "")
    if isinstance(prompt, list):
        text_parts = []
        for block in prompt:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "\n".join(text_parts)
    else:
        text = str(prompt)
    return text[:max_len].strip()


def select_diverse(
    samples: list[dict],
    sources: list[dict],
    n: int,
    seed: int,
) -> list[tuple[int, dict, dict]]:
    """Stratified sample: take roughly proportional counts per (map, side, phase)
    bucket, dedup to one (demo, round). Return list of (idx, sample, source).
    """
    rng = random.Random(seed)

    by_bucket: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, (sample, source) in enumerate(zip(samples, sources)):
        key = stratify_key(sample, source)
        by_bucket[key].append(i)

    # Round-robin over buckets, picking one sample per bucket until we have n.
    # Within each bucket, prefer earlier (more diverse) indices.
    bucket_keys = sorted(by_bucket.keys())
    rng.shuffle(bucket_keys)
    for k in bucket_keys:
        rng.shuffle(by_bucket[k])

    selected: list[tuple[int, dict, dict]] = []
    seen_rounds: set[tuple[str, int]] = set()
    bucket_iters = {k: iter(by_bucket[k]) for k in bucket_keys}

    while len(selected) < n and bucket_iters:
        exhausted = []
        for k in list(bucket_iters.keys()):
            if len(selected) >= n:
                break
            try:
                while True:
                    idx = next(bucket_iters[k])
                    rkey = round_key(sources[idx])
                    if rkey in seen_rounds:
                        continue
                    seen_rounds.add(rkey)
                    selected.append((idx, samples[idx], sources[idx]))
                    break
            except StopIteration:
                exhausted.append(k)
        for k in exhausted:
            del bucket_iters[k]

    return selected


def build_record(idx: int, sample: dict, source: dict) -> dict:
    gt = sample.get("ground_truth", {})
    return {
        "id": idx,
        "demo_stem": source.get("demo_stem"),
        "round_num": source.get("round_num"),
        "tick": source.get("tick"),
        "player_name": source.get("player_name"),
        "stratum": list(stratify_key(sample, source)),
        "state_summary": state_summary(sample),
        "game_state": gt.get("game_state", {}),
        "pro_action": gt.get("pro_action", {}),
        "round_won": gt.get("round_won"),
        "candidates": {
            "A_correct": "TODO: write advice that matches the pro_action with concrete reasoning",
            "B_anti_pro": "TODO: write advice that recommends the opposite of pro_action",
            "C_generic": "TODO: write generic vague advice (no specific action)",
            "D_plausible_wrong": "TODO: write polished CS2 advice that's tactically wrong here",
        },
        "_construction_meaning": {
            "A_correct": "should score HIGH",
            "B_anti_pro": "should score LOW (when round_won=true)",
            "C_generic": "should score MEDIUM-LOW",
            "D_plausible_wrong": "should score LOW; separates polish from tactics",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="smoke_test.jsonl with prompt + ground_truth")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="smoke_test_source.jsonl with (demo_stem, round_num, tick, player)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="stub output path (you hand-author the candidates afterward)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"number of states to sample (default {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    samples = load_jsonl(args.data)
    sources = load_jsonl(args.source)
    if len(samples) != len(sources):
        raise ValueError(
            f"data ({len(samples)}) and source ({len(sources)}) row counts disagree; "
            "they must be aligned by line index"
        )

    selected = select_diverse(samples, sources, args.n, args.seed)
    print(f"Selected {len(selected)} states from {len(samples)} candidates "
          f"(seed={args.seed}, stratified by map×side×phase, deduped on (demo, round))")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for idx, sample, source in selected:
            f.write(json.dumps(build_record(idx, sample, source)) + "\n")

    # Print stratum coverage so the author can sanity-check before authoring.
    counts: dict[tuple, int] = collections.Counter()
    for _, sample, source in selected:
        counts[stratify_key(sample, source)] += 1
    print("\nStratum coverage:")
    for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:>14}  {k[1]:>4}  {k[2]:>16}  n={c}")

    print(f"\nWrote stub: {args.output}")
    print(f"Next step: open the file and hand-author the four candidates per record.")
    print(f"After authoring: python scripts/eval_scorer.py {args.output}")


if __name__ == "__main__":
    main()
