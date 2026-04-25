"""Build a queue of GRPO completion-pair candidates for human (expert CS2 player) labeling.

Reads f09 GRPO audit rows (`useful_jumps.jsonl`) and joins them with the smoke-test
prompt dataset + source metadata to emit one preference-pair per output line. The
labeling tool downstream consumes the locked schema documented in the README.

Filters (each exists for a specific reason):
  1. Drop pairs where both completions are format-fails (reward == 0.0). The judge
     gave them zero credit, so neither is "preferred"; labeling these pairs would
     train on noise.
  2. Drop near-duplicate pairs: |reward_diff| < 0.02 AND completion text similarity
     > 0.95. The judge can't separate them and they read the same -- no signal for
     a human either, just labeling fatigue.
  3. Cap at 500 pairs total, sorted by `informativeness_score` DESC, so the most
     useful pairs surface first when the human starts labeling.

`informativeness_score = 0.5*(1 - |reward_diff|) + 0.5*(1 - completion_similarity)`.
The intuition: pairs where the judge wasn't confident (small |reward_diff|) but the
two completions are textually different are precisely where human judgment adds
information the auto-judge missed.

Similarity backend: sentence-transformers all-MiniLM-L6-v2 if importable, else a
character-trigram Jaccard fallback (no extra deps required).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

DEFAULT_AUDIT = "outputs/grpo/f09/useful_jumps.jsonl"
DEFAULT_DATASET = "data/training/grpo/smoke_test.jsonl"
DEFAULT_SOURCE = "data/training/grpo/smoke_test_source.jsonl"
DEFAULT_OUTPUT = "outputs/labels/candidate_pairs.jsonl"
SOURCE_RUN = "f09"
MAX_OUTPUT_PAIRS = 500
DUPLICATE_REWARD_EPS = 0.02
DUPLICATE_SIM_THRESH = 0.95


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def char_trigrams(text: str) -> set[str]:
    text = text or ""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def jaccard(a: str, b: str) -> float:
    ta, tb = char_trigrams(a), char_trigrams(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def build_similarity_fn(completions_pool: list[str]):
    """Returns (sim_fn, backend_name). sim_fn(text_a, text_b) -> float in [0,1]."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Cache embeddings keyed by text identity (id) for the run.
        embed_cache: dict[int, Any] = {}

        def encode(text: str):
            key = id(text)
            if key not in embed_cache:
                vec = model.encode([text], normalize_embeddings=True)[0]
                embed_cache[key] = np.asarray(vec, dtype=np.float32)
            return embed_cache[key]

        def sim(a: str, b: str) -> float:
            va, vb = encode(a), encode(b)
            return float(np.dot(va, vb))

        return sim, "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return jaccard, "char-trigram-jaccard"


def make_pair_id(sample_idx: int, i: int, j: int) -> str:
    return f"{SOURCE_RUN}_{sample_idx}_{i}_{j}"


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def histogram(scores: list[float], buckets: int = 5) -> list[tuple[str, int]]:
    edges = [i / buckets for i in range(buckets + 1)]
    counts = [0] * buckets
    for s in scores:
        # clamp to [0, 1)
        idx = min(int(s * buckets), buckets - 1) if s < 1.0 else buckets - 1
        idx = max(0, idx)
        counts[idx] += 1
    return [(f"[{edges[i]:.2f},{edges[i + 1]:.2f})", counts[i]) for i in range(buckets)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--audit", default=DEFAULT_AUDIT, type=Path)
    p.add_argument("--dataset", default=DEFAULT_DATASET, type=Path)
    p.add_argument("--source", default=DEFAULT_SOURCE, type=Path)
    p.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    p.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.exists() and not args.force:
        print(f"refusing to overwrite existing {args.output} (pass --force)", file=sys.stderr)
        return 2

    audit_rows = load_jsonl(args.audit)
    dataset_rows = load_jsonl(args.dataset)
    source_rows = load_jsonl(args.source)

    dataset_by_idx = {i: row for i, row in enumerate(dataset_rows)}
    source_by_idx: dict[int, dict[str, Any]] = {}
    for row in source_rows:
        key = row.get("idx")
        if key is None:
            continue
        source_by_idx[key] = row

    # Collect every completion text first so the similarity backend can warm up.
    all_texts: list[str] = []
    for row in audit_rows:
        comps = row.get("completions_preview") or row.get("completions_first200") or []
        all_texts.extend(comps)
    sim_fn, sim_backend = build_similarity_fn(all_texts)

    considered = 0
    dropped_double_format_fail = 0
    dropped_duplicate = 0
    pairs: list[dict[str, Any]] = []

    for row in audit_rows:
        sample_idx = row["sample_idx"]
        rewards = row.get("rewards") or []
        comps = row.get("completions_preview") or row.get("completions_first200") or []
        if len(comps) < 2:
            continue

        sample = dataset_by_idx.get(sample_idx)
        gt = (sample or {}).get("ground_truth", {}) if sample else {}
        state = gt.get("game_state", {}) or {}
        pro_action = gt.get("pro_action", {}) or {}
        src = source_by_idx.get(sample_idx, {})
        context = {
            "demo_stem": src.get("demo_stem"),
            "round_num": src.get("round_num"),
            "tick": src.get("tick"),
            "player_name": src.get("player_name"),
            "pro_action_categories": pro_action.get("categories", []),
            "pro_action_description": pro_action.get("description"),
            "round_won": gt.get("round_won"),
        }

        for i, j in combinations(range(len(comps)), 2):
            considered += 1
            r_a = rewards[i] if i < len(rewards) else None
            r_b = rewards[j] if j < len(rewards) else None

            # Filter 1: drop pairs where both are format-fails.
            if (r_a == 0.0) and (r_b == 0.0):
                dropped_double_format_fail += 1
                continue

            text_a, text_b = comps[i], comps[j]
            sim = sim_fn(text_a, text_b)
            dissim = max(0.0, 1.0 - sim)

            # Reward diff: when one side is None we treat |diff| as 1.0 (max informativeness on that axis).
            if r_a is None or r_b is None:
                reward_diff_abs = 1.0
            else:
                reward_diff_abs = abs(r_a - r_b)

            # Filter 2: near-duplicate by both reward AND text.
            if reward_diff_abs < DUPLICATE_REWARD_EPS and sim > DUPLICATE_SIM_THRESH:
                dropped_duplicate += 1
                continue

            informativeness = 0.5 * (1.0 - min(reward_diff_abs, 1.0)) + 0.5 * dissim

            pair = {
                "pair_id": make_pair_id(sample_idx, i, j),
                "sample_idx": sample_idx,
                "source_run": SOURCE_RUN,
                "state": state,
                "context": context,
                "completion_a": text_a,
                "completion_b": text_b,
                "judge_reward_a": r_a,
                "judge_reward_b": r_b,
                "informativeness_score": round(informativeness, 6),
            }
            pairs.append(pair)

    # Sort by informativeness DESC; tiebreak with stable hash for determinism.
    pairs.sort(key=lambda p: (-p["informativeness_score"], stable_hash(p["pair_id"])))
    capped = pairs[:MAX_OUTPUT_PAIRS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for pair in capped:
            fh.write(json.dumps(pair) + "\n")

    scores = [p["informativeness_score"] for p in capped]
    print(f"similarity backend: {sim_backend}")
    print(f"audit rows: {len(audit_rows)}")
    print(f"pairs considered: {considered}")
    print(f"dropped (both format-fail): {dropped_double_format_fail}")
    print(f"dropped (near-duplicate): {dropped_duplicate}")
    print(f"pairs after filtering: {len(pairs)}")
    print(f"pairs written (capped at {MAX_OUTPUT_PAIRS}): {len(capped)}")
    print("informativeness histogram (5 buckets over [0,1]):")
    for label, count in histogram(scores):
        print(f"  {label}: {count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
