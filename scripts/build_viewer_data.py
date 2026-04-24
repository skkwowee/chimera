#!/usr/bin/env python3
"""Emit a JSONL the cs2-demo-viewer can overlay on round replays:
for each GRPO training sample, the model's advices + scores + ground truth +
*K nearest neighbor ticks* under three different state-equivalence metrics.

The viewer reads this and, when on a tick (demo, round, tick), shows:
  - the 4 candidate advices the model produced + judge scores + format-pass
  - the pro action + round outcome
  - K neighbors per metric, each with a (demo, round, tick) link the user
    can click to jump to that replay frame

The "neighbors" are the equivalence judgment we've been arguing about. Three
metrics so the user can compare:
  1. tactical_19d  -- the hand-engineered tactical_embedding RECALL used.
                      The "previously hacked our states" baseline.
  2. sentence_emb  -- sentence-transformer embedding of game_state JSON.
                      Captures lexical-semantic similarity but not
                      tactical decision equivalence.
  3. coarse_key    -- exact match on (player_side, alive_t, alive_e,
                      bomb_status, round_phase). Discrete, interpretable.
                      Often returns 0 or many; useful as a sanity check.

Inputs:
  --audit       Path to useful_jumps.jsonl from a training run.
  --source      Path to smoke_test_source.jsonl (recover_source_metadata.py).
  --dataset     Path to smoke_test.jsonl (full ground truth).
  --output      Path to viewer JSONL to emit.
  --k           Neighbors per metric (default 10).

Output schema (one line per audit sample):
  {
    "step_at_use": int,
    "sample_idx": int,
    "source": {demo_stem, round_num, tick, player_name, player_side, map_name},
    "query_state": {...game_state...},
    "pro_action": {...},
    "round_won": bool,
    "advices": [
      {"text_preview": "...", "reward": float, "passed_format": bool},
      ...G entries
    ],
    "neighbors": {
      "tactical_19d":  [{sample_idx, distance, source: {...}}, ...K],
      "sentence_emb":  [...K],
      "coarse_key":    [...K]
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable regardless of cwd / PYTHONPATH so the workflow
# documented in docs/viewer-grpo-overlay.md just works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# We import these lazily so the script's --help doesn't pull torch.
def _load_tactical_embedding():
    from src.training.recall import tactical_embedding
    return tactical_embedding


def _load_sentence_embedder(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def coarse_key(gs: dict) -> tuple:
    return (
        gs.get("player_side"),
        gs.get("alive_teammates"),
        gs.get("alive_enemies"),
        gs.get("bomb_status"),
        gs.get("round_phase"),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    print("Loading dataset...")
    dataset = [json.loads(l) for l in open(args.dataset)]
    print(f"  {len(dataset)} samples")

    print("Loading source metadata...")
    src_by_idx: dict[int, dict] = {}
    for line in open(args.source):
        rec = json.loads(line)
        src_by_idx[rec["idx"]] = rec
    print(f"  {len(src_by_idx)} samples have source metadata")

    print("Loading audit...")
    audit = [json.loads(l) for l in open(args.audit)]
    print(f"  {len(audit)} audit rows")

    # --- Pre-compute embeddings for the FULL dataset (not just audit samples)
    # so neighbor candidates span everything, not just touched samples.
    print("Computing tactical_19d embeddings for dataset...")
    tactical_embedding = _load_tactical_embedding()
    n = len(dataset)
    tact_embs = np.zeros((n, 19), dtype=np.float32)
    for i, s in enumerate(dataset):
        gs = s.get("ground_truth", {}).get("game_state", {})
        tact_embs[i] = np.asarray(tactical_embedding(gs), dtype=np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(tact_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tact_norm = tact_embs / norms

    print("Computing sentence embeddings for dataset...")
    sbert = _load_sentence_embedder()
    sent_texts = []
    for s in dataset:
        gs = s.get("ground_truth", {}).get("game_state", {})
        sent_texts.append(json.dumps(gs, sort_keys=True))
    sent_embs = sbert.encode(
        sent_texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=128,
    )

    print("Indexing coarse keys...")
    coarse_idx: dict[tuple, list[int]] = {}
    for i, s in enumerate(dataset):
        gs = s.get("ground_truth", {}).get("game_state", {})
        coarse_idx.setdefault(coarse_key(gs), []).append(i)

    # --- Helpers to pull K nearest neighbors per metric, excluding the query itself
    def topk_cosine(query_vec: np.ndarray, all_vecs: np.ndarray, query_idx: int, k: int) -> list[tuple[int, float]]:
        sims = all_vecs @ query_vec
        sims[query_idx] = -np.inf  # exclude self
        order = np.argpartition(-sims, k)[:k]
        order = order[np.argsort(-sims[order])]
        return [(int(i), float(1.0 - sims[i])) for i in order]  # distance = 1 - sim

    def coarse_neighbors(query_idx: int, k: int) -> list[tuple[int, float]]:
        gs = dataset[query_idx].get("ground_truth", {}).get("game_state", {})
        key = coarse_key(gs)
        peers = [i for i in coarse_idx.get(key, []) if i != query_idx]
        # Distance 0.0 (exact key match), truncate to k
        return [(i, 0.0) for i in peers[:k]]

    # --- Build viewer JSONL
    print(f"Writing viewer data to {args.output}...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_no_source = 0
    with out_path.open("w") as out_f:
        for row in audit:
            si = row.get("sample_idx")
            if si is None or si >= len(dataset):
                continue
            src = src_by_idx.get(si)
            if src is None:
                skipped_no_source += 1
                continue

            gt = dataset[si].get("ground_truth", {})
            gs = gt.get("game_state", {})

            advices = []
            # Back-compat: old audits used `completions_first200`; new runs use
            # `completions_preview` (first 800 chars).
            completions = row.get("completions_preview") or row.get("completions_first200", [])
            for c, r in zip(completions, row.get("rewards", [])):
                advices.append({
                    "text_preview": c,
                    "reward": r,
                    "passed_format": abs(r) > 1e-9,
                })

            # Neighbors per metric
            tact_n = topk_cosine(tact_norm[si], tact_norm, si, args.k)
            sent_n = topk_cosine(sent_embs[si], sent_embs, si, args.k)
            coarse_n = coarse_neighbors(si, args.k)

            def to_neighbor_block(items):
                out = []
                for n_idx, dist in items:
                    n_src = src_by_idx.get(n_idx)
                    if n_src is None:
                        continue
                    out.append({
                        "sample_idx": n_idx,
                        "distance": dist,
                        "source": {k: n_src[k] for k in
                                    ("demo_stem", "round_num", "tick",
                                     "player_name", "player_side", "map_name")},
                    })
                return out

            record = {
                "step_at_use": row.get("step_at_use"),
                "sample_idx": si,
                "source": {k: src[k] for k in
                            ("demo_stem", "round_num", "tick",
                             "player_name", "player_side", "map_name")},
                "query_state": gs,
                "pro_action": gt.get("pro_action", {}),
                "round_won": gt.get("round_won"),
                "advices": advices,
                "neighbors": {
                    "tactical_19d": to_neighbor_block(tact_n),
                    "sentence_emb": to_neighbor_block(sent_n),
                    "coarse_key":   to_neighbor_block(coarse_n),
                },
            }
            out_f.write(json.dumps(record) + "\n")
            written += 1

    print(f"\nWrote {written} records ({skipped_no_source} skipped: no source metadata)")
    print(f"Viewer can read: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
