#!/usr/bin/env python3
"""Fine-tune a sentence-transformer on CS2 game states so neighbors are
strategically equivalent, not just surface-feature-similar.

Why: the 19-dim hand-engineered embedding RECALL uses returns 82% same-round
neighbors (hack). Baseline sentence-transformer on game_state JSON does 81%
(lexical similarity, not tactics). Goal: learn an embedding where two states
that call for the same kind of pro decision -- and where that decision actually
worked -- cluster together, regardless of demo/round/surface features.

Signal: two samples are "strategically equivalent" if
  1. Their pro_action.categories (as a set) match, AND
  2. Their round_won matches (decision worked equally -- success or failure).

Triplets: (anchor, positive, negative)
  anchor, positive -- same categories-set AND same round_won
  anchor, negative -- different categories-set

Encoder: starts from all-MiniLM-L6-v2 so we inherit language prior (knows
"AWP" and "de_inferno" are tokens). Fine-tune with triplet margin loss.

After training: embed all 2013 states, compute K-nearest for each, measure
  - same-round rate  (red -- bad, RECALL's failure mode)
  - cross-round rate (green -- good, within-demo generalization)
  - cross-demo rate  (yellow -- strongest, cross-situation generalization)
  - same-category rate (sanity: did we learn anything about decisions?)

Compare all four: tactical_19d (RECALL), baseline_sent (off-the-shelf
MiniLM), learned_sent (ours), and optionally joint of all three.

Usage:
  python scripts/train_state_embedding.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --source  data/training/grpo/smoke_test_source.jsonl \\
      --audit   /workspace/outputs/grpo/f09/useful_jumps.jsonl \\
      --output  /workspace/outputs/embedding/learned_v1 \\
      --epochs 3 --triplets-per-anchor 5 --k 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_tactical_embedding():
    from src.training.recall import tactical_embedding
    return tactical_embedding


def _state_text(gs: dict) -> str:
    """Serialize game_state deterministically for the sentence encoder."""
    return json.dumps(gs, sort_keys=True, ensure_ascii=False)


def _category_key(pro_action: dict) -> frozenset:
    cats = pro_action.get("categories", []) or []
    return frozenset(cats)


def build_triplets(
    dataset: list[dict],
    triplets_per_anchor: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Emit (anchor_text, positive_text, negative_text) triples.

    Positive = same category-set AND same round_won.
    Negative = different category-set (any).
    Anchors that have <1 positive available are skipped.
    """
    # Index by (category_set, round_won) bucket
    buckets: dict[tuple[frozenset, bool], list[int]] = defaultdict(list)
    states: list[str] = []
    cats_by_idx: list[frozenset] = []
    for i, s in enumerate(dataset):
        gt = s.get("ground_truth", {})
        gs = gt.get("game_state", {})
        pa = gt.get("pro_action", {})
        rw = bool(gt.get("round_won", False))
        ckey = _category_key(pa)
        buckets[(ckey, rw)].append(i)
        states.append(_state_text(gs))
        cats_by_idx.append(ckey)

    all_indices = list(range(len(dataset)))
    triplets: list[tuple[str, str, str]] = []

    for i in range(len(dataset)):
        key = (cats_by_idx[i], bool(dataset[i].get("ground_truth", {}).get("round_won", False)))
        pos_pool = [j for j in buckets[key] if j != i]
        if not pos_pool:
            continue
        for _ in range(triplets_per_anchor):
            j = rng.choice(pos_pool)
            # Sample negatives until we hit one with different category-set.
            # Usually one draw suffices.
            for _attempt in range(10):
                k = rng.choice(all_indices)
                if k != i and cats_by_idx[k] != cats_by_idx[i]:
                    triplets.append((states[i], states[j], states[k]))
                    break
    return triplets


def train_encoder(
    base_model: str,
    triplets: list[tuple[str, str, str]],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Fine-tune the encoder on triplet margin loss and save to output_dir."""
    from sentence_transformers import (
        InputExample,
        SentenceTransformer,
        losses,
    )
    from torch.utils.data import DataLoader

    print(f"Loading base encoder: {base_model}")
    model = SentenceTransformer(base_model)

    examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(model=model, triplet_margin=0.3)

    print(f"Training: {len(examples)} triplets, {epochs} epochs, batch={batch_size}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=max(1, len(loader) // 10),
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    print(f"Saved encoder to {output_dir}")


def embed_dataset(model_name_or_path: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path)
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=False,
    )
    return embs.astype(np.float32)


def embed_tactical(dataset: list[dict]) -> np.ndarray:
    tactical_embedding = _load_tactical_embedding()
    n = len(dataset)
    embs = np.zeros((n, 19), dtype=np.float32)
    for i, s in enumerate(dataset):
        gs = s.get("ground_truth", {}).get("game_state", {})
        embs[i] = np.asarray(tactical_embedding(gs), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def neighbor_stats(
    name: str,
    embs: np.ndarray,
    dataset: list[dict],
    src_by_idx: dict[int, dict],
    cats_by_idx: list[frozenset],
    k: int,
) -> dict:
    """For each sample that has source metadata, compute neighbor mix.

    Vectorized: one (n,n) similarity matrix, argpartition per row. n=2013 is
    trivial — 16MB of float32. Self-excluded by setting diagonal to -inf.
    """
    n = embs.shape[0]
    sims = embs @ embs.T  # (n, n); unit vectors, so dot = cosine sim
    np.fill_diagonal(sims, -np.inf)
    # Top-k indices per row
    topk_unsorted = np.argpartition(-sims, k, axis=1)[:, :k]
    # Sort within top-k
    topk = np.zeros_like(topk_unsorted)
    for i in range(n):
        row = topk_unsorted[i]
        topk[i] = row[np.argsort(-sims[i, row])]

    total = dict(same_round=0, cross_round=0, cross_demo=0, no_src=0, same_cat=0, total=0)
    for q_idx in range(n):
        q_src = src_by_idx.get(q_idx)
        if q_src is None:
            continue
        q_cat = cats_by_idx[q_idx]
        for nb in topk[q_idx]:
            nb = int(nb)
            nb_src = src_by_idx.get(nb)
            if nb_src is None:
                total["no_src"] += 1
                total["total"] += 1
                continue
            if nb_src["demo_stem"] == q_src["demo_stem"]:
                if nb_src["round_num"] == q_src["round_num"]:
                    total["same_round"] += 1
                else:
                    total["cross_round"] += 1
            else:
                total["cross_demo"] += 1
            if cats_by_idx[nb] == q_cat:
                total["same_cat"] += 1
            total["total"] += 1
    tot = max(total["total"], 1)
    pct = {key: total[key] / tot for key in ("same_round", "cross_round", "cross_demo", "same_cat", "no_src")}
    return {"metric": name, "k": k, "counts": total, "pct": pct}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True, help="Dir to save fine-tuned encoder")
    p.add_argument("--base-model", default="all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--triplets-per-anchor", type=int, default=5)
    p.add_argument("--k", type=int, default=10, help="K for neighbor eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, just eval the encoder at --output")
    args = p.parse_args()

    rng = random.Random(args.seed)

    print("Loading dataset...")
    dataset = [json.loads(l) for l in open(args.dataset)]
    print(f"  {len(dataset)} samples")

    print("Loading source metadata...")
    src_by_idx = {r["idx"]: r for r in (json.loads(l) for l in open(args.source))}
    print(f"  {len(src_by_idx)} samples with source metadata")

    cats_by_idx = [
        _category_key(s.get("ground_truth", {}).get("pro_action", {}))
        for s in dataset
    ]

    output_dir = Path(args.output)

    if not args.eval_only:
        print("\nBuilding triplets...")
        triplets = build_triplets(dataset, args.triplets_per_anchor, rng)
        print(f"  {len(triplets)} triplets generated")
        if len(triplets) < 100:
            print("ERROR: too few triplets. Check category distribution.")
            return 1

        print("\nTraining encoder...")
        train_encoder(
            base_model=args.base_model,
            triplets=triplets,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    print("\n=== Evaluation ===")
    texts = [_state_text(s.get("ground_truth", {}).get("game_state", {})) for s in dataset]

    print("\nComputing tactical_19d embeddings...")
    tact = embed_tactical(dataset)

    print("Computing baseline sentence-transformer embeddings...")
    base = embed_dataset(args.base_model, texts)

    print("Computing learned encoder embeddings...")
    learned = embed_dataset(str(output_dir), texts)

    print(f"\nNeighbor stats (k={args.k}, excluding self):")
    print(f"{'metric':<18} {'same_round':>10} {'cross_round':>11} {'cross_demo':>10} {'same_cat':>10} {'total':>10}")
    print("-" * 75)
    for name, embs in [
        ("tactical_19d", tact),
        ("baseline_sent", base),
        ("learned_sent", learned),
    ]:
        r = neighbor_stats(name, embs, dataset, src_by_idx, cats_by_idx, args.k)
        c = r["counts"]
        p = r["pct"]
        print(
            f"{name:<18} "
            f"{p['same_round']*100:9.1f}% "
            f"{p['cross_round']*100:10.1f}% "
            f"{p['cross_demo']*100:9.1f}% "
            f"{p['same_cat']*100:9.1f}% "
            f"{c['total']:>10d}"
        )

    summary_path = output_dir / "neighbor_eval.json"
    results = []
    for name, embs in [
        ("tactical_19d", tact),
        ("baseline_sent", base),
        ("learned_sent", learned),
    ]:
        results.append(neighbor_stats(name, embs, dataset, src_by_idx, cats_by_idx, args.k))
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
