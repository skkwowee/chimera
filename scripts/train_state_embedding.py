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


# Lexical shortcuts — fields that leak demo/round/tick identity and let the
# encoder cluster by surface similarity instead of tactical structure.
#   player_name/teammates/team -> player identity; same demo same round →
#     same strings → lexical hit
#   round_number/round           -> round identity
#   time_*/round_time/tick       -> monotonic within-round signal
#   score                        -> within a round, identical across ticks
# Kept (genuine tactical features): map_name, player_side, player_health,
# player_armor, weapon_primary, alive_teammates, alive_enemies, bomb_status,
# round_phase, economy.
_REDACT_KEYS = frozenset({
    "player_name", "player", "teammates", "team", "opponent_team",
    "round_number", "round",
    "time_remaining", "time", "round_time", "tick",
    "score", "current_player",
})


def _redact(obj, redact: bool):
    if not redact:
        return obj
    if isinstance(obj, dict):
        return {k: _redact(v, redact) for k, v in obj.items() if k not in _REDACT_KEYS}
    if isinstance(obj, list):
        return [_redact(v, redact) for v in obj]
    return obj


def _state_text(gs: dict, redact: bool = False) -> str:
    """Serialize game_state deterministically for the sentence encoder."""
    return json.dumps(_redact(gs, redact), sort_keys=True, ensure_ascii=False)


def _category_key(pro_action: dict) -> frozenset:
    cats = pro_action.get("categories", []) or []
    return frozenset(cats)


def _alive_bucket(n: int) -> str:
    """Coarse bucketing so 4v5 and 1v1 fall into different equivalence classes.
    3 buckets/axis = 9 combinations total, enough positives per bucket."""
    if n <= 1:
        return "clutch"  # 0 or 1 left — literally clutching
    if n <= 3:
        return "low"     # 2 or 3 left — mid round attrition
    return "high"        # 4 or 5 left — full team / early situation


def _alive_key(game_state: dict) -> tuple[str, str]:
    t = int(game_state.get("alive_teammates", 0) or 0)
    e = int(game_state.get("alive_enemies", 0) or 0)
    return (_alive_bucket(t), _alive_bucket(e))


def build_triplets(
    dataset: list[dict],
    triplets_per_anchor: int,
    rng: random.Random,
    redact: bool = False,
    alive_bucket: bool = False,
    counterfactual: bool = False,
    positive_rule: str | None = None,
) -> list[tuple[str, str, str]]:
    """Emit (anchor_text, positive_text, negative_text) triples.

    Standard mode (counterfactual=False, the original v1/v2/v3 setup):
        Positive = same category-set AND same round_won [AND same alive-bucket].
        Negative = different category-set OR different alive-bucket.
    Failure mode: encoder learns to cluster (categories, round_won) jointly.
    The kNN of any state then has near-zero outcome variance — RECALL's V̂
    saturates and advantage carries no signal. Measured directly in
    scripts/recall_variance_diagnostic.py: median neighbor std = 0.000 for
    learned_v2_redact, vs 0.39 for off-the-shelf MiniLM.

    Counterfactual mode (counterfactual=True):
        Positive = same category-set [AND same alive-bucket] AND
                   *different* round_won.
        Negative = different category-set [OR different alive-bucket].
    Goal: keep tactical clustering, but force each cluster to span both wins
    AND losses. Then V̂ has natural variance and Q̂ filtered by action picks
    out which actions actually shifted the outcome away from baseline.

    The alive-bucket flag fixes the 4v5 <-> 1v1 failure mode: without it,
    'hold'+won covers both full-team post-plant and 1v1 clutches, and the
    encoder learns those are the same tactical situation.
    """
    # Bucket key always splits by round_won so we can mix-and-match modes:
    # standard mode samples positives from same (cat, rw, akey) bucket;
    # counterfactual mode samples positives from same (cat, akey) but flips rw.
    buckets: dict[tuple, list[int]] = defaultdict(list)
    states: list[str] = []
    cats_by_idx: list[frozenset] = []
    alive_by_idx: list[tuple[str, str]] = []
    rw_by_idx: list[bool] = []
    for i, s in enumerate(dataset):
        gt = s.get("ground_truth", {})
        gs = gt.get("game_state", {})
        pa = gt.get("pro_action", {})
        rw = bool(gt.get("round_won", False))
        ckey = _category_key(pa)
        akey = _alive_key(gs) if alive_bucket else None
        buckets[(ckey, rw, akey)].append(i)
        states.append(_state_text(gs, redact=redact))
        cats_by_idx.append(ckey)
        alive_by_idx.append(_alive_key(gs))
        rw_by_idx.append(rw)

    all_indices = list(range(len(dataset)))
    triplets: list[tuple[str, str, str]] = []

    def different_enough(i: int, k: int) -> bool:
        if cats_by_idx[k] != cats_by_idx[i]:
            return True
        # Category-sets match, but if alive-bucket differs that's still a
        # valid hard-negative signal when the flag is on.
        return alive_bucket and alive_by_idx[k] != alive_by_idx[i]

    # Resolve effective rule: explicit positive_rule wins, else counterfactual
    # flag, else default standard.
    if positive_rule is None:
        rule = "same_cat_diff_outcome" if counterfactual else "same_cat_same_outcome"
    else:
        rule = positive_rule

    skipped_no_pos = 0
    for i in range(len(dataset)):
        akey = alive_by_idx[i] if alive_bucket else None
        if rule == "same_cat_diff_outcome":
            key = (cats_by_idx[i], not rw_by_idx[i], akey)
            pos_pool = [j for j in buckets[key] if j != i]
        elif rule == "same_cat_same_outcome":
            key = (cats_by_idx[i], rw_by_idx[i], akey)
            pos_pool = [j for j in buckets[key] if j != i]
        elif rule == "category_only":
            # Same category-set + same alive-bucket; outcome ignored.
            pos_pool = [
                j for j in range(len(dataset))
                if j != i and cats_by_idx[j] == cats_by_idx[i]
                and (not alive_bucket or alive_by_idx[j] == akey)
            ]
        elif rule == "outcome_only":
            # Same outcome only; category ignored. Should produce maximally
            # outcome-aligned clusters (extreme F2 baseline).
            pos_pool = [
                j for j in range(len(dataset))
                if j != i and rw_by_idx[j] == rw_by_idx[i]
            ]
        else:
            raise ValueError(f"unknown positive_rule: {rule}")
        if not pos_pool:
            skipped_no_pos += 1
            continue
        for _ in range(triplets_per_anchor):
            j = rng.choice(pos_pool)
            for _attempt in range(10):
                k = rng.choice(all_indices)
                if k != i and different_enough(i, k):
                    triplets.append((states[i], states[j], states[k]))
                    break
    if skipped_no_pos:
        print(f"  skipped {skipped_no_pos} anchors (no positive available "
              f"for (cat{', akey' if alive_bucket else ''}, "
              f"{'flipped' if counterfactual else 'same'} round_won))")
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
    # Split same-cat counts by region so we can see which region is doing the
    # tactical lift.
    cat_by_region = dict(same_round=0, cross_round=0, cross_demo=0)
    region_counts = dict(same_round=0, cross_round=0, cross_demo=0)
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
                    region = "same_round"
                else:
                    region = "cross_round"
            else:
                region = "cross_demo"
            total[region] += 1
            region_counts[region] += 1
            if cats_by_idx[nb] == q_cat:
                total["same_cat"] += 1
                cat_by_region[region] += 1
            total["total"] += 1
    tot = max(total["total"], 1)
    pct = {key: total[key] / tot for key in ("same_round", "cross_round", "cross_demo", "same_cat", "no_src")}
    # Per-region same-cat rate: of the neighbors in this region, what fraction
    # share the pro-action category?
    cat_rate_by_region = {r: (cat_by_region[r] / region_counts[r]) if region_counts[r] else 0.0
                          for r in region_counts}
    return {
        "metric": name, "k": k, "counts": total, "pct": pct,
        "cat_rate_by_region": cat_rate_by_region,
        "region_counts": region_counts,
    }


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
    p.add_argument("--redact", action="store_true",
                   help="Strip lexical shortcuts (player_name, round_num, score, "
                        "time, etc.) from game_state before encoding. Force the "
                        "embedding to learn from tactical features, not identity.")
    p.add_argument("--alive-bucket", action="store_true",
                   help="Include (alive_teammates, alive_enemies) bucketing in "
                        "the positive-pair rule. Prevents 4v5 and 1v1 from being "
                        "pulled together during training.")
    p.add_argument("--counterfactual", action="store_true",
                   help="Flip the positive-pair rule: positives are same "
                        "category-set [AND same alive-bucket] but with "
                        "DIFFERENT round_won. Forces each cluster to span "
                        "wins AND losses so RECALL's V̂ has natural variance. "
                        "See scripts/recall_variance_diagnostic.py for why "
                        "the standard rule collapses neighbor outcome std to 0.")
    p.add_argument("--positive-rule", default=None,
                   choices=[None, "same_cat_same_outcome", "same_cat_diff_outcome",
                            "category_only", "outcome_only"],
                   help="Override positive-pair rule directly (overrides "
                        "--counterfactual). 'category_only' ignores outcome "
                        "(positives have same cat, any outcome). 'outcome_only' "
                        "ignores category (positives have same outcome, any cat). "
                        "Used to ablate the F2 mechanism: F2 predicts that "
                        "outcome-correlated positives collapse retrieval variance "
                        "proportional to the correlation strength.")
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

    if args.redact:
        print("Redaction ON: stripping lexical shortcuts from game_state")
    if args.alive_bucket:
        print("Alive-bucket ON: positive pairs must match (team, enemy) alive buckets")
    if args.counterfactual:
        print("Counterfactual ON: positives have FLIPPED round_won "
              "(same tactic, different outcome)")

    if not args.eval_only:
        print("\nBuilding triplets...")
        triplets = build_triplets(
            dataset, args.triplets_per_anchor, rng,
            redact=args.redact, alive_bucket=args.alive_bucket,
            counterfactual=args.counterfactual,
        )
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
    texts = [_state_text(s.get("ground_truth", {}).get("game_state", {}), redact=args.redact) for s in dataset]

    print("\nComputing tactical_19d embeddings...")
    tact = embed_tactical(dataset)

    print("Computing baseline sentence-transformer embeddings...")
    base = embed_dataset(args.base_model, texts)

    print("Computing learned encoder embeddings...")
    learned = embed_dataset(str(output_dir), texts)

    print(f"\nNeighbor stats (k={args.k}, excluding self):")
    print(f"{'metric':<18} {'same_round':>10} {'cross_round':>11} {'cross_demo':>10} {'same_cat':>10}")
    print("-" * 65)
    stats_all = []
    for name, embs in [
        ("tactical_19d", tact),
        ("baseline_sent", base),
        ("learned_sent", learned),
    ]:
        r = neighbor_stats(name, embs, dataset, src_by_idx, cats_by_idx, args.k)
        stats_all.append(r)
        p = r["pct"]
        print(
            f"{name:<18} "
            f"{p['same_round']*100:9.1f}% "
            f"{p['cross_round']*100:10.1f}% "
            f"{p['cross_demo']*100:9.1f}% "
            f"{p['same_cat']*100:9.1f}%"
        )

    # Per-region same-cat rate: does the embedding cluster by TACTIC even when
    # it reaches across rounds/demos? Random baseline for same-cat is ~15-18%
    # (category-set distribution is top-heavy: 'hold' alone is 28% of data).
    print(f"\nSame-category rate BY REGION (cross-demo is what we care about):")
    print(f"{'metric':<18} {'same_round':>10} {'cross_round':>11} {'cross_demo':>10}")
    print("-" * 55)
    for r in stats_all:
        c = r["cat_rate_by_region"]
        print(
            f"{r['metric']:<18} "
            f"{c['same_round']*100:9.1f}% "
            f"{c['cross_round']*100:10.1f}% "
            f"{c['cross_demo']*100:9.1f}%"
        )

    summary_path = output_dir / "neighbor_eval.json"
    summary_path.write_text(json.dumps(stats_all, indent=2))
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
