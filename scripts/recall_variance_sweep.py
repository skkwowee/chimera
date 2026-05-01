#!/usr/bin/env python3
"""Reviewer-fix sweeps for the variance diagnostic.

Addresses three reviewer attacks against the headline saturation numbers:

  (a) k cherry-pick: report std percentiles + saturation at k ∈ {8, 16, 32, 64, 128}
  (b) threshold arbitrariness: report saturation at std < {0.05, 0.10, 0.20, 0.30}
  (c) whole-demo holdout: clean F1 control. Mask out neighbors from the same
      DEMO as the query (not just the same round). Adjacent rounds within the
      same demo share map/economy/lineup, so same-round masking alone leaves
      these confounds in. Whole-demo holdout removes them.

Operates on the existing per-query-stats dump? No — those were computed at
fixed k. We need a fresh sweep. Reuses the embedding computation path from
recall_variance_diagnostic.py via direct import.

Usage:
  python3.11 scripts/recall_variance_sweep.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --source data/training/grpo/smoke_test_source.jsonl \\
      --learned-encoder outputs/embedding/counterfactual_v1 \\
      --output outputs/embedding/variance_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the embedding helpers from the original diagnostic — same exact paths
# guarantee the sweep measures the same quantity at different settings.
from scripts.recall_variance_diagnostic import (  # noqa: E402
    embed_tactical, embed_sentence, _state_text,
)


def per_query_std(
    sims: np.ndarray, outcomes: np.ndarray, k: int,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-row top-k neighbor outcome std. exclude_mask[i,j] True → drop j
    from i's neighbor list. Self always excluded."""
    n = sims.shape[0]
    work = sims.copy()
    np.fill_diagonal(work, -np.inf)
    if exclude_mask is not None:
        work[exclude_mask] = -np.inf
    topk = np.argpartition(-work, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    chosen = work[rows, topk]
    valid_mask = chosen > -np.inf
    stds = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        keep = valid_mask[i]
        if keep.sum() < 2:
            continue
        stds[i] = float(outcomes[topk[i][keep]].std())
    return stds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--learned-encoder", default=None)
    p.add_argument("--baseline-encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--output", required=True)
    p.add_argument("--redact", action="store_true")
    p.add_argument("--k-list", nargs="+", type=int,
                   default=[8, 16, 32, 64, 128])
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.05, 0.10, 0.20, 0.30])
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}...")
    dataset = [json.loads(l) for l in open(args.dataset)]
    n = len(dataset)
    outcomes = np.array(
        [1.0 if s.get("ground_truth", {}).get("round_won") else 0.0 for s in dataset],
        dtype=np.float32,
    )
    base_rate = float(outcomes.mean())
    print(f"  n={n}, base rate={base_rate:.3f}, ceiling std={(base_rate*(1-base_rate))**0.5:.3f}")

    # Same-round + same-demo masks (broadcast).
    src_by_idx = {}
    with open(args.source) as f:
        for line in f:
            r = json.loads(line)
            src_by_idx[r["idx"]] = r
    same_round_mask = np.zeros((n, n), dtype=bool)
    same_demo_mask = np.zeros((n, n), dtype=bool)
    by_round: dict[tuple, list[int]] = {}
    by_demo: dict[str, list[int]] = {}
    for i in range(n):
        s = src_by_idx.get(i)
        if s is None:
            continue
        rk = (s.get("demo_stem"), s.get("round_num"))
        dk = s.get("demo_stem")
        by_round.setdefault(rk, []).append(i)
        by_demo.setdefault(dk, []).append(i)
    for inds in by_round.values():
        if len(inds) > 1:
            for i in inds:
                same_round_mask[i, inds] = True
    for inds in by_demo.values():
        if len(inds) > 1:
            for i in inds:
                same_demo_mask[i, inds] = True
    print(f"  same-round pairs: {(same_round_mask.sum() - n) // 2}")
    print(f"  same-demo pairs:  {(same_demo_mask.sum() - n) // 2}")

    # Compute embeddings once
    print("\nComputing embeddings...")
    e_tact = embed_tactical(dataset)
    texts = [_state_text(s.get("ground_truth", {}).get("game_state", {}), redact=args.redact) for s in dataset]
    e_base = embed_sentence(args.baseline_encoder, texts)
    embeddings = [("tactical_19d", e_tact), ("baseline_sent", e_base)]
    if args.learned_encoder:
        e_learn = embed_sentence(args.learned_encoder, texts)
        embeddings.append(("counterfactual_sent", e_learn))

    conditions = [
        ("all_neighbors", None),
        ("cross_round_only", same_round_mask),
        ("cross_demo_only", same_demo_mask),
    ]

    # Sweep k × threshold × condition × embedding
    rows = []
    for name, embs in embeddings:
        sims = embs @ embs.T
        for cond_name, mask in conditions:
            for k in args.k_list:
                stds = per_query_std(sims, outcomes, k, mask)
                valid = ~np.isnan(stds)
                s = stds[valid]
                row = {
                    "embedding": name,
                    "condition": cond_name,
                    "k": k,
                    "valid_queries": int(valid.sum()),
                    "std_p25": float(np.percentile(s, 25)) if len(s) else None,
                    "std_p50": float(np.percentile(s, 50)) if len(s) else None,
                    "std_p75": float(np.percentile(s, 75)) if len(s) else None,
                }
                for thr in args.thresholds:
                    row[f"sat_lt_{thr:.2f}"] = float((s < thr).mean()) if len(s) else None
                rows.append(row)

    # Write summary JSON
    out_json = out_dir / "sweep.json"
    out_json.write_text(json.dumps({
        "n": n, "base_rate": base_rate,
        "ceiling_std": (base_rate * (1 - base_rate)) ** 0.5,
        "k_list": args.k_list,
        "thresholds": args.thresholds,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_json}")

    # Print compact tables — one per condition. Rows are k, columns are
    # (embedding × p50, sat<0.10, sat<0.20).
    for cond_name, _ in conditions:
        print(f"\n=== condition: {cond_name} ===")
        # Header
        emb_names = [name for name, _ in embeddings]
        head = f"  {'k':>4}  "
        for emb in emb_names:
            head += f"  {emb[:14]:<14}    "
        print(head)
        head2 = f"  {'':>4}  "
        for _ in emb_names:
            head2 += f"  {'p50  s<.10  s<.20':<19}    "
        print(head2)
        print("  " + "-" * (4 + len(emb_names) * 24))
        for k in args.k_list:
            line = f"  {k:>4}  "
            for emb in emb_names:
                row = next(r for r in rows
                           if r["embedding"] == emb and r["condition"] == cond_name and r["k"] == k)
                p50 = row["std_p50"]
                s10 = row["sat_lt_0.10"] * 100
                s20 = row["sat_lt_0.20"] * 100
                line += f"  {p50:.3f} {s10:>5.1f}% {s20:>5.1f}%    "
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
