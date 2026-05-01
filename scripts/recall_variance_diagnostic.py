#!/usr/bin/env python3
"""Pure-retrieval variance diagnostic for RECALL embeddings.

The hypothesis: RECALL's advantage signal Q̂(s,a) − V̂(s) collapses on this
dataset because the kNN neighborhood of any query state has near-zero outcome
variance. If neighbors all share the same `round_won`, V̂ saturates at 0 or 1
and the advantage carries no signal.

This script measures that directly without running the model. For each
indexed state, it pulls top-k state neighbors under each candidate embedding
and reports the distribution of neighbor outcome std/mean across queries.

Compares:
  - tactical_19d:    hand-engineered RECALL embedding (default)
  - baseline_sent:   off-the-shelf MiniLM on game_state JSON
  - learned_sent:    fine-tuned encoder at --learned-encoder

For each embedding we also compute a same-round-excluded variant: when the
neighbor list is filtered to only cross-round/cross-demo states (using
smoke_test_source.jsonl to identify same-round duplicates), how does the
variance distribution shift? This isolates "is the collapse just because we're
retrieving sibling ticks of the same round?" from "is it deeper than that?"

Outputs (under --output):
  variance_diag.json    summary stats per embedding × condition
  variance_hist.png     histogram of per-query neighbor outcome std
  saturation_table.txt  human-readable summary

Reference: with binary outcomes and base rate p, the max achievable per-query
std is √(p(1−p)). For this dataset p≈0.60 so max std ≈ 0.49. Anything below
0.1 means the neighborhood is effectively outcome-homogeneous.

Usage:
  python scripts/recall_variance_diagnostic.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --source  data/training/grpo/smoke_test_source.jsonl \\
      --learned-encoder outputs/embedding/learned_v2_redact \\
      --output  outputs/embedding/variance_diag \\
      --k 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_REDACT_KEYS = frozenset({
    "player_name", "player", "teammates", "team", "opponent_team",
    "round_number", "round",
    "time_remaining", "time", "round_time", "tick",
    "score", "current_player",
})


def _redact(obj: Any, redact: bool) -> Any:
    if not redact:
        return obj
    if isinstance(obj, dict):
        return {k: _redact(v, redact) for k, v in obj.items() if k not in _REDACT_KEYS}
    if isinstance(obj, list):
        return [_redact(v, redact) for v in obj]
    return obj


def _state_text(gs: dict, redact: bool) -> str:
    return json.dumps(_redact(gs, redact), sort_keys=True, ensure_ascii=False)


# Inlined from src/training/recall.py — that module uses PEP 604 type unions
# which don't parse on Python 3.9, so we can't import it directly. Kept in
# byte-for-byte feature parity so this diagnostic measures what production
# RECALL actually retrieves.
_MAP_IDS = {
    "de_dust2": 0, "de_mirage": 1, "de_inferno": 2, "de_nuke": 3,
    "de_overpass": 4, "de_ancient": 5, "de_anubis": 6, "de_vertigo": 7,
}


def tactical_embedding(game_state: dict) -> np.ndarray:
    vec = np.zeros(19, dtype=np.float32)
    side = str(game_state.get("player_side", "")).upper()
    vec[0] = 1.0 if side == "CT" else 0.0
    phase = str(game_state.get("round_phase", "")).lower()
    vec[1] = 1.0 if phase == "post-plant" else 0.0
    map_name = str(game_state.get("map_name", "")).lower()
    map_id = _MAP_IDS.get(map_name)
    if map_id is not None:
        vec[2 + map_id] = 1.0
    alive_t = int(game_state.get("alive_teammates", 0))
    alive_e = int(game_state.get("alive_enemies", 0))
    vec[10] = (alive_t - alive_e) / 5.0
    weapon = game_state.get("weapon_primary")
    vec[11] = 1.0 if weapon else 0.0
    vec[12] = float(game_state.get("player_health", 0)) / 100.0
    vec[13] = float(game_state.get("player_armor", 0)) / 100.0
    utility = game_state.get("utility", [])
    vec[14] = min(len(utility), 4) / 4.0 if isinstance(utility, list) else 0.0
    bomb = str(game_state.get("bomb_status", "")).lower()
    vec[15] = 0.5 if bomb == "planted" else (1.0 if bomb == "dropped" else 0.0)
    vec[16] = min(alive_t, 4) / 4.0
    vec[17] = min(alive_e, 5) / 5.0
    vec[18] = min(int(game_state.get("visible_enemies", 0)), 5) / 5.0
    return vec


def embed_tactical(dataset: list[dict]) -> np.ndarray:
    n = len(dataset)
    embs = np.zeros((n, 19), dtype=np.float32)
    for i, s in enumerate(dataset):
        gs = s.get("ground_truth", {}).get("game_state", {})
        embs[i] = np.asarray(tactical_embedding(gs), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def embed_sentence(model_path: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_path)
    return m.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(np.float32)


def topk_with_mask(
    sims: np.ndarray, k: int, exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Top-k indices per row of `sims`. If `exclude_mask[i, j]` is True,
    index j is treated as -inf when finding the top-k of row i.

    Returns int array (n, k). Self is always excluded (diagonal -inf).
    """
    n = sims.shape[0]
    work = sims.copy()
    np.fill_diagonal(work, -np.inf)
    if exclude_mask is not None:
        work[exclude_mask] = -np.inf
    # argpartition on the negated array gives the indices of the k largest.
    # Some queries may have <k valid neighbors after masking; in that case
    # the trailing slots are still real indices but their sims are -inf, so
    # we mark them with -1.
    topk = np.argpartition(-work, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    # Mask out slots where the chosen neighbor's sim is -inf (no valid neighbor).
    rows = np.arange(n)[:, None]
    chosen_sims = work[rows, topk]
    topk = np.where(chosen_sims > -np.inf, topk, -1)
    return topk


def per_query_outcome_stats(topk: np.ndarray, outcomes: np.ndarray) -> dict[str, np.ndarray]:
    """Given (n, k) neighbor indices (-1 = invalid), return per-query mean,
    std, count, and saturation flag."""
    n, k = topk.shape
    means = np.zeros(n, dtype=np.float32)
    stds = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        valid = topk[i][topk[i] >= 0]
        if len(valid) == 0:
            means[i] = np.nan
            stds[i] = np.nan
            counts[i] = 0
            continue
        ovec = outcomes[valid]
        means[i] = ovec.mean()
        stds[i] = ovec.std()
        counts[i] = len(valid)
    return {"mean": means, "std": stds, "count": counts}


def summarize(stats: dict[str, np.ndarray], base_rate: float) -> dict[str, float]:
    """Aggregate per-query stats into single numbers."""
    stds = stats["std"]
    means = stats["mean"]
    counts = stats["count"]
    valid = ~np.isnan(stds) & (counts > 0)
    if not valid.any():
        return {"valid_queries": 0}
    s = stds[valid]
    m = means[valid]
    max_std = (base_rate * (1 - base_rate)) ** 0.5
    return {
        "valid_queries": int(valid.sum()),
        "mean_neighbor_count": float(counts[valid].mean()),
        # Per-query neighbor outcome std distribution
        "std_p25": float(np.percentile(s, 25)),
        "std_p50": float(np.percentile(s, 50)),
        "std_p75": float(np.percentile(s, 75)),
        "std_mean": float(s.mean()),
        # Saturation: % of queries where neighbors are nearly homogeneous
        "frac_std_below_0_10": float((s < 0.10).mean()),
        "frac_std_below_0_20": float((s < 0.20).mean()),
        "frac_std_below_0_30": float((s < 0.30).mean()),
        # How extreme is the mean? Far from base rate = strong (possibly
        # overconfident) prediction; near base rate = uninformative.
        "mean_distance_from_base": float(np.abs(m - base_rate).mean()),
        "frac_mean_within_0_05_of_base": float((np.abs(m - base_rate) < 0.05).mean()),
        "frac_mean_extreme": float(((m < 0.10) | (m > 0.90)).mean()),
        "max_possible_std": float(max_std),
        "base_rate": float(base_rate),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True, help="smoke_test_source.jsonl for same-round exclusion")
    p.add_argument("--learned-encoder", default=None,
                   help="Path to fine-tuned sentence-transformer (optional)")
    p.add_argument("--baseline-encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--output", required=True, help="Output dir")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--redact", action="store_true",
                   help="Apply same redaction as learned_v2_redact training (recommended when learned encoder is redacted)")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}...")
    dataset = [json.loads(l) for l in open(args.dataset)]
    n = len(dataset)
    print(f"  {n} samples")

    outcomes = np.array(
        [1.0 if s.get("ground_truth", {}).get("round_won") else 0.0
         for s in dataset],
        dtype=np.float32,
    )
    base_rate = float(outcomes.mean())
    print(f"  round_won base rate: {base_rate:.3f}, max std = {(base_rate*(1-base_rate))**0.5:.3f}")

    # Load source metadata for same-round exclusion
    print(f"Loading {args.source}...")
    src_by_idx: dict[int, dict] = {}
    with open(args.source) as f:
        for line in f:
            r = json.loads(line)
            src_by_idx[r["idx"]] = r
    print(f"  {len(src_by_idx)} samples have source metadata")

    # Build a (n, n) boolean: same_round_mask[i, j] iff i and j are from the
    # same (demo, round). We'll use this to mask out same-round neighbors in
    # the cross-round-only condition.
    print("Building same-round mask...")
    round_keys = []
    for i in range(n):
        src = src_by_idx.get(i)
        if src is None:
            round_keys.append((None, None))
        else:
            round_keys.append((src.get("demo_stem"), src.get("round_num")))
    # Build mask with broadcast; small enough at n=2013 (~16MB bool).
    same_round_mask = np.zeros((n, n), dtype=bool)
    by_key: dict[tuple, list[int]] = {}
    for i, k in enumerate(round_keys):
        if k[0] is not None:
            by_key.setdefault(k, []).append(i)
    for indices in by_key.values():
        if len(indices) > 1:
            for i in indices:
                same_round_mask[i, indices] = True
    n_same_round_pairs = int((same_round_mask.sum() - same_round_mask.shape[0]) // 2)
    print(f"  {n_same_round_pairs} same-round pairs")

    # Compute embeddings
    print("\nComputing tactical_19d embeddings...")
    e_tact = embed_tactical(dataset)

    texts = [_state_text(s.get("ground_truth", {}).get("game_state", {}), redact=args.redact)
             for s in dataset]

    print(f"\nComputing baseline_sent embeddings ({args.baseline_encoder})...")
    e_base = embed_sentence(args.baseline_encoder, texts)

    e_learned = None
    if args.learned_encoder:
        print(f"\nComputing learned_sent embeddings ({args.learned_encoder})...")
        e_learned = embed_sentence(args.learned_encoder, texts)

    # Compute (n, n) similarity, then top-k under two conditions, then per-query
    # outcome stats.
    embeddings = [("tactical_19d", e_tact), ("baseline_sent", e_base)]
    if e_learned is not None:
        embeddings.append(("learned_sent", e_learned))

    summary = {"k": args.k, "n": n, "base_rate": base_rate, "results": []}
    per_query_dump: dict[str, dict[str, list]] = {}

    for name, embs in embeddings:
        print(f"\n--- {name} ---")
        sims = embs @ embs.T  # cosine sim since embeddings are unit-normalized

        for cond_name, exclude in [
            ("all_neighbors", None),
            ("cross_round_only", same_round_mask),
        ]:
            topk = topk_with_mask(sims, args.k, exclude)
            stats = per_query_outcome_stats(topk, outcomes)
            summ = summarize(stats, base_rate)
            entry = {"embedding": name, "condition": cond_name, "summary": summ}
            summary["results"].append(entry)
            print(f"  [{cond_name}]")
            print(f"    valid queries: {summ['valid_queries']}/{n}, "
                  f"mean k = {summ['mean_neighbor_count']:.1f}")
            print(f"    neighbor std percentiles: "
                  f"p25={summ['std_p25']:.3f} p50={summ['std_p50']:.3f} p75={summ['std_p75']:.3f}  "
                  f"(max possible = {summ['max_possible_std']:.3f})")
            print(f"    saturation: <0.10 std = {summ['frac_std_below_0_10']*100:.1f}%, "
                  f"<0.20 = {summ['frac_std_below_0_20']*100:.1f}%, "
                  f"<0.30 = {summ['frac_std_below_0_30']*100:.1f}%")
            print(f"    mean extreme (<0.1 or >0.9): {summ['frac_mean_extreme']*100:.1f}%")
            per_query_dump.setdefault(name, {})[cond_name] = {
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
                "count": stats["count"].tolist(),
            }

    # Save summary JSON
    summary_path = output_dir / "variance_diag.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")

    # Save per-query stats for plot
    per_query_path = output_dir / "per_query_stats.json"
    per_query_path.write_text(json.dumps(per_query_dump))
    print(f"Wrote per-query stats to {per_query_path}")

    # Plot
    print("\nPlotting...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot")
    else:
        n_emb = len(embeddings)
        fig, axes = plt.subplots(n_emb, 2, figsize=(11, 3.2 * n_emb), sharex=True, sharey=True)
        if n_emb == 1:
            axes = axes.reshape(1, -1)
        max_std = (base_rate * (1 - base_rate)) ** 0.5
        for row, (name, _) in enumerate(embeddings):
            for col, cond in enumerate(["all_neighbors", "cross_round_only"]):
                stats = per_query_dump[name][cond]
                stds = np.array(stats["std"])
                stds = stds[~np.isnan(stds)]
                ax = axes[row, col]
                ax.hist(stds, bins=40, range=(0, max_std + 0.05), color="steelblue",
                        edgecolor="black", linewidth=0.4)
                ax.axvline(max_std, color="red", linestyle="--", linewidth=1,
                           label=f"max possible ({max_std:.2f})")
                ax.axvline(0.10, color="orange", linestyle=":", linewidth=1,
                           label="saturation threshold (0.10)")
                ax.set_title(f"{name} | {cond}", fontsize=10)
                if row == n_emb - 1:
                    ax.set_xlabel("per-query neighbor outcome std")
                if col == 0:
                    ax.set_ylabel("# queries")
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(
            f"RECALL neighbor outcome std — k={args.k}, n={n}, base rate p={base_rate:.2f}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        plot_path = output_dir / "variance_hist.png"
        fig.savefig(plot_path, dpi=130, bbox_inches="tight")
        print(f"  Wrote {plot_path}")

    # Saturation table
    table_lines = [
        f"RECALL neighbor outcome variance diagnostic",
        f"  dataset: {args.dataset}",
        f"  n = {n}, base rate p(round_won) = {base_rate:.3f}",
        f"  max achievable per-query std = √(p(1-p)) = {(base_rate*(1-base_rate))**0.5:.3f}",
        f"  saturated query = neighbor outcome std < 0.10",
        f"  k = {args.k}",
        "",
        f"{'embedding':<16} {'condition':<20} {'std p50':>9} {'sat<0.10':>10} {'mean extreme':>14}",
        "-" * 72,
    ]
    for r in summary["results"]:
        s = r["summary"]
        table_lines.append(
            f"{r['embedding']:<16} {r['condition']:<20} "
            f"{s['std_p50']:>9.3f} "
            f"{s['frac_std_below_0_10']*100:>9.1f}% "
            f"{s['frac_mean_extreme']*100:>13.1f}%"
        )
    table_path = output_dir / "saturation_table.txt"
    table_path.write_text("\n".join(table_lines))
    print(f"Wrote {table_path}")
    print("\n" + "\n".join(table_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
