#!/usr/bin/env python3
"""Qualitative inspection of cross-round and cross-demo jumps for a learned
state-equivalence embedding.

For each of N diverse query states (sampled across map × round_phase buckets),
dump:
  - the query's tactical summary (side, HP, alive, bomb, categories, won)
  - its top-K cross-round neighbors (same demo, different round)
  - its top-K cross-demo neighbors (different demo entirely)

Human reads the output and judges whether "similar embedding neighbors"
translates to "similar tactical situation." Aggregate metrics (same_cat
rates, red/green/yellow ratios) tell you the embedding has *some* signal;
this script is for deciding whether that signal is actually tactical.

Usage:
  python scripts/inspect_cross_jumps.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --source  data/training/grpo/smoke_test_source.jsonl \\
      --encoder /workspace/outputs/embedding/learned_v2_redact \\
      --n-queries 5 --k 3 --output cross_jumps.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _state_text(gs: dict) -> str:
    return json.dumps(gs, sort_keys=True, ensure_ascii=False)


def _fmt_state(i: int, dataset: list[dict], src_by_idx: dict[int, dict]) -> str:
    gs = dataset[i]["ground_truth"]["game_state"]
    pa = dataset[i]["ground_truth"]["pro_action"]
    rw = dataset[i]["ground_truth"].get("round_won")
    src = src_by_idx.get(i, {})
    return (
        f"  [{i:4d}] {src.get('demo_stem','?')[:12]:12s} r{src.get('round_num','?'):>2} "
        f"{src.get('player_name','?')[:8]:8s} | "
        f"{gs.get('map_name','?'):10s} {gs.get('round_phase','?'):10s} "
        f"{gs.get('player_side','?')} HP={gs.get('player_health',0):3d} "
        f"A={gs.get('alive_teammates',0)}v{gs.get('alive_enemies',0)} "
        f"{gs.get('bomb_status','?'):8s} | "
        f"cat={sorted(pa.get('categories',[]))} won={rw}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--encoder", required=True,
                   help="Path to SentenceTransformer model (learned encoder)")
    p.add_argument("--n-queries", type=int, default=5)
    p.add_argument("--k", type=int, default=3, help="Neighbors per region")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="-",
                   help="Write formatted report here ('-' for stdout)")
    args = p.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    dataset = [json.loads(l) for l in open(args.dataset)]
    src_by_idx = {r["idx"]: r for r in (json.loads(l) for l in open(args.source))}

    model = SentenceTransformer(args.encoder)
    texts = [_state_text(s["ground_truth"]["game_state"]) for s in dataset]
    embs = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=128, show_progress_bar=False,
    ).astype(np.float32)

    sims = embs @ embs.T
    np.fill_diagonal(sims, -np.inf)

    # Diverse queries: one per (map, round_phase) bucket, seeded RNG.
    rng = random.Random(args.seed)
    buckets: dict[tuple, list[int]] = {}
    for i in range(len(dataset)):
        gs = dataset[i]["ground_truth"]["game_state"]
        key = (gs.get("map_name"), gs.get("round_phase"))
        buckets.setdefault(key, []).append(i)
    queries: list[int] = []
    for key in sorted(buckets.keys()):
        if len(buckets[key]) > 5:
            queries.append(rng.choice(buckets[key]))
    queries = queries[:args.n_queries]

    lines: list[str] = []
    lines.append(f"Encoder: {args.encoder}")
    lines.append(f"Seed: {args.seed}  |  Queries: {len(queries)}  |  K per region: {args.k}")
    lines.append("")

    for q in queries:
        q_src = src_by_idx.get(q, {})
        q_demo = q_src.get("demo_stem")
        q_round = q_src.get("round_num")
        row = sims[q]
        order = np.argsort(-row)

        cross_round: list[tuple[int, float]] = []
        cross_demo: list[tuple[int, float]] = []
        for j in order:
            j = int(j)
            if j == q:
                continue
            j_src = src_by_idx.get(j, {})
            if j_src.get("demo_stem") == q_demo:
                if j_src.get("round_num") != q_round:
                    cross_round.append((j, float(row[j])))
            else:
                cross_demo.append((j, float(row[j])))
            if len(cross_round) >= args.k and len(cross_demo) >= args.k:
                break

        lines.append("=" * 100)
        lines.append("QUERY:")
        lines.append(_fmt_state(q, dataset, src_by_idx))
        lines.append("")
        lines.append(f"Top {args.k} cross-round neighbors (same demo, different round):")
        for j, s in cross_round[:args.k]:
            lines.append(f"  sim={s:.3f}")
            lines.append(_fmt_state(j, dataset, src_by_idx))
        lines.append("")
        lines.append(f"Top {args.k} cross-demo neighbors (different demo entirely):")
        for j, s in cross_demo[:args.k]:
            lines.append(f"  sim={s:.3f}")
            lines.append(_fmt_state(j, dataset, src_by_idx))
        lines.append("")

    report = "\n".join(lines)
    if args.output == "-":
        print(report)
    else:
        Path(args.output).write_text(report)
        print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
