#!/usr/bin/env python3
"""End-to-end smoke test of v6 encoder → RECALL pipeline.

Builds a RECALLIndex backed by Level-2 encoder embeddings from the
smoke_test.jsonl rows, then queries a few samples to verify:

  1. Cache lookup hits for the smoke test's source rows
  2. Index builds with d=512 (encoder embedding dim), not 19
  3. kNN retrieval returns sensible neighbors
  4. Same-round mask still drops sibling ticks

This is the final L2→L3 integration check before wiring v6 into the
GRPO trainer end-to-end. Cheap (~10s) — runs entirely on CPU after the
cache is loaded.

Usage:
    python scripts/test_encoder_recall.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.perception.encoder_cache import EncoderEmbeddingCache
from src.training.recall import RECALLIndex


CACHE_PATH = REPO / "outputs" / "round_encoder" / "v6_81demos" / "embedding_cache.pt"
SOURCE_JSONL = REPO / "data" / "training" / "grpo" / "smoke_test_source.jsonl"
DATA_JSONL = REPO / "data" / "training" / "grpo" / "smoke_test.jsonl"


def main():
    print(f"Loading encoder embedding cache: {CACHE_PATH}")
    cache = EncoderEmbeddingCache(CACHE_PATH)
    print(f"  rounds: {cache.n_rounds}, total ticks: {cache.n_ticks:,}, "
          f"d_model={cache.d_model}")
    print()

    print(f"Loading smoke test ({DATA_JSONL.name} + {SOURCE_JSONL.name})...")
    sources = [json.loads(l) for l in SOURCE_JSONL.open()]
    data = [json.loads(l) for l in DATA_JSONL.open()]
    assert len(sources) == len(data), \
        f"row count mismatch: {len(sources)} vs {len(data)}"
    # Merge: attach source provenance to each data row
    samples = []
    for src, d in zip(sources, data):
        merged = dict(d)
        merged["source"] = {
            "demo_stem": src["demo_stem"],
            "round_num": src["round_num"],
            "tick": src["tick"],
        }
        samples.append(merged)
    print(f"  {len(samples)} samples merged.")
    print()

    # Cache-hit audit before doing anything else
    hits = 0
    misses_by_demo: dict[str, int] = {}
    for s in samples:
        src = s["source"]
        emb = cache.lookup(src["demo_stem"], src["round_num"], src["tick"])
        if emb is not None:
            hits += 1
        else:
            misses_by_demo[src["demo_stem"]] = misses_by_demo.get(src["demo_stem"], 0) + 1
    print(f"Cache hits: {hits}/{len(samples)}  ({100*hits/len(samples):.1f}%)")
    if misses_by_demo:
        print(f"Misses (by demo):")
        for k, v in misses_by_demo.items():
            print(f"  {k}: {v}")
    print()

    # Build the embedder closure
    def state_embedder_full(sample):
        src = sample.get("source") or sample.get("ground_truth", {}).get("source")
        if not src:
            return None
        return cache.lookup(src["demo_stem"], int(src["round_num"]), int(src["tick"]))

    print("Building RECALLIndex with Level-2 encoder embeddings...")
    idx = RECALLIndex(state_embedder_full=state_embedder_full)
    idx.build_from_samples(samples)
    print(f"  index size: {idx.size}, expected ~{hits} (cache hits)")
    print()

    # Spot-check a query
    print("Querying sample 0 against the index...")
    s0 = samples[0]
    gt = s0["ground_truth"]
    q_src = (s0["source"]["demo_stem"], int(s0["source"]["round_num"]))
    q, v, conf = idx.query(
        gt["game_state"],
        gt.get("pro_action", {}).get("behavior", {}),
        k=32, k_min=5,
        query_source_key=q_src,
        query_sample=s0,
    )
    print(f"  query 0 (demo={q_src[0]}, round={q_src[1]}, "
          f"won={gt.get('round_won')}): Q={q:.3f} V={v:.3f} confident={conf}")
    print()

    # Sanity: confirm V_hat varies across queries (no collapse)
    print("Sampling 50 queries to check V̂ spread...")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(50, len(samples)), replace=False)
    vs = []
    qs = []
    confs = 0
    for i in pick:
        s = samples[i]
        gt = s["ground_truth"]
        q_src = (s["source"]["demo_stem"], int(s["source"]["round_num"]))
        q, v, conf = idx.query(
            gt["game_state"],
            gt.get("pro_action", {}).get("behavior", {}),
            k=32, k_min=5,
            query_source_key=q_src,
            query_sample=s,
        )
        qs.append(q); vs.append(v); confs += int(conf)
    vs = np.array(vs); qs = np.array(qs)
    print(f"  V̂: mean={vs.mean():.3f}, std={vs.std():.3f}, "
          f"min={vs.min():.3f}, max={vs.max():.3f}")
    print(f"  Q̂: mean={qs.mean():.3f}, std={qs.std():.3f}")
    print(f"  Confident: {confs}/{len(pick)}")
    print()

    # Compare to baseline 19-d tactical_embedding RECALL on same samples
    print("Same samples with 19-d tactical_embedding (baseline) for spread check...")
    idx_baseline = RECALLIndex()
    idx_baseline.build_from_samples(samples)
    vs_b = []
    qs_b = []
    for i in pick:
        s = samples[i]
        gt = s["ground_truth"]
        q_src = (s["source"]["demo_stem"], int(s["source"]["round_num"]))
        q, v, _ = idx_baseline.query(
            gt["game_state"],
            gt.get("pro_action", {}).get("behavior", {}),
            k=32, k_min=5,
            query_source_key=q_src,
        )
        qs_b.append(q); vs_b.append(v)
    vs_b = np.array(vs_b); qs_b = np.array(qs_b)
    print(f"  baseline V̂: mean={vs_b.mean():.3f}, std={vs_b.std():.3f}")
    print(f"  baseline Q̂: mean={qs_b.mean():.3f}, std={qs_b.std():.3f}")
    print()

    advantage_v6 = qs - vs
    advantage_b = qs_b - vs_b
    print(f"Advantage signal A = Q̂ - V̂:")
    print(f"  v6 encoder:     mean={advantage_v6.mean():+.3f}, "
          f"std={advantage_v6.std():.3f}, "
          f"non-zero={int((np.abs(advantage_v6) > 0.01).sum())}/{len(advantage_v6)}")
    print(f"  19-d baseline:  mean={advantage_b.mean():+.3f}, "
          f"std={advantage_b.std():.3f}, "
          f"non-zero={int((np.abs(advantage_b) > 0.01).sum())}/{len(advantage_b)}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
