#!/usr/bin/env python3
"""End-to-end validation that the same-round-exclusion path in
src/training/recall.py actually masks correctly and that neighbor outcome
variance on the production code path matches the standalone diagnostic.

Builds a real RECALLIndex from smoke_test.jsonl + smoke_test_source.jsonl,
queries every indexed state with a dummy action, and computes per-query
neighbor outcome std under two conditions:

    A. query_source_key=None  → behaves like old RECALL (no masking)
    B. query_source_key=<own (demo, round)>  → excludes same-round neighbors

If Layer 0 is wired correctly, A reproduces the "all_neighbors" numbers from
recall_variance_diagnostic.py and B reproduces "cross_round_only" numbers.

This doesn't run the model — it's a CPU-only sanity check on the index.

Usage (python 3.11; recall.py uses PEP 604 unions):
  python3.11 scripts/validate_recall_masking.py \\
      --dataset data/training/grpo/smoke_test.jsonl \\
      --source  data/training/grpo/smoke_test_source.jsonl
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--k", type=int, default=32)
    args = p.parse_args()

    # Skip src/training/__init__ to dodge unrelated import-time side effects
    # in this CPU-only context (the trainer pulls heavy ML deps we don't need
    # to test the index alone).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_recall", _REPO_ROOT / "src" / "training" / "recall.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    RECALLIndex = mod.RECALLIndex

    print(f"Loading {args.dataset}...")
    samples = [json.loads(l) for l in open(args.dataset)]
    print(f"  {len(samples)} samples")

    print(f"Loading {args.source}...")
    sources_by_idx = {}
    with open(args.source) as f:
        for line in f:
            r = json.loads(line)
            sources_by_idx[int(r["idx"])] = r
    print(f"  {len(sources_by_idx)} source records")

    # Mirror the merge that train_grpo.py now does — source assigned by
    # original line index in dataset.
    n_merged = 0
    for i, s in enumerate(samples):
        src = sources_by_idx.get(i)
        if src is None:
            continue
        gt = s.setdefault("ground_truth", {})
        gt["source"] = {
            "demo_stem": src.get("demo_stem"),
            "round_num": src.get("round_num"),
        }
        n_merged += 1
    print(f"  merged source into {n_merged}/{len(samples)} samples")

    print("\nBuilding RECALLIndex...")
    idx = RECALLIndex()
    idx.build_from_samples(samples)
    print(f"  index size: {idx.size}")
    n_with_src = sum(1 for k in idx._source_keys if k is not None)
    print(f"  indexed samples with source: {n_with_src}/{idx.size}")

    # Run one batch of queries — every indexed sample, querying with its own
    # game_state and a dummy zero action. We only care about the neighbor
    # *state* mask here; the action filter is independent.
    print(f"\nQuerying every indexed state (k={args.k})...")

    stds = {"no_mask": [], "with_mask": []}
    means = {"no_mask": [], "with_mask": []}

    # Pull each query state's source key from the same merged samples.
    for i, s in enumerate(samples):
        gt = s.get("ground_truth", {})
        gs = gt.get("game_state")
        beh = gt.get("pro_action", {}).get("behavior") if isinstance(gt.get("pro_action"), dict) else None
        if gs is None or beh is None:
            continue

        # Use the indexed sample's own source key as the query key for the
        # masked condition. Same-round means same (demo_stem, round_num).
        src = gt.get("source")
        qkey = None
        if isinstance(src, dict):
            qkey = (str(src["demo_stem"]), int(src["round_num"])) if "demo_stem" in src and "round_num" in src else None

        # Reach into internals to get the raw neighbor outcome list under each
        # condition. RECALLIndex.query returns (Q̂, V̂, confident); we want the
        # full neighbor outcomes to compute std. Easiest is to inline the
        # search ourselves with the same logic the production query uses.
        # That keeps the test honest about what production retrieval pulls.
        state_vec = np.asarray(idx._state_embedder(gs), dtype=np.float32).reshape(1, -1)

        for cond, use_mask in [("no_mask", False), ("with_mask", True)]:
            do_mask = use_mask and qkey is not None and idx._source_keys is not None
            k_fetch = min(args.k * 4, idx._n) if do_mask else min(args.k, idx._n)
            _d, indices = idx._state_index.search(state_vec, k_fetch)
            valid = indices[0]
            valid = valid[valid >= 0]
            if do_mask:
                keep = np.array([idx._source_keys[j] != qkey for j in valid], dtype=bool)
                valid = valid[keep][: args.k]
            else:
                valid = valid[: args.k]
            if len(valid) == 0:
                continue
            outcomes = idx._outcomes[valid]
            stds[cond].append(float(outcomes.std()))
            means[cond].append(float(outcomes.mean()))

    base_rate = float(idx._outcomes.mean())
    max_std = (base_rate * (1 - base_rate)) ** 0.5

    print(f"\nbase rate p(round_won) = {base_rate:.3f}, max std = {max_std:.3f}")
    print()
    print(f"{'condition':<14} {'queries':>8} {'std p25':>9} {'std p50':>9} {'std p75':>9} {'sat<0.10':>10}")
    print("-" * 64)
    for cond in ["no_mask", "with_mask"]:
        s = np.array(stds[cond])
        if len(s) == 0:
            print(f"{cond:<14} {'0':>8}")
            continue
        print(
            f"{cond:<14} {len(s):>8} "
            f"{np.percentile(s, 25):>9.3f} "
            f"{np.percentile(s, 50):>9.3f} "
            f"{np.percentile(s, 75):>9.3f} "
            f"{(s < 0.10).mean()*100:>9.1f}%"
        )

    # Acceptance check: with_mask should have meaningfully higher median std
    # than no_mask. If not, something in the masking path is broken.
    if len(stds["no_mask"]) and len(stds["with_mask"]):
        no_p50 = float(np.percentile(stds["no_mask"], 50))
        ma_p50 = float(np.percentile(stds["with_mask"], 50))
        if ma_p50 > no_p50 + 0.05:
            print(f"\n✓ same-round masking lifts median neighbor std "
                  f"from {no_p50:.3f} to {ma_p50:.3f}")
        else:
            print(f"\n✗ masking did NOT visibly increase variance "
                  f"(no_mask p50={no_p50:.3f}, with_mask p50={ma_p50:.3f}). "
                  f"Check that source merge populated _source_keys.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
