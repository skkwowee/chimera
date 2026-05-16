#!/usr/bin/env python3
"""Pro-action probe on event-pooled v6 embeddings.

The L2-G2 probe (round_won) is easy to hill-climb — outcomes are nearly
inferrable from "X alive vs Y alive" bookkeeping. The harder, more
research-relevant question is whether the encoder captures **what pros
actually do** in a given situation — the signal RECALL needs to learn
"in similar states, this action historically worked."

Targets — pro behavior at the labeled tick (from F05 prep, smoke_test.jsonl):
  - movement_direction:    3-way {-1, 0, +1}   (away / hold / toward objective)
  - objective_direction:   3-way {-1, 0, +1}   (disengage / hold / push site)
  - initiated_engagement:  2-way {False, True}
  - used_utility:          2-way {no / yes}    (derived from utility_used list)

Embedding: event_embedding(demo, round, tick, window=128 raw ticks =
2 seconds causal buildup) from the v6 cache. NOT the single-tick lookup
— pro actions take time to telegraph; a buildup window is the right unit.

Split: demo-disjoint. smoke_test.jsonl spans 4 furia-vs-vitality demos;
3 → train (1585 samples), 1 → val (varies). Default val = m3-nuke (224
samples — smallest, leaves max train).

Usage:
    python scripts/probe_pro_action.py
    python scripts/probe_pro_action.py --val-demo furia-vs-vitality-m4-overpass
    python scripts/probe_pro_action.py --cache outputs/round_encoder/v7_1024d_salience
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.perception.encoder_cache import EncoderEmbeddingCache

DEFAULT_CACHE = REPO / "outputs" / "round_encoder" / "v6_81demos" / "embedding_cache.pt"
SOURCE_JSONL = REPO / "data" / "training" / "grpo" / "smoke_test_source.jsonl"
DATA_JSONL = REPO / "data" / "training" / "grpo" / "smoke_test.jsonl"


def load_samples_with_embeddings(cache: EncoderEmbeddingCache,
                                  window_ticks: int) -> list[dict]:
    """Load every smoke_test row, compute event_embedding from v6 cache."""
    sources = [json.loads(l) for l in SOURCE_JSONL.open()]
    data = [json.loads(l) for l in DATA_JSONL.open()]
    out = []
    misses = 0
    for s, d in zip(sources, data):
        beh = d.get("ground_truth", {}).get("pro_action", {}).get("behavior")
        if not isinstance(beh, dict):
            continue
        emb = cache.event_embedding(
            s["demo_stem"], s["round_num"], s["tick"],
            window_ticks=window_ticks,
        )
        if emb is None:
            misses += 1
            continue
        out.append({
            "demo_stem": s["demo_stem"],
            "round_num": s["round_num"],
            "tick": s["tick"],
            "embedding": emb,
            "movement_direction": int(beh.get("movement_direction", 0)),
            "objective_direction": int(beh.get("objective_direction", 0)),
            "initiated_engagement": bool(beh.get("initiated_engagement", False)),
            "used_utility": bool(
                isinstance(beh.get("utility_used"), list)
                and len(beh["utility_used"]) > 0
            ),
            "engagement_delay": float(beh.get("engagement_delay", 1.0)),
        })
    print(f"Loaded {len(out)} samples (skipped {misses} cache misses)")
    return out


class Probe(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(tx, ty, vx, vy, n_classes, epochs, device,
                      hidden=128, lr=1e-3, batch_size=128, seeds=(0, 1, 2)):
    """Train a tiny MLP probe over multiple seeds, return mean best val acc."""
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        d_in = tx.shape[1]
        probe = Probe(d_in, n_classes, hidden).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        txt = torch.from_numpy(tx).to(device)
        tyt = torch.from_numpy(ty).to(device)
        vxt = torch.from_numpy(vx).to(device)
        vyt = torch.from_numpy(vy).to(device)
        best = 0.0
        for ep in range(epochs):
            probe.train()
            perm = torch.randperm(len(tyt), device=device)
            for i in range(0, len(tyt), batch_size):
                idx = perm[i:i + batch_size]
                loss = nn.functional.cross_entropy(probe(txt[idx]), tyt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                v = float((probe(vxt).argmax(-1) == vyt).float().mean().item())
            if v > best: best = v
        accs.append(best)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="path to embedding_cache.pt or a directory of chunks")
    ap.add_argument("--window-ticks", type=int, default=128,
                    help="raw 64Hz ticks for causal event-window pooling "
                         "(default 128 = 2.0s)")
    ap.add_argument("--val-demo", type=str,
                    default="furia-vs-vitality-m3-nuke",
                    help="which demo to hold out as val")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cache: {args.cache}")
    cache = EncoderEmbeddingCache(args.cache)
    print(f"  d_model={cache.d_model}, rounds={cache.n_rounds}")
    print()

    samples = load_samples_with_embeddings(cache, args.window_ticks)
    if not samples:
        sys.exit("no samples — bail")

    by_demo = Counter(s["demo_stem"] for s in samples)
    print(f"\nPer-demo counts: {dict(by_demo)}")
    if args.val_demo not in by_demo:
        sys.exit(f"val_demo {args.val_demo!r} not in samples")

    train_s = [s for s in samples if s["demo_stem"] != args.val_demo]
    val_s = [s for s in samples if s["demo_stem"] == args.val_demo]
    print(f"\nSplit: train demos={sorted(set(s['demo_stem'] for s in train_s))}")
    print(f"  train={len(train_s)} samples, val={len(val_s)} samples")
    print()

    tx_full = np.stack([s["embedding"] for s in train_s]).astype(np.float32)
    vx_full = np.stack([s["embedding"] for s in val_s]).astype(np.float32)

    targets = {
        "movement_direction":   {"vals": [-1, 0, 1], "n_classes": 3},
        "objective_direction":  {"vals": [-1, 0, 1], "n_classes": 3},
        "initiated_engagement": {"vals": [False, True], "n_classes": 2},
        "used_utility":         {"vals": [False, True], "n_classes": 2},
    }

    results = {}
    print("=" * 72)
    print(f"{'target':25s}  {'classes':>7s}  {'majority':>9s}  "
          f"{'probe':>7s}  {'lift':>6s}  {'std':>5s}  result")
    print("=" * 72)
    for name, spec in targets.items():
        vals = spec["vals"]
        ty = np.array([vals.index(s[name]) for s in train_s], dtype=np.int64)
        vy = np.array([vals.index(s[name]) for s in val_s], dtype=np.int64)
        # Majority baseline on val
        most_common_class = Counter(vy.tolist()).most_common(1)[0][0]
        majority = float(np.mean(vy == most_common_class))
        mean_acc, std_acc = train_classifier(
            tx_full, ty, vx_full, vy, spec["n_classes"],
            epochs=args.epochs, device=device, seeds=args.seeds,
        )
        lift = mean_acc - majority
        verdict = "PASS" if lift > 0.05 else "weak" if lift > 0.0 else "FAIL"
        print(f"  {name:25s}  {spec['n_classes']:>7d}  {majority:>9.4f}  "
              f"{mean_acc:>7.4f}  {lift:>+6.4f}  {std_acc:>5.3f}  {verdict}")
        results[name] = {
            "n_classes": spec["n_classes"],
            "majority_baseline": majority,
            "probe_val_acc_mean": mean_acc,
            "probe_val_acc_std": std_acc,
            "lift_over_majority": lift,
            "verdict": verdict,
        }
    print()

    out_path = args.cache.parent / "probe_pro_action.json" \
        if args.cache.is_file() else args.cache / "probe_pro_action.json"
    out_path.write_text(json.dumps({
        "cache": str(args.cache),
        "val_demo": args.val_demo,
        "window_ticks": args.window_ticks,
        "train_n": len(train_s),
        "val_n": len(val_s),
        "results": results,
    }, indent=2))
    print(f"→ wrote {out_path}")


if __name__ == "__main__":
    main()
