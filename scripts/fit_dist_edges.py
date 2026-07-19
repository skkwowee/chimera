#!/usr/bin/env python3
"""Fit DIST_EDGES_U for the k=4 (500 ms) canonical retrain (Knob 5b).

Rule (pinned in docs/retrain-recipe.md):
  - class 0 (stationary) threshold stays at 8u — it is an absolute positional
    noise floor, not a function of the horizon;
  - interior ring edges e1..e5 = {1/6..5/6} quantiles of the ALIVE-masked,
    MOVING (>=8u) |dxy| distribution at k=4 over the CLEAN train split;
  - open-ring representative magnitude = median of the top sextile
    (replaces the hardcoded 700u that was fit for k=8).

Alive mask matches the training loss: alive at BOTH t and t+k.
Output: the six edges (rounded to integer game units) + per-map quantiles
for the datasheet. CPU-only, minutes.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _corpus import clean_blob

K = 4                    # 500 ms at 8 Hz
XY_NORM = 3000.0
FLOOR_U = 8.0            # stationary threshold, absolute (not horizon-scaled)
ALIVE_DIM = 13           # per_player_layout index of "alive"
TRAIN_PT = "data/processed/tick_sequences/train_v2m.pt"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_PT
    print(f"loading {path} ...")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    clean_blob(blob, tag="fit_dist_edges")
    ppd = blob.get("per_player_dim", 56)
    per_map = defaultdict(list)
    mags_all = []
    for r, m in zip(blob["tensors"], blob["metas"]):
        t = r.numpy()
        if t.shape[0] <= K:
            continue
        P = 10
        pb = t[:, : P * ppd].reshape(t.shape[0], P, ppd)
        xy = pb[:, :, 0:2] * XY_NORM
        alive = pb[:, :, ALIVE_DIM] > 0.5
        d = xy[K:] - xy[:-K]                          # [T-K, P, 2]
        mask = alive[K:] & alive[:-K]                 # alive at t AND t+k
        mag = np.linalg.norm(d, axis=-1)[mask].astype(np.float32)
        mags_all.append(mag)
        per_map[m.get("map_name", "?")].append(mag)
    mags = np.concatenate(mags_all)
    moving = mags[mags >= FLOOR_U]
    q = np.quantile(moving, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    edges = [FLOOR_U] + [round(float(v)) for v in q]
    open_med = float(np.median(moving[moving >= q[-1]]))
    print(f"\nalive player-frame pairs: {len(mags):,}")
    print(f"stationary (<{FLOOR_U:.0f}u): {100 * (len(mags) - len(moving)) / len(mags):.1f}%")
    print(f"DIST_EDGES_U (k={K}, 500ms) = {edges}")
    print(f"open-ring representative magnitude = {open_med:.0f}u (median of top sextile)")
    print("\nper-map mover quantiles (1/6..5/6):")
    for mp in sorted(per_map):
        v = np.concatenate(per_map[mp]); v = v[v >= FLOOR_U]
        print(f"  {mp:14s} n={len(v):>10,}  " +
              " ".join(f"{x:6.0f}" for x in np.quantile(v, [1/6, 2/6, 3/6, 4/6, 5/6])))


if __name__ == "__main__":
    main()
