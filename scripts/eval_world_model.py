#!/usr/bin/env python3
"""Per-feature-GROUP eval for the world model — because the aggregate loss is
dominated by easy, static, not-the-goal dims. Positioning is the target.

Splits the 597-d frame into:
  - position  (x,y,z per player)            <- THE GOAL
  - aim       (sin/cos yaw, sin/cos pitch)  <- where they're looking (positioning-adjacent)
  - other     (hp/armor/weapon/econ/global) <- nice-to-have, mostly static/easy

For each group, reports the model's residual Huber vs two trivial baselines:
  - COPY        : predict no motion (residual 0)
  - CONST-VEL   : linear extrapolation k*(x - x_prev)  <- the STRONG baseline for
                  position (players move smoothly); beating it on position is the
                  real bar, the same one MLMove had to clear.
skill = (baseline - model) / baseline  (higher = better; positive = beats baseline).

Usage: python scripts/eval_world_model.py --ckpt outputs/world_model/h8/best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import RoundWindows, build_model, N_PLAYERS  # noqa
from _corpus import load_corpus


def group_indices(ppd, fdim):
    """Predicted dims only: position / aim / other (v2 raw). Excludes the 9 derived
    perception dims per player (input-only, not forecast) and keeps global in 'other'."""
    pos, aim, other = [], [], []
    for p in range(N_PLAYERS):
        b = p * ppd
        pos += [b + 0, b + 1, b + 2]            # x, y, z
        aim += [b + 3, b + 4, b + 5, b + 6]     # sin/cos yaw, sin/cos pitch
        other += list(range(b + 7, b + 56))     # v2 'other' (hp/armor/weapon/util); EXCLUDE 56:ppd derived
    other += list(range(N_PLAYERS * ppd, fdim))  # global block
    return {"position": pos, "aim": aim, "other": other}


def huber_idx(pred, target, idx):
    return F.smooth_l1_loss(pred[..., idx], target[..., idx], beta=1.0).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/world_model/h8/best.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # copy the checkpoint first (training may be writing best.pt concurrently)
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "ckpt.pt"
        shutil.copy(args.ckpt, safe)
        ck = torch.load(safe, map_location="cpu", weights_only=False)
    a = ck["args"]
    horizon = a["horizon"]
    ppd = ck.get("per_player_dim", 56)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd)
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    print(f"ckpt step {ck.get('step')}  arch={a['arch']}  horizon={horizon} "
          f"({horizon*125}ms)  per_player={ppd}  val_ns(agg)={ck.get('val_ns', float('nan')):.4f}  "
          f"value_auc={ck.get('value_auc', float('nan')):.3f}")

    blob = load_corpus(args.val_pt, tag="val")
    ds = RoundWindows(blob["tensors"], blob["metas"], a["window"], horizon, crops_per_round=16)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False)
    groups = group_indices(ppd, ck["feature_dim"])

    agg = {g: dict(model=0.0, copy=0.0, cv=0.0) for g in groups}
    n = 0
    for bi, (x, y, x_prev, _) in enumerate(ld):
        if bi >= args.max_batches:
            break
        x, y, x_prev = x.to(args.device), y.to(args.device), x_prev.to(args.device)
        true_res = y - x
        pred_res = model(x)
        cv_res = horizon * (x - x_prev)
        zero = torch.zeros_like(true_res)
        for g, idx in groups.items():
            agg[g]["model"] += huber_idx(pred_res, true_res, idx)
            agg[g]["copy"] += huber_idx(zero, true_res, idx)
            agg[g]["cv"] += huber_idx(cv_res, true_res, idx)
        n += 1

    print(f"\neval over {n} val batches ({len(groups['position'])} pos dims, "
          f"{len(groups['aim'])} aim, {len(groups['other'])} other):\n")
    print(f"{'group':10s} {'model':>9s} {'copy':>9s} {'const-vel':>9s}  "
          f"{'skill vs copy':>14s} {'skill vs CV':>12s}")
    for g in ("position", "aim", "other"):
        m = agg[g]["model"] / n
        c = agg[g]["copy"] / n
        cv = agg[g]["cv"] / n
        sc = (c - m) / c * 100 if c > 0 else 0.0
        scv = (cv - m) / cv * 100 if cv > 0 else 0.0
        print(f"{g:10s} {m:9.5f} {c:9.5f} {cv:9.5f}  {sc:13.1f}% {scv:11.1f}%")
    print("\nThe POSITION row is the one that matters. Beating const-vel there = the model "
          "predicts movement better than smooth extrapolation (the MLMove bar).")


if __name__ == "__main__":
    main()
