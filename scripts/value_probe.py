#!/usr/bin/env python3
"""Value probe: does the world-model LATENT make round-outcome learnable in a way
RAW features don't? This is the decisive "probe transfer" test for the value
stack (see docs/world-model-design.md). Demo-disjoint by construction (train.pt
and val.pt hold disjoint matches).

Target: V(state) = P(CT wins the round | state)   [label = winner == 'ct'].
We fit a LINEAR probe (frozen representation -> logistic regression) on train
rounds and report val AUC. Three representations, head-to-head:
  raw_last  : the current 597-d frame (no history)           <- the "no world model" baseline
  raw_mean  : mean of the 96-frame window (history, but no learned dynamics)
  latent    : world-model latent at the current frame         <- THE TEST

A linear probe is deliberate: it measures how LINEARLY-DECODABLE the outcome is
from each representation, i.e. how much value-relevant structure the world model
baked in -- not how powerful a downstream head is.

Bucketed by round-progress (0.25 / 0.5 / 0.75): early-round value is the hard,
meaningful part (late-round, everyone predicts the winner). If latent > raw
*early*, the world model is carrying real value signal.

Usage: python scripts/value_probe.py --ckpt outputs/world_model/h8/best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model  # noqa

FRACS = [0.25, 0.5, 0.75]


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC (Mann-Whitney). scores,labels: 1-D tensors."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels > 0.5
    n_pos = pos.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def build_reps(blob, model, window, device, batch=64):
    """Return dict of representation tensors + labels + frac-bucket per sample."""
    tensors, metas = blob["tensors"], blob["metas"]
    windows, raw_last, raw_mean, labels, fracs = [], [], [], [], []
    for r, m in zip(tensors, metas):
        T = r.shape[0]
        y = 1.0 if m["winner"] == "ct" else 0.0
        for f in FRACS:
            end = int(f * T)
            if end < window:
                continue
            w = r[end - window:end]                       # [window, F]
            windows.append(w)
            raw_last.append(r[end - 1])
            raw_mean.append(w.mean(0))
            labels.append(y)
            fracs.append(f)
    lat = []
    for i in range(0, len(windows), batch):
        wb = torch.stack(windows[i:i + batch]).to(device)  # [B, window, F]
        lat.append(model.latent(wb)[:, -1, :].cpu())       # last-timestep latent
    return {
        "raw_last": torch.stack(raw_last),
        "raw_mean": torch.stack(raw_mean),
        "latent": torch.cat(lat),
        "labels": torch.tensor(labels),
        "fracs": torch.tensor(fracs),
    }


def fit_probe(Xtr, ytr, Xva, epochs=300, lr=0.05, wd=1e-3, device="cpu"):
    """Standardize on train, fit logistic regression, return val logits."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device)
    Xva = ((Xva - mu) / sd).to(device)
    ytr = ytr.to(device)
    lin = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(Xtr).squeeze(1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(Xva).squeeze(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/world_model/h8/best.pt")
    ap.add_argument("--train-pt", default="data/processed/tick_sequences/train.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"
        shutil.copy(args.ckpt, safe)
        ck = torch.load(safe, map_location="cpu", weights_only=False)
    a = ck["args"]; window = a["window"]
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"])
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    print(f"ckpt step {ck.get('step')}  arch={a['arch']}  latent_dim={a['d_model']}")

    tr = build_reps(torch.load(args.train_pt, map_location="cpu", weights_only=False),
                    model, window, args.device)
    va = build_reps(torch.load(args.val_pt, map_location="cpu", weights_only=False),
                    model, window, args.device)
    base = va["labels"].float().mean().item()
    print(f"train samples {len(tr['labels'])}  val {len(va['labels'])}  "
          f"val CT-win base rate {base:.2f}\n")

    reps = ["raw_last", "raw_mean", "latent"]
    val_logits = {r: fit_probe(tr[r], tr["labels"], va[r], device=args.device) for r in reps}

    print(f"{'representation':12s} {'AUC all':>8s} " + " ".join(f"{'AUC@'+str(f):>8s}" for f in FRACS))
    for r in reps:
        row = [auc(val_logits[r], va["labels"])]
        for f in FRACS:
            mask = va["fracs"] == f
            row.append(auc(val_logits[r][mask], va["labels"][mask]))
        tag = "  <- TEST" if r == "latent" else ""
        print(f"{r:12s} " + " ".join(f"{v:8.3f}" for v in row) + tag)
    print("\nIf latent AUC > raw_* (especially @0.25, early-round), the world-model latent "
          "carries value signal raw features don't -> value head belongs on the latent.")


if __name__ == "__main__":
    main()
