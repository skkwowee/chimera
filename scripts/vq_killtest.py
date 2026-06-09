#!/usr/bin/env python3
"""VQ kill-test for Path B: can a codebook represent a player's state — especially
POSITION — without blurring it? Trains a per-player VQ-VAE (encode 56-d raw state ->
quantize to a codebook -> decode) and reports:
  - position reconstruction RMSE in GAME UNITS (the precision question),
  - codebook utilization (collapse check),
  - per-map recon (does one shared codebook suffice, or do we need per-map?).

Pass -> Path B viable (build autoregressive-over-codes). Fail (positions blur) ->
stay continuous + add a distributional head.

Usage: python scripts/vq_killtest.py --codebook 1024 --maps de_mirage,de_dust2,de_inferno
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_PPD, NP_, PPD = 56, 10, 65   # VQ the 56 raw dims (exclude the 9 derived)


class VQVAE(nn.Module):
    def __init__(self, din=RAW_PPD, d=128, K=1024, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, din))
        self.codebook = nn.Embedding(K, d)
        self.codebook.weight.data.uniform_(-1.0 / K, 1.0 / K)
        self.K = K

    def forward(self, x):
        z = self.enc(x)                                  # [B, d]
        dist = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        zq = self.codebook(idx)
        vq_loss = F.mse_loss(zq, z.detach()) + 0.25 * F.mse_loss(z, zq.detach())
        zq_st = z + (zq - z).detach()                    # straight-through
        return self.dec(zq_st), idx, vq_loss, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(args.pt, map_location="cpu", weights_only=False)
    keep = set(args.maps.split(","))
    # flatten to alive per-player 56-d vectors, tagged by map
    vecs, vmap = [], []
    for t, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        a = t.numpy()[:, :NP_ * PPD].reshape(t.shape[0], NP_, PPD)[:, :, :RAW_PPD]  # [T,10,56]
        alive = a[:, :, 13] > 0.5
        v = a[alive]                                                     # [n,56]
        vecs.append(v); vmap += [m["map_name"]] * len(v)
    X = np.concatenate(vecs, 0).astype(np.float32)
    vmap = np.array(vmap)
    print(f"alive player-vectors: {len(X)}  maps={sorted(keep)}  codebook K={args.codebook}")

    # round-agnostic split (90/10) — fine for a reconstruction kill-test
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g).numpy()
    X = X[perm]; vmap = vmap[perm]
    ntr = int(0.9 * len(X))
    Xtr = torch.from_numpy(X[:ntr]); Xte = torch.from_numpy(X[ntr:]); vte = vmap[ntr:]

    dev = torch.device(args.device)
    model = VQVAE(d=args.d, K=args.codebook).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    Xtr_d = Xtr.to(dev)
    usage = torch.zeros(args.codebook, device=dev)
    for step in range(args.steps):
        bidx = torch.randint(0, len(Xtr_d), (args.batch,), device=dev)
        x = Xtr_d[bidx]
        rec, idx, vql, z = model(x)
        loss = F.mse_loss(rec, x) + vql
        opt.zero_grad(); loss.backward(); opt.step()
        usage.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        if step % 1000 == 999:                          # dead-code revival (anti-collapse)
            dead = (usage == 0).nonzero().squeeze(1)
            if len(dead):
                r = torch.randint(0, len(z), (len(dead),), device=dev)
                model.codebook.weight.data[dead] = z.detach()[r]
            usage.zero_()
        if step % 1000 == 0:
            print(f"  step {step}  recon {F.mse_loss(rec,x).item():.5f}  vq {vql.item():.5f}")

    # ---- eval reconstruction on held-out vectors ----
    model.eval()
    with torch.no_grad():
        rec, idx, _, _ = model(Xte.to(dev))
    rec = rec.cpu(); used = len(torch.unique(idx))
    # de-normalized POSITION rmse (x,y /3000, z /500) -> game units
    pos_xy = (((rec[:, 0:2] - Xte[:, 0:2]) * 3000) ** 2).mean().sqrt().item()
    pos_z = (((rec[:, 2] - Xte[:, 2]) * 500) ** 2).mean().sqrt().item()
    hp_err = ((rec[:, 7] - Xte[:, 7]).abs().mean() * 100).item()
    print(f"\n=== VQ kill-test result (held-out {len(Xte)} vectors) ===")
    print(f"codebook utilization: {used}/{args.codebook} ({100*used/args.codebook:.0f}%) "
          f"{'<- COLLAPSE' if used < 0.2*args.codebook else 'ok'}")
    print(f"POSITION recon RMSE: xy={pos_xy:.1f} units, z={pos_z:.1f} units")
    print(f"  (CS player radius ~32u; <~50u xy = positions survive VQ -> Path B viable)")
    print(f"hp recon MAE: {hp_err:.1f}")
    # per-map position recon
    print("per-map position xy-RMSE (units):")
    for mp in sorted(keep):
        mask = vte == mp
        if mask.sum() == 0: continue
        e = (((rec[mask, 0:2] - Xte[mask, 0:2]) * 3000) ** 2).mean().sqrt().item()
        print(f"  {mp:12s} {e:.1f}  (n={int(mask.sum())})")
    print("\nverdict: shared codebook across 3 maps; if per-map RMSE varies wildly or "
          "is >~80u, per-map codebooks (or VQ-the-latent) are needed.")


if __name__ == "__main__":
    main()
