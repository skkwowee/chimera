#!/usr/bin/env python3
"""Probe ablations: zero specific input-feature dims, re-encode, re-probe.

The L2-G2 probe (scripts/probe_features.py) hit val_acc 0.763 on v6
embeddings — well above the 0.65 gate. But the encoder's input is a
582-d feature vector that includes bookkeeping fields (score, round_num,
round_time, bomb_state) which trivially correlate with round_won. If
zeroing those features keeps val_acc >0.65, the encoder really did learn
tactical structure; if it craters, the probe was reading bookkeeping.

The encoder is frozen, so the ablation happens at the INPUT side: zero
specific dims in (B, T, 582), re-encode → train a fresh probe → report
val_acc.

Index ranges derived from data/processed/tick_sequences/feature_schema_v1.json.

Usage:
    python scripts/probe_ablations.py
    python scripts/probe_ablations.py --epochs 30 --queries-per-round 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v6_81demos" / "best.pt"

PER_PLAYER_DIM = 56
N_PLAYERS = 10
GLOBAL_START = N_PLAYERS * PER_PLAYER_DIM  # = 560

# Index helpers — per-player offsets within a 56-d block
PP = {
    "pos": [0, 1, 2],
    "view": [3, 4, 5, 6],
    "hp": [7],
    "armor": [8],
    "helmet": [9],
    "defuser": [10],
    "balance": [11],
    "equip": [12],
    "alive": [13],
    "has_c4": [14],
    "primary": list(range(15, 33)),  # 18 dims
    "secondary": list(range(33, 51)),
    "util": list(range(51, 56)),
}

# Global indices (absolute)
G_MAP = list(range(GLOBAL_START + 0, GLOBAL_START + 7))      # map_onehot
G_PHASE = list(range(GLOBAL_START + 7, GLOBAL_START + 11))   # phase_onehot
G_SCORE = [GLOBAL_START + 11, GLOBAL_START + 12]             # score_t, score_ct
G_ROUND_NUM = [GLOBAL_START + 13]
G_ROUND_TIME = [GLOBAL_START + 14]
G_BOMB_STATE = list(range(GLOBAL_START + 15, GLOBAL_START + 19))
G_BOMB_POS = [GLOBAL_START + 19, GLOBAL_START + 20]
G_BOMB_AGE = [GLOBAL_START + 21]


def player_dims(field: str) -> list[int]:
    """All 10-player absolute indices for a per-player field name."""
    offsets = PP[field]
    out = []
    for p in range(N_PLAYERS):
        base = p * PER_PLAYER_DIM
        out.extend(base + o for o in offsets)
    return out


ABLATIONS = {
    "baseline":              [],
    "no_score":              G_SCORE,
    "no_round_num":          G_ROUND_NUM,
    "no_round_time":         G_ROUND_TIME,
    "no_bookkeeping":        G_SCORE + G_ROUND_NUM + G_ROUND_TIME,
    "no_bomb":               G_BOMB_STATE + G_BOMB_POS + G_BOMB_AGE,
    "no_alive":              player_dims("alive"),
    "no_hp_armor":           player_dims("hp") + player_dims("armor"),
    "no_alive_hp_armor":     player_dims("alive") + player_dims("hp") + player_dims("armor"),
    "no_phase":              G_PHASE,
    # Aggressive: keep ONLY per-player x,y,z + map. Tests if pure spatial
    # info alone is enough.
    "spatial_only":          (
        [i for i in range(GLOBAL_START) if i % PER_PLAYER_DIM not in (0, 1, 2)]
        + G_PHASE + G_SCORE + G_ROUND_NUM + G_ROUND_TIME
        + G_BOMB_STATE + G_BOMB_POS + G_BOMB_AGE
    ),
    # Another aggressive: keep ONLY bookkeeping (score, round_num, round_time,
    # bomb_state, bomb_pos, bomb_age, phase, map). Tests if those alone are
    # sufficient — the upper bound on a "bookkeeping cheating" hypothesis.
    "bookkeeping_only":      [i for i in range(GLOBAL_START)]  # zero all per-player
                              + [],
}


def load_encoder(ckpt_path: Path, device):
    sys.path.insert(0, str(REPO / "scripts"))
    from train_round_encoder import RoundEncoder, TrainConfig
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in TrainConfig.__dataclass_fields__})
    enc = RoundEncoder(cfg).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    return enc, cfg


def sample_query_positions(T: int, n: int) -> np.ndarray:
    skip = min(20, max(0, T // 10))
    if T - skip <= n:
        return np.arange(skip, T)
    return np.linspace(skip, T - 1, n, dtype=np.int64)


def encode_split_with_ablation(split_path, encoder, device, queries_per_round,
                                zero_dims: list[int]):
    """Encode every round AFTER zeroing the listed input dims, return per-tick
    embeddings + round_won label per query."""
    blob = torch.load(split_path, weights_only=False)
    tensors = blob["tensors"]
    metas = blob["metas"]
    zero_idx = torch.tensor(zero_dims, dtype=torch.long, device=device) \
        if zero_dims else None
    embs, labels = [], []
    with torch.no_grad():
        for tensor, meta in zip(tensors, metas):
            T = tensor.shape[0]
            if T < 30:
                continue
            x = tensor.unsqueeze(0).to(device)  # (1, T, 582)
            if zero_idx is not None:
                x[:, :, zero_idx] = 0.0
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
            h = h.squeeze(0).float().cpu().numpy()
            qpos = sample_query_positions(T, queries_per_round)
            label = 1 if meta.get("winner") == "t" else 0
            for p in qpos:
                embs.append(h[p])
                labels.append(label)
    return np.stack(embs).astype(np.float32), np.array(labels, dtype=np.int64)


class Probe(nn.Module):
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(tx, ty, vx, vy, d_hidden, epochs, lr, batch_size, device,
                seeds: list[int]) -> float:
    """Train probe over multiple seeds, return MEAN best val_acc (more stable)."""
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        probe = Probe(tx.shape[1], d_hidden).to(device)
        optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        tx_t = torch.from_numpy(tx).to(device)
        ty_t = torch.from_numpy(ty).to(device)
        vx_t = torch.from_numpy(vx).to(device)
        vy_t = torch.from_numpy(vy).to(device)

        best = 0.0
        N = tx_t.shape[0]
        for ep in range(epochs):
            probe.train()
            perm = torch.randperm(N, device=device)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                loss = nn.functional.cross_entropy(probe(tx_t[idx]), ty_t[idx])
                optim.zero_grad(); loss.backward(); optim.step()
            probe.eval()
            with torch.no_grad():
                v = float((probe(vx_t).argmax(-1) == vy_t).float().mean().item())
            if v > best:
                best = v
        accs.append(best)
    return float(np.mean(accs))


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}\n")

    encoder, cfg = load_encoder(args.ckpt, device)
    print(f"Encoder: d_model={cfg.d_model}, n_layers={cfg.n_layers}\n")

    selected = args.only.split(",") if args.only else list(ABLATIONS.keys())
    invalid = [s for s in selected if s not in ABLATIONS]
    if invalid:
        sys.exit(f"unknown ablations: {invalid}\n  available: {list(ABLATIONS)}")

    results = {}
    for name in selected:
        zero_dims = ABLATIONS[name]
        print(f"[{name}] zeroing {len(zero_dims)} dims of 582...")
        tx, ty = encode_split_with_ablation(
            DATA_DIR / "train.pt", encoder, device,
            args.queries_per_round, zero_dims,
        )
        vx, vy = encode_split_with_ablation(
            DATA_DIR / "val.pt", encoder, device,
            args.queries_per_round, zero_dims,
        )
        majority = float(max(vy.mean(), 1 - vy.mean()))
        val_acc = train_probe(
            tx, ty, vx, vy,
            d_hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=device, seeds=args.seeds,
        )
        results[name] = {
            "zero_count": len(zero_dims),
            "mean_val_acc": val_acc,
            "majority": majority,
            "lift_over_majority": val_acc - majority,
        }
        print(f"  → mean val_acc (over {len(args.seeds)} seeds) = {val_acc:.4f} "
              f"(majority {majority:.3f}, lift +{val_acc-majority:.3f})")
        print()

    # Summary
    base = results.get("baseline", {}).get("mean_val_acc")
    print("=" * 64)
    print("Probe ablation summary (mean val_acc over seeds)")
    print("=" * 64)
    print(f"  {'ablation':24s}  {'val_acc':>8s}  {'Δ vs base':>10s}  {'gate(0.65)':>11s}")
    for name, r in results.items():
        acc = r["mean_val_acc"]
        delta = (acc - base) if base is not None else 0.0
        gate = "PASS" if acc >= 0.65 else "FAIL"
        print(f"  {name:24s}  {acc:>8.4f}  {delta:+10.4f}  {gate:>11s}")
    print()

    out = args.ckpt.parent / "probe_ablations.json"
    out.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "seeds": args.seeds,
        "queries_per_round": args.queries_per_round,
        "results": results,
    }, indent=2))
    print(f"→ wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--queries-per-round", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds for probe training (mean reported)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated ablation names (default: all)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
