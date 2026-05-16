#!/usr/bin/env python3
"""Level-2 gate L2-G2: probe-outcome MLP on frozen encoder embeddings.

Per docs/alignment-delta.md §8: train a small MLP on per-tick encoder
embeddings to predict round_won, on a (demo, round)-disjoint split. The
encoder was trained C1-clean (never saw round_won), so anything the probe
can recover from its frozen embeddings is honest evidence the encoder has
captured outcome-relevant structure. Gate threshold: **val acc ≥ 0.65**
(base rate ~0.47–0.53 in our dataset, so 0.65 is meaningfully informative).

Reports overall + per-position-bin (early/mid/late within round). σ_s
already showed late dominates clustering, so the per-bin breakdown tells
us whether the probe is doing real work or just memorizing endgames.

Usage:
    python scripts/probe_features.py
    python scripts/probe_features.py --ckpt outputs/round_encoder/<run>/best.pt
    python scripts/probe_features.py --queries-per-round 16 --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v6_81demos" / "best.pt"

GATE_THRESHOLD = 0.65  # L2-G2


def load_encoder(ckpt_path: Path, device: torch.device):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from train_round_encoder import RoundEncoder, TrainConfig
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in TrainConfig.__dataclass_fields__})
    encoder = RoundEncoder(cfg).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, cfg, ckpt


def sample_query_positions(T: int, n: int) -> np.ndarray:
    skip = min(20, max(0, T // 10))
    if T - skip <= n:
        return np.arange(skip, T)
    return np.linspace(skip, T - 1, n, dtype=np.int64)


def encode_split(split_path: Path, encoder, device, queries_per_round: int):
    """Encode every round, return (embeddings (N,d), labels (N,), positions (N,)
    where position is the fraction within round."""
    blob = torch.load(split_path, weights_only=False)
    tensors = blob["tensors"]
    metas = blob["metas"]
    embs, labels, positions = [], [], []
    with torch.no_grad():
        for tensor, meta in zip(tensors, metas):
            T = tensor.shape[0]
            if T < 2:
                continue
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
            h = h.squeeze(0).float().cpu().numpy()
            qpos = sample_query_positions(T, queries_per_round)
            label = 1 if meta.get("winner") == "t" else 0
            for p in qpos:
                embs.append(h[p])
                labels.append(label)
                positions.append(p / max(1, T - 1))
    return (np.stack(embs).astype(np.float32),
            np.array(labels, dtype=np.int64),
            np.array(positions, dtype=np.float32))


class Probe(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(train_x, train_y, val_x, val_y, val_pos,
                d_hidden: int, epochs: int, lr: float, batch_size: int,
                device: torch.device) -> dict:
    probe = Probe(train_x.shape[1], d_hidden).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(val_x).to(device)
    vy = torch.from_numpy(val_y).to(device)

    best_val_acc = 0.0
    best_val_metrics = {}
    N = tx.shape[0]
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(tx[idx])
            loss = nn.functional.cross_entropy(logits, ty[idx])
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item()) * len(idx)
        train_loss = total_loss / N

        probe.eval()
        with torch.no_grad():
            vl = probe(vx)
            vpred = vl.argmax(dim=-1)
            val_acc = float((vpred == vy).float().mean().item())
            # Per-position bins
            vpred_np = vpred.cpu().numpy()
            vy_np = vy.cpu().numpy()
            bin_accs = {}
            for lo, hi, name in [(0.0, 0.33, "early"),
                                  (0.33, 0.67, "mid"),
                                  (0.67, 1.01, "late")]:
                mask = (val_pos >= lo) & (val_pos < hi)
                if mask.any():
                    bin_accs[name] = float((vpred_np[mask] == vy_np[mask]).mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_metrics = {"val_acc": val_acc, "epoch": ep,
                                "bin_acc": bin_accs.copy()}
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  e{ep:3d} train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                  f"(early={bin_accs.get('early',0):.3f} "
                  f"mid={bin_accs.get('mid',0):.3f} "
                  f"late={bin_accs.get('late',0):.3f})")
    return best_val_metrics


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}")

    encoder, cfg, ckpt = load_encoder(args.ckpt, device)
    print(f"Encoder: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"feature_dim={cfg.feature_dim}")
    print()

    print(f"Encoding train queries (n_queries/round = {args.queries_per_round})...")
    train_x, train_y, train_pos = encode_split(
        DATA_DIR / "train.pt", encoder, device, args.queries_per_round,
    )
    print(f"  train: {train_x.shape[0]} queries, balance: "
          f"t={int(train_y.sum())} / ct={int((1 - train_y).sum())}")

    print(f"Encoding val queries...")
    val_x, val_y, val_pos = encode_split(
        DATA_DIR / "val.pt", encoder, device, args.queries_per_round,
    )
    base_rate = float(val_y.mean())
    majority = max(base_rate, 1 - base_rate)
    print(f"  val:   {val_x.shape[0]} queries, balance: "
          f"t={int(val_y.sum())} / ct={int((1 - val_y).sum())}")
    print(f"  base rate (t-win): {base_rate:.3f}, majority-class baseline: {majority:.3f}")
    print()

    print(f"Training probe (hidden={args.hidden}, lr={args.lr}, epochs={args.epochs})...")
    metrics = train_probe(
        train_x, train_y, val_x, val_y, val_pos,
        d_hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=device,
    )
    print()

    val_acc = metrics["val_acc"]
    over_majority = val_acc - majority
    bin_accs = metrics["bin_acc"]

    if val_acc >= GATE_THRESHOLD:
        verdict = f"PASS (val_acc={val_acc:.4f} ≥ {GATE_THRESHOLD})"
    else:
        verdict = f"FAIL (val_acc={val_acc:.4f} < {GATE_THRESHOLD})"

    print("=" * 64)
    print(f"L2-G2 probe_outcome — encoder {args.ckpt.name}")
    print("=" * 64)
    print(f"  Gate threshold:          {GATE_THRESHOLD}")
    print(f"  Majority-class baseline: {majority:.4f}")
    print(f"  Best val_acc:            {val_acc:.4f}  (at epoch {metrics['epoch']})")
    print(f"  Over majority:           {over_majority:+.4f}")
    print()
    print(f"  Per-position breakdown:")
    print(f"    early (0.00–0.33):    {bin_accs.get('early', 0):.4f}")
    print(f"    mid   (0.33–0.67):    {bin_accs.get('mid', 0):.4f}")
    print(f"    late  (0.67–1.00):    {bin_accs.get('late', 0):.4f}")
    print()
    print(f"  VERDICT: {verdict}")
    print()

    out = args.ckpt.parent / f"probe_features_{args.ckpt.stem}.json"
    out.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "gate_threshold": GATE_THRESHOLD,
        "majority_class_baseline": majority,
        "base_rate_t": base_rate,
        "best_val_acc": val_acc,
        "over_majority": over_majority,
        "best_epoch": metrics["epoch"],
        "bin_acc": bin_accs,
        "n_train_queries": int(train_x.shape[0]),
        "n_val_queries": int(val_x.shape[0]),
        "verdict": verdict,
    }, indent=2))
    print(f"  → wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--queries-per-round", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128,
                    help="probe MLP hidden dim (default 128 — keep small)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
