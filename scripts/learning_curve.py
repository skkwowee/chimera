#!/usr/bin/env python3
"""Learning curve: train v6-style encoder on subsets, measure probe_acc.

Answers "how much data is ideal?" by training the encoder on N ∈ {16, 32,
48, 64, 81} demos worth of rounds and reporting probe_outcome val_acc per
subset. Val split is fixed (12 demos) across all runs, so probe numbers
are directly comparable.

The curve shape tells us where we are:
  - Steep slope at N=81 → more demos will keep helping
  - Flat slope at N=81 → diminishing returns; data isn't the bottleneck

Per-subset training: 15 epochs (v6 best is at epoch ~2-5 anyway, so 15
captures peak val_acc with margin). Encoder config matches v6 exactly
(d_model=512, n_layers=4, salience off).

Usage:
    python scripts/learning_curve.py
    python scripts/learning_curve.py --subsets 16 32 48 64 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_round_encoder import (
    RoundEncoder, TrainConfig, ForwardPredHead, NextEventHead, TimeToEventHead,
    collate, compute_loss, lr_lambda,
)

DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
OUT_DIR = REPO / "outputs" / "round_encoder" / "learning_curve"


class SubsetDataset(Dataset):
    """In-memory subset of the tick-sequence blob — share the underlying
    tensor list, just select by index."""

    def __init__(self, tensors, metas, event_labels, event_times, indices):
        self.tensors = [tensors[i] for i in indices]
        self.metas = [metas[i] for i in indices]
        if event_labels is not None:
            self.event_labels = [event_labels[i] for i in indices]
            self.event_times = [event_times[i] for i in indices]
        else:
            self.event_labels = None
            self.event_times = None
        self.feature_dim = int(tensors[0].shape[1])

    def has_events(self):
        return self.event_labels is not None

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        el = self.event_labels[i] if self.event_labels is not None else None
        et = self.event_times[i] if self.event_times is not None else None
        return self.tensors[i], self.metas[i], el, et


def select_indices_by_demo(metas, n_demos, seed=0):
    """Pick all rounds belonging to the first `n_demos` distinct demos
    (after a seeded shuffle), so subsets are nested by demo identity."""
    rng = np.random.default_rng(seed)
    demos = sorted({m["demo_stem"] for m in metas})
    rng.shuffle(demos)
    chosen = set(demos[:n_demos])
    return [i for i, m in enumerate(metas) if m["demo_stem"] in chosen], chosen


def train_encoder_for_subset(train_ds, val_ds, epochs, device):
    """Train a v6-equivalent encoder. Returns the trained encoder + best
    val_acc_event_only (used as proxy for early stopping)."""
    cfg = TrainConfig(
        feature_dim=train_ds.feature_dim,
        d_model=512, n_layers=4, n_heads=8, d_ff=2048, dropout=0.15,
        max_seq_len=2048,
        epochs=epochs, batch_size=8, num_workers=0,
        lr=3e-4, weight_decay=0.01, warmup_steps=min(100, epochs * 5),
        log_every=10_000_000,  # silence per-step logging
        use_salience=False,
    )
    encoder = RoundEncoder(cfg).to(device)
    heads = nn.ModuleList([
        ForwardPredHead(cfg.d_model, cfg.feature_dim).to(device)
        for _ in cfg.horizons
    ])
    event_head = NextEventHead(cfg.d_model, cfg.n_event_classes).to(device) \
        if train_ds.has_events() else None
    time_head = TimeToEventHead(cfg.d_model).to(device) \
        if train_ds.has_events() else None
    class_weights = None
    if train_ds.has_events():
        cw = torch.ones(cfg.n_event_classes, dtype=torch.float32)
        cw[cfg.none_event_idx] = cfg.none_class_weight
        class_weights = cw.to(device)

    params = list(encoder.parameters()) + list(heads.parameters())
    if event_head is not None:
        params += list(event_head.parameters()) + list(time_head.parameters())

    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                betas=(0.9, 0.95))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)
    total_steps = max(1, cfg.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: lr_lambda(step, cfg.warmup_steps, total_steps),
    )

    best_acc = 0.0
    best_state = None
    for ep in range(epochs):
        encoder.train(); heads.train()
        if event_head: event_head.train(); time_head.train()
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss, _m = compute_loss(encoder, heads, event_head, time_head,
                                          class_weights, batch, cfg, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step(); scheduler.step()

        # Val: only the metric we care about for the curve
        encoder.eval()
        if event_head:
            event_head.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    mask = batch["mask"].to(device)
                    ev = batch["event_labels"].to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "cuda"):
                        h = encoder(x, key_padding_mask=~mask)
                        logits = event_head(h)
                    pred = logits.argmax(dim=-1)
                    valid = mask & (ev != cfg.none_event_idx) & (ev >= 0)
                    if valid.any():
                        correct += int((pred[valid] == ev[valid]).sum())
                        total += int(valid.sum())
            event_acc = correct / max(1, total)
        else:
            event_acc = 0.0

        if event_acc > best_acc:
            best_acc = event_acc
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
            }

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
    return encoder, cfg, best_acc


def run_probe(encoder, train_ds, val_ds, device, queries_per_round=12):
    """Re-encode train + val, train a small MLP to predict round_won, return
    best val_acc (mean over 2 seeds)."""

    def encode_split(ds):
        embs, labels = [], []
        with torch.no_grad():
            for t, m, _e, _et in [ds[i] for i in range(len(ds))]:
                T = t.shape[0]
                if T < 30: continue
                x = t.unsqueeze(0).to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    h = encoder(x)
                h = h.squeeze(0).float().cpu().numpy()
                skip = min(20, max(0, T // 10))
                qpos = (np.linspace(skip, T - 1, queries_per_round, dtype=np.int64)
                        if T - skip > queries_per_round else np.arange(skip, T))
                label = 1 if m.get("winner") == "t" else 0
                for p in qpos:
                    embs.append(h[p]); labels.append(label)
        return (np.stack(embs).astype(np.float32),
                np.array(labels, dtype=np.int64))

    encoder.eval()
    tx, ty = encode_split(train_ds)
    vx, vy = encode_split(val_ds)
    accs = []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        d_in = tx.shape[1]
        probe = nn.Sequential(
            nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        ).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
        txt = torch.from_numpy(tx).to(device)
        tyt = torch.from_numpy(ty).to(device)
        vxt = torch.from_numpy(vx).to(device)
        vyt = torch.from_numpy(vy).to(device)
        best = 0.0
        for ep in range(15):
            probe.train()
            perm = torch.randperm(len(tyt), device=device)
            for i in range(0, len(tyt), 256):
                idx = perm[i:i+256]
                loss = nn.functional.cross_entropy(probe(txt[idx]), tyt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                v = float((probe(vxt).argmax(-1) == vyt).float().mean().item())
            if v > best: best = v
        accs.append(best)
    return float(np.mean(accs))


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading full train.pt + val.pt once...")
    tb = torch.load(DATA_DIR / "train.pt", weights_only=False)
    vb = torch.load(DATA_DIR / "val.pt", weights_only=False)
    print(f"  train: {len(tb['tensors'])} rounds, {len(set(m['demo_stem'] for m in tb['metas']))} demos")
    print(f"  val:   {len(vb['tensors'])} rounds, {len(set(m['demo_stem'] for m in vb['metas']))} demos")
    print()

    val_ds = SubsetDataset(
        vb["tensors"], vb["metas"],
        vb.get("event_labels"), vb.get("event_times"),
        list(range(len(vb["tensors"]))),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    n_demos_available = len(set(m["demo_stem"] for m in tb["metas"]))
    subsets = [n for n in args.subsets if n <= n_demos_available]
    if n_demos_available not in subsets:
        subsets.append(n_demos_available)

    for n_demos in sorted(subsets):
        print(f"=== n_demos = {n_demos} ===")
        indices, demos = select_indices_by_demo(tb["metas"], n_demos, seed=0)
        train_ds = SubsetDataset(
            tb["tensors"], tb["metas"],
            tb.get("event_labels"), tb.get("event_times"),
            indices,
        )
        print(f"  selected {len(demos)} demos, {len(train_ds)} rounds")

        t0 = time.time()
        encoder, cfg, best_event_acc = train_encoder_for_subset(
            train_ds, val_ds, epochs=args.epochs, device=device,
        )
        train_dt = time.time() - t0
        print(f"  trained in {train_dt:.1f}s, best val_acc_event_only={best_event_acc:.4f}")

        t0 = time.time()
        probe_acc = run_probe(encoder, train_ds, val_ds, device)
        probe_dt = time.time() - t0
        print(f"  probe acc {probe_acc:.4f} (took {probe_dt:.1f}s)")
        print()

        results.append({
            "n_demos": n_demos,
            "n_train_rounds": len(train_ds),
            "best_val_acc_event_only": best_event_acc,
            "probe_acc": probe_acc,
            "train_seconds": train_dt,
            "probe_seconds": probe_dt,
        })
        del encoder
        torch.cuda.empty_cache()

    out_path = OUT_DIR / f"curve_epochs{args.epochs}.json"
    out_path.write_text(json.dumps({
        "epochs_per_run": args.epochs,
        "val_demos": len(set(m["demo_stem"] for m in vb["metas"])),
        "val_rounds": len(val_ds),
        "results": results,
    }, indent=2))

    # Summary
    print("=" * 64)
    print("Learning curve summary")
    print("=" * 64)
    print(f"  {'n_demos':>7s}  {'rounds':>6s}  {'event_acc':>10s}  {'probe_acc':>10s}")
    for r in results:
        print(f"  {r['n_demos']:>7d}  {r['n_train_rounds']:>6d}  "
              f"{r['best_val_acc_event_only']:>10.4f}  {r['probe_acc']:>10.4f}")
    print()
    if len(results) >= 2:
        last_two = results[-2:]
        d_demos = last_two[1]["n_demos"] - last_two[0]["n_demos"]
        d_probe = last_two[1]["probe_acc"] - last_two[0]["probe_acc"]
        slope_per_10 = (d_probe / d_demos) * 10 if d_demos else 0
        print(f"  Top-end slope: +{slope_per_10:+.4f} probe_acc per +10 demos")
    print(f"  → wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--subsets", type=int, nargs="+",
                    default=[16, 32, 48, 64])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
