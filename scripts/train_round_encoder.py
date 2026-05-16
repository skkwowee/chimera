#!/usr/bin/env python3
"""Train the Level-2 round encoder (v4: causal decoder).

Per docs/round-encoder-design.md v4:
  - Round-as-sequence-of-downsampled-ticks (8 Hz, ~960 tokens per round)
  - Causal self-attention so h_T is a function of ticks 0..T only (F2-safe)
  - Predict-forward SSL objectives — none consume round_won

This v1 of the script ships TWO of the four planned SSL objectives:
  - L_next:  MSE on per-tick feature vector at T+1
  - L_multi: MSE on per-tick feature vector at T+H for H ∈ {8, 32}
             (1s and 4s ahead — multi-horizon prevents short-horizon shortcut)

The other two objectives (time-to-next-event regression, next-event-type CE)
need per-tick event labels that build_tick_sequences.py doesn't currently
emit. Adding them is incremental — extend the data prep with kills.json /
bomb.json walkthroughs, plug new heads into the loss sum here.

Usage:
    # Smoke test (small model, few epochs) — verify training works
    python scripts/train_round_encoder.py --d-model 256 --n-layers 2 --epochs 3

    # The intended config (~13M params, ~50 epochs)
    python scripts/train_round_encoder.py --d-model 512 --n-layers 4 --epochs 50

Output:
    outputs/round_encoder/<run-id>/
      config.json       — full hyperparameters used
      train_log.jsonl   — per-step + per-epoch metrics
      best.pt           — checkpoint with lowest val total loss
      last.pt           — most-recent checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
OUTPUT_ROOT = REPO / "outputs" / "round_encoder"


@dataclass
class TrainConfig:
    # Model
    feature_dim: int = 582  # set from feature_schema_v1.json
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.15
    max_seq_len: int = 2048  # > any round at 8Hz (max observed: 1547)

    # SSL horizons (in 8Hz ticks): 1 = 0.125s, 8 = 1s, 32 = 4s
    horizons: list[int] = field(default_factory=lambda: [1, 8, 32])
    horizon_weights: list[float] = field(default_factory=lambda: [1.0, 0.7, 0.4])

    # Optim
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    batch_size: int = 8
    epochs: int = 30
    num_workers: int = 2

    # Bookkeeping
    log_every: int = 10
    val_every_epochs: int = 1
    seed: int = 42
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RoundDataset(Dataset):
    """Wraps the list of per-round (T, F) tensors from train.pt / val.pt."""

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, weights_only=False)
        self.tensors: list[torch.Tensor] = blob["tensors"]
        self.metas: list[dict] = blob["metas"]
        self.feature_dim: int = int(blob["feature_dim"])

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
        return self.tensors[i], self.metas[i]


def collate(batch: list[tuple[torch.Tensor, dict]]) -> dict:
    """Pad rounds to max-T-in-batch; build attention mask (1=real, 0=pad)."""
    tensors, metas = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    T_max = int(lengths.max().item())
    F_dim = tensors[0].shape[1]
    B = len(tensors)
    out = torch.zeros(B, T_max, F_dim, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return {"x": out, "mask": mask, "lengths": lengths, "metas": list(metas)}


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


# ---------------------------------------------------------------------------
# Causal-decoder transformer + prediction head
# ---------------------------------------------------------------------------
class RoundEncoder(nn.Module):
    """Causal self-attention transformer over per-tick feature vectors.

    h_T at output position T is a function of input ticks 0..T only — no
    future leakage. F2-safe by construction (per docs/round-encoder-design.md
    v4 design history).
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.norm_in = nn.LayerNorm(cfg.d_model)
        self.pos_emb = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, feature_dim). key_padding_mask: (B, T) where True = PAD.
        Returns h: (B, T, d_model).
        """
        h = self.norm_in(self.input_proj(x))
        h = self.pos_emb(h)
        T = h.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=h.device, dtype=h.dtype,
        )
        for layer in self.layers:
            h = layer(
                h,
                src_mask=causal_mask,
                src_key_padding_mask=key_padding_mask,
                is_causal=True,
            )
        return self.norm_out(h)


class ForwardPredHead(nn.Module):
    """Two-layer MLP head that predicts a feature vector from h_T."""

    def __init__(self, d_model: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(
    encoder: RoundEncoder,
    heads: nn.ModuleList,
    batch: dict,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-horizon forward-prediction loss.

    For each horizon H, the target at position t is the input feature vector
    at position t+H. Padded positions and out-of-bounds positions are excluded.

    Returns (total_loss, per_horizon_dict).
    """
    x = batch["x"].to(device, non_blocking=True)        # (B, T, F)
    mask = batch["mask"].to(device, non_blocking=True)  # (B, T) — True = real
    key_padding_mask = ~mask                            # (B, T) — True = PAD

    h = encoder(x, key_padding_mask=key_padding_mask)   # (B, T, d_model)

    total = torch.zeros((), device=device, dtype=h.dtype)
    per_horizon: dict[str, float] = {}
    _, T, _ = x.shape

    for w, H, head in zip(cfg.horizon_weights, cfg.horizons, heads):
        if H >= T:
            per_horizon[f"L_h{H}"] = 0.0
            continue
        h_pred = head(h[:, : T - H, :])           # (B, T-H, F)
        target = x[:, H:, :]                       # (B, T-H, F)
        valid = mask[:, : T - H] & mask[:, H:]     # (B, T-H)
        if not valid.any():
            per_horizon[f"L_h{H}"] = 0.0
            continue
        diff = (h_pred - target) ** 2              # (B, T-H, F)
        per_pos = diff.mean(dim=-1)                # (B, T-H)
        loss_h = (per_pos * valid.to(per_pos.dtype)).sum() / valid.to(per_pos.dtype).sum()
        total = total + w * loss_h
        per_horizon[f"L_h{H}"] = float(loss_h.item())

    return total, per_horizon


# ---------------------------------------------------------------------------
# LR schedule (linear warmup → cosine decay)
# ---------------------------------------------------------------------------
def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def evaluate(encoder: RoundEncoder, heads: nn.ModuleList, val_loader: DataLoader,
             cfg: TrainConfig, device: torch.device) -> dict:
    encoder.eval()
    heads.eval()
    total = 0.0
    n = 0
    horizon_sums: dict[str, float] = {f"L_h{H}": 0.0 for H in cfg.horizons}
    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss, per_h = compute_loss(encoder, heads, batch, cfg, device)
            total += float(loss.item())
            n += 1
            for k, v in per_h.items():
                horizon_sums[k] = horizon_sums.get(k, 0.0) + v
    if n == 0:
        return {"val_total": float("nan")}
    out = {"val_total": total / n}
    for k, v in horizon_sums.items():
        out[f"val_{k}"] = v / n
    return out


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"Device: {device}{dev_name}")

    train_ds = RoundDataset(DATA_DIR / "train.pt")
    val_ds = RoundDataset(DATA_DIR / "val.pt")
    cfg.feature_dim = train_ds.feature_dim
    print(f"Train: {len(train_ds)} rounds | Val: {len(val_ds)} rounds | "
          f"feature_dim={cfg.feature_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    encoder = RoundEncoder(cfg).to(device)
    heads = nn.ModuleList([
        ForwardPredHead(cfg.d_model, cfg.feature_dim).to(device)
        for _ in cfg.horizons
    ])
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_heads = sum(p.numel() for p in heads.parameters())
    print(f"Params: encoder {n_enc/1e6:.2f}M + heads {n_heads/1e6:.2f}M "
          f"= {(n_enc + n_heads)/1e6:.2f}M total")

    params = list(encoder.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.95))
    total_steps = max(1, cfg.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, cfg.warmup_steps, total_steps),
    )
    print(f"Total steps: {total_steps}, warmup: {cfg.warmup_steps}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a")
    print(f"Output: {out_dir}")
    print()
    print("=" * 60)
    print("Training start")
    print("=" * 60)

    step = 0
    best_val = float("inf")
    for epoch in range(cfg.epochs):
        encoder.train()
        heads.train()
        epoch_t0 = time.time()
        epoch_total = 0.0
        epoch_n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss, per_h = compute_loss(encoder, heads, batch, cfg, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_total += float(loss.item())
            epoch_n += 1

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                msg = {"step": step, "epoch": epoch, "loss": float(loss.item()),
                       "lr": lr, **per_h}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                horizon_str = " ".join(f"{k}={v:.4f}" for k, v in per_h.items())
                print(f"  [e{epoch} s{step}] loss={loss.item():.4f} lr={lr:.2e} "
                      f"{horizon_str}", flush=True)

        epoch_loss = epoch_total / max(1, epoch_n)
        epoch_dt = time.time() - epoch_t0

        val_metrics: dict = {}
        if (epoch + 1) % cfg.val_every_epochs == 0 or epoch == cfg.epochs - 1:
            val_metrics = evaluate(encoder, heads, val_loader, cfg, device)

        msg = {"step": step, "epoch": epoch, "epoch_train_loss": epoch_loss,
               "epoch_seconds": epoch_dt, **val_metrics}
        log_f.write(json.dumps(msg) + "\n")
        log_f.flush()
        val_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"  epoch {epoch} done: train_loss={epoch_loss:.4f} "
              f"({epoch_dt:.0f}s) {val_str}", flush=True)

        ckpt = {
            "epoch": epoch, "step": step,
            "encoder": encoder.state_dict(),
            "heads": heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(cfg),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        cur_val = val_metrics.get("val_total", float("inf"))
        if cur_val < best_val:
            best_val = cur_val
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  → new best val_total={best_val:.4f}", flush=True)

    log_f.close()
    print()
    print(f"Done. best val_total={best_val:.4f} | last.pt + best.pt at {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--horizon-weights", type=float, nargs="+", default=[1.0, 0.7, 0.4])
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", type=str, default=None,
                    help="output dir name (default: timestamped)")
    args = ap.parse_args()

    if len(args.horizon_weights) != len(args.horizons):
        ap.error(f"--horizons ({len(args.horizons)}) and --horizon-weights "
                 f"({len(args.horizon_weights)}) must be the same length")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / run_id

    cfg = TrainConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        horizons=args.horizons,
        horizon_weights=args.horizon_weights,
        log_every=args.log_every,
        seed=args.seed,
        output_dir=str(out_dir),
    )
    train(cfg)


if __name__ == "__main__":
    main()
