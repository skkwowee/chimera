#!/usr/bin/env python3
"""Train the Level-2 round encoder (causal decoder over per-tick features).

The encoder consumes one CS2 round at a time as a sequence of downsampled
(8 Hz) per-tick feature vectors, and is trained with four causal-honest
self-supervised objectives:

  1. Forward tick prediction         — h_T -> features at T+1   (MSE)
  2. Multi-horizon forward state     — h_T -> features at T+8, T+32  (MSE)
  3. Time-to-next-event regression   — h_T -> seconds-to-next-event  (Smooth L1)
  4. Next-event-type prediction      — h_T -> argmax over 6 types    (CE + smoothing)

ALL objectives are predict-forward (no future-leakage) and NONE consume
round_won (F2 collapse safety, hard constraint).

Two sizes: --smoke (~2-3M params, 256-dim/2-block, for the de-risk on 4 demos)
and --full (~50M params, 1024-dim/4-block, for the real run on 40+ demos).

Usage:
    python scripts/train_round_encoder.py --smoke
    python scripts/train_round_encoder.py --full --epochs 80
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Event types — must match build_tick_sequences.py EVENT_TYPES order
EVENT_TYPES = ["kill_T", "kill_CT", "plant", "defuse", "freeze_end", "round_end"]
N_EVENT_TYPES = len(EVENT_TYPES) + 1  # +1 for "no future event" sentinel
NO_EVENT_IDX = len(EVENT_TYPES)

# Forward-prediction horizons (in downsampled-tick units). 1 = next tick,
# 8 = ~1s ahead, 32 = ~4s ahead at 8 Hz downsample.
FORWARD_HORIZONS = (1, 8, 32)

# Time-to-next-event regression: predict seconds, clipped to this max
TIME_TO_EVENT_MAX = 30.0


@dataclass
class EncoderConfig:
    """Architecture + training hyperparameters."""

    # Architecture
    d_model: int = 1024
    n_blocks: int = 4
    n_heads: int = 8
    d_ff: int = 4096
    dropout: float = 0.15
    max_seq_len: int = 1024

    # Training
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.10
    grad_clip: float = 1.0

    # Loss weights (forward_tick, multi_horizon, time_to_event, next_event_type)
    loss_weights: tuple[float, float, float, float] = (1.0, 0.5, 0.5, 0.5)

    # Regularization
    player_slot_dropout: float = 0.10  # randomly zero entire player slots
    label_smoothing: float = 0.1

    # Logging / checkpointing
    log_every: int = 20
    val_every: int = 200
    save_every: int = 500


def smoke_config() -> EncoderConfig:
    """De-risk smoke-test config (~2-3M params)."""
    return EncoderConfig(
        d_model=256,
        n_blocks=2,
        n_heads=4,
        d_ff=1024,
        batch_size=16,
        epochs=200,
        log_every=10,
        val_every=50,
        save_every=200,
    )


def full_config() -> EncoderConfig:
    """Full-encoder config (~50M params)."""
    return EncoderConfig()  # defaults are the full spec


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention via SDPA."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, d_model); key_padding_mask: (B, L) bool, True = padding (ignore)
        b, l, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, head_dim)

        # Build attention mask: causal + padding
        # SDPA expects an additive mask of shape (B, H, L, L) or (L, L)
        # is_causal=True handles the causal part; padding mask must be added
        if key_padding_mask is not None:
            # (B, L) -> (B, 1, 1, L), float with -inf at padded keys
            attn_mask = torch.zeros(b, 1, 1, l, device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            # SDPA: when both is_causal and attn_mask given, only attn_mask is used,
            # so we have to build the causal mask manually.
            causal = torch.triu(
                torch.full((l, l), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
            )
            attn_mask = attn_mask + causal[None, None, :, :]
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(b, l, -1)
        x = x + self.dropout(self.attn_out(attn_out))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


def _sinusoidal_position_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encoding [seq_len, d_model]."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


def _time_encoding(tick_seconds: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding over round-relative seconds [B, L, d_model]."""
    device = tick_seconds.device
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    # tick_seconds: (B, L) -> (B, L, 1)
    pos = tick_seconds.unsqueeze(-1)  # (B, L, 1)
    enc = torch.zeros(*tick_seconds.shape, d_model, device=device)
    enc[..., 0::2] = torch.sin(pos * div_term)
    enc[..., 1::2] = torch.cos(pos * div_term)
    return enc


class RoundDecoder(nn.Module):
    """Causal decoder over per-tick features.

    Input:  (B, L, input_dim) per-tick features + (B, L) round-relative seconds
    Output: (B, L, d_model) per-tick contextualized hidden states + 4 prediction heads
    """

    def __init__(self, input_dim: int, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.ln_out = nn.LayerNorm(cfg.d_model)

        # Prediction heads (each a small MLP)
        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, out_dim),
            )

        self.head_forward_tick = _head(input_dim)        # predict raw features at T+1
        self.head_forward_h8 = _head(input_dim)          # predict features at T+8
        self.head_forward_h32 = _head(input_dim)         # predict features at T+32
        self.head_time_to_event = _head(1)               # scalar: seconds to next event
        self.head_next_event_type = _head(N_EVENT_TYPES)  # logits over event types

    def forward(
        self,
        features: torch.Tensor,
        tick_seconds: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # features: (B, L, F); tick_seconds: (B, L); key_padding_mask: (B, L) bool
        b, l, _ = features.shape
        x = self.input_proj(features)
        x = x + _sinusoidal_position_encoding(l, self.cfg.d_model, features.device)[None, :, :]
        x = x + _time_encoding(tick_seconds, self.cfg.d_model)
        x = self.input_dropout(x)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        h = self.ln_out(x)  # (B, L, d_model)

        return {
            "hidden": h,
            "pred_t1": self.head_forward_tick(h),       # (B, L, F)
            "pred_h8": self.head_forward_h8(h),         # (B, L, F)
            "pred_h32": self.head_forward_h32(h),       # (B, L, F)
            "pred_time": self.head_time_to_event(h).squeeze(-1),  # (B, L)
            "pred_event_type": self.head_next_event_type(h),  # (B, L, N_EVENT_TYPES)
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TickSequenceDataset(Dataset):
    """Wraps the list-of-round-dicts produced by build_tick_sequences.py.

    Pre-computes per-position labels: time-to-next-event and next-event-type.
    """

    def __init__(self, rounds_path: Path, target_hz: int = 8):
        self.rounds: list[dict] = torch.load(rounds_path, weights_only=False)
        self.target_hz = target_hz
        # Pre-compute event labels per round
        for r in self.rounds:
            tick_indices = r["tick_indices"]  # (L,)
            events = r["events"]
            r["time_to_event"], r["next_event_type"] = self._compute_event_labels(
                tick_indices, events
            )

    @staticmethod
    def _compute_event_labels(
        tick_indices: torch.Tensor, events: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each position, find seconds-to-next-event and next-event-type."""
        L = tick_indices.shape[0]
        time_to = torch.full((L,), float("nan"))  # nan = no future event (mask out)
        type_idx = torch.full((L,), -1, dtype=torch.long)  # -1 = ignore in CE loss

        if not events:
            return time_to, type_idx

        # Walk positions and find next event > current tick
        ev_ticks = [e["tick"] for e in events]
        ev_types = [EVENT_TYPES.index(e["type"]) for e in events]
        ev_idx = 0
        for i in range(L):
            t = int(tick_indices[i].item())
            # Advance ev_idx to the first event strictly after t
            while ev_idx < len(events) and ev_ticks[ev_idx] <= t:
                ev_idx += 1
            if ev_idx < len(events):
                dt_seconds = (ev_ticks[ev_idx] - t) / 64.0
                time_to[i] = min(dt_seconds, TIME_TO_EVENT_MAX)
                type_idx[i] = ev_types[ev_idx]
            else:
                # No future event: label as "no event" sentinel for the type CE,
                # leave time-to NaN (mask out in regression loss)
                type_idx[i] = NO_EVENT_IDX

        return time_to, type_idx

    def __len__(self) -> int:
        return len(self.rounds)

    def __getitem__(self, idx: int) -> dict:
        return self.rounds[idx]


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length rounds to the longest in the batch."""
    L_max = max(r["features"].shape[0] for r in batch)
    F = batch[0]["features"].shape[1]
    B = len(batch)

    features = torch.zeros(B, L_max, F)
    tick_seconds = torch.zeros(B, L_max)
    time_to_event = torch.full((B, L_max), float("nan"))
    next_event_type = torch.full((B, L_max), -1, dtype=torch.long)
    key_padding_mask = torch.ones(B, L_max, dtype=torch.bool)  # True = pad

    for i, r in enumerate(batch):
        L = r["features"].shape[0]
        features[i, :L] = r["features"]
        tick_seconds[i, :L] = r["tick_seconds"]
        time_to_event[i, :L] = r["time_to_event"]
        next_event_type[i, :L] = r["next_event_type"]
        key_padding_mask[i, :L] = False

    return {
        "features": features,
        "tick_seconds": tick_seconds,
        "time_to_event": time_to_event,
        "next_event_type": next_event_type,
        "key_padding_mask": key_padding_mask,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: EncoderConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the four objectives. Returns (total_loss, per-loss-dict for logging)."""
    features = batch["features"]               # (B, L, F)
    valid = ~batch["key_padding_mask"]         # (B, L) True where real
    B, L, F_dim = features.shape

    losses: dict[str, torch.Tensor] = {}

    # --- 1. Forward tick prediction (T -> T+1) ---
    # Target: features shifted left by 1. Last position has no target.
    target_t1 = torch.zeros_like(features)
    target_t1[:, :-1] = features[:, 1:]
    valid_t1 = valid.clone()
    valid_t1[:, -1:] = False
    valid_t1 = valid_t1 & valid.roll(-1, dims=1)  # also need T+1 to be real
    diff = (out["pred_t1"] - target_t1) ** 2
    diff = diff.mean(dim=-1)  # (B, L)
    if valid_t1.any():
        losses["forward_tick"] = diff[valid_t1].mean()
    else:
        losses["forward_tick"] = torch.zeros((), device=features.device)

    # --- 2. Multi-horizon forward state (T -> T+8, T+32) ---
    horizon_loss = 0.0
    horizon_count = 0
    for horizon, pred_key in zip((8, 32), ("pred_h8", "pred_h32")):
        if L <= horizon:
            continue
        target = features.roll(-horizon, dims=1)
        valid_h = valid.clone()
        valid_h[:, -horizon:] = False
        # Also require target position to be real
        valid_target = valid.roll(-horizon, dims=1)
        valid_target[:, -horizon:] = False
        valid_h = valid_h & valid_target
        diff = (out[pred_key] - target) ** 2
        diff = diff.mean(dim=-1)
        if valid_h.any():
            horizon_loss = horizon_loss + diff[valid_h].mean()
            horizon_count += 1
    if horizon_count > 0:
        losses["multi_horizon"] = horizon_loss / horizon_count
    else:
        losses["multi_horizon"] = torch.zeros((), device=features.device)

    # --- 3. Time-to-next-event regression (Smooth L1) ---
    time_target = batch["time_to_event"]  # (B, L), NaN where no future event
    time_valid = valid & ~torch.isnan(time_target)
    if time_valid.any():
        pred_time = out["pred_time"]
        # Normalize by TIME_TO_EVENT_MAX for stable regression
        pred_norm = pred_time / TIME_TO_EVENT_MAX
        target_norm = time_target / TIME_TO_EVENT_MAX
        # Replace NaN in target with 0 (will be masked out anyway)
        target_norm = torch.where(time_valid, target_norm, torch.zeros_like(target_norm))
        l1 = F.smooth_l1_loss(pred_norm, target_norm, reduction="none")
        losses["time_to_event"] = l1[time_valid].mean()
    else:
        losses["time_to_event"] = torch.zeros((), device=features.device)

    # --- 4. Next-event-type prediction (CE) ---
    event_target = batch["next_event_type"]  # (B, L), -1 where ignore
    type_valid = valid & (event_target >= 0)
    if type_valid.any():
        # Cross-entropy with ignore_index=-1
        logits = out["pred_event_type"]  # (B, L, N_EVENT_TYPES)
        ce = F.cross_entropy(
            logits.reshape(-1, N_EVENT_TYPES),
            event_target.reshape(-1),
            ignore_index=-1,
            label_smoothing=cfg.label_smoothing,
        )
        losses["next_event_type"] = ce
    else:
        losses["next_event_type"] = torch.zeros((), device=features.device)

    # Weighted sum
    w_tick, w_horizon, w_time, w_type = cfg.loss_weights
    total = (
        w_tick * losses["forward_tick"]
        + w_horizon * losses["multi_horizon"]
        + w_time * losses["time_to_event"]
        + w_type * losses["next_event_type"]
    )

    return total, {k: float(v.item()) for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Per-player slot dropout (input-side regularization)
# ---------------------------------------------------------------------------

PER_PLAYER_DIM = 30
N_PLAYER_SLOTS = 10


def apply_player_slot_dropout(features: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Randomly zero entire player slots (per round, per slot) with probability p.

    features: (B, L, F=N_player_slots*per_player_dim + global_dim)
    """
    if not training or p <= 0:
        return features
    B, L, F = features.shape
    # Per-batch, per-slot dropout decision (consistent across the round)
    # Shape: (B, N_PLAYER_SLOTS) -> mask each slot's chunk in features
    mask = (torch.rand(B, N_PLAYER_SLOTS, device=features.device) > p).float()
    # Build a (B, F) feature-level mask: 1 for global, dropout for player slots
    feat_mask = torch.ones(B, F, device=features.device)
    for slot in range(N_PLAYER_SLOTS):
        start = slot * PER_PLAYER_DIM
        end = start + PER_PLAYER_DIM
        feat_mask[:, start:end] = mask[:, slot:slot + 1]
    return features * feat_mask[:, None, :]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_at_step(step: int, total_steps: int, peak_lr: float, warmup_frac: float, min_lr_frac: float) -> float:
    warmup_steps = max(1, int(warmup_frac * total_steps))
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace, cfg: EncoderConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = Path(args.data_dir)
    schema_path = data_dir / "feature_schema_v1.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"feature schema not found at {schema_path}; run build_tick_sequences.py first")
    schema = json.loads(schema_path.read_text())
    input_dim = schema["feature_dim"]
    print(f"Feature dim: {input_dim}")

    train_ds = TickSequenceDataset(data_dir / "train.pt")
    val_ds = TickSequenceDataset(data_dir / "val.pt")
    print(f"Train rounds: {len(train_ds)}, val rounds: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=1
    )

    # Model
    model = RoundDecoder(input_dim=input_dim, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.2f}M params ({n_trainable/1e6:.2f}M trainable)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs * len(train_loader) // cfg.grad_accum)
    print(f"Total optimizer steps: {total_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({**cfg.__dict__, "input_dim": input_dim}, indent=2, default=str))

    log_path = output_dir / "train_log.jsonl"
    log_f = open(log_path, "a")

    use_amp = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    step = 0
    accum_count = 0
    accum_loss = 0.0
    accum_components: dict[str, float] = {k: 0.0 for k in ("forward_tick", "multi_horizon", "time_to_event", "next_event_type")}
    optimizer.zero_grad()
    t_step_start = time.time()
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Player-slot dropout (input-side regularization)
            batch["features"] = apply_player_slot_dropout(
                batch["features"], cfg.player_slot_dropout, model.training
            )

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                out = model(batch["features"], batch["tick_seconds"], batch["key_padding_mask"])
                loss, components = compute_losses(out, batch, cfg)

            (loss / cfg.grad_accum).backward()
            accum_loss += loss.item()
            for k in accum_components:
                accum_components[k] += components[k]
            accum_count += 1

            if accum_count >= cfg.grad_accum:
                # LR schedule
                lr = lr_at_step(step, total_steps, cfg.lr, cfg.warmup_frac, cfg.min_lr_frac)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % cfg.log_every == 0:
                    elapsed = time.time() - t_step_start
                    avg_loss = accum_loss / accum_count
                    avg_comp = {k: v / accum_count for k, v in accum_components.items()}
                    rec = {
                        "step": step,
                        "epoch": epoch,
                        "lr": lr,
                        "loss": avg_loss,
                        "grad_norm": float(grad_norm),
                        "elapsed_s": elapsed,
                        **avg_comp,
                    }
                    print(
                        f"step {step}/{total_steps} | loss {avg_loss:.4f} "
                        f"| fwd {avg_comp['forward_tick']:.4f} h_pool {avg_comp['multi_horizon']:.4f} "
                        f"time {avg_comp['time_to_event']:.4f} type {avg_comp['next_event_type']:.4f} "
                        f"| lr {lr:.2e} grad {float(grad_norm):.2f} | {elapsed:.1f}s"
                    )
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    t_step_start = time.time()

                if step % cfg.val_every == 0:
                    val_loss = evaluate(model, val_loader, cfg, device, autocast_dtype, use_amp)
                    print(f"  [val] step {step}: loss {val_loss:.4f}")
                    log_f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                    log_f.flush()
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(model, cfg, output_dir / "best.pt", step, val_loss)
                    model.train()

                if step % cfg.save_every == 0:
                    save_checkpoint(model, cfg, output_dir / f"checkpoint_{step}.pt", step, None)

                # Reset accumulators
                accum_count = 0
                accum_loss = 0.0
                accum_components = {k: 0.0 for k in accum_components}

    save_checkpoint(model, cfg, output_dir / "final.pt", step, None)
    log_f.close()
    print(f"\nDone. Best val loss: {best_val:.4f}. Outputs in {output_dir}")


@torch.no_grad()
def evaluate(model, loader, cfg, device, autocast_dtype, use_amp) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            out = model(batch["features"], batch["tick_seconds"], batch["key_padding_mask"])
            loss, _ = compute_losses(out, batch, cfg)
        total += loss.item()
        n += 1
    return total / max(1, n)


def save_checkpoint(model, cfg, path: Path, step: int, val_loss: float | None) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": {**cfg.__dict__, "_step": step, "_val_loss": val_loss},
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/tick_sequences"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/round_encoder"))
    ap.add_argument("--smoke", action="store_true", help="use smoke-test config (~2-3M params)")
    ap.add_argument("--full", action="store_true", help="use full config (~50M params)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.smoke or args.full):
        ap.error("specify --smoke or --full")
    cfg = smoke_config() if args.smoke else full_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Config: {'smoke' if args.smoke else 'full'}")
    print(f"  d_model={cfg.d_model}, n_blocks={cfg.n_blocks}, n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"  batch={cfg.batch_size}, grad_accum={cfg.grad_accum}, epochs={cfg.epochs}, lr={cfg.lr}")
    print()

    train(args, cfg)


if __name__ == "__main__":
    main()
