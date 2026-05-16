#!/usr/bin/env python3
"""Train the Level-2 round encoder (v5: causal decoder + event objectives).

Per docs/round-encoder-design.md v4 architecture, v5 SSL objectives:
  - Round-as-sequence-of-downsampled-ticks (8 Hz, ~960 tokens per round)
  - Causal self-attention so h_T is a function of ticks 0..T only (F2-safe)
  - All SSL objectives are C1-clean (never consume round_won)

SSL objectives (5 in total):
  - L_h1, L_h8, L_h32: MSE on the input feature vector H ticks ahead
                        (multi-horizon forward prediction — v4 inherits)
  - L_event_ce:        7-way CE on next-event type within EVENT_HORIZON_TICKS
                        ({kill_t, kill_ct, bomb_planted, bomb_defused,
                          bomb_exploded, round_end, none}). "none" is
                        down-weighted to keep the rare events visible.
  - L_time_to_event:   Huber regression on log(1+ticks) until next event.
                        Masked to non-"none" positions only — otherwise the
                        model trivially learns "predict the cap everywhere".

v4 (forward-pred only) failed σ_s gate because forward prediction alone
clusters states by visual continuity, not strategic structure. Adding
event objectives forces the encoder to differentiate states by what's
about to happen — kills, plants, etc. — which is exactly the signal a
downstream RL policy needs.

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

    # Event objectives (v5)
    n_event_classes: int = 7  # matches EVENT_VOCAB
    none_event_idx: int = 6
    event_ce_weight: float = 1.0       # weight in total loss
    event_time_weight: float = 0.5     # weight in total loss (regression scale)
    none_class_weight: float = 0.1     # CE weight for the dominant "none" class
    event_horizon_ticks: float = 256.0 # for log-time normalization (raw 64Hz)

    # v7: salience-biased causal attention. When True, each tick learns a
    # scalar salience score; that score is added as additive bias to
    # attention logits for that tick as a KEY. Net effect: model can
    # learn to weight ticks variably (freeze-time low salience, kills/
    # plants high salience) without hand-specifying which ticks matter.
    use_salience: bool = False

    # v8: window-forecast SSL — at tick T predict aggregates over the next
    # WINDOW_DOWNSAMPLED ticks (default 16 = 2s at 8Hz). Forces per-tick
    # output to encode "what's about to happen in this window" instead of
    # just "what's the very next event". Targets:
    #   - kills_t_in_window (count, regression)
    #   - kills_ct_in_window (count, regression)
    #   - any_plant_in_window (binary)
    #   - any_defuse_in_window (binary)
    #   - any_explode_in_window (binary)
    #   - any_round_end_in_window (binary)
    # Labels derived at training time from the per-tick event_labels list.
    window_forecast_ticks: int = 16  # downsampled ticks; at 8Hz = 2 seconds
    window_forecast_weight: float = 0.3
    use_window_forecast: bool = False

    # v9: change-point segmentation. Per-tick boundary head predicts where
    # an "event" ends; cumsum of boundaries → segment ID per tick. Aggregate
    # head forecasts per-segment event counts. Density penalty around a
    # TARGET density (not just minimization — that collapses to 0 boundaries
    # with 1 giant segment per round). Forecast loss prevents over-merging.
    # Equilibrium: boundaries at real state transitions, ~target_density.
    #
    # Initialization: boundary head bias starts at logit(target_density) so
    # the model begins around the desired regime instead of having to climb
    # there against the density-around-target gradient.
    use_change_point: bool = False
    change_point_max_segments: int = 64
    change_point_weight: float = 1.0
    change_point_target_density: float = 0.02
    # MSE around the target — penalty = weight * (mean(b_prob) - target)^2.
    # Pulls density toward target from either side; stable equilibrium.
    change_point_density_weight: float = 5.0

    # v10: change-point loss variants. v9 (forecast) gave the boundary head
    # gradient ONLY from density penalty (seg_id is detached after STE).
    # That left placement up to chance + the auxiliary signal of the
    # density target, hence anti-correlated boundaries. v10 variants add
    # an aux "boundary at large state-change" term that gives DIRECT
    # gradient to b_prob, then replace the segment-level objective with
    # one that's better aligned with "natural" event boundaries:
    #   forecast    — v9 baseline (per-segment event counts)
    #   contrastive — InfoNCE: same-segment ticks pull together
    #   variance    — intra-segment h-variance penalty (k-means-style)
    #   outcome     — forecast events in the NEXT segment (shifted target)
    change_point_loss: str = "forecast"
    change_point_aux_align_weight: float = 0.0
    contrastive_temperature: float = 0.1

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
    """Wraps the list of per-round (T, F) tensors from train.pt / val.pt.

    For v5+ datasets, also exposes per-tick event labels and times (used by
    the event-prediction SSL heads). Falls back gracefully if a v4 blob is
    loaded — event_labels/event_times become None, and the trainer skips the
    event loss.
    """

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, weights_only=False)
        self.tensors: list[torch.Tensor] = blob["tensors"]
        self.metas: list[dict] = blob["metas"]
        self.feature_dim: int = int(blob["feature_dim"])
        self.event_labels: list[torch.Tensor] | None = blob.get("event_labels")
        self.event_times: list[torch.Tensor] | None = blob.get("event_times")
        self.event_vocab: list[str] | None = blob.get("event_vocab")

    def has_events(self) -> bool:
        return self.event_labels is not None and self.event_times is not None

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict, torch.Tensor | None,
                                           torch.Tensor | None]:
        ev_l = self.event_labels[i] if self.event_labels is not None else None
        ev_t = self.event_times[i] if self.event_times is not None else None
        return self.tensors[i], self.metas[i], ev_l, ev_t


def collate(batch: list[tuple[torch.Tensor, dict, torch.Tensor | None, torch.Tensor | None]]) -> dict:
    """Pad rounds to max-T-in-batch; build attention mask (1=real, 0=pad)."""
    tensors, metas, ev_lbls, ev_times = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    T_max = int(lengths.max().item())
    F_dim = tensors[0].shape[1]
    B = len(tensors)
    out = torch.zeros(B, T_max, F_dim, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    has_events = ev_lbls[0] is not None
    if has_events:
        ev_l_pad = torch.full((B, T_max), -100, dtype=torch.long)  # -100 = ignore_index
        ev_t_pad = torch.zeros(B, T_max, dtype=torch.float32)
    for i, t in enumerate(tensors):
        T_i = t.shape[0]
        out[i, : T_i] = t
        mask[i, : T_i] = True
        if has_events:
            ev_l_pad[i, : T_i] = ev_lbls[i]
            ev_t_pad[i, : T_i] = ev_times[i]
    res = {"x": out, "mask": mask, "lengths": lengths, "metas": list(metas)}
    if has_events:
        res["event_labels"] = ev_l_pad
        res["event_times"] = ev_t_pad
    return res


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
# Salience-biased causal attention block (v7)
# ---------------------------------------------------------------------------
class SalienceCausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention + per-key salience.

    Standard MHA computes attention logits = (Q @ K.T) / sqrt(d). We add a
    learned per-key scalar `salience[K]` to those logits, so when any query
    Q attends, it sees keys K weighted by both content similarity AND a
    content-derived salience score. The salience score is computed once
    (per-tick, shared across layers + heads + queries) from the post-input
    projected representation — so the encoder learns "ticks where the
    state changes meaningfully" without hand-specifying event boundaries.

    Bias is additive in log-space; sigmoid would have been multiplicative
    in probability-space but that bounds salience to [0,1] which is more
    restrictive than what the model can express here.

    Causal mask + padding mask are composed into the same additive bias.
    Uses F.scaled_dot_product_attention so it dispatches to fused kernels
    where available.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                              # (B, T, D)
        salience: torch.Tensor,                       # (B, T) — per-key bias
        key_padding_mask: torch.Tensor | None = None, # (B, T) — True = PAD
    ) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (B, n_heads, T, d_head)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Combined additive mask: (B, 1, T, T)
        # - causal triangle: future positions get -inf
        # - per-key salience bias: broadcast across queries
        # - padding: PAD keys get -inf
        neg_inf = torch.finfo(q.dtype).min
        causal = torch.full((T, T), neg_inf, device=x.device, dtype=q.dtype)
        causal = torch.triu(causal, diagonal=1)   # (T, T)
        bias = salience.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        if key_padding_mask is not None:
            pad_bias = torch.where(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                neg_inf, 0.0,
            ).to(q.dtype)                          # (B, 1, 1, T)
            bias = bias + pad_bias
        attn_mask = causal.unsqueeze(0).unsqueeze(0) + bias  # (B, 1, T, T)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = x + self.drop(out)

        # Pre-norm FFN
        h2 = self.norm2(x)
        x = x + self.drop(self.ffn(h2))
        return x


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
        self.use_salience = bool(getattr(cfg, "use_salience", False))
        self.input_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.norm_in = nn.LayerNorm(cfg.d_model)
        self.pos_emb = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        if self.use_salience:
            self.layers = nn.ModuleList([
                SalienceCausalBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ])
            # Salience head: content-derived per-tick scalar. Computed once
            # from input projection, shared across layers + queries + heads.
            self.salience_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )
        else:
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
        if self.use_salience:
            # Per-tick salience score; padding positions get -inf so they
            # neither attract nor get attended (combined with key_padding
            # bias inside the block).
            salience = self.salience_head(h).squeeze(-1)   # (B, T)
            for layer in self.layers:
                h = layer(h, salience=salience,
                          key_padding_mask=key_padding_mask)
        else:
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

    def compute_salience(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return per-tick salience scores for diagnostic/visualization use.
        None when the encoder wasn't trained with salience."""
        if not self.use_salience:
            return None
        h = self.norm_in(self.input_proj(x))
        h = self.pos_emb(h)
        return self.salience_head(h).squeeze(-1)


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


class NextEventHead(nn.Module):
    """Predicts the type of the next event (within EVENT_HORIZON_TICKS)."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class TimeToEventHead(nn.Module):
    """Predicts ticks-until-next-event in log space (real-valued scalar)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class WindowForecastHead(nn.Module):
    """Predicts an aggregate over the next-W-tick window from h_T.

    Output dims (6):
      0: kills_t_count       (regression, target log1p)
      1: kills_ct_count      (regression, target log1p)
      2: any_plant_in_window (binary, sigmoid)
      3: any_defuse_in_window (binary)
      4: any_explode_in_window (binary)
      5: any_round_end_in_window (binary)

    The model is forced to encode "what's about to happen in this window",
    not just "what's the very next event". A retake-style situation looks
    structurally different from a save-style situation in terms of these
    window aggregates, even if the very next event is the same in both.
    """

    OUT_DIM = 6
    KILLS_T = 0
    KILLS_CT = 1
    BIN_FIRST = 2  # indices [2..5] are binary

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.OUT_DIM),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ChangePointHead(nn.Module):
    """Learned event segmentation: per-tick boundary head + per-segment forecast.

    Boundary head produces a sigmoid probability b_T at each tick. Hard
    boundaries (b_T > 0.5) are used for segmentation in forward pass; the
    gradient flows through the soft sigmoid via straight-through estimator.

    Segment IDs are cumsum of hard boundaries (T,) → values in [0, max_segments).
    Per-segment embedding = mean of h within the segment (scatter_mean).
    Forecast head predicts per-segment aggregate counts of {kill_t, kill_ct,
    plant, defuse, explode, round_end}.

    Density penalty on b_prob.mean() keeps boundaries from collapsing to
    "boundary every tick" (degenerate). The forecast loss keeps boundaries
    from collapsing to "no boundaries" (one giant segment → poor per-event
    aggregation).

    Equilibrium: boundaries emerge at real state transitions where keeping
    the prior ticks together costs more forecast loss than the marginal
    boundary-density cost.
    """

    OUT_DIM = WindowForecastHead.OUT_DIM
    KILLS_T = WindowForecastHead.KILLS_T
    KILLS_CT = WindowForecastHead.KILLS_CT
    BIN_FIRST = WindowForecastHead.BIN_FIRST

    def __init__(self, d_model: int, max_segments: int = 64,
                 init_target_density: float = 0.02):
        super().__init__()
        self.max_segments = max_segments
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Initialize boundary bias so the model STARTS around the target
        # density (logit(p) = log(p/(1-p))). Without this, with zero-init
        # sigmoid(0)=0.5 we'd start with half the ticks being boundaries —
        # the model collapses straight to "1 giant segment per round" as
        # the first move because the per-batch density penalty dominates
        # the per-segment forecast signal at start.
        p = max(min(init_target_density, 0.99), 1e-4)
        init_bias = float(math.log(p / (1.0 - p)))
        with torch.no_grad():
            self.boundary_head[-1].bias.fill_(init_bias)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.OUT_DIM),
        )

    def forward(
        self,
        h: torch.Tensor,             # (B, T, d_model)
        key_padding_mask: torch.Tensor | None = None,  # (B, T) True=PAD
    ) -> dict:
        """Returns dict with:
          'b_logits':   (B, T)  raw boundary logits
          'b_prob':     (B, T)  sigmoid(logits) — soft probability
          'seg_id':     (B, T)  long, segment index per tick (after STE)
          'seg_pred':   (B, max_segments, OUT_DIM) per-segment forecast
          'seg_mask':   (B, max_segments) True for segments that have ticks
        """
        B, T, D = h.shape
        b_logits = self.boundary_head(h).squeeze(-1)  # (B, T)
        b_prob = torch.sigmoid(b_logits)
        # Discrete sample. Training: Bernoulli (stochastic, matches density);
        # eval: top-K positions where K ≈ E[b_prob] × T (reproducible, same
        # expected segment count). Threshold>0.5 would under-sample whenever
        # the target density is below 0.5 (always, in our setup).
        if self.training:
            b_hard = torch.bernoulli(b_prob)
        else:
            # Pick the round-mean-density top-K positions per batch row
            k = int(torch.clamp(b_prob.mean(dim=1).mean() * T, min=1.0,
                                  max=float(T - 1)).item())
            # topk indices, then scatter 1.0
            _, idx = torch.topk(b_prob, k, dim=1)
            b_hard = torch.zeros_like(b_prob)
            b_hard.scatter_(1, idx, 1.0)
        # Straight-through estimator: forward uses hard, backward uses soft.
        b_ste = b_hard.detach() + b_prob - b_prob.detach()  # (B, T)

        # Segment IDs = cumsum of boundaries up to (and including) this tick.
        # Each tick belongs to the segment that "starts" after the previous
        # boundary. Convention: tick 0 is segment 0; first boundary opens
        # segment 1; etc.
        seg_id_soft = torch.cumsum(b_ste, dim=1)  # (B, T) float, gradient-carrying
        seg_id = seg_id_soft.detach().long().clamp(max=self.max_segments - 1)

        # Pool h within each segment (scatter_mean).
        seg_pred = self._segment_pool_and_forecast(h, seg_id, key_padding_mask)
        seg_mask = self._segment_mask(seg_id, key_padding_mask)
        return {
            "b_logits": b_logits,
            "b_prob": b_prob,
            "b_ste": b_ste,
            "seg_id": seg_id,
            "seg_pred": seg_pred,
            "seg_mask": seg_mask,
        }

    def _segment_pool_and_forecast(
        self,
        h: torch.Tensor,             # (B, T, D)
        seg_id: torch.Tensor,        # (B, T) long
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, D = h.shape
        S = self.max_segments
        sum_h = torch.zeros(B, S, D, device=h.device, dtype=h.dtype)
        cnt = torch.zeros(B, S, 1, device=h.device, dtype=h.dtype)
        if key_padding_mask is not None:
            weight = (~key_padding_mask).to(h.dtype).unsqueeze(-1)  # (B, T, 1)
        else:
            weight = torch.ones(B, T, 1, device=h.device, dtype=h.dtype)
        sum_h.scatter_add_(1, seg_id.unsqueeze(-1).expand(-1, -1, D), h * weight)
        cnt.scatter_add_(1, seg_id.unsqueeze(-1), weight)
        pooled = sum_h / cnt.clamp(min=1.0)  # (B, S, D)
        return self.forecast_head(pooled)             # (B, S, OUT_DIM)

    def _segment_mask(
        self,
        seg_id: torch.Tensor,        # (B, T) long
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """True where the segment has at least one non-pad tick."""
        B, T = seg_id.shape
        S = self.max_segments
        cnt = torch.zeros(B, S, device=seg_id.device, dtype=torch.float32)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float()
        else:
            valid = torch.ones(B, T, device=seg_id.device, dtype=torch.float32)
        cnt.scatter_add_(1, seg_id, valid)
        return cnt > 0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _build_window_forecast_targets(
    ev_labels: torch.Tensor,    # (B, T)
    mask: torch.Tensor,          # (B, T)
    window: int,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each tick T predict aggregates over (T+1, T+window].

    Targets shape (B, T, 6): kills_t_count, kills_ct_count, plant, defuse,
    explode, round_end. Valid mask (B, T): True where (T+window) is within
    the round AND the position itself is real (not pad).

    Built once per batch using vectorized scatter/sum. Cheap.
    """
    B, T = ev_labels.shape
    device = ev_labels.device

    # One-hot the event types per tick (B, T, 7) — we then sum/max over window.
    # int64 → onehot float32. Pad / -100 positions are masked later.
    safe = ev_labels.clamp(min=0)  # avoid -100 indexing
    onehot = nn.functional.one_hot(safe, num_classes=cfg.n_event_classes).float()  # (B, T, 7)
    # Zero out pad rows so they contribute nothing to window sums
    onehot = onehot * mask.float().unsqueeze(-1)

    # Window aggregation: for each position p, sum over (p+1)..(p+window).
    # Use cumulative sum then difference.
    cs = torch.cumsum(onehot, dim=1)  # (B, T, 7)
    # window_sum[p] = cs[p+window] - cs[p], clamped at T-1
    end_idx = torch.clamp(torch.arange(T, device=device) + window, max=T - 1)
    win_sum = cs[:, end_idx, :] - cs                       # (B, T, 7)

    # Build target (B, T, 6)
    tgt = torch.zeros(B, T, WindowForecastHead.OUT_DIM, device=device, dtype=torch.float32)
    tgt[:, :, WindowForecastHead.KILLS_T] = win_sum[:, :, 0]   # kill_t
    tgt[:, :, WindowForecastHead.KILLS_CT] = win_sum[:, :, 1]  # kill_ct
    # Binary flags from "any" → sum > 0
    tgt[:, :, WindowForecastHead.BIN_FIRST + 0] = (win_sum[:, :, 2] > 0).float()  # plant
    tgt[:, :, WindowForecastHead.BIN_FIRST + 1] = (win_sum[:, :, 3] > 0).float()  # defuse
    tgt[:, :, WindowForecastHead.BIN_FIRST + 2] = (win_sum[:, :, 4] > 0).float()  # explode
    tgt[:, :, WindowForecastHead.BIN_FIRST + 3] = (win_sum[:, :, 5] > 0).float()  # round_end

    # Valid: position must be real AND window must not extend past the round.
    # We require the full window to land inside non-pad ticks.
    valid_window = torch.zeros(B, T, dtype=torch.bool, device=device)
    # End-of-window must be a real position (mask[end_idx]) AND start must be too
    valid_window = mask & mask[:, end_idx]
    # Also exclude the very last `window` ticks where the window would be truncated
    # (cumsum-diff approach clamps, so labels would be biased low — easier to mask)
    if window < T:
        valid_window[:, -window:] = False
    return tgt, valid_window


def _contrastive_segment_loss(
    h: torch.Tensor,          # (B, T, D)
    seg_id: torch.Tensor,     # (B, T) long, detached
    mask: torch.Tensor,       # (B, T) True=real
    temperature: float,
) -> torch.Tensor:
    """InfoNCE: for each anchor tick, positives are other ticks in the same
    segment (and same round); negatives are all other valid ticks. Maximizes
    intra-segment cosine similarity and minimizes cross-segment similarity.

    Gradient through h goes to the encoder; gradient through which ticks
    are positives is blocked by the detached seg_id (the auxiliary
    boundary-alignment term handles boundary-head gradient separately).
    """
    B, T, D = h.shape
    h_norm = nn.functional.normalize(h, dim=-1)
    sim = torch.bmm(h_norm, h_norm.transpose(1, 2)) / temperature  # (B, T, T)
    eye = torch.eye(T, device=h.device, dtype=torch.bool).unsqueeze(0)
    valid_keys = mask.unsqueeze(1)         # (B, 1, T)
    valid_anchors = mask.unsqueeze(2)      # (B, T, 1)
    same_seg = (seg_id.unsqueeze(1) == seg_id.unsqueeze(2))  # (B, T, T)
    pos_mask = same_seg & valid_keys & valid_anchors & (~eye)
    all_mask = valid_keys & valid_anchors & (~eye)
    neg_inf = torch.finfo(sim.dtype).min
    sim_all = sim.masked_fill(~all_mask, neg_inf)
    sim_pos = sim.masked_fill(~pos_mask, neg_inf)
    log_p_all = torch.logsumexp(sim_all, dim=-1)  # (B, T)
    log_p_pos = torch.logsumexp(sim_pos, dim=-1)  # (B, T)
    has_pos = pos_mask.any(dim=-1)
    valid_anchor = mask & has_pos
    if not valid_anchor.any():
        return torch.zeros((), device=h.device, dtype=h.dtype)
    nll = -(log_p_pos - log_p_all)
    return nll[valid_anchor].mean()


def _intra_segment_variance(
    h: torch.Tensor,          # (B, T, D)
    seg_id: torch.Tensor,     # (B, T) long, detached
    mask: torch.Tensor,       # (B, T)
    max_segments: int,
) -> torch.Tensor:
    """Mean squared deviation of each tick's h from its segment's centroid.
    K-means objective with fixed (STE) assignments. Encoder is rewarded for
    placing ticks of the same segment near each other in embedding space.
    """
    B, T, D = h.shape
    S = max_segments
    weight = mask.to(h.dtype).unsqueeze(-1)         # (B, T, 1)
    sum_h = torch.zeros(B, S, D, device=h.device, dtype=h.dtype)
    cnt = torch.zeros(B, S, 1, device=h.device, dtype=h.dtype)
    sid_full = seg_id.unsqueeze(-1).expand(-1, -1, D)
    sum_h.scatter_add_(1, sid_full, h * weight)
    cnt.scatter_add_(1, seg_id.unsqueeze(-1), weight)
    mean_h = sum_h / cnt.clamp(min=1.0)              # (B, S, D)
    mean_per_tick = torch.gather(mean_h, 1, sid_full)  # (B, T, D)
    dev2 = (h - mean_per_tick).pow(2).mean(dim=-1)   # (B, T)
    if not mask.any():
        return torch.zeros((), device=h.device, dtype=h.dtype)
    return dev2[mask].mean()


def _boundary_align_aux(
    h: torch.Tensor,          # (B, T, D)
    b_prob: torch.Tensor,     # (B, T)
    mask: torch.Tensor,       # (B, T)
) -> torch.Tensor:
    """Negative correlation between b_prob[t] and ||h[t] - h[t-1]||.

    Gives the boundary head DIRECT gradient: where the encoder representation
    changes a lot between adjacent ticks, b_prob should be high. h is detached
    so this term only updates the boundary head (not the encoder).

    L_aux = -E[ b_prob[t] * detach(||Δh[t]||) ] for t >= 1, masked to ticks
    where both t and t-1 are real. Negative sign because higher correlation
    → lower loss → boundaries align with state-changes.
    """
    B, T, D = h.shape
    if T < 2:
        return torch.zeros((), device=h.device, dtype=h.dtype)
    with torch.no_grad():
        delta = (h[:, 1:, :] - h[:, :-1, :]).norm(dim=-1)  # (B, T-1)
        # Normalize per-batch so different rounds are comparable
        delta_mean = delta.mean(dim=1, keepdim=True).clamp(min=1e-6)
        delta_norm = delta / delta_mean
    valid = mask[:, 1:] & mask[:, :-1]
    if not valid.any():
        return torch.zeros((), device=h.device, dtype=h.dtype)
    score = (b_prob[:, 1:] * delta_norm)[valid].mean()
    return -score


def _shifted_segment_targets(tgt_seg: torch.Tensor, seg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift per-segment targets by +1 along segment dim → predict events in
    the NEXT segment from the current segment's embedding. Last real segment
    has no next-segment target → drop from loss."""
    # tgt_seg: (B, S, 6); seg_mask: (B, S)
    B, S, K = tgt_seg.shape
    shifted = torch.zeros_like(tgt_seg)
    shifted[:, :-1, :] = tgt_seg[:, 1:, :]
    valid = seg_mask.clone()
    # Drop segments whose next segment is invalid (i.e., out of range or no ticks)
    valid_next = torch.zeros_like(seg_mask)
    valid_next[:, :-1] = seg_mask[:, 1:]
    return shifted, valid & valid_next


def _build_segment_targets(
    event_labels: torch.Tensor,    # (B, T)
    event_times: torch.Tensor,     # (B, T)
    mask: torch.Tensor,             # (B, T) True=real
    seg_id: torch.Tensor,           # (B, T) long
    downsample: int,
    n_event_classes: int,
    none_idx: int,
    max_segments: int,
) -> torch.Tensor:
    """Per-segment count tensor (B, S, 6) of {kill_t, kill_ct, plant, defuse,
    explode, round_end}. An event is "at" a tick when its tick-distance to
    the next event is less than the downsample step (i.e., it falls in this
    tick's downsample window). Sum over ticks in segment = segment count."""
    B, T = event_labels.shape
    is_event_at_tick = (
        (event_labels != none_idx)
        & (event_labels >= 0)
        & (event_times < downsample)
        & mask
    )                                                  # (B, T) bool
    safe = event_labels.clamp(min=0)                   # avoid -100
    # One-hot the event type per tick, zero rows where no event-at-this-tick
    onehot = nn.functional.one_hot(safe, num_classes=n_event_classes).to(torch.float32)
    onehot = onehot * is_event_at_tick.float().unsqueeze(-1)   # (B, T, 7)
    # Drop the "none" column — only forecast real events
    onehot = onehot[..., :n_event_classes - 1]                  # (B, T, 6)
    # Scatter-add into segments
    targets = torch.zeros(B, max_segments, onehot.shape[-1],
                           device=event_labels.device, dtype=torch.float32)
    sid = seg_id.unsqueeze(-1).expand(-1, -1, onehot.shape[-1])
    targets.scatter_add_(1, sid, onehot)
    return targets


def compute_loss(
    encoder: RoundEncoder,
    forward_heads: nn.ModuleList,
    event_head: NextEventHead | None,
    time_head: TimeToEventHead | None,
    window_head: "WindowForecastHead | None",
    change_point_head: "ChangePointHead | None",
    class_weights: torch.Tensor | None,
    batch: dict,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-horizon forward-prediction loss + (optional) event objectives.

    Forward-pred: for each horizon H, target at position t is the input
    feature vector at t+H. Padded and out-of-bounds positions excluded.

    Event-CE: 7-way CE over the next event type within EVENT_HORIZON_TICKS,
    with class_weights down-weighting "none" (the dominant class).

    Time-to-event: smoothed L1 on log(1+ticks) until next event, masked to
    non-"none" positions only — otherwise the model trivially predicts the
    cap.

    Returns (total_loss, metrics_dict).
    """
    x = batch["x"].to(device, non_blocking=True)        # (B, T, F)
    mask = batch["mask"].to(device, non_blocking=True)  # (B, T) — True = real
    key_padding_mask = ~mask                            # (B, T) — True = PAD

    h = encoder(x, key_padding_mask=key_padding_mask)   # (B, T, d_model)

    total = torch.zeros((), device=device, dtype=h.dtype)
    metrics: dict[str, float] = {}
    _, T, _ = x.shape

    # --- Forward prediction ---
    for w, H, head in zip(cfg.horizon_weights, cfg.horizons, forward_heads):
        if H >= T:
            metrics[f"L_h{H}"] = 0.0
            continue
        h_pred = head(h[:, : T - H, :])           # (B, T-H, F)
        target = x[:, H:, :]                       # (B, T-H, F)
        valid = mask[:, : T - H] & mask[:, H:]     # (B, T-H)
        if not valid.any():
            metrics[f"L_h{H}"] = 0.0
            continue
        diff = (h_pred - target) ** 2              # (B, T-H, F)
        per_pos = diff.mean(dim=-1)                # (B, T-H)
        loss_h = (per_pos * valid.to(per_pos.dtype)).sum() / valid.to(per_pos.dtype).sum()
        total = total + w * loss_h
        metrics[f"L_h{H}"] = float(loss_h.item())

    # --- Event objectives ---
    if event_head is not None and "event_labels" in batch:
        ev_labels = batch["event_labels"].to(device, non_blocking=True)  # (B, T)
        ev_times = batch["event_times"].to(device, non_blocking=True)    # (B, T)
        # Mask out PAD (already -100 from collate, but be defensive)
        ev_labels = torch.where(mask, ev_labels, torch.full_like(ev_labels, -100))

        # CE over 7 classes — flatten and use ignore_index=-100, weight handles
        # class imbalance (none gets a small weight)
        logits = event_head(h)                      # (B, T, n_classes)
        loss_ce = nn.functional.cross_entropy(
            logits.reshape(-1, cfg.n_event_classes),
            ev_labels.reshape(-1),
            weight=class_weights,
            ignore_index=-100,
        )
        if torch.isfinite(loss_ce):
            total = total + cfg.event_ce_weight * loss_ce
            metrics["L_event_ce"] = float(loss_ce.item())
            # Track event-only accuracy too (mask "none")
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                non_none = (ev_labels != cfg.none_event_idx) & (ev_labels != -100)
                if non_none.any():
                    acc_event = (pred[non_none] == ev_labels[non_none]).float().mean()
                    metrics["acc_event_only"] = float(acc_event.item())

    if time_head is not None and "event_times" in batch:
        # log(1 + ticks) target, valid only at non-"none", non-pad positions
        ev_labels = batch["event_labels"].to(device, non_blocking=True)
        ev_times = batch["event_times"].to(device, non_blocking=True)
        time_valid = mask & (ev_labels != cfg.none_event_idx) & (ev_labels >= 0)
        if time_valid.any():
            # Normalize to log space so the loss isn't dominated by far events
            log_target = torch.log1p(ev_times) / math.log(1.0 + cfg.event_horizon_ticks)
            t_pred = time_head(h)
            loss_time = nn.functional.smooth_l1_loss(
                t_pred[time_valid], log_target[time_valid], beta=0.1,
            )
            if torch.isfinite(loss_time):
                total = total + cfg.event_time_weight * loss_time
                metrics["L_time_to_event"] = float(loss_time.item())

    # --- v9 / v10: change-point segmentation ---
    if change_point_head is not None and "event_labels" in batch:
        ev_labels = batch["event_labels"].to(device, non_blocking=True)
        ev_times = batch["event_times"].to(device, non_blocking=True)
        cp_out = change_point_head(h, key_padding_mask=key_padding_mask)
        seg_id = cp_out["seg_id"]
        seg_pred = cp_out["seg_pred"]      # (B, S, 6)
        seg_mask = cp_out["seg_mask"]      # (B, S)
        b_prob = cp_out["b_prob"]          # (B, T)
        downsample = int(batch.get("downsample", 8)) if isinstance(batch.get("downsample"), (int, float)) else 8

        # Density penalty is always on (anchors the rate).
        density = b_prob[mask].mean()
        density_pen = cfg.change_point_density_weight * (density - cfg.change_point_target_density) ** 2

        # Aux: direct gradient to boundary head, telling it "place boundaries
        # where adjacent h changes a lot". v9 had NO gradient to b_prob from
        # the forecast loss (seg_id is detached after STE) — only density
        # penalty influenced placement. That's the root cause of anti-alignment.
        aux_align = torch.zeros((), device=device, dtype=h.dtype)
        if cfg.change_point_aux_align_weight > 0.0:
            aux_align = _boundary_align_aux(h, b_prob, mask)
            metrics["L_cp_align"] = float(aux_align.item())

        cp_loss_aux = density_pen + cfg.change_point_aux_align_weight * aux_align

        loss_variant = cfg.change_point_loss
        if loss_variant in ("forecast", "outcome") and seg_mask.any():
            tgt = _build_segment_targets(
                ev_labels, ev_times, mask, seg_id,
                downsample=downsample,
                n_event_classes=cfg.n_event_classes,
                none_idx=cfg.none_event_idx,
                max_segments=change_point_head.max_segments,
            )                                   # (B, S, 6)
            if loss_variant == "outcome":
                tgt, valid_seg = _shifted_segment_targets(tgt, seg_mask)
            else:
                valid_seg = seg_mask
            cnt_pred = seg_pred[..., :ChangePointHead.BIN_FIRST]
            cnt_tgt = torch.log1p(tgt[..., :ChangePointHead.BIN_FIRST])
            bin_pred = seg_pred[..., ChangePointHead.BIN_FIRST:]
            bin_tgt = (tgt[..., ChangePointHead.BIN_FIRST:] > 0).float()
            if valid_seg.any():
                cnt_loss = nn.functional.smooth_l1_loss(
                    cnt_pred[valid_seg], cnt_tgt[valid_seg], beta=0.5,
                )
                bin_loss = nn.functional.binary_cross_entropy_with_logits(
                    bin_pred[valid_seg], bin_tgt[valid_seg],
                )
                seg_loss = cnt_loss + bin_loss
                cp_loss = cfg.change_point_weight * seg_loss + cp_loss_aux
                metrics["L_cp_count"] = float(cnt_loss.item())
                metrics["L_cp_bin"] = float(bin_loss.item())
            else:
                cp_loss = cp_loss_aux
        elif loss_variant == "contrastive":
            con_loss = _contrastive_segment_loss(
                h, seg_id, mask, temperature=cfg.contrastive_temperature,
            )
            cp_loss = cfg.change_point_weight * con_loss + cp_loss_aux
            metrics["L_cp_contrastive"] = float(con_loss.item())
        elif loss_variant == "variance":
            var_loss = _intra_segment_variance(
                h, seg_id, mask, change_point_head.max_segments,
            )
            cp_loss = cfg.change_point_weight * var_loss + cp_loss_aux
            metrics["L_cp_variance"] = float(var_loss.item())
        else:
            cp_loss = cp_loss_aux

        if torch.isfinite(cp_loss):
            total = total + cp_loss
            metrics["cp_density"] = float(density.item())
            metrics["cp_density_target"] = float(cfg.change_point_target_density)
            metrics["cp_n_segments_avg"] = float(seg_mask.float().sum(dim=1).mean().item())

    # --- v8: window-forecast objective ---
    if window_head is not None and "event_labels" in batch:
        ev_labels = batch["event_labels"].to(device, non_blocking=True)
        tgt, valid_w = _build_window_forecast_targets(
            ev_labels, mask, cfg.window_forecast_ticks, cfg,
        )
        if valid_w.any():
            pred = window_head(h)  # (B, T, 6)
            # Counts: regress log1p (so 0, 1, 2 kills aren't unequal in magnitude)
            cnt_pred = pred[..., :WindowForecastHead.BIN_FIRST]
            cnt_tgt = torch.log1p(tgt[..., :WindowForecastHead.BIN_FIRST])
            cnt_loss = nn.functional.smooth_l1_loss(
                cnt_pred[valid_w], cnt_tgt[valid_w], beta=0.5,
            )
            # Binary: BCE-with-logits per flag
            bin_pred = pred[..., WindowForecastHead.BIN_FIRST:]
            bin_tgt = tgt[..., WindowForecastHead.BIN_FIRST:]
            bin_loss = nn.functional.binary_cross_entropy_with_logits(
                bin_pred[valid_w], bin_tgt[valid_w],
            )
            loss_window = cnt_loss + bin_loss
            if torch.isfinite(loss_window):
                total = total + cfg.window_forecast_weight * loss_window
                metrics["L_window_count"] = float(cnt_loss.item())
                metrics["L_window_bin"] = float(bin_loss.item())

    return total, metrics


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
def evaluate(encoder: RoundEncoder, forward_heads: nn.ModuleList,
             event_head: NextEventHead | None, time_head: TimeToEventHead | None,
             window_head: "WindowForecastHead | None",
             change_point_head: "ChangePointHead | None",
             class_weights: torch.Tensor | None, val_loader: DataLoader,
             cfg: TrainConfig, device: torch.device) -> dict:
    encoder.eval()
    forward_heads.eval()
    if event_head is not None:
        event_head.eval()
    if time_head is not None:
        time_head.eval()
    if window_head is not None:
        window_head.eval()
    if change_point_head is not None:
        change_point_head.eval()
    total = 0.0
    n = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    with torch.no_grad():
        for batch in val_loader:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss, m = compute_loss(
                    encoder, forward_heads, event_head, time_head, window_head,
                    change_point_head, class_weights, batch, cfg, device,
                )
            total += float(loss.item())
            n += 1
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1
    if n == 0:
        return {"val_total": float("nan")}
    out = {"val_total": total / n}
    for k, v in metric_sums.items():
        out[f"val_{k}"] = v / metric_counts[k]
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
    use_events = train_ds.has_events()
    event_head: NextEventHead | None = None
    time_head: TimeToEventHead | None = None
    window_head: WindowForecastHead | None = None
    change_point_head: ChangePointHead | None = None
    class_weights: torch.Tensor | None = None
    if use_events:
        event_head = NextEventHead(cfg.d_model, cfg.n_event_classes).to(device)
        time_head = TimeToEventHead(cfg.d_model).to(device)
        # Class weights: down-weight the dominant "none" class so rare events
        # actually drive gradient.
        cw = torch.ones(cfg.n_event_classes, dtype=torch.float32)
        cw[cfg.none_event_idx] = cfg.none_class_weight
        class_weights = cw.to(device)
        print(f"Event objectives ON (vocab: {train_ds.event_vocab})")
        print(f"  none class weight = {cfg.none_class_weight}")
        if cfg.use_window_forecast:
            window_head = WindowForecastHead(cfg.d_model).to(device)
            print(f"Window forecast ON (window={cfg.window_forecast_ticks} "
                  f"downsampled ticks, weight={cfg.window_forecast_weight})")
        if cfg.use_change_point:
            change_point_head = ChangePointHead(
                cfg.d_model, max_segments=cfg.change_point_max_segments,
                init_target_density=cfg.change_point_target_density,
            ).to(device)
            print(f"Change-point segmentation ON (max_segments="
                  f"{cfg.change_point_max_segments}, seg_weight="
                  f"{cfg.change_point_weight}, "
                  f"target_density={cfg.change_point_target_density}, "
                  f"density_weight={cfg.change_point_density_weight})")
    else:
        print("Event objectives OFF — dataset has no event_labels")

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_heads = sum(p.numel() for p in heads.parameters())
    n_event = (sum(p.numel() for p in event_head.parameters())
               + sum(p.numel() for p in time_head.parameters())) if use_events else 0
    n_window = sum(p.numel() for p in window_head.parameters()) if window_head else 0
    n_cp = sum(p.numel() for p in change_point_head.parameters()) if change_point_head else 0
    print(f"Params: encoder {n_enc/1e6:.2f}M + forward heads {n_heads/1e6:.2f}M"
          f"{' + event heads ' + f'{n_event/1e6:.2f}M' if use_events else ''}"
          f"{' + window head ' + f'{n_window/1e6:.2f}M' if window_head else ''}"
          f"{' + cp head ' + f'{n_cp/1e6:.2f}M' if change_point_head else ''}"
          f" = {(n_enc + n_heads + n_event + n_window + n_cp)/1e6:.2f}M total")

    params = list(encoder.parameters()) + list(heads.parameters())
    if use_events:
        params += list(event_head.parameters()) + list(time_head.parameters())
    if window_head is not None:
        params += list(window_head.parameters())
    if change_point_head is not None:
        params += list(change_point_head.parameters())
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
        if event_head is not None:
            event_head.train()
        if time_head is not None:
            time_head.train()
        if window_head is not None:
            window_head.train()
        if change_point_head is not None:
            change_point_head.train()
        epoch_t0 = time.time()
        epoch_total = 0.0
        epoch_n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss, m = compute_loss(
                    encoder, heads, event_head, time_head, window_head,
                    change_point_head, class_weights, batch, cfg, device,
                )
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
                       "lr": lr, **m}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                metric_str = " ".join(f"{k}={v:.4f}" for k, v in m.items())
                print(f"  [e{epoch} s{step}] loss={loss.item():.4f} lr={lr:.2e} "
                      f"{metric_str}", flush=True)

        epoch_loss = epoch_total / max(1, epoch_n)
        epoch_dt = time.time() - epoch_t0

        val_metrics: dict = {}
        if (epoch + 1) % cfg.val_every_epochs == 0 or epoch == cfg.epochs - 1:
            val_metrics = evaluate(
                encoder, heads, event_head, time_head, window_head,
                change_point_head, class_weights, val_loader, cfg, device,
            )

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
            "event_head": event_head.state_dict() if event_head is not None else None,
            "time_head": time_head.state_dict() if time_head is not None else None,
            "window_head": window_head.state_dict() if window_head is not None else None,
            "change_point_head": change_point_head.state_dict() if change_point_head is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(cfg),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        # Early-stopping signal selection: if the event objectives are on,
        # val_total gets dominated and quickly polluted by event-CE overfit
        # (train→0, val→huge while forward-pred stays stable). The true
        # generalization signal is val_acc_event_only — pick the ckpt that
        # maximizes that. When event objectives are off, fall back to val_total.
        if "val_acc_event_only" in val_metrics:
            cur_signal = -float(val_metrics["val_acc_event_only"])  # negate so "lower=better"
            signal_name = "val_acc_event_only"
        else:
            cur_signal = float(val_metrics.get("val_total", float("inf")))
            signal_name = "val_total"
        if cur_signal < best_val:
            best_val = cur_signal
            torch.save(ckpt, out_dir / "best.pt")
            display = -best_val if signal_name == "val_acc_event_only" else best_val
            print(f"  → new best ({signal_name})={display:.4f}", flush=True)

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
    ap.add_argument("--salience", action="store_true",
                    help="enable per-tick learned salience bias on attention "
                         "(v7+; default off preserves v6 architecture)")
    ap.add_argument("--window-forecast", action="store_true",
                    help="enable window-forecast SSL objective (v8+; predicts "
                         "kill counts + bomb-event flags over next 2s window)")
    ap.add_argument("--window-forecast-ticks", type=int, default=16,
                    help="window size in downsampled ticks (default 16 = 2s)")
    ap.add_argument("--window-forecast-weight", type=float, default=0.3,
                    help="weight on window-forecast loss in total")
    ap.add_argument("--change-point", action="store_true",
                    help="enable change-point segmentation head (v9+; learned "
                         "event boundaries via straight-through estimator)")
    ap.add_argument("--change-point-max-segments", type=int, default=64,
                    help="max segments per round for change-point head")
    ap.add_argument("--change-point-weight", type=float, default=1.0)
    ap.add_argument("--change-point-target-density", type=float, default=0.02,
                    help="target boundary density (mean b_prob across ticks); "
                         "0.02 ≈ 1 boundary per 50 ticks ≈ 6.25s")
    ap.add_argument("--change-point-density-weight", type=float, default=5.0,
                    help="MSE penalty weight around target density")
    ap.add_argument("--change-point-loss", type=str, default="forecast",
                    choices=["forecast", "contrastive", "variance", "outcome"],
                    help="v10 segment loss variant. forecast=v9 baseline "
                         "(per-segment event counts); contrastive=InfoNCE on "
                         "same-segment ticks; variance=intra-segment h-variance; "
                         "outcome=forecast NEXT segment's event counts.")
    ap.add_argument("--change-point-aux-align-weight", type=float, default=0.0,
                    help="weight on aux 'boundary at state-change' term that "
                         "gives DIRECT gradient to boundary head. v9 was 0; "
                         "v10 variants typically use 0.1.")
    ap.add_argument("--contrastive-temperature", type=float, default=0.1,
                    help="InfoNCE softmax temperature for contrastive variant")
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
        use_salience=args.salience,
        use_window_forecast=args.window_forecast,
        window_forecast_ticks=args.window_forecast_ticks,
        window_forecast_weight=args.window_forecast_weight,
        use_change_point=args.change_point,
        change_point_max_segments=args.change_point_max_segments,
        change_point_weight=args.change_point_weight,
        change_point_target_density=args.change_point_target_density,
        change_point_density_weight=args.change_point_density_weight,
        change_point_loss=args.change_point_loss,
        change_point_aux_align_weight=args.change_point_aux_align_weight,
        contrastive_temperature=args.contrastive_temperature,
        log_every=args.log_every,
        seed=args.seed,
        output_dir=str(out_dir),
    )
    train(cfg)


if __name__ == "__main__":
    main()
