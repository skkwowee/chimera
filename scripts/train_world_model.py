#!/usr/bin/env python3
"""World model: next-state prediction over CS2 game-state frames.

This is the foundation of the post-pivot architecture (see docs/world-model-design.md):
a causal transformer trained state->state (NO text), like LM pretraining but over
597-d game-state frames. It learns the DYNAMICS of CS2; events fall out of
prediction surprise, and value/policy/language attach later as heads on the latent.

Data: data/processed/tick_sequences/{train,val}.pt
  dict with 'tensors' = list of [T, 597] float32 (one per ROUND, 8 Hz), 'metas'.
  We IGNORE the baked-in event_* fields (from the superseded change-point work).

Key design choices (and the traps they avoid):
  - ROUND-SCOPED windows: a round is a "document"; we never attend across the
    round reset (which is an engine discontinuity, not dynamics).
  - HORIZON SWEEP (--horizon, in 8 Hz steps): predict frame t+k. k=1 (125 ms) is
    MLMove-style inertia and learns ~nothing strategic; tactics show up at longer
    horizons. Sweeping k IS the experiment. Run once per k.
  - RESIDUAL target: predict (frame_{t+k} - frame_t), so the model must learn what
    CHANGES, not copy the current frame (the high-frequency inertia trap).
  - HONEST BASELINES: every eval reports the model's loss against two trivial
    predictors -- COPY (zero motion) and CONST-VELOCITY (linear extrapolation).
    A low absolute loss means nothing; beating const-velocity is the real bar.
    (Final acceptance is probe transfer, not loss -- that's a separate harness.)

TODO (next iterations, intentionally not in v1):
  - Distributional head (discretize / GMM) on the multimodal continuous dims to
    avoid mode-averaging blur. v1 uses Huber regression as the runnable baseline.
  - Per-player token factoring (relational attention a la MLMove) instead of a
    flat 597-d frame. v1 treats the frame as one token.
  - Categorical-vs-continuous split head (one-hot dims want cross-entropy).

Usage:
  smoke (CPU, tiny, ~30s):   python scripts/train_world_model.py --smoke
  real (pod, one horizon):   python scripts/train_world_model.py --horizon 8 \
                                 --d-model 512 --layers 6 --window 128 --steps 20000
  sweep = 4 runs with --horizon 1 / 4 / 8 / 16.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data/processed/tick_sequences")


# --------------------------------------------------------------------------- data
class RoundWindows(Dataset):
    """Random fixed-length crops from per-round frame sequences.

    For horizon k and window L, a crop of length L+k gives, at each position i in
    [0, L), an input frame x_i = crop[i] and a target frame y_i = crop[i+k].
    The model sees x_{0..i} (causal) and predicts the RESIDUAL y_i - x_i.
    Rounds shorter than L+k+1 are dropped (need one extra frame for the
    const-velocity baseline).
    """

    def __init__(self, tensors, window: int, horizon: int, crops_per_round: int):
        self.window = window
        self.horizon = horizon
        need = window + horizon + 1
        self.rounds = [t for t in tensors if t.shape[0] >= need]
        self.crops_per_round = crops_per_round
        self.dropped = len(tensors) - len(self.rounds)

    def __len__(self):
        return len(self.rounds) * self.crops_per_round

    def __getitem__(self, idx):
        r = self.rounds[idx % len(self.rounds)]
        L, k = self.window, self.horizon
        hi = r.shape[0] - (L + k)
        # start >= 1 so the const-velocity baseline can read crop[i-1] at i=0
        start = 1 + int(torch.randint(0, hi, (1,)).item()) if hi > 0 else 1
        x = r[start : start + L]                     # [L, F]
        y = r[start + k : start + k + L]             # [L, F]  (frame t+k)
        x_prev = r[start - 1 : start - 1 + L]        # [L, F]  (for const-velocity)
        return x, y, x_prev


# ----------------------------------------------------------------------- model
class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                              # x: [B, L, D]
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class WorldModelFlat(nn.Module):
    """Baseline: causal transformer treating the whole 597-d frame as ONE token."""

    def __init__(self, feature_dim, d_model=512, layers=6, heads=8, ff=2048, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos = SinusoidalPositions(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feature_dim)

    def _run(self, x):
        L = x.shape[1]
        h = self.pos(self.in_proj(x))
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        return self.norm(self.encoder(h, mask=mask))

    def forward(self, x):
        return self.head(self._run(x))                # [B,L,F] residual

    def latent(self, x):
        return self._run(x)


# 597-d frame layout (feature_schema_v2): 10 players x 56, then 37 global (player-major).
N_PLAYERS, PER_PLAYER_DIM, GLOBAL_DIM = 10, 56, 37
PLAYER_BLOCK = N_PLAYERS * PER_PLAYER_DIM            # 560


class WorldModelPlayers(nn.Module):
    """MLMove-style FACTORED model: each player is its own token + a global token,
    with attention over BOTH players (relational) and time (causal). Per-player
    heads predict each player's own next-k residual; a global head the global part.

    This is the per-player-TOKEN half of MLMove's feature engineering, on top of the
    geometric per-player features already in the schema (x/y/z/yaw/...). The other
    half -- the DERIVED VISIBILITY channel (who-can-see/shoot-whom, LOS through
    smokes) -- needs a geometry recompute in build_tick_sequences and is the next
    feature pass (see docs/world-model-design.md). Borrow perception, learn tactics.
    """

    def __init__(self, feature_dim, d_model=512, layers=6, heads=8, ff=2048, dropout=0.1):
        super().__init__()
        assert feature_dim == PLAYER_BLOCK + GLOBAL_DIM, feature_dim
        self.d = d_model
        self.player_proj = nn.Linear(PER_PLAYER_DIM, d_model)
        self.global_proj = nn.Linear(GLOBAL_DIM, d_model)
        self.slot_emb = nn.Embedding(N_PLAYERS + 1, d_model)   # 10 player slots + global
        self.tpos = SinusoidalPositions(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.player_head = nn.Linear(d_model, PER_PLAYER_DIM)
        self.global_head = nn.Linear(d_model, GLOBAL_DIM)
        self._mask_cache = {}

    def _tokens(self, x):
        B, L, _ = x.shape
        players = x[..., :PLAYER_BLOCK].reshape(B, L, N_PLAYERS, PER_PLAYER_DIM)
        glob = x[..., PLAYER_BLOCK:]
        tok = torch.cat([self.player_proj(players),
                         self.global_proj(glob).unsqueeze(2)], dim=2)   # [B,L,11,d]
        slot = self.slot_emb(torch.arange(N_PLAYERS + 1, device=x.device))
        tok = tok + slot.view(1, 1, N_PLAYERS + 1, self.d)
        tpos = self.tpos(torch.zeros(1, L, self.d, device=x.device)).squeeze(0)  # [L,d]
        tok = tok + tpos.view(1, L, 1, self.d)
        return tok.reshape(B, L * (N_PLAYERS + 1), self.d), L

    def _mask(self, L, device):
        key = (L, str(device))
        if key not in self._mask_cache:
            P = N_PLAYERS + 1
            t = torch.arange(L * P, device=device) // P
            self._mask_cache[key] = t.unsqueeze(0) > t.unsqueeze(1)     # True=disallow future time
        return self._mask_cache[key]

    def _run(self, x):
        seq, L = self._tokens(x)
        return self.norm(self.encoder(seq, mask=self._mask(L, x.device))), L

    def forward(self, x):
        h, L = self._run(x)
        B = x.shape[0]; P = N_PLAYERS + 1
        h = h.reshape(B, L, P, self.d)
        pres = self.player_head(h[:, :, :N_PLAYERS, :]).reshape(B, L, PLAYER_BLOCK)
        gres = self.global_head(h[:, :, N_PLAYERS, :])
        return torch.cat([pres, gres], dim=-1)                          # [B,L,597] residual

    def latent(self, x):
        h, L = self._run(x)
        B = x.shape[0]; P = N_PLAYERS + 1
        return h.reshape(B, L, P, self.d).mean(dim=2)                   # [B,L,d] per-time summary


def build_model(arch, feature_dim, d_model, layers, heads):
    return {"flat": WorldModelFlat, "player": WorldModelPlayers}[arch](
        feature_dim, d_model, layers, heads)


# -------------------------------------------------------------------- train/eval
def huber(pred, target):
    return F.smooth_l1_loss(pred, target, beta=1.0)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    m_loss = copy_loss = cv_loss = n = 0.0
    for bi, (x, y, x_prev) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y, x_prev = x.to(device), y.to(device), x_prev.to(device)
        true_res = y - x
        pred_res = model(x)
        m_loss += huber(pred_res, true_res).item()
        copy_loss += huber(torch.zeros_like(true_res), true_res).item()     # predict no motion
        cv_loss += huber(model.horizon * (x - x_prev), true_res).item()     # linear extrapolation
        n += 1
    model.train()
    return m_loss / n, copy_loss / n, cv_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["flat", "player"], default="player",
                    help="player = MLMove-style per-player tokens + relational attention")
    ap.add_argument("--horizon", type=int, default=8, help="predict t+k, k in 8Hz steps (1=125ms,8=1s,16=2s)")
    ap.add_argument("--window", type=int, default=128, help="context length (frames)")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--crops-per-round", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--train-pt", default=str(DATA_DIR / "train.pt"))
    ap.add_argument("--val-pt", default=str(DATA_DIR / "val.pt"))
    ap.add_argument("--out", default="outputs/world_model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run on val.pt")
    args = ap.parse_args()

    if args.smoke:
        args.d_model, args.layers, args.heads = 128, 2, 4
        args.window, args.batch, args.steps = 64, 8, 30
        args.eval_every, args.crops_per_round = 15, 4
        args.train_pt = args.val_pt           # reuse val for a light self-contained run
        args.device = "cpu"
        print("[smoke] tiny CPU run on val.pt")

    dev = torch.device(args.device)
    print(f"loading {args.train_pt} ...")
    train_blob = torch.load(args.train_pt, map_location="cpu", weights_only=False)
    val_blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    fdim = train_blob["feature_dim"]
    assert fdim == val_blob["feature_dim"]

    tr_ds = RoundWindows(train_blob["tensors"], args.window, args.horizon, args.crops_per_round)
    va_ds = RoundWindows(val_blob["tensors"], args.window, args.horizon, args.crops_per_round)
    print(f"feature_dim={fdim}  horizon={args.horizon} ({args.horizon*125}ms)  "
          f"window={args.window}  train_rounds={len(tr_ds.rounds)} (dropped {tr_ds.dropped}) "
          f"val_rounds={len(va_ds.rounds)}")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=0 if args.smoke else 4)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=0 if args.smoke else 2)

    model = build_model(args.arch, fdim, args.d_model, args.layers, args.heads).to(dev)
    model.horizon = args.horizon          # used by const-velocity baseline in eval
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch}  model params: {n_params/1e6:.1f}M  device={dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = Path(args.out) / f"h{args.horizon}"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    step = 0
    t0 = time.time()
    data_iter = iter(tr_ld)
    while step < args.steps:
        try:
            x, y, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(tr_ld)
            x, y, _ = next(data_iter)
        x, y = x.to(dev), y.to(dev)
        true_res = y - x
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            pred_res = model(x)
            loss = huber(pred_res, true_res)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            m, c, cv = evaluate(model, va_ld, dev, max_batches=10 if args.smoke else 50)
            skill = (cv - m) / cv * 100 if cv > 0 else 0.0   # % better than const-velocity
            print(f"step {step:6d}  train {loss.item():.4f}  val {m:.4f}  "
                  f"[copy {c:.4f}  const-vel {cv:.4f}]  skill-vs-CV {skill:+.1f}%  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            if m < best:
                best = m
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "feature_dim": fdim, "val_loss": m, "step": step},
                           out / "best.pt")
    print(f"done. best val {best:.4f} -> {out/'best.pt'}")
    print("NOTE: low loss != understanding. The real test is probe transfer "
          "(value/event probes on model.latent()) vs the old encoder -- separate harness.")


if __name__ == "__main__":
    main()
