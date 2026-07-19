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

Heads beyond the v1 Huber regression:
  - --dist-head: DISTRIBUTIONAL player-xy displacement head (classify-then-refine,
    97 classes = stationary + 6 magnitude rings x 16 directions, plus per-class xy
    refine offset). Fixes regression mode-averaging: stationary jitter, hard-turn
    means landing between modes, jitter feedback in closed-loop rollout. Decode via
    model.gen_residual() (argmax or sampled class) — eval scripts use it everywhere.
TODO (next iterations, intentionally not in v1):
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
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus import load_corpus

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

    def __init__(self, tensors, metas, window: int, horizon: int, crops_per_round: int):
        self.window = window
        self.horizon = horizon
        need = window + horizon + 1
        keep = [(t, m) for t, m in zip(tensors, metas) if t.shape[0] >= need]
        self.rounds = [t for t, _ in keep]
        # value label: 1.0 if CT won the round, else 0.0 (P(CT win) target)
        self.won = [1.0 if (m.get("winner") == "ct") else 0.0 for _, m in keep]
        self.crops_per_round = crops_per_round
        self.dropped = len(tensors) - len(self.rounds)

    def __len__(self):
        return len(self.rounds) * self.crops_per_round

    def __getitem__(self, idx):
        # NOTE: crops below are VIEWS into (possibly mmap'd, load_corpus) round
        # tensors — never write into x/y/x_prev in place. Default collate stacks
        # them into fresh batch tensors, so downstream code gets copies anyway.
        ri = idx % len(self.rounds)
        r = self.rounds[ri]
        L, k = self.window, self.horizon
        hi = r.shape[0] - (L + k)
        # start >= 1 so the const-velocity baseline can read crop[i-1] at i=0
        start = 1 + int(torch.randint(0, hi, (1,)).item()) if hi > 0 else 1
        x = r[start : start + L]                     # [L, F]
        y = r[start + k : start + k + L]             # [L, F]  (frame t+k)
        x_prev = r[start - 1 : start - 1 + L]        # [L, F]  (for const-velocity)
        return x, y, x_prev, torch.tensor(self.won[ri], dtype=torch.float32)


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

    def gen_residual(self, x, sample=False, temperature=1.0, generator=None):
        return self.forward(x)

    def latent(self, x):
        return self._run(x)


N_PLAYERS = 10   # frame is player-major: n_players x per_player_dim, then global
RAW_PPD = 56     # raw per-player dims; dims [RAW_PPD:ppd] are derived perception
                 # (input-only: masked out of every loss, so head outputs there
                 # are UNTRAINED — decode must zero them or rollouts feed noise)

# ---- distributional displacement head (classify-then-refine over player xy) ----
# Ring edges in GAME UNITS (xy norm = units/3000). Class 0 = stationary (<8u);
# else ring (log-spaced magnitude) x 16 direction bins of 22.5deg.
XY_NORM = 3000.0
DIST_EDGES_U = [8.0, 24.0, 56.0, 120.0, 248.0, 512.0]
DIST_RINGS, DIST_DIRS = 6, 16
DIST_C = 1 + DIST_RINGS * DIST_DIRS                  # 97


def dist_centers():
    """[C, 2] class centers in NORMALIZED units. Ring magnitude = geometric mean
    of its edges (open last ring: 700u). Direction bin d centered at d*22.5deg."""
    mags = [math.sqrt(DIST_EDGES_U[r] * DIST_EDGES_U[r + 1]) for r in range(DIST_RINGS - 1)] + [700.0]
    c = torch.zeros(DIST_C, 2)
    for r, m in enumerate(mags):
        for d in range(DIST_DIRS):
            th = d * (2 * math.pi / DIST_DIRS)
            c[1 + r * DIST_DIRS + d, 0] = m * math.cos(th) / XY_NORM
            c[1 + r * DIST_DIRS + d, 1] = m * math.sin(th) / XY_NORM
    return c


def dist_class(d):
    """d: [..., 2] normalized xy displacement -> class ids [...] (long)."""
    mag = d.norm(dim=-1) * XY_NORM
    ring = torch.bucketize(mag, torch.tensor(DIST_EDGES_U[1:], device=d.device))
    ang = torch.atan2(d[..., 1], d[..., 0])
    # round to nearest bin center so bin 0 spans [-11.25, +11.25)deg
    dirb = torch.round(ang / (2 * math.pi / DIST_DIRS)).long() % DIST_DIRS
    cls = 1 + ring * DIST_DIRS + dirb
    return torch.where(mag < DIST_EDGES_U[0], torch.zeros_like(cls), cls)


class WorldModelPlayers(nn.Module):
    """MLMove-style FACTORED model: each player is its own token + a global token,
    with attention over BOTH players (relational) and time (causal). Per-player
    heads predict each player's own next-k residual; a global head the global part.
    A VALUE head on the per-frame latent predicts P(CT wins) — co-trained with
    next-state so the latent RETAINS outcome structure (the fix for the v2 value-
    probe failure, where pure next-state compressed value away).

    per_player_dim is read from the data schema (v2=56, v3=65 with the derived
    visibility/perception dims), so the same model handles either feature book.
    """

    def __init__(self, feature_dim, d_model=512, layers=6, heads=8, ff=2048,
                 dropout=0.1, per_player_dim=56, n_players=N_PLAYERS, dist=False):
        super().__init__()
        self.n_players = n_players
        self.dist = dist
        self.ppd = per_player_dim
        self.player_block = n_players * per_player_dim
        self.global_dim = feature_dim - self.player_block
        assert self.global_dim > 0, (feature_dim, self.player_block)
        self.d = d_model
        self.player_proj = nn.Linear(per_player_dim, d_model)
        self.global_proj = nn.Linear(self.global_dim, d_model)
        self.slot_emb = nn.Embedding(n_players + 1, d_model)   # player slots + global
        self.tpos = SinusoidalPositions(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.player_head = nn.Linear(d_model, per_player_dim)
        self.global_head = nn.Linear(d_model, self.global_dim)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Linear(d_model // 2, 1))     # P(CT win) logit
        if dist:
            self.dist_head = nn.Linear(d_model, DIST_C * 3)  # C logits + C xy offsets
            self.register_buffer("centers", dist_centers())  # [C, 2] normalized
        self._mask_cache = {}

    def _tokens(self, x):
        B, L, _ = x.shape
        P = self.n_players
        players = x[..., :self.player_block].reshape(B, L, P, self.ppd)
        glob = x[..., self.player_block:]
        tok = torch.cat([self.player_proj(players),
                         self.global_proj(glob).unsqueeze(2)], dim=2)   # [B,L,P+1,d]
        slot = self.slot_emb(torch.arange(P + 1, device=x.device))
        tok = tok + slot.view(1, 1, P + 1, self.d)
        tpos = self.tpos(torch.zeros(1, L, self.d, device=x.device)).squeeze(0)
        tok = tok + tpos.view(1, L, 1, self.d)
        return tok.reshape(B, L * (P + 1), self.d), L

    def _mask(self, L, device):
        key = (L, str(device))
        if key not in self._mask_cache:
            P = self.n_players + 1
            t = torch.arange(L * P, device=device) // P
            self._mask_cache[key] = t.unsqueeze(0) > t.unsqueeze(1)
        return self._mask_cache[key]

    def _grid(self, x):
        """Run encoder, return contextualized tokens reshaped to [B,L,P+1,d]."""
        seq, L = self._tokens(x)
        h = self.norm(self.encoder(seq, mask=self._mask(L, x.device)))
        return h.reshape(x.shape[0], L, self.n_players + 1, self.d), L

    def forward(self, x):
        h, L = self._grid(x)
        P = self.n_players
        pres = self.player_head(h[:, :, :P, :]).reshape(x.shape[0], L, self.player_block)
        gres = self.global_head(h[:, :, P, :])
        return torch.cat([pres, gres], dim=-1)                          # [B,L,F] residual

    def heads(self, x):
        """Multi-task forward: next-state residual + per-frame value logit
        (+ dist logits/offsets over player xy displacement when enabled)."""
        h, L = self._grid(x)
        P = self.n_players
        pres = self.player_head(h[:, :, :P, :]).reshape(x.shape[0], L, self.player_block)
        gres = self.global_head(h[:, :, P, :])
        residual = torch.cat([pres, gres], dim=-1)
        value = self.value_head(h.mean(dim=2)).squeeze(-1)              # [B,L] P(CT win) logit
        out = {"residual": residual, "value": value}
        if self.dist:
            do = self.dist_head(h[:, :, :P, :])                         # [B,L,P,C*3]
            out["dist_logits"] = do[..., :DIST_C]
            out["dist_off"] = do[..., DIST_C:].reshape(*do.shape[:-1], DIST_C, 2)
        return out

    def gen_residual(self, x, sample=False, temperature=1.0, generator=None):
        """Decode-time residual: like forward(), but with the player xy dims
        OVERWRITTEN by the distributional head (class center + refine offset).
        Falls back to forward() when the dist head is absent, so callers can
        use it unconditionally on any checkpoint."""
        if not self.dist:
            res = self.forward(x)
            if self.ppd > RAW_PPD:                       # freeze input-only derived dims
                B, L_, _ = res.shape
                res = res.clone()
                res[..., :self.player_block].reshape(B, L_, self.n_players, self.ppd)[..., RAW_PPD:] = 0
            return res
        h, L = self._grid(x)
        P = self.n_players
        pres = self.player_head(h[:, :, :P, :])                         # [B,L,P,ppd]
        gres = self.global_head(h[:, :, P, :])
        do = self.dist_head(h[:, :, :P, :])
        logits = do[..., :DIST_C]
        off = do[..., DIST_C:].reshape(*do.shape[:-1], DIST_C, 2)
        if sample:
            probs = F.softmax(logits / temperature, dim=-1)
            cls = torch.multinomial(probs.reshape(-1, DIST_C), 1,
                                    generator=generator).reshape(logits.shape[:-1])
        else:
            cls = logits.argmax(dim=-1)                                 # [B,L,P]
        off_c = off.gather(-2, cls[..., None, None].expand(*cls.shape, 1, 2)).squeeze(-2)
        pres = pres.clone()
        pres[..., 0:2] = self.centers[cls] + off_c
        if self.ppd > RAW_PPD:                           # freeze input-only derived dims
            pres[..., RAW_PPD:] = 0
        pres = pres.reshape(x.shape[0], L, self.player_block)
        return torch.cat([pres, gres], dim=-1)                          # [B,L,F]

    def latent(self, x):
        h, L = self._grid(x)
        return h.mean(dim=2)                                            # [B,L,d] per-frame summary


def build_model(arch, feature_dim, d_model, layers, heads, per_player_dim=56, dist=False):
    if arch == "flat":
        return WorldModelFlat(feature_dim, d_model, layers, heads)
    return WorldModelPlayers(feature_dim, d_model, layers, heads,
                             per_player_dim=per_player_dim, dist=dist)


# -------------------------------------------------------------------- train/eval
def huber(pred, target):
    return F.smooth_l1_loss(pred, target, beta=1.0)


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC (Mann-Whitney)."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels > 0.5
    npos = int(pos.sum().item()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum().item() - npos * (npos + 1) / 2) / (npos * nneg)


@torch.no_grad()
def evaluate(model, loader, device, pred_mask, freeze_col, end_col, max_batches=50, cv_residual=False):
    model.eval()
    m_loss = copy_loss = cv_loss = n = 0.0
    vlog, vlab = [], []
    for bi, (x, y, x_prev, won) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y, x_prev = x.to(device), y.to(device), x_prev.to(device)
        true_res = (y - x)[..., pred_mask]                       # predict RAW state only
        out = model.heads(x) if hasattr(model, "heads") else {}
        cv_base = (model.horizon * (x - x_prev)) if cv_residual else 0.0
        # decode path (dist head decodes class+offset for xy; plain forward otherwise)
        pred_res = (model.gen_residual(x) + cv_base)[..., pred_mask]
        live = x[..., freeze_col] < 0.5                          # D3: freeze frames out of ns metrics
        if live.any():
            m_loss += huber(pred_res[live], true_res[live]).item()
            copy_loss += huber(torch.zeros_like(true_res[live]), true_res[live]).item()
            cv_loss += huber((model.horizon * (x - x_prev))[..., pred_mask][live], true_res[live]).item()
            n += 1
        if "value" in out:
            keep = x[:, -1, end_col] < 0.5                       # O3: end-phase windows out of AUC
            if keep.any():
                vlog.append(out["value"][keep, -1].float().cpu()); vlab.append(won[keep])
    model.train()
    v_auc = auc(torch.cat(vlog), torch.cat(vlab)) if vlog else float("nan")
    if n == 0:
        return float("nan"), float("nan"), float("nan"), v_auc
    return m_loss / n, copy_loss / n, cv_loss / n, v_auc


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
    ap.add_argument("--value-weight", type=float, default=0.3,
                    help="weight of the value (P CT-win) loss co-trained with next-state")
    ap.add_argument("--maps", default="",
                    help="comma-sep map_name filter, e.g. de_mirage,de_dust2,de_inferno (empty=all)")
    ap.add_argument("--cv-residual", action="store_true",
                    help="predict the CORRECTION over a const-velocity prior (pred = cv_base + head). "
                         "Head learns ~0 on straight frames, so easy-frame jitter dies and the "
                         "tactical correction shows through. See scripts/decision_eval.py.")
    ap.add_argument("--dist-head", action="store_true",
                    help="DISTRIBUTIONAL player-xy head: classify displacement into 97 classes "
                         "(stationary + 6 rings x 16 dirs) + per-class refine offset. Fixes "
                         "regression mode-averaging (stationary jitter, between-mode means).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-pt", default=str(DATA_DIR / "train_v3m.pt"))
    ap.add_argument("--val-pt", default=str(DATA_DIR / "val_v3m.pt"))
    ap.add_argument("--out", default="outputs/world_model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run on val.pt")
    args = ap.parse_args()
    assert not (args.dist_head and args.cv_residual), "--dist-head and --cv-residual are mutually exclusive"

    if args.smoke:
        args.d_model, args.layers, args.heads = 128, 2, 4
        args.window, args.batch, args.steps = 64, 8, 30
        args.eval_every, args.crops_per_round = 15, 4
        args.train_pt = args.val_pt           # reuse val for a light self-contained run
        args.device = "cpu"
        print("[smoke] tiny CPU run on val.pt")

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"seed={args.seed}")

    dev = torch.device(args.device)
    print(f"loading {args.train_pt} ...")
    # load_corpus = mmap + clean_blob + --maps keep-set; tensors stay file-backed.
    # In --smoke, train_pt == val_pt (val reuse): two mmap loads of the same file
    # cost ~nothing, so the reuse behavior is preserved.
    train_blob = load_corpus(args.train_pt, maps=args.maps or None, tag="train")
    val_blob = load_corpus(args.val_pt, maps=args.maps or None, tag="val")
    fdim = train_blob["feature_dim"]
    assert fdim == val_blob["feature_dim"]
    ppd = train_blob.get("per_player_dim", 56)

    # phase flags (build_tick_sequences.encode_global: global block starts at
    # 10*ppd, map_onehot then phase_onehot = [freeze, live, post_plant, end])
    n_maps = len(train_blob.get("map_vocab", [None] * 7))
    freeze_col = N_PLAYERS * ppd + n_maps
    end_col = freeze_col + 3
    _ph = val_blob["tensors"][0][:, freeze_col:end_col + 1]
    assert _ph.max() <= 1.0 and ((_ph.sum(1) - 1.0).abs() < 1e-4).all(), "phase one-hot layout mismatch"

    tr_ds = RoundWindows(train_blob["tensors"], train_blob["metas"], args.window, args.horizon, args.crops_per_round)
    va_ds = RoundWindows(val_blob["tensors"], val_blob["metas"], args.window, args.horizon, args.crops_per_round)
    print(f"feature_dim={fdim} per_player={ppd}  horizon={args.horizon} ({args.horizon*125}ms)  "
          f"window={args.window}  train_rounds={len(tr_ds.rounds)} (dropped {tr_ds.dropped}) "
          f"val_rounds={len(va_ds.rounds)}  value_weight={args.value_weight}")
    dl_gen = torch.Generator(); dl_gen.manual_seed(args.seed)
    def _winit(wid):
        ws = torch.initial_seed() % 2**32
        random.seed(ws); np.random.seed(ws)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=0 if args.smoke else 4,
                       generator=dl_gen, worker_init_fn=_winit)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=0 if args.smoke else 2,
                       generator=dl_gen, worker_init_fn=_winit)

    model = build_model(args.arch, fdim, args.d_model, args.layers, args.heads,
                        per_player_dim=ppd, dist=args.dist_head).to(dev)
    model.horizon = args.horizon          # used by const-velocity baseline in eval

    # Prediction target = RAW state dims only. The derived perception dims (v3:
    # last 9 of each 65-dim player block) are INPUT-ONLY — they're a deterministic
    # function of state, so forecasting them is redundant and wastes capacity on a
    # hard, irrelevant target. Mask them out of the next-state loss.
    RAW_PPD = 56
    n_der = ppd - RAW_PPD
    pred_mask = torch.ones(fdim, dtype=torch.bool)
    if n_der > 0:
        for p in range(N_PLAYERS):
            pred_mask[p * ppd + RAW_PPD : (p + 1) * ppd] = False
    pred_mask = pred_mask.to(dev)
    print(f"next-state target: {int(pred_mask.sum())}/{fdim} dims predicted "
          f"({n_der} derived/player are input-only perception)")
    # --dist-head: player xy leaves the REGRESSION loss (classified instead);
    # pred_mask (eval) keeps xy IN so printed skill stays comparable across runs.
    reg_mask = pred_mask.clone()
    xy_idx = torch.tensor([[p * ppd, p * ppd + 1] for p in range(N_PLAYERS)], device=dev)
    if args.dist_head:
        reg_mask[xy_idx.reshape(-1)] = False
        print(f"dist head: {DIST_C} classes over player xy; regression dims now {int(reg_mask.sum())}")
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

    out = Path(args.out) / f"h{args.horizon}_mt"
    out.mkdir(parents=True, exist_ok=True)
    best_v, best_ns = float("-inf"), float("inf")   # value peaks early, position late ->
                                                    # save BOTH (best.pt = value, best_ns.pt = next-state)
    step = 0
    t0 = time.time()
    data_iter = iter(tr_ld)
    while step < args.steps:
        try:
            x, y, x_prev, won = next(data_iter)
        except StopIteration:
            data_iter = iter(tr_ld)
            x, y, x_prev, won = next(data_iter)
        x, y, x_prev, won = x.to(dev), y.to(dev), x_prev.to(dev), won.to(dev)
        true_res = y - x
        # cv_base = const-velocity residual; in --cv-residual the head learns the
        # correction on top of it (pred_residual = cv_base + head), so straight
        # frames cost ~0 and the head spends capacity only on tactics.
        cv_base = (args.horizon * (x - x_prev)) if args.cv_residual else 0.0
        live_m = x[..., freeze_col] < 0.5     # [B,L] D3: freeze frames are immobile — out of ns losses
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            o = model.heads(x)
            pred_res = o["residual"] + cv_base
            if live_m.any():
                reg_loss = huber(pred_res[..., reg_mask][live_m], true_res[..., reg_mask][live_m])
            else:
                reg_loss = pred_res.sum() * 0.0                          # all-freeze batch
            if args.dist_head:
                d_true = true_res[..., xy_idx]                           # [B,L,P,2]
                cls = dist_class(d_true)                                 # [B,L,P]
                pm = live_m.unsqueeze(-1).expand_as(cls)                 # [B,L,P]
                if pm.any():
                    ce_loss = F.cross_entropy(o["dist_logits"][pm].float(), cls[pm])
                    off = o["dist_off"].gather(
                        -2, cls[..., None, None].expand(*cls.shape, 1, 2)).squeeze(-2)
                    ref_loss = huber(off[pm], (d_true - model.centers[cls])[pm])
                else:
                    ce_loss = ref_loss = pred_res.sum() * 0.0
                ns_loss = reg_loss + 0.1 * ce_loss + ref_loss
            else:
                ns_loss = reg_loss
            v_tgt = won.unsqueeze(1).expand_as(o["value"])
            # O3: end-phase frames trivially label the winner — out of value loss.
            # Freeze frames stay in (freeze-time value is legitimate).
            keep_v = x[..., end_col] < 0.5
            v_all = F.binary_cross_entropy_with_logits(o["value"].float(), v_tgt, reduction="none")
            v_loss = (v_all * keep_v).sum() / keep_v.sum().clamp(min=1)
            loss = ns_loss + args.value_weight * v_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            m, c, cv, vauc = evaluate(model, va_ld, dev, pred_mask, freeze_col, end_col,
                                      max_batches=10 if args.smoke else 50,
                                      cv_residual=args.cv_residual)
            skill = (cv - m) / cv * 100 if cv > 0 else 0.0
            comp = (f"[reg {reg_loss.item():.4f} ce {ce_loss.item():.3f} ref {ref_loss.item():.4f}] "
                    if args.dist_head else "")
            print(f"step {step:6d}  ns {ns_loss.item():.4f} {comp}v {v_loss.item():.3f}  "
                  f"val_ns {m:.4f} [copy {c:.4f} cv {cv:.4f}] skill {skill:+.1f}%  "
                  f"VALUE_AUC {vauc:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            meta = {"model": model.state_dict(), "args": vars(args), "seed": args.seed,
                    "feature_dim": fdim, "per_player_dim": ppd,
                    "val_ns": m, "value_auc": vauc, "step": step}
            if not math.isnan(vauc) and vauc > best_v:
                best_v = vauc; torch.save(meta, out / "best.pt")           # best VALUE
            if m < best_ns:
                best_ns = m; torch.save(meta, out / "best_ns.pt")          # best NEXT-STATE
    print(f"done. best value_AUC {best_v:.3f} (best.pt)  best val_ns {best_ns:.4f} (best_ns.pt)")
    print("GATE: this VALUE_AUC must beat the raw-feature / v2-latent baseline "
          "(run scripts/value_probe.py) — that's the test the v2 latent FAILED.")


if __name__ == "__main__":
    main()
