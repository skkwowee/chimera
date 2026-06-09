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

    def latent(self, x):
        return self._run(x)


N_PLAYERS = 10   # frame is player-major: n_players x per_player_dim, then global


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
                 dropout=0.1, per_player_dim=56, n_players=N_PLAYERS):
        super().__init__()
        self.n_players = n_players
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
        """Multi-task forward: next-state residual + per-frame value logit."""
        h, L = self._grid(x)
        P = self.n_players
        pres = self.player_head(h[:, :, :P, :]).reshape(x.shape[0], L, self.player_block)
        gres = self.global_head(h[:, :, P, :])
        residual = torch.cat([pres, gres], dim=-1)
        value = self.value_head(h.mean(dim=2)).squeeze(-1)              # [B,L] P(CT win) logit
        return {"residual": residual, "value": value}

    def latent(self, x):
        h, L = self._grid(x)
        return h.mean(dim=2)                                            # [B,L,d] per-frame summary


def build_model(arch, feature_dim, d_model, layers, heads, per_player_dim=56):
    if arch == "flat":
        return WorldModelFlat(feature_dim, d_model, layers, heads)
    return WorldModelPlayers(feature_dim, d_model, layers, heads,
                             per_player_dim=per_player_dim)


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
def evaluate(model, loader, device, pred_mask, max_batches=50, cv_residual=False):
    model.eval()
    m_loss = copy_loss = cv_loss = n = 0.0
    vlog, vlab = [], []
    for bi, (x, y, x_prev, won) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y, x_prev = x.to(device), y.to(device), x_prev.to(device)
        true_res = (y - x)[..., pred_mask]                       # predict RAW state only
        out = model.heads(x) if hasattr(model, "heads") else {"residual": model(x)}
        cv_base = (model.horizon * (x - x_prev)) if cv_residual else 0.0
        pred_res = (out["residual"] + cv_base)[..., pred_mask]
        m_loss += huber(pred_res, true_res).item()
        copy_loss += huber(torch.zeros_like(true_res), true_res).item()
        cv_loss += huber((model.horizon * (x - x_prev))[..., pred_mask], true_res).item()
        n += 1
        if "value" in out:
            vlog.append(out["value"][:, -1].float().cpu()); vlab.append(won)
    model.train()
    v_auc = auc(torch.cat(vlog), torch.cat(vlab)) if vlog else float("nan")
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
    ap.add_argument("--train-pt", default=str(DATA_DIR / "train_v3.pt"))
    ap.add_argument("--val-pt", default=str(DATA_DIR / "val_v3.pt"))
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
    ppd = train_blob.get("per_player_dim", 56)

    if args.maps:
        keep = set(args.maps.split(","))
        def _filter(blob):
            idx = [i for i, m in enumerate(blob["metas"]) if m.get("map_name") in keep]
            blob["tensors"] = [blob["tensors"][i] for i in idx]
            blob["metas"] = [blob["metas"][i] for i in idx]
            return len(idx)
        nt, nv = _filter(train_blob), _filter(val_blob)
        print(f"map filter {sorted(keep)}: train {nt} rounds, val {nv} rounds")

    tr_ds = RoundWindows(train_blob["tensors"], train_blob["metas"], args.window, args.horizon, args.crops_per_round)
    va_ds = RoundWindows(val_blob["tensors"], val_blob["metas"], args.window, args.horizon, args.crops_per_round)
    print(f"feature_dim={fdim} per_player={ppd}  horizon={args.horizon} ({args.horizon*125}ms)  "
          f"window={args.window}  train_rounds={len(tr_ds.rounds)} (dropped {tr_ds.dropped}) "
          f"val_rounds={len(va_ds.rounds)}  value_weight={args.value_weight}")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=0 if args.smoke else 4)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=0 if args.smoke else 2)

    model = build_model(args.arch, fdim, args.d_model, args.layers, args.heads,
                        per_player_dim=ppd).to(dev)
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
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            o = model.heads(x)
            pred_res = o["residual"] + cv_base
            ns_loss = huber(pred_res[..., pred_mask], true_res[..., pred_mask])
            v_tgt = won.unsqueeze(1).expand_as(o["value"])
            v_loss = F.binary_cross_entropy_with_logits(o["value"].float(), v_tgt)
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
            m, c, cv, vauc = evaluate(model, va_ld, dev, pred_mask, max_batches=10 if args.smoke else 50,
                                      cv_residual=args.cv_residual)
            skill = (cv - m) / cv * 100 if cv > 0 else 0.0
            print(f"step {step:6d}  ns {ns_loss.item():.4f} v {v_loss.item():.3f}  "
                  f"val_ns {m:.4f} [copy {c:.4f} cv {cv:.4f}] skill {skill:+.1f}%  "
                  f"VALUE_AUC {vauc:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s")
            meta = {"model": model.state_dict(), "args": vars(args),
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
