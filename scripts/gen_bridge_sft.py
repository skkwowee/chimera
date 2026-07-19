#!/usr/bin/env python3
"""Templated-from-predictive-heads SFT-set generator (bridge-design.md §5 / §7 step 2).

Produces single-moment `(latent-input, predictive-channels, target-text)` pairs for
the Phase-2 bridge SFT, locally on the 4090 (no pod). The target text is grounded in
the world model's PREDICTIVE outputs — the foresight that raw features don't contain:

  - value head        P(CT win) at the current frame,
  - sampled rollout   value TRAJECTORY at +2s / +4s (mean + spread over K rollouts),
  - dist head         per-player predicted next movement (hold / reposition / push).

This is the §5 "A'" decision: static state ("mid-round, CT side") is allowed as
qualitative scaffolding but is NOT the grounding — the grounding is the foresight,
which is not computable from the current frame without the forward model. That is
exactly what ablate-the-latent must reward (latent-on >> latent-off). The quantities
in the text come ONLY through the latent/soft tokens, so the soft tokens are
load-bearing by construction.

Output cache (torch.save), per §4 (ship to the pod; no world model needed there):
  grid [N,11,512] f16, channels [N,11,3] f16, value_logit [N], won [N],
  v_future {2s,4s} mean/spread, prompt[N] str, target[N] str, meta[N].

Usage:
  python scripts/gen_bridge_sft.py --ckpt outputs/wm_3map_dist_v3m/h8_mt/best.pt \
      --split data/processed/tick_sequences/train_v3m.pt --max-pairs 30000 \
      --out data/processed/bridge_sft/train_single.pt
  python scripts/gen_bridge_sft.py --smoke      # tiny CPU run on val, prints samples
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus import load_corpus

from src.bridge.wm_interface import N_PRED_CHANNELS, extract, load_world_model

DATA = Path("data/processed/tick_sequences")
DIST_EDGES_U = [8.0, 24.0, 56.0, 120.0, 248.0, 512.0]  # ring edges (game units), mirrors train_world_model


# --------------------------------------------------------------- rollout foresight
@torch.no_grad()
def rollout_value(model, x, steps, record, n_samples, chunk=128):
    """Sampled autoregressive value rollout for a dist-head checkpoint.

    Mirrors rollout_eval.roll_step for the cv_residual=False case (dist head):
    pred = cur + sampled-dist-decode residual, slide window, re-read value. Returns
    {step: value_logit [B, n_samples]} for the recorded steps.

    The WM attends over the flattened L*(P+1)=1056-token sequence, so attention
    scores scale with (rows * seq^2) — process rows in chunks to bound memory
    independent of the frame batch. Rows are independent, so chunking is exact."""
    B = x.shape[0]
    rows = x.unsqueeze(1).expand(B, n_samples, *x.shape[1:]).reshape(B * n_samples, *x.shape[1:])
    acc = {s: [] for s in record}
    for c in range(0, rows.shape[0], chunk):
        buf = rows[c:c + chunk].clone()
        for s in range(1, steps + 1):
            res = model.gen_residual(buf, sample=True)[:, -1, :]
            pred = buf[:, -1] + res
            buf = torch.cat([buf[:, 1:], pred[:, None]], dim=1)
            if s in record:
                acc[s].append(model.heads(buf)["value"][:, -1])
    return {s: torch.cat(acc[s]).reshape(B, n_samples) for s in record}


# --------------------------------------------------------------------- templating
def _pct(logit):
    return 100.0 / (1.0 + math.exp(-logit))


def value_phrase(rng, pct):
    side = "CT" if pct >= 50 else "T"
    edge = abs(pct - 50)
    if edge < 6:
        s = rng.choice(["the round is roughly even", "this is close to a coin-flip",
                        "neither side holds a clear edge"])
    elif edge < 18:
        s = rng.choice([f"{side} is modestly favoured", f"{side} holds a slight edge",
                        f"the model leans {side}"])
    else:
        s = rng.choice([f"{side} is in a strong position", f"{side} is clearly favoured",
                        f"the model reads this as a {side}-sided round"])
    return f"{s} (CT win probability ~{pct:.0f}%)"


def trajectory_phrase(rng, now, fut4):
    """Brief, sign-only note — DEMOTED: the rollout value-trajectory is near-flat
    (the value head is insensitive to short-horizon movement), so only mention it
    when there is a real shift, never as quantified boilerplate."""
    delta = fut4 - now
    if abs(delta) < 3:
        return ""                                  # no boilerplate when nothing moves
    drift = "edge up slightly" if delta > 0 else "soften slightly"
    return rng.choice([f" The model expects CT's read to {drift} over the next few seconds.",
                       f" Over the next few seconds the model sees this {drift}."])


def _class_to_mag_dir(c):
    """dist class id -> (displacement magnitude in game units over the horizon,
    bearing in radians). Class 0 = stationary -> (0, None). Mirrors train_world_model
    dist_centers(): 6 rings (log-spaced) x 16 direction bins of 22.5deg."""
    if c == 0:
        return 0.0, None
    ring, d = (c - 1) // 16, (c - 1) % 16
    mag = math.sqrt(DIST_EDGES_U[ring] * DIST_EDGES_U[ring + 1]) if ring < len(DIST_EDGES_U) - 1 else 700.0
    return mag, d * (2 * math.pi / 16)


def _resultant(bearings):
    """Mean resultant length in [0,1]: 1 = all moving the same way (coordinated),
    ~0 = scattered. The foresight 'is this a coordinated push' signal."""
    c = sum(math.cos(b) for b in bearings) / len(bearings)
    s = sum(math.sin(b) for b in bearings) / len(bearings)
    return math.hypot(c, s)


def _team_clause(rng, team, movers, fast):
    """Verb-fragment that reads cleanly after 'The model expects '."""
    if not movers:
        return rng.choice([f"{team} to hold position", f"{team} to stay put"])
    if len(fast) >= 2:
        if _resultant([b for _, b in fast]) > 0.6:
            return rng.choice([f"a coordinated {team} push ({len(fast)} players)",
                              f"{len(fast)} {team} players pushing together"])
        return rng.choice([f"{len(fast)} {team} players pushing from spread positions",
                          f"scattered fast movement from {len(fast)} {team} players"])
    n = len(movers)
    return rng.choice([f"{n} {team} player{'s' if n > 1 else ''} repositioning",
                      f"{n} {team} player{'s' if n > 1 else ''} on the move"])


def movement_phrase(rng, top_classes, alive):
    """Team-aware, alive-masked predicted-movement foresight (slots 0-4=T, 5-9=CT,
    per build_v3_features.py). PRIMARY grounding now the value-trajectory is flat:
    which side is predicted to hold vs push, how many, and whether the pushers move
    coherently — all from the dist head, none of it in the current frame's text."""
    teams = {"T": [], "CT": []}
    for p, (c, al) in enumerate(zip(top_classes, alive)):
        if al:
            teams["T" if p < 5 else "CT"].append(_class_to_mag_dir(c))
    clauses = []
    for team in ("CT", "T"):
        if not teams[team]:
            continue
        movers = [(m, b) for m, b in teams[team] if m >= 56.0]      # > hold threshold
        fast = [(m, b) for m, b in movers if m >= 248.0]
        clauses.append(_team_clause(rng, team, movers, fast))
    if not clauses:
        return "The model predicts both sides largely holding position."
    return "The model expects " + "; ".join(clauses) + "."


PROMPTS = ["Assess this moment for the CT side.",
           "What is the tactical read on the current situation?",
           "Evaluate the state of this round right now."]


def _cap(s):  # uppercase first char only (str.capitalize would lowercase "CT"/"T")
    return s[0].upper() + s[1:] if s else s


def build_target(rng, pct, fut4, top_classes, alive):
    return (_cap(value_phrase(rng, pct)) + "."
            + trajectory_phrase(rng, pct, fut4)
            + " " + movement_phrase(rng, top_classes, alive))


# --------------------------------------------------------------------- generation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map_dist_v3m/h8_mt/best.pt")
    ap.add_argument("--split", default=str(DATA / "train_v3m.pt"))
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--out", default="data/processed/bridge_sft/train_single.pt")
    ap.add_argument("--max-pairs", type=int, default=30000)
    ap.add_argument("--per-round", type=int, default=12, help="max anchors sampled per round")
    ap.add_argument("--stride", type=int, default=16, help="frames between candidate anchors")
    ap.add_argument("--frame-batch", type=int, default=64)
    ap.add_argument("--rollout-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.split = str(DATA / "val_v3m.pt"); args.max_pairs = 24; args.per_round = 4
        args.frame_batch = 8; args.rollout_samples = 4; args.device = "cpu"
        print("[smoke] tiny CPU run")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model, meta = load_world_model(args.ckpt, device=args.device)
    assert meta["dist"], "generator needs the distributional head (movement foresight)"
    L = meta["window"]
    keep = set(args.maps.split(","))
    # read-only consumer (windows are torch.stack copies) -> safe on load_corpus mmap
    blob = load_corpus(args.split, maps=keep, tag="sft-src")
    # record steps: horizon k=8 -> 1s/step, so +2s = step 2, +4s = step 4
    REC2, REC4 = 2, 4
    dev = torch.device(args.device)
    print(f"ckpt window={L} maps={sorted(keep)} -> {args.out}  (target {args.max_pairs} pairs)")

    # ---- gather candidate (round, anchor-tick) anchors ----
    anchors = []
    for ri, (r, m) in enumerate(zip(blob["tensors"], blob["metas"])):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        ts = list(range(L - 1, T - 1, args.stride))
        rng.shuffle(ts)
        for t in ts[:args.per_round]:
            anchors.append((ri, t, 1.0 if m.get("winner") == "ct" else 0.0, m.get("map_name")))
    rng.shuffle(anchors)
    anchors = anchors[:args.max_pairs]
    print(f"sampling {len(anchors)} anchors from {len(blob['tensors'])} rounds")

    grids, chans, vlogits, wons, f2m, f4m, f4s, prompts, targets, metas = ([] for _ in range(10))
    for i in range(0, len(anchors), args.frame_batch):
        batch = anchors[i:i + args.frame_batch]
        x = torch.stack([blob["tensors"][ri][t - L + 1:t + 1] for ri, t, _, _ in batch]).to(dev)
        grid, _, channels = extract(model, x)                      # [B,11,512],[B,11,3]
        out = model.heads(x)
        vlog = out["value"][:, -1]                                 # [B]
        top = out["dist_logits"][:, -1].argmax(-1)                 # [B,10]
        # alive per slot from the last frame (per-player dim 13, build_v3_features.py)
        players_last = x[:, -1, :model.player_block].reshape(x.shape[0], model.n_players, model.ppd)
        alive = (players_last[..., 13] > 0.5)                      # [B,10] bool
        if args.rollout_samples > 0:
            roll = rollout_value(model, x, steps=REC4, record={REC2, REC4}, n_samples=args.rollout_samples)
            v2 = torch.sigmoid(roll[REC2]) * 100.0                 # [B,K]
            v4 = torch.sigmoid(roll[REC4]) * 100.0
        else:                                                      # skip rollout (fast, e.g. val set)
            vp = (torch.sigmoid(vlog) * 100.0).unsqueeze(1)        # [B,1]; trajectory note then never fires
            v2 = v4 = vp
        for b, (ri, t, won, mp) in enumerate(batch):
            pct = _pct(vlog[b].item())
            fut2, fut4 = v2[b].mean().item(), v4[b].mean().item()
            sp4 = v4[b].std().item() if v4.shape[1] > 1 else 0.0
            tgt = build_target(rng, pct, fut4, top[b].tolist(), alive[b].tolist())
            grids.append(grid[b].half().cpu()); chans.append(channels[b].half().cpu())
            vlogits.append(vlog[b].item()); wons.append(won)
            f2m.append(fut2); f4m.append(fut4); f4s.append(sp4)
            prompts.append(rng.choice(PROMPTS)); targets.append(tgt)
            metas.append({"round": int(ri), "tick": int(t), "map": mp})
        if (i // args.frame_batch) % 20 == 0:
            print(f"  {len(targets)}/{len(anchors)}")

    cache = {
        "grid": torch.stack(grids), "channels": torch.stack(chans),
        "value_logit": torch.tensor(vlogits), "won": torch.tensor(wons),
        "v_future_2s": torch.tensor(f2m), "v_future_4s": torch.tensor(f4m),
        "v_future_4s_spread": torch.tensor(f4s),
        "prompt": prompts, "target": targets, "meta": metas,
        "schema": {"latent_dim": meta["d_model"], "n_tokens": 11,
                   "n_pred_channels": N_PRED_CHANNELS, "ckpt": args.ckpt,
                   "rollout_samples": args.rollout_samples, "horizon_s_per_step": 1},
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, outp)
    sz = outp.stat().st_size / 1e6
    print(f"\nwrote {len(targets)} pairs -> {outp} ({sz:.1f} MB)")

    # readability guard (§3): always eyeball a few pairs
    print("\n--- sample pairs (readability check) ---")
    sidx = list(range(len(targets))); rng.shuffle(sidx)
    samples = []
    for j in sidx[:8]:
        print(f"\n[{metas[j]['map']} r{metas[j]['round']} t{metas[j]['tick']}]  "
              f"won={'CT' if wons[j] else 'T'}")
        print(f"  Q: {prompts[j]}")
        print(f"  A: {targets[j]}")
        samples.append({"prompt": prompts[j], "target": targets[j], **metas[j],
                        "won_ct": bool(wons[j])})
    side = outp.with_suffix(".samples.jsonl")
    side.write_text("\n".join(json.dumps(s) for s in samples))
    print(f"\nsamples -> {side}")


if __name__ == "__main__":
    main()
