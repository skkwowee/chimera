#!/usr/bin/env python3
"""Decision-frame eval — does the world model learn anything BEYOND momentum?

SELECTION-EFFECT FIX. The previous version bucketed frames by deviation-from-
const-velocity, which is *literally the const-vel error*: within a bucket
[lo, hi) CV's error is bounded by construction while the model's is not. Any
predictor with roughly constant error E looks terrible in buckets < E and
great in buckets > E — the old -915% (0-25u) and +39/57% (150u+) skills were
partly that artifact, not evidence. The fix is twofold:

1) Buckets are now defined ONLY by properties of the TRUE trajectory — the
   turn angle theta between the smoothed past velocity v_past (mean of the
   last 4 frame deltas) and the true displacement d_true = pos(t+k)-pos(t).
   No predictor's error appears in the bucket key, so no baseline is
   mechanically clamped inside a bucket:
       stationary : speed_past < 5 u/frame and |d_true| < 25 u (angle undefined)
       straight   : theta <  20 deg   (moving)
       mild turn  : 20-60 deg
       hard turn  : 60-120 deg
       reversal   : > 120 deg
2) The model is compared against the BEST of four baselines per bucket, not
   just raw 1-frame CV:
       copy        : pos(t)                       (no motion)
       const-vel   : pos + k*(pos(t)-pos(t-1))    (1-frame velocity)
       smoothed-CV : pos + k*v_past               (4-frame mean velocity)
       damped-CV   : pos + alpha*k*v_past, alpha fit by least squares on
                     TRAIN data (closed form: sum(d_true.d_cv)/sum(d_cv.d_cv))
   damped-CV is the bar: if the model only beats raw CV but not a tuned
   scalar damping of momentum, "learned corrections" doesn't stand.

All errors are mean xy distance in game units (normalized*3000) over alive
player-frames. Skill = (best_baseline - model)/best_baseline per bucket.

Usage: python scripts/decision_eval.py --ckpt outputs/wm_3map/h8_mt/best_ns.pt
       [--max-rounds 3 --device cpu]   # smoke test
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS  # noqa
from _corpus import load_corpus

MAPS = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train"]
BNAMES = ["stationary", "straight", "mild turn", "hard turn", "reversal"]
THETA_EDGES = torch.tensor([20.0, 60.0, 120.0])     # deg -> bucket 1..4 via bucketize+1
STAT_SPEED, STAT_DISP = 5.0, 25.0                   # u/frame, u — stationary gate
SMOOTH = 4                                          # frames in the velocity average


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


def xy_units(frame, idx):
    """frame[idx] -> game units (normalized * 3000)."""
    return frame[idx] * 3000


def fit_alpha(train_pt, keep, k, px, py, alive_i, L, max_rounds=50, stride=8):
    """Least-squares damping for smoothed-CV: alpha = sum(d_true.d_cv)/sum(d_cv.d_cv).

    Streams a capped subsample of TRAIN rounds (alive players, same maps filter)
    so alpha is fit on data the model trained on, never on val.
    """
    blob = load_corpus(train_pt, maps=keep, tag="train")
    num = den = 0.0; used = 0
    for r, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        if L + k > T:
            continue
        for t in range(L - 1, T - k, stride):
            cur, fut = r[t], r[t + k]
            alive = cur[alive_i] > 0.5
            if alive.sum() == 0:
                continue
            vx = (xy_units(cur, px) - xy_units(r[t - SMOOTH], px)) / SMOOTH
            vy = (xy_units(cur, py) - xy_units(r[t - SMOOTH], py)) / SMOOTH
            dtx = xy_units(fut, px) - xy_units(cur, px)
            dty = xy_units(fut, py) - xy_units(cur, py)
            cvx, cvy = k * vx, k * vy
            num += float((dtx[alive]*cvx[alive] + dty[alive]*cvy[alive]).sum())
            den += float((cvx[alive]**2 + cvy[alive]**2).sum())
        used += 1
        if used >= max_rounds:
            break
    del blob; gc.collect()
    if den == 0:
        raise RuntimeError("alpha fit: no usable train frames")
    return num / den, used


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map/h8_mt/best_ns.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--train-pt", default="data/processed/tick_sequences/train_v3.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--stride", type=int, default=4, help="frame stride within a round")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-rounds", type=int, default=0, help="cap val rounds (0 = all)")
    ap.add_argument("--alpha-rounds", type=int, default=50, help="train rounds for alpha fit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck.get("per_player_dim", 56)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))
    cv_res = bool(a.get("cv_residual", False))
    print(f"ckpt step {ck.get('step')}  horizon k={k} ({k*125}ms)  per_player={ppd}  "
          f"maps={sorted(keep)}  cv_residual={cv_res}")

    px = torch.tensor([p*ppd+0 for p in range(N_PLAYERS)])
    py = torch.tensor([p*ppd+1 for p in range(N_PLAYERS)])
    alive_i = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])

    alpha, a_rounds = fit_alpha(args.train_pt, keep, k, px, py, alive_i, L,
                                max_rounds=args.alpha_rounds)
    print(f"damped-CV alpha = {alpha:.4f}  (LS fit on {a_rounds} train rounds, stride 8)")

    blob = load_corpus(args.val_pt, maps=keep, tag="val")
    # per alive-player-frame: bucket id, the 5 errors, map index
    bid_all, errs_all, mapidx = [], [], []
    n_rounds = 0
    for r, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        ts = list(range(L - 1, T - k, args.stride))
        if not ts:
            continue
        n_rounds += 1
        if args.max_rounds and n_rounds > args.max_rounds:
            break
        for i in range(0, len(ts), args.batch):
            chunk = ts[i:i+args.batch]
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)   # [b,L,F]
            res = model.gen_residual(wins)[:, -1, :].cpu()                      # [b,F] residual
            for j, t in enumerate(chunk):
                cur, prev, fut = r[t], r[t-1], r[t+k]
                pred = cur + res[j] + (k * (cur - prev) if cv_res else 0.0)
                alive = cur[alive_i] > 0.5
                if alive.sum() == 0:
                    continue
                cx, cy = xy_units(cur, px), xy_units(cur, py)
                fx, fy = xy_units(fut, px), xy_units(fut, py)
                gx, gy = xy_units(pred, px), xy_units(pred, py)
                v1x, v1y = cx - xy_units(prev, px), cy - xy_units(prev, py)       # 1-frame vel
                vsx = (cx - xy_units(r[t-SMOOTH], px)) / SMOOTH                   # smoothed vel
                vsy = (cy - xy_units(r[t-SMOOTH], py)) / SMOOTH
                dtx, dty = fx - cx, fy - cy                                       # true displacement
                # truth-only bucket key: stationary gate, else turn angle
                speed = (vsx**2 + vsy**2).sqrt()
                disp = (dtx**2 + dty**2).sqrt()
                cos = (vsx*dtx + vsy*dty) / (speed*disp + 1e-9)
                theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
                bid = torch.bucketize(theta, THETA_EDGES) + 1                     # 1..4
                bid[(speed < STAT_SPEED) & (disp < STAT_DISP)] = 0
                errs = torch.stack([
                    disp,                                                         # copy
                    ((dtx - k*v1x)**2 + (dty - k*v1y)**2).sqrt(),                 # const-vel
                    ((dtx - k*vsx)**2 + (dty - k*vsy)**2).sqrt(),                 # smoothed-CV
                    ((dtx - alpha*k*vsx)**2 + (dty - alpha*k*vsy)**2).sqrt(),     # damped-CV
                    ((fx - gx)**2 + (fy - gy)**2).sqrt(),                         # MODEL
                ], dim=1)                                                         # [10,5]
                bid_all.append(bid[alive]); errs_all.append(errs[alive])
                mapidx += [MAPS.index(m["map_name"])] * int(alive.sum())

    bid = torch.cat(bid_all); errs = torch.cat(errs_all); mapidx = torch.tensor(mapidx)
    N = len(bid)
    COLS = ["copy", "const-vel", "smooth-CV", "damped-CV", "MODEL"]
    used = min(n_rounds, args.max_rounds) if args.max_rounds else n_rounds
    print(f"\nalive player-frame samples: {N}  (val rounds used: {used})\n")
    hdr = f"{'bucket':12s} {'n':>7s} {'%frm':>6s}  " + " ".join(f"{c:>9s}" for c in COLS) \
          + f"  {'best-base':>9s} {'skill':>7s}"
    print(hdr)

    def row(name, sel):
        n = int(sel.sum())
        if n == 0:
            print(f"{name:12s} {0:>7d}"); return
        mu = errs[sel].mean(dim=0)                       # [5]
        best = mu[:4].min().item(); mo = mu[4].item()
        skill = (best - mo) / best * 100 if best > 0 else 0.0
        print(f"{name:12s} {n:>7d} {n/N*100:5.1f}%  "
              + " ".join(f"{v:8.0f}u" for v in mu.tolist())
              + f"  {COLS[int(mu[:4].argmin())]:>9s} {skill:6.1f}%")

    for bi, bn in enumerate(BNAMES):
        row(bn, bid == bi)
    row("ALL", torch.ones(N, dtype=torch.bool))

    # hard-turn + reversal per map: does any 'beyond momentum' signal hold across maps?
    hi_sel = bid >= 3
    print("\nhard turn + reversal frames — per map (model vs best baseline on that subset):")
    print(f"{'map':12s} {'n':>7s} {'best-base':>10s} {'MODEL':>8s} {'skill':>7s}")
    for mp in sorted(keep):
        ms = hi_sel & (mapidx == MAPS.index(mp))
        n = int(ms.sum())
        if n == 0:
            continue
        mu = errs[ms].mean(dim=0)
        best = mu[:4].min().item(); mo = mu[4].item()
        print(f"{mp:12s} {n:>7d} {best:9.0f}u {mo:7.0f}u {(best-mo)/best*100:6.1f}%")

    print("\nread: buckets are truth-trajectory-based (turn angle), so no baseline is bounded "
          "by construction inside a bucket. The claim 'model learned corrections momentum "
          "can't give' requires positive skill vs the BEST baseline — especially damped-CV — "
          "in the turn/reversal buckets, not just vs raw const-vel.")


if __name__ == "__main__":
    main()
