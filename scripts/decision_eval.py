#!/usr/bin/env python3
"""Decision-frame eval — strip out the momentum-trivial frames and ask whether the
world model learns anything BEYOND const-velocity.

Aggregate position loss is dominated by frames where players walk in a straight line
(momentum = the whole answer; model and const-vel both win, which proves nothing).
The honest test is: on frames where the player does something OTHER than continue
straight — a turn, a stop, a start — does the model still beat const-velocity?

We bucket every alive-player/frame by its DEVIATION-FROM-STRAIGHT in the next k frames:
    deviation = || truth_pos(t+k) - [pos(t) + k*(pos(t)-pos(t-1))] ||   (game units)
i.e. how far the player ends up from where const-velocity says they'd be. Low bucket =
boring straight motion. High bucket = a decision. For each bucket we report mean xy
error (game units) for:
    copy       : assume no motion         (truth displacement magnitude)
    const-vel  : straight extrapolation   (the momentum baseline)
    MODEL      : the world model
plus skill = (const_vel - model)/const_vel. If model skill stays positive (or grows)
in the HIGH-deviation buckets, it learned corrections momentum can't give. If model
error tracks const-vel up the buckets, it IS basically a velocity integrator.

Usage: python scripts/decision_eval.py --ckpt outputs/wm_3map/h8_mt/best_ns.pt
"""
from __future__ import annotations
import argparse, shutil, sys, tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS  # noqa

MAPS = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train"]
BUCKETS = [0, 25, 75, 150, 300, 1e9]   # deviation-from-straight, game units
BNAMES = ["0-25u", "25-75u", "75-150u", "150-300u", "300u+"]


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map/h8_mt/best_ns.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--stride", type=int, default=4, help="frame stride within a round")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck.get("per_player_dim", 56)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd)
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))
    cv_res = bool(a.get("cv_residual", False))
    print(f"ckpt step {ck.get('step')}  horizon k={k} ({k*125}ms)  per_player={ppd}  "
          f"maps={sorted(keep)}  cv_residual={cv_res}")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    # gather per-sample errors tagged by deviation bucket; also a per-map breakdown on the top bucket
    dev_all, m_all, cv_all, cp_all, mapidx = [], [], [], [], []
    px = torch.tensor([p*ppd+0 for p in range(N_PLAYERS)])
    py = torch.tensor([p*ppd+1 for p in range(N_PLAYERS)])
    alive_i = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])

    for r, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        ts = list(range(L - 1, T - k, args.stride))
        if not ts:
            continue
        for i in range(0, len(ts), args.batch):
            chunk = ts[i:i+args.batch]
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)   # [b,L,F]
            res = model(wins)[:, -1, :].cpu()                                   # [b,F] residual
            for j, t in enumerate(chunk):
                cur, prev, fut = r[t], r[t-1], r[t+k]
                pred = cur + res[j] + (k * (cur - prev) if cv_res else 0.0)
                alive = cur[alive_i] > 0.5
                if alive.sum() == 0:
                    continue
                # xy in game units
                cx, cy = cur[px]*3000, cur[py]*3000
                vx, vy = (cur[px]-prev[px])*3000, (cur[py]-prev[py])*3000   # per-tick vel
                fx, fy = fut[px]*3000, fut[py]*3000
                gx, gy = pred[px]*3000, pred[py]*3000
                cvx, cvy = cx + k*vx, cy + k*vy
                dev = ((fx-cvx)**2 + (fy-cvy)**2).sqrt()           # deviation from straight
                m_e = ((fx-gx)**2 + (fy-gy)**2).sqrt()            # model error
                cv_e = ((fx-cvx)**2 + (fy-cvy)**2).sqrt()         # = dev
                cp_e = ((fx-cx)**2 + (fy-cy)**2).sqrt()           # copy (displacement)
                sel = alive
                dev_all.append(dev[sel]); m_all.append(m_e[sel])
                cv_all.append(cv_e[sel]); cp_all.append(cp_e[sel])
                mapidx += [MAPS.index(m["map_name"])] * int(sel.sum())

    dev = torch.cat(dev_all); me = torch.cat(m_all); cve = torch.cat(cv_all); cpe = torch.cat(cp_all)
    mapidx = torch.tensor(mapidx)
    N = len(dev)
    print(f"\nalive player-frame samples: {N}\n")
    print(f"{'deviation bucket':16s} {'n':>7s} {'%mass':>6s}  "
          f"{'copy':>8s} {'const-vel':>9s} {'MODEL':>8s}  {'skill vs CV':>11s}")
    total_mass = me.sum().item()
    for bi in range(len(BNAMES)):
        lo, hi = BUCKETS[bi], BUCKETS[bi+1]
        sel = (dev >= lo) & (dev < hi)
        n = int(sel.sum())
        if n == 0:
            print(f"{BNAMES[bi]:16s} {0:>7d}"); continue
        cp, cv, mo = cpe[sel].mean().item(), cve[sel].mean().item(), me[sel].mean().item()
        mass = me[sel].sum().item()/total_mass*100
        skill = (cv-mo)/cv*100 if cv > 0 else 0.0
        print(f"{BNAMES[bi]:16s} {n:>7d} {mass:5.1f}%  {cp:7.0f}u {cv:8.0f}u {mo:7.0f}u  {skill:10.1f}%")

    ov_sk = (cve.mean()-me.mean())/cve.mean()*100
    print(f"\noverall: model {me.mean():.0f}u vs const-vel {cve.mean():.0f}u  "
          f"(skill {ov_sk:.1f}%), copy {cpe.mean():.0f}u")
    # high-deviation bucket per map (does the 'tactics' signal hold across maps?)
    hi_sel = dev >= 150
    print(f"\nhigh-deviation frames (>=150u, the decisions) — per map:")
    print(f"{'map':12s} {'n':>7s} {'const-vel':>9s} {'MODEL':>8s} {'skill':>7s}")
    for mp in sorted(keep):
        ms = hi_sel & (mapidx == MAPS.index(mp))
        n = int(ms.sum())
        if n == 0: continue
        cv, mo = cve[ms].mean().item(), me[ms].mean().item()
        print(f"{mp:12s} {n:>7d} {cv:8.0f}u {mo:7.0f}u {(cv-mo)/cv*100:6.1f}%")
    print("\nread: if MODEL skill stays clearly positive in 150-300u / 300u+ buckets, the model "
          "learned corrections momentum can't give. If skill collapses toward 0 there, it's a "
          "velocity integrator that only wins on the easy straight frames.")


if __name__ == "__main__":
    main()
