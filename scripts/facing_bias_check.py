#!/usr/bin/env python3
"""Does the dist head over-rely on FACING direction (a learned shortcut)?

Facing (sin/cos yaw, dims 3-4) is a direct input feature; velocity must be
derived from the window. Hypothesis: for ambiguous frames the model defaults
to "facing, one ring out" — wrong exactly when movement is lateral (A/D
strafing) or against the crosshair.

Test on alive, MOVING player-frames (true displacement >= 25u):
  For each frame, three candidate directions: FACING (yaw), MOMENTUM
  (smoothed past velocity), TRUTH (actual displacement t->t+k).
  On CONFLICT frames — facing vs momentum >60 deg apart — count:
    truth closer to facing vs momentum     (what reality does)
    PREDICTION closer to facing vs momentum (what the model does)
  Model% facing >> truth% facing  ==> shortcut bias, quantified.
Also reports mean |angle(pred, truth)| split by conflict/no-conflict.

Usage: python scripts/facing_bias_check.py --ckpt outputs/wm_3map_dist/h8_mt/best_ns.pt
"""
from __future__ import annotations
import argparse, math, shutil, sys, tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS  # noqa

SMOOTH = 4
MOVE_MIN_U = 25.0          # ignore near-stationary truth (direction undefined)
SPEED_MIN_U = 3.0          # momentum direction undefined below this (u/frame)
CONFLICT_DEG = 60.0


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


def ang_between(ax, ay, bx, by):
    """Absolute angle (deg) between 2-D vectors, elementwise tensors."""
    dot = ax*bx + ay*by
    na = (ax**2 + ay**2).sqrt(); nb = (bx**2 + by**2).sqrt()
    c = (dot / (na*nb + 1e-9)).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map_dist/h8_mt/best_ns.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-rounds", type=int, default=0)
    ap.add_argument("--corrupt-yaw", choices=["none", "shuffle"], default="none",
                    help="shuffle = permute sin/cos yaw across the batch (permutation "
                         "importance): if facing-following collapses, the shortcut is "
                         "CAUSAL on the yaw input dims, not just correlated")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck["per_player_dim"]
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))
    print(f"ckpt step {ck.get('step')}  dist={a.get('dist_head', False)}  horizon {k*125}ms  "
          f"corrupt_yaw={args.corrupt_yaw}")
    if args.corrupt_yaw == "shuffle":
        print("(input yaw permuted across batch; TRUTH/MODEL rows still scored against "
              "the REAL facing — if MODEL's facing% collapses, the shortcut is causal)")

    px = torch.tensor([p*ppd+0 for p in range(N_PLAYERS)])
    py = torch.tensor([p*ppd+1 for p in range(N_PLAYERS)])
    syaw = torch.tensor([p*ppd+3 for p in range(N_PLAYERS)])   # sin(yaw)
    cyaw = torch.tensor([p*ppd+4 for p in range(N_PLAYERS)])   # cos(yaw)
    alive_i = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])

    # accumulators over conflict frames
    truth_face = truth_mom = pred_face = pred_mom = 0
    ang_pt_conf, ang_pt_noconf = [], []
    n_rounds = n_frames = 0

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False, mmap=True)
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
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)
            if args.corrupt_yaw == "shuffle" and wins.shape[0] > 1:
                g = torch.Generator(device="cpu").manual_seed(i)
                perm = torch.randperm(wins.shape[0], generator=g).to(args.device)
                yd = torch.tensor([p*ppd+d for p in range(N_PLAYERS) for d in (3, 4)],
                                  device=args.device)
                wins[..., yd] = wins[perm][..., yd]       # batch-permuted facing
            res = model.gen_residual(wins)[:, -1, :].cpu()
            for j, t in enumerate(chunk):
                cur, fut = r[t], r[t+k]
                alive = cur[alive_i] > 0.5
                # vectors (game units)
                dtx, dty = (fut[px]-cur[px])*3000, (fut[py]-cur[py])*3000     # truth
                pdx, pdy = res[j][px]*3000, res[j][py]*3000                   # prediction
                vx = (cur[px] - r[t-SMOOTH][px])*3000/SMOOTH                  # momentum
                vy = (cur[py] - r[t-SMOOTH][py])*3000/SMOOTH
                # facing: yaw stored as sin/cos; forward = (cos, sin)
                fx, fy = cur[cyaw], cur[syaw]
                dmag = (dtx**2+dty**2).sqrt(); spd = (vx**2+vy**2).sqrt()
                ok = alive & (dmag >= MOVE_MIN_U) & (spd >= SPEED_MIN_U)
                if ok.sum() == 0:
                    continue
                a_fm = ang_between(fx, fy, vx, vy)                 # facing vs momentum
                a_tf = ang_between(dtx, dty, fx, fy)               # truth vs facing
                a_tm = ang_between(dtx, dty, vx, vy)               # truth vs momentum
                a_pf = ang_between(pdx, pdy, fx, fy)               # pred vs facing
                a_pm = ang_between(pdx, pdy, vx, vy)               # pred vs momentum
                a_pt = ang_between(pdx, pdy, dtx, dty)             # pred vs truth
                conf = ok & (a_fm > CONFLICT_DEG)
                noconf = ok & ~conf
                truth_face += int((a_tf[conf] < a_tm[conf]).sum())
                truth_mom  += int((a_tf[conf] >= a_tm[conf]).sum())
                pred_face  += int((a_pf[conf] < a_pm[conf]).sum())
                pred_mom   += int((a_pf[conf] >= a_pm[conf]).sum())
                ang_pt_conf.append(a_pt[conf]); ang_pt_noconf.append(a_pt[noconf])
                n_frames += int(ok.sum())

    nc = truth_face + truth_mom
    print(f"\nmoving alive player-frames: {n_frames}  "
          f"conflict frames (facing vs momentum >{CONFLICT_DEG:.0f}deg): {nc} ({100*nc/max(1,n_frames):.0f}%)\n")
    print("on CONFLICT frames, direction followed:")
    print(f"  {'':12s} {'facing':>8s} {'momentum':>9s}")
    print(f"  {'TRUTH':12s} {100*truth_face/max(1,nc):7.1f}% {100*truth_mom/max(1,nc):8.1f}%")
    print(f"  {'MODEL':12s} {100*pred_face/max(1,nc):7.1f}% {100*pred_mom/max(1,nc):8.1f}%")
    bias = 100*pred_face/max(1,nc) - 100*truth_face/max(1,nc)
    apc = torch.cat(ang_pt_conf).mean() if ang_pt_conf else float("nan")
    apn = torch.cat(ang_pt_noconf).mean() if ang_pt_noconf else float("nan")
    print(f"\nfacing-shortcut bias (model% - truth% on facing): {bias:+.1f} pp")
    print(f"mean |angle(pred, truth)|: conflict {apc:.0f}deg vs aligned {apn:.0f}deg")
    print("\nread: MODEL following facing substantially more than TRUTH does on conflict "
          "frames = learned facing shortcut. If MODEL ~ TRUTH, the facing-lean you see "
          "is the data's own statistics, not a model bias.")


if __name__ == "__main__":
    main()
