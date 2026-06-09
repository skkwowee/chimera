#!/usr/bin/env python3
"""Walk through ONE concrete generation, in human-readable game units — "what does
the world model actually output?"

1) Seed the model with a real 128-frame context window ending at frame t.
2) GENERATE the next state (t+horizon, i.e. +1s): pred_frame = last_frame + residual.
3) Print, per player, CURRENT -> GENERATED -> TRUTH position (game units) + the
   error, plus the value head's P(CT win) vs the actual round outcome.
4) Then do a short AUTOREGRESSIVE rollout (feed predictions back) and show how
   position error grows vs a const-velocity baseline — what drift looks like.

Usage: python scripts/gen_demo.py --ckpt outputs/wm_3map/h8_mt/best_ns.pt --round 0 --frame 200
"""
from __future__ import annotations
import argparse, math, shutil, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS  # noqa

MAPS = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train"]
SLOTS = ["T1","T2","T3","T4","T5","CT1","CT2","CT3","CT4","CT5"]


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map/h8_mt/best_ns.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--round", type=int, default=None, help="round index (default: first long enough)")
    ap.add_argument("--frame", type=int, default=None, help="anchor frame t (default: mid-round)")
    ap.add_argument("--rollout", type=int, default=8, help="autoregressive steps to show drift")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck.get("per_player_dim", 56)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    cv_res = bool(a.get("cv_residual", False))
    print(f"ckpt step {ck.get('step')}  window={L}  horizon k={k} ({k*125}ms)  per_player={ppd}  "
          f"cv_residual={cv_res}  val_ns={ck.get('val_ns', float('nan')):.4f}\n")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    need = L + args.rollout * k + 1
    # pick round
    if args.round is not None:
        ridx = args.round
    else:
        ridx = next(i for i, t in enumerate(blob["tensors"]) if t.shape[0] >= need)
    r = blob["tensors"][ridx]; meta = blob["metas"][ridx]
    T = r.shape[0]
    if T < L + k + 1:
        print(f"round {ridx} too short ({T} frames); need >= {L+k+1}"); return
    t = args.frame if args.frame is not None else min(T - k - 1, L + (T - L) // 2)
    t = max(L - 1, min(t, T - k - 1))
    mp = MAPS[int(r[t].numpy()[N_PLAYERS*ppd:N_PLAYERS*ppd+7].argmax())]
    print(f"round {ridx}: {meta.get('demo_stem','?')} rnd {meta.get('round_num','?')}  "
          f"map={mp}  winner={meta.get('winner','?')}  T={T} frames  anchor t={t}\n")

    # --- one-step generation ---
    win = r[t - L + 1:t + 1].unsqueeze(0).to(args.device)            # [1,L,F] ending at t
    res = model.gen_residual(win)[0, -1].cpu()                       # residual (dist decode)
    vlogit = model.heads(win)["value"][0, -1].item()
    cur = r[t]; true = r[t + k]
    pred = cur + res + (k * (cur - r[t-1]) if cv_res else 0.0)
    vp = 1 / (1 + math.exp(-vlogit))

    def pos(frame, p):                                               # (x,y,z) game units
        b = frame.numpy()[p*ppd:p*ppd+3]
        return b[0]*3000, b[1]*3000, b[2]*500

    print(f"GENERATED next state  (t -> t+{k} = +{k*125}ms)")
    print(f"{'slot':5} {'aliv':>4}  {'-- current --':>22}  {'-- GENERATED --':>22}  "
          f"{'-- truth --':>22}  {'gen_err':>8} {'cv_err':>8}")
    pos_err_gen = pos_err_cv = nalive = 0.0
    velf = r[t] - r[t-1]
    for p in range(N_PLAYERS):
        alive = cur.numpy()[p*ppd+13] > 0.5
        cx,cy,cz = pos(cur,p); gx,gy,gz = pos(pred,p); tx,ty,tz = pos(true,p)
        cvx = cx + float(velf.numpy()[p*ppd+0])*3000*k
        cvy = cy + float(velf.numpy()[p*ppd+1])*3000*k
        ge = math.dist((gx,gy),(tx,ty)); ce = math.dist((cvx,cvy),(tx,ty))
        if alive:
            pos_err_gen += ge; pos_err_cv += ce; nalive += 1
        print(f"{SLOTS[p]:5} {int(alive):>4}  ({cx:6.0f},{cy:6.0f},{cz:5.0f})  "
              f"({gx:6.0f},{gy:6.0f},{gz:5.0f})  ({tx:6.0f},{ty:6.0f},{tz:5.0f})  "
              f"{ge:7.0f}u {ce:7.0f}u" + ("" if alive else "  (dead)"))
    if nalive:
        print(f"\nmean alive-player position error:  GENERATED {pos_err_gen/nalive:6.1f}u   "
              f"const-vel {pos_err_cv/nalive:6.1f}u   (lower=better)")
    print(f"value head: P(CT win)={vp:.2f}   actual winner={meta.get('winner','?')}  "
          f"{'OK' if (vp>0.5)==(meta.get('winner')=='ct') else 'MISS'}")

    # --- autoregressive rollout (compounding drift) ---
    R = args.rollout
    if T >= L + R * k + 1 and R > 0:
        buf = r[t - L + 1:t + 1].clone().unsqueeze(0).to(args.device)
        last = r[t].clone(); vel = (r[t] - r[t-1])
        pos_idx = torch.tensor([p*ppd+d for p in range(N_PLAYERS) for d in (0,1,2)])
        print(f"\nAUTOREGRESSIVE ROLLOUT (feed predictions back; {R} steps = {R*k*125}ms):")
        print(f"{'step':>4} {'t(ms)':>6} {'gen_pos_err':>12} {'const-vel':>10}  {'gen better':>10}")
        for step in range(1, R + 1):
            res = model.gen_residual(buf)[:, -1, :]
            cvb = (k * (buf[0, -1] - buf[0, -2])) if cv_res else 0.0
            predf = buf[0, -1] + res[0] + cvb
            truef = r[t + step * k].to(args.device)
            cvf = (last + (step*k)*vel).to(args.device)
            ge = (predf[pos_idx]-truef[pos_idx]).abs().mean().item()*3000/ (3000)  # normalized units
            ge_u = ((predf.cpu()[pos_idx]-truef.cpu()[pos_idx]).reshape(-1,3)[:, :2]*3000).pow(2).sum(1).sqrt().mean().item()
            ce_u = ((cvf.cpu()[pos_idx]-truef.cpu()[pos_idx]).reshape(-1,3)[:, :2]*3000).pow(2).sum(1).sqrt().mean().item()
            better = (ce_u-ge_u)/ce_u*100 if ce_u>0 else 0
            print(f"{step:>4} {step*k*125:>6} {ge_u:11.1f}u {ce_u:9.1f}u  {better:9.1f}%")
            buf = torch.cat([buf[:, 1:, :], predf.view(1,1,-1)], dim=1)
        print("\n(rollout caveat: predictions for static/categorical dims are fed back too, so "
              "this is a conservative read; position is what we track.)")


if __name__ == "__main__":
    main()
