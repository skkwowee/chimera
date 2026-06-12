#!/usr/bin/env python3
"""Bake a world-model GENERATION into JSON for the gen viewer (viewer/gen_viewer.html).

For each frame t of a chosen round it records, per player:
  - now        : ground-truth position at t            (solid dot)
  - gen        : model's GENERATED position at t+k      (ghost ring; +1s belief)
  - future     : ground-truth position at t+k           (faint X; what really happened)
plus the value head's P(CT win) and a separate AUTOREGRESSIVE rollout trail from a
chosen anchor (feed predictions back, R steps). Copies the awpy radar PNG so the
viewer is fully self-contained.

Usage:
  python scripts/export_gen_viewer.py --ckpt outputs/wm_3map/h8_mt/best_ns.pt --round 0
  # then open viewer/gen_viewer.html in a browser
"""
from __future__ import annotations
import argparse, json, math, shutil, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, dist_class, N_PLAYERS  # noqa
from awpy.data.map_data import MAP_DATA

MAPS = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train"]
SLOTS = ["T1","T2","T3","T4","T5","CT1","CT2","CT3","CT4","CT5"]
RADAR_DIR = Path.home() / ".awpy" / "maps"


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map/h8_mt/best_ns.pt")
    ap.add_argument("--value-ckpt", default=None,
                    help="separate ckpt for the value head (use best.pt — value AUC peaks "
                         "early and the late-step best_ns value head is overfit/saturated)")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--round", type=int, default=None, help="round index (default: first usable)")
    ap.add_argument("--anchor", type=int, default=None, help="rollout anchor frame (default: mid)")
    ap.add_argument("--rollout", type=int, default=12, help="autoregressive steps for the drift trail")
    ap.add_argument("--out", default="viewer/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck.get("per_player_dim", 56)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    cv_res = bool(a.get("cv_residual", False))

    vmodel = model
    if args.value_ckpt:
        vck = load_ckpt(args.value_ckpt); va = vck["args"]
        vmodel = build_model(va["arch"], vck["feature_dim"], va["d_model"], va["layers"],
                             va["heads"], per_player_dim=vck.get("per_player_dim", 56),
                             dist=va.get("dist_head", False))
        vmodel.load_state_dict(vck["model"]); vmodel.to(args.device).eval()
        print(f"value head from {args.value_ckpt} (step {vck.get('step')}, "
              f"value_auc {vck.get('value_auc', float('nan')):.3f})")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    ridx = args.round if args.round is not None else \
        next(i for i, t in enumerate(blob["tensors"]) if t.shape[0] >= L + k + 1)
    r = blob["tensors"][ridx]; meta = blob["metas"][ridx]; T = r.shape[0]
    mp = MAPS[int(r[L].numpy()[N_PLAYERS*ppd:N_PLAYERS*ppd+7].argmax())]
    md = MAP_DATA[mp]
    print(f"round {ridx}: {meta.get('demo_stem','?')} rnd {meta.get('round_num','?')}  "
          f"map={mp}  winner={meta.get('winner','?')}  T={T}  horizon={k} ({k*125}ms)")

    def gpos(frame, p):                       # game-unit (x,y) for player p
        b = frame[p*ppd:p*ppd+2]
        return float(b[0])*3000, float(b[1])*3000

    # ---- per-frame single-step generation (batched) ----
    starts = list(range(L - 1, T - k))        # anchor t with full window + a future
    frames_json = []
    xy_idx = torch.tensor([[p*ppd, p*ppd+1] for p in range(N_PLAYERS)])
    alive_idx = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])
    B = 128
    for i in range(0, len(starts), B):
        chunk = starts[i:i+B]
        wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)   # [b,L,F]
        out = model.heads(wins)
        res = model.gen_residual(wins)[:, -1, :].cpu()                     # [b,F] (dist decode)
        vout = out if vmodel is model else vmodel.heads(wins)
        val = torch.sigmoid(vout["value"][:, -1]).cpu()                    # [b]
        # surprise = NLL of the true displacement class (dist ckpts only)
        logp = (F.log_softmax(out["dist_logits"][:, -1].float(), -1).cpu()
                if "dist_logits" in out else None)
        for j, t in enumerate(chunk):
            cur = r[t]; fut = r[t+k]; prev = r[max(0, t-1)]
            pred = cur + res[j] + (k * (cur - prev) if cv_res else 0.0)
            nll = None
            if logp is not None:
                tc = dist_class((fut - cur)[xy_idx])                      # true class [P]
                nll = -logp[j].gather(1, tc.unsqueeze(1)).squeeze(1)
                nll[cur[alive_idx] <= 0.5] = 0.0
            players = []
            for p in range(N_PLAYERS):
                alive = float(cur[p*ppd+13]) > 0.5
                nx, ny = gpos(cur, p); gx, gy = gpos(pred, p); fx, fy = gpos(fut, p)
                pvx, pvy = gpos(prev, p)
                cvx, cvy = nx + (nx - pvx) * k, ny + (ny - pvy) * k   # const-vel +k
                players.append({
                    "slot": SLOTS[p], "side": "t" if p < 5 else "ct", "alive": alive,
                    "hp": round(float(cur[p*ppd+7])*100),
                    "x": round(nx), "y": round(ny),
                    "gx": round(gx), "gy": round(gy),
                    "fx": round(fx), "fy": round(fy),
                    "cvx": round(cvx), "cvy": round(cvy),
                    "err": round(math.dist((gx, gy), (fx, fy))),
                    "cverr": round(math.dist((cvx, cvy), (fx, fy))),
                    "nats": round(float(nll[p]), 1) if nll is not None else 0.0,
                })
            frames_json.append({"t": int(t), "value": round(float(val[j]), 3),
                                "surp": round(float(nll.sum()), 1) if nll is not None else 0.0,
                                "players": players})

    # ---- autoregressive rollout trail from an anchor ----
    anchor = args.anchor if args.anchor is not None else max(L - 1, min(T - args.rollout*k - 1, T // 2))
    rollout = None
    if T >= anchor + args.rollout * k + 1 and anchor >= L - 1:
        buf = r[anchor-L+1:anchor+1].clone().unsqueeze(0).to(args.device)
        steps = []
        for s in range(1, args.rollout + 1):
            res = model.gen_residual(buf)[:, -1, :]
            cvb = (k * (buf[0, -1] - buf[0, -2])) if cv_res else 0.0
            predf = (buf[0, -1] + res[0] + cvb).cpu()
            truef = r[anchor + s*k]
            steps.append({
                "t_ms": s*k*125,
                "gen": [[round(gpos(predf,p)[0]), round(gpos(predf,p)[1])] for p in range(N_PLAYERS)],
                "truth": [[round(gpos(truef,p)[0]), round(gpos(truef,p)[1])] for p in range(N_PLAYERS)],
            })
            buf = torch.cat([buf[:, 1:, :], predf.to(args.device).view(1,1,-1)], dim=1)
        rollout = {"anchor": int(anchor), "steps": steps,
                   "sides": ["t" if p < 5 else "ct" for p in range(N_PLAYERS)]}
        print(f"rollout anchor t={anchor}, {args.rollout} steps")

    # ---- write JSON + copy radar ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    radar_src = RADAR_DIR / f"{mp}.png"
    if radar_src.exists():
        shutil.copy(radar_src, out / f"{mp}.png")
    payload = {
        "map": mp, "radar": f"{mp}.png",
        "map_data": {"pos_x": md["pos_x"], "pos_y": md["pos_y"], "scale": md["scale"]},
        "meta": {"demo": meta.get("demo_stem","?"), "round_num": meta.get("round_num","?"),
                 "winner": meta.get("winner","?"), "horizon_ms": k*125, "ckpt_step": ck.get("step")},
        "frames": frames_json, "rollout": rollout,
    }
    (out / "round.json").write_text(json.dumps(payload))
    print(f"wrote {out/'round.json'}  ({len(frames_json)} frames) + radar.  "
          f"Open viewer/gen_viewer.html")


if __name__ == "__main__":
    main()
