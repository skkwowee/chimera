#!/usr/bin/env python3
"""Bake world-model GENERATIONS into JSON for the gen viewer (viewer/gen_viewer.html).

For each frame t of each chosen round it records, per player:
  - now    : ground-truth position at t              (solid dot)
  - gen    : one-step generation at t+k              (ghost ring; the MODE of the
             displacement distribution — a monitoring view, not the model's claim)
  - future : ground-truth position at t+k            (faint X; what really happened)
plus the detached value head's sigmoid (MONITORING ONLY — the reportable value
number is the frozen-trunk linear probe, scripts/value_probe.py), a per-frame
`end` phase flag (value is end-phase-masked per Knob 5), and a K-sample
AUTOREGRESSIVE rollout fan from an anchor: K trajectories advanced with SAMPLED
steps (Knob 1: rollout-native + sampled; the honest claim is coverage/minADE-K
over the fan, never a single argmax trail — that trail is exported separately,
labeled "mode"). All horizon labels in the viewer derive from the checkpoint's
`horizon` (k * 125 ms); nothing assumes h8/1s.

Multi-round exports write data/round_<i>.json + data/index.json (map/OOD-aware
round picker); a single data/round.json (first round) is kept for back-compat.

Usage:
  python scripts/export_gen_viewer.py --ckpt <best_ns.pt> --val-pt <val_blob.pt> \
      [--rounds 4] [--K 16] [--temperature 1.0] [--rollout-seed 0]
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
OOD_MAP = "de_overpass"          # Knob 4: held out entirely (367 rounds, OOD set)


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


def detect_historical(a: dict) -> list[str]:
    """Pre-reset checkpoint detection (locked recipe: k=4/500ms, detached value,
    scheduled sampling — Knobs 1/5/6). Returns list of reasons; empty = current."""
    reasons = []
    if a.get("horizon") != 4:
        reasons.append(f"horizon={a.get('horizon')} (locked recipe is k=4 / 500 ms)")
    markers = [k for k in a if "detach" in k or k.startswith("ss") or "sched" in k]
    if not markers:
        reasons.append("ckpt args carry no Knob-5/6 markers (detached value head / "
                       "scheduled sampling) — written by the pre-reset trainer")
    return reasons


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="world-model checkpoint. Canonical (post-reset): the v2 "
                         "seed-0 best_ns.pt from the Knob 1-7 retrain. Pre-reset "
                         "checkpoints export with a HISTORICAL banner.")
    ap.add_argument("--value-ckpt", default=None,
                    help="optional separate ckpt for the value readout, e.g. the "
                         "Phase-3 head-only refit on the frozen trunk. Default: the "
                         "detached monitoring head inside --ckpt. NOTE: the head "
                         "sigmoid is monitoring only; the reportable value number is "
                         "the frozen-trunk linear probe (value_probe.py).")
    ap.add_argument("--val-pt", required=True,
                    help="val blob. Canonical: the runbook-[1] PATCHED val_v2m.pt "
                         "(v2 597-d); v3 687-d is the ablation arm.")
    ap.add_argument("--round", type=int, default=None, help="single round index")
    ap.add_argument("--rounds", type=int, default=1,
                    help="bake the first N usable rounds (ignored if --round/--round-list)")
    ap.add_argument("--round-list", default=None,
                    help="comma-separated round indices to bake (overrides --round/--rounds)")
    ap.add_argument("--anchor", type=int, default=None, help="rollout anchor frame (default: mid)")
    ap.add_argument("--rollout", type=int, default=12, help="autoregressive steps for the fan")
    ap.add_argument("--K", type=int, default=16,
                    help="sampled trajectories in the rollout fan (locked metric is minADE-16)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--rollout-seed", type=int, default=0, help="RNG seed for the fan (reproducible)")
    ap.add_argument("--zero-map-id", action="store_true",
                    help="Knob-4 control: zero the map one-hot in every model input")
    ap.add_argument("--out", default="viewer/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck.get("per_player_dim", 56)
    hist_reasons = detect_historical(a)
    if hist_reasons:
        print("!" * 74)
        print("!! HISTORICAL CHECKPOINT — pre-reset model, baselines only. Reasons:")
        for rr in hist_reasons:
            print(f"!!   - {rr}")
        print("!! The viewer will show a persistent HISTORICAL banner (meta.historical).")
        print("!" * 74)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    cv_res = bool(a.get("cv_residual", False))
    has_dist = bool(getattr(model, "dist", False))
    K = args.K
    if not has_dist and K > 1:
        print("WARNING: checkpoint has no distributional head — sampling is impossible, "
              "all fan members would be identical. Forcing K=1 (deterministic trail, "
              "labeled 'mode' in the viewer).")
        K = 1

    vmodel = model
    if args.value_ckpt:
        vck = load_ckpt(args.value_ckpt); va = vck["args"]
        vmodel = build_model(va["arch"], vck["feature_dim"], va["d_model"], va["layers"],
                             va["heads"], per_player_dim=vck.get("per_player_dim", 56),
                             dist=va.get("dist_head", False))
        vmodel.load_state_dict(vck["model"]); vmodel.to(args.device).eval()
        print(f"value readout from {args.value_ckpt} (step {vck.get('step')})")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    vocab = blob.get("map_vocab", MAPS); n_maps = len(vocab)
    gbase = N_PLAYERS * ppd                       # global block start
    # encode_global layout: map_onehot(n_maps) then phase_onehot [freeze,live,post_plant,end]
    end_col = gbase + n_maps + 3

    usable = [i for i, t in enumerate(blob["tensors"]) if t.shape[0] >= L + k + 1]
    if args.round_list:
        sel = [int(x) for x in args.round_list.split(",")]
    elif args.round is not None:
        sel = [args.round]
    else:
        sel = usable[:max(1, args.rounds)]
    if not sel:
        sys.exit("no usable rounds (need T >= window + horizon + 1)")

    xy_idx = torch.tensor([[p*ppd, p*ppd+1] for p in range(N_PLAYERS)])
    alive_idx = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])

    def gpos(frame, p):                       # game-unit (x,y) for player p
        b = frame[p*ppd:p*ppd+2]
        return float(b[0])*3000, float(b[1])*3000

    def export_round(ridx):
        r = blob["tensors"][ridx]; meta = blob["metas"][ridx]; T = r.shape[0]
        mp = vocab[int(r[L].numpy()[gbase:gbase+n_maps].argmax())]
        md = MAP_DATA[mp]
        r_in = r
        if args.zero_map_id:
            r_in = r.clone(); r_in[:, gbase:gbase+n_maps] = 0.0
        print(f"round {ridx}: {meta.get('demo_stem','?')} rnd {meta.get('round_num','?')}  "
              f"map={mp}  winner={meta.get('winner','?')}  T={T}  horizon={k} ({k*125}ms)"
              + ("  [map-ID zeroed]" if args.zero_map_id else ""))

        # ---- per-frame single-step generation (batched; mode decode, monitoring view) ----
        starts = list(range(L - 1, T - k))     # anchor t with full window + a future
        frames_json = []
        B = 128
        for i in range(0, len(starts), B):
            chunk = starts[i:i+B]
            wins = torch.stack([r_in[t-L+1:t+1] for t in chunk]).to(args.device)   # [b,L,F]
            out = model.heads(wins)
            res = model.gen_residual(wins)[:, -1, :].cpu()                 # [b,F] mode decode
            vout = out if vmodel is model else vmodel.heads(wins)
            val = torch.sigmoid(vout["value"][:, -1]).cpu()                # [b]
            # surprise = NLL of the true displacement class (dist ckpts only)
            logp = (F.log_softmax(out["dist_logits"][:, -1].float(), -1).cpu()
                    if "dist_logits" in out else None)
            for j, t in enumerate(chunk):
                cur = r[t]; fut = r[t+k]; prev = r[max(0, t-1)]
                pred = cur + res[j] + (k * (cur - prev) if cv_res else 0.0)
                nll = None
                if logp is not None:
                    tc = dist_class((fut - cur)[xy_idx])                   # true class [P]
                    nll = -logp[j].gather(1, tc.unsqueeze(1)).squeeze(1)
                    nll[cur[alive_idx] <= 0.5] = 0.0
                players = []
                for p in range(N_PLAYERS):
                    alive = float(cur[p*ppd+13]) > 0.5
                    nx, ny = gpos(cur, p); gx, gy = gpos(pred, p); fx, fy = gpos(fut, p)
                    pvx, pvy = gpos(prev, p)
                    cvx, cvy = nx + (nx - pvx) * k, ny + (ny - pvy) * k   # naive const-vel +k
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
                                    "end": bool(float(r[t][end_col]) > 0.5),
                                    "surp": round(float(nll.sum()), 1) if nll is not None else 0.0,
                                    "players": players})

        # ---- K-sample autoregressive rollout fan from an anchor (the honest object) ----
        anchor = args.anchor if args.anchor is not None else \
            max(L - 1, min(T - args.rollout*k - 1, T // 2))
        rollout = None
        if T >= anchor + args.rollout * k + 1 and anchor >= L - 1:
            g = torch.Generator(device=args.device); g.manual_seed(args.rollout_seed)
            win0 = r_in[anchor-L+1:anchor+1].to(args.device)
            buf = win0.unsqueeze(0).repeat(K, 1, 1)              # [K,L,F] sampled fan
            mbuf = win0.unsqueeze(0).clone()                     # [1,L,F] argmax "mode" trail
            steps = []
            ade_sum = torch.zeros(K)
            for s in range(1, args.rollout + 1):
                res = model.gen_residual(buf, sample=has_dist, temperature=args.temperature,
                                         generator=g)[:, -1, :]
                cvb = (k * (buf[:, -1] - buf[:, -2])) if cv_res else 0.0
                predf = (buf[:, -1] + res + cvb)                                 # [K,F]
                mres = model.gen_residual(mbuf)[:, -1, :]
                mcvb = (k * (mbuf[:, -1] - mbuf[:, -2])) if cv_res else 0.0
                mpredf = (mbuf[:, -1] + mres + mcvb)                             # [1,F]
                truef = r[anchor + s*k]
                alive = [bool(float(truef[p*ppd+13]) > 0.5) for p in range(N_PLAYERS)]
                pxy = predf.cpu()[:, xy_idx]                                     # [K,P,2]
                txy = truef[xy_idx]                                              # [P,2]
                d = ((pxy - txy.unsqueeze(0)) * 3000).norm(dim=-1)               # [K,P] game units
                am = torch.tensor(alive, dtype=torch.bool)
                dm = d[:, am].mean(dim=1) if am.any() else d.mean(dim=1)         # [K]
                ade_sum += dm
                pc, mc = predf.cpu(), mpredf.cpu()[0]
                steps.append({
                    "t_ms": s*k*125,
                    "truth": [[round(gpos(truef, p)[0]), round(gpos(truef, p)[1])]
                              for p in range(N_PLAYERS)],
                    "gens": [[[round(gpos(pc[i], p)[0]), round(gpos(pc[i], p)[1])]
                              for p in range(N_PLAYERS)] for i in range(K)],
                    "mode": [[round(gpos(mc, p)[0]), round(gpos(mc, p)[1])]
                             for p in range(N_PLAYERS)],
                    "alive": alive,
                    "minade": round(float(dm.min())),
                })
                buf = torch.cat([buf[:, 1:, :], predf.unsqueeze(1)], dim=1)
                mbuf = torch.cat([mbuf[:, 1:, :], mpredf.unsqueeze(1)], dim=1)
            rollout = {"anchor": int(anchor), "steps": steps,
                       "sides": ["t" if p < 5 else "ct" for p in range(N_PLAYERS)],
                       "K": K, "temperature": args.temperature, "seed": args.rollout_seed,
                       "best": int(ade_sum.argmin()), "sampled": has_dist}
            # TODO(runbook [4]): once rollout_eval.py ships the fair stochastic baseline
            # (per-bucket damped-CV + fitted residual covariance, K=16, scored identically),
            # bake baseline_gens / baseline_minade / baseline_best alongside — coverage
            # numbers are UNREPORTABLE without it (CHANGE B).
            print(f"rollout anchor t={anchor}, {args.rollout} steps, K={K} "
                  f"temp={args.temperature} seed={args.rollout_seed} "
                  f"minADE-{K}@final={steps[-1]['minade']}u")

        payload = {
            "map": mp, "radar": f"{mp}.png",
            "map_data": {"pos_x": md["pos_x"], "pos_y": md["pos_y"], "scale": md["scale"]},
            "meta": {"demo": meta.get("demo_stem","?"), "round_num": meta.get("round_num","?"),
                     "winner": meta.get("winner","?"), "horizon_ms": k*125,
                     "ckpt_step": ck.get("step"),
                     "historical": bool(hist_reasons),
                     "historical_reasons": hist_reasons,
                     "zero_map_id": bool(args.zero_map_id),
                     "ood": mp == OOD_MAP},
            "frames": frames_json, "rollout": rollout,
        }
        return payload, mp

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = []
    for n, ridx in enumerate(sel):
        payload, mp = export_round(ridx)
        fname = f"round_{ridx}.json"
        (out / fname).write_text(json.dumps(payload))
        if n == 0:                                          # back-compat single-round path
            (out / "round.json").write_text(json.dumps(payload))
        radar_src = RADAR_DIR / f"{mp}.png"
        if radar_src.exists() and not (out / f"{mp}.png").exists():
            shutil.copy(radar_src, out / f"{mp}.png")
        index.append({"i": int(ridx), "file": fname, "map": mp,
                      "demo": payload["meta"]["demo"],
                      "round_num": payload["meta"]["round_num"],
                      "winner": payload["meta"]["winner"],
                      "ood": mp == OOD_MAP})
    (out / "index.json").write_text(json.dumps(index))
    print(f"wrote {len(index)} round(s) + index.json to {out}.  Open viewer/gen_viewer.html")


if __name__ == "__main__":
    main()
