#!/usr/bin/env python3
"""Distribution COVERAGE eval (minADE-K) for the distributional head.

Point-prediction on multimodal frames is partly unwinnable: on turn/reversal
frames 'copy' IS one of the modes (stay), and beating it requires knowing the
player's choice — information not in the state. The distributional head's job
is different: put mass ON the modes so that sampling covers what actually
happened. That's exactly what GRPO group-generation needs (diverse plausible
futures), and the standard metric is minADE-K: draw K samples, score the best.

Per truth-trajectory bucket (same turn-angle buckets as decision_eval) reports
mean xy error in game units for:
  copy      : the stay mode (baseline)
  argmax    : single-point decode (mode commit)
  minADE-K  : best of K sampled decodes      <- coverage
  medADE-K  : median of K (sanity: spread)
If minADE-K << copy on turn/reversal buckets while argmax loses, the
distribution covers the modes it cannot pick between — the head is doing its
actual job and 'generate the group' gets diverse futures for free.

Usage: python scripts/dist_coverage_eval.py --ckpt outputs/wm_3map_dist/h8_mt/best_ns.pt --k 16
"""
from __future__ import annotations
import argparse, math, shutil, sys, tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, dist_class, N_PLAYERS  # noqa
from _corpus import clean_blob  # noqa

BNAMES = ["stationary", "straight", "mild turn", "hard turn", "reversal"]
THETA_EDGES = torch.tensor([20.0, 60.0, 120.0])
STAT_SPEED, STAT_DISP = 5.0, 25.0
SMOOTH = 4


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/wm_3map_dist/h8_mt/best_ns.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--maps", default="de_mirage,de_dust2,de_inferno")
    ap.add_argument("--k", type=int, default=16, help="samples per frame")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max-rounds", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck["per_player_dim"]
    assert a.get("dist_head"), "coverage eval needs a --dist-head checkpoint"
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=True)
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))
    print(f"ckpt step {ck.get('step')}  horizon {k*125}ms  K={args.k} samples @ T={args.temperature}")

    px = torch.tensor([p*ppd+0 for p in range(N_PLAYERS)])
    py = torch.tensor([p*ppd+1 for p in range(N_PLAYERS)])
    alive_i = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])
    centers = model.centers.cpu()                                     # [C,2] normalized

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    clean_blob(blob, tag="val")  # datasheet §5 D1/D2
    maplist = sorted(keep)
    bid_all, cp_all, am_all, mn_all, md_all, mapidx = [], [], [], [], [], []
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
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)
            out = model.heads(wins)
            logits = out["dist_logits"][:, -1].float()                # [b,P,C]
            offs = out["dist_off"][:, -1].float().cpu()               # [b,P,C,2]
            prob = torch.softmax(logits / args.temperature, -1)
            b, P, C = prob.shape
            samp = torch.multinomial(prob.reshape(b*P, C), args.k, replacement=True)
            samp = samp.reshape(b, P, args.k).cpu()                   # [b,P,K]
            amax = logits.argmax(-1).cpu()                            # [b,P]
            for j, t in enumerate(chunk):
                cur, fut = r[t], r[t+k]
                alive = cur[alive_i] > 0.5
                if alive.sum() == 0:
                    continue
                # truth displacement + bucket (same scheme as decision_eval)
                dtx = (fut[px] - cur[px]) * 3000
                dty = (fut[py] - cur[py]) * 3000
                dmag = (dtx**2 + dty**2).sqrt()
                vx = (cur[px] - r[t-SMOOTH][px]) * 3000 / SMOOTH
                vy = (cur[py] - r[t-SMOOTH][py]) * 3000 / SMOOTH
                spd = (vx**2 + vy**2).sqrt()
                cosang = ((vx*dtx + vy*dty) / (spd*dmag + 1e-6)).clamp(-1, 1)
                theta = torch.rad2deg(torch.acos(cosang))
                bid = 1 + torch.bucketize(theta, THETA_EDGES)
                bid[(spd < STAT_SPEED) & (dmag < STAT_DISP)] = 0
                # decode errors (game units)
                am_d = (centers[amax[j]] + offs[j].gather(
                    1, amax[j].view(P, 1, 1).expand(P, 1, 2)).squeeze(1)) * 3000  # [P,2]
                sm = samp[j]                                                     # [P,K]
                sm_d = (centers[sm] + offs[j].gather(
                    1, sm.unsqueeze(-1).expand(P, args.k, 2))) * 3000            # [P,K,2]
                tr = torch.stack([dtx, dty], -1)                                 # [P,2]
                e_am = (am_d - tr).norm(dim=-1)                                  # [P]
                e_sm = (sm_d - tr.unsqueeze(1)).norm(dim=-1)                     # [P,K]
                sel = alive
                bid_all.append(bid[sel])
                cp_all.append(dmag[sel])                       # copy error = |displacement|
                am_all.append(e_am[sel])
                mn_all.append(e_sm[sel].min(dim=1).values)
                md_all.append(e_sm[sel].median(dim=1).values)
                mapidx += [maplist.index(m["map_name"])] * int(sel.sum())

    bid = torch.cat(bid_all); cp = torch.cat(cp_all); am = torch.cat(am_all)
    mn = torch.cat(mn_all); md = torch.cat(md_all); mapidx = torch.tensor(mapidx)
    print(f"\nalive player-frame samples: {len(bid)}  (rounds: {n_rounds})\n")
    print(f"{'bucket':12s} {'n':>7s} {'copy':>7s} {'argmax':>8s} {'minADE-'+str(args.k):>9s} "
          f"{'medADE':>7s}  {'cover vs copy':>13s}")
    for bi, name in enumerate(BNAMES):
        sel = bid == bi
        n = int(sel.sum())
        if n == 0:
            continue
        c, a_, mi, me = cp[sel].mean(), am[sel].mean(), mn[sel].mean(), md[sel].mean()
        cov = (c - mi) / c * 100 if c > 0 else 0.0
        print(f"{name:12s} {n:>7d} {c:6.0f}u {a_:7.0f}u {mi:8.0f}u {me:6.0f}u  {cov:12.1f}%")
    c, a_, mi, me = cp.mean(), am.mean(), mn.mean(), md.mean()
    print(f"{'ALL':12s} {len(bid):>7d} {c:6.0f}u {a_:7.0f}u {mi:8.0f}u {me:6.0f}u  "
          f"{(c-mi)/c*100:12.1f}%")

    # per-map breakdown (datasheet mandate: per-map, never pooled — mirrors
    # decision_eval): all buckets pooled, plus the headline turn+reversal cover
    hi_sel = bid >= 3
    print(f"\nper map (all buckets; turn+ = hard turn + reversal subset):")
    print(f"{'map':12s} {'n':>7s} {'copy':>7s} {'argmax':>8s} {'minADE-'+str(args.k):>9s} "
          f"{'medADE':>7s}  {'cover':>6s} {'n(turn+)':>9s} {'cover(turn+)':>12s}")
    for mpi, mp in enumerate(maplist):
        sel = mapidx == mpi
        n = int(sel.sum())
        if n == 0:
            continue
        c, a_, mi, me = cp[sel].mean(), am[sel].mean(), mn[sel].mean(), md[sel].mean()
        hs = sel & hi_sel
        hcov = ((cp[hs].mean() - mn[hs].mean()) / cp[hs].mean() * 100) if hs.any() else float("nan")
        print(f"{mp:12s} {n:>7d} {c:6.0f}u {a_:7.0f}u {mi:8.0f}u {me:6.0f}u  "
              f"{(c-mi)/c*100:5.1f}% {int(hs.sum()):>9d} {hcov:11.1f}%")

    print("\nread: minADE-K << copy on turn/reversal = the distribution COVERS the modes "
          "(the head knows the option set even when no point prediction can know the "
          "choice). medADE >> minADE = healthy spread, not collapsed.")


if __name__ == "__main__":
    main()
