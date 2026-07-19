#!/usr/bin/env python3
"""Does the surprise signal (dist-head NLL, nats) detect EVENT BOUNDARIES?

Falsifiable test against ground truth: align per-frame nats to every KILL in
the val rounds (kills carry exact ticks; frame = (tick - first_tick)/downsample)
and report:
  1. the event-triggered average — mean nats at offsets around the kill frame.
     NOTE the expected LEAD: s_t scores displacement over t -> t+8 (1s ahead),
     so frames in [kill-8, kill] have the kill's movement inside their target
     window — a ramp starting ~1s BEFORE the kill is correct behavior, not
     leakage.
  2. detection AUC — score = nats(t), positive = a kill lands in (t, t+8],
     negative = no kill within [t-16, t+24] (clean negatives). Mann-Whitney.
  3. the quiet floor for contrast.

Usage: python scripts/event_boundary_check.py --ckpt outputs/wm_3map_dist/h8_mt/best_ns.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, dist_class, N_PLAYERS, auc  # noqa
from _corpus import load_corpus

DEMO_DIR = Path("data/processed/demos")
OFF_LO, OFF_HI = -32, 17          # offsets in frames (-4s .. +2s)


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
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]; ppd = ck["per_player_dim"]
    assert a.get("dist_head")
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"],
                        per_player_dim=ppd, dist=True)
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    keep = set(args.maps.split(","))

    xy = torch.tensor([[p*ppd, p*ppd+1] for p in range(N_PLAYERS)])
    alive_i = torch.tensor([p*ppd+13 for p in range(N_PLAYERS)])
    kills_cache: dict[str, list] = {}

    curve_sum = torch.zeros(OFF_HI - OFF_LO); curve_n = torch.zeros(OFF_HI - OFF_LO)
    pos_scores, neg_scores, all_scores = [], [], []
    pos_maps, neg_maps = [], []
    n_kills_used = rounds = 0

    blob = load_corpus(args.val_pt, maps=keep, tag="val")
    for r, m in zip(blob["tensors"], blob["metas"]):
        if m.get("map_name") not in keep:
            continue
        T = r.shape[0]
        if L + k + 2 > T:
            continue
        rounds += 1
        if args.max_rounds and rounds > args.max_rounds:
            break
        stem = m["demo_stem"]
        if stem not in kills_cache:
            f = DEMO_DIR / f"{stem}_kills.json"
            kills_cache[stem] = json.loads(f.read_text()) if f.exists() else []
        ds = m.get("downsample", 8)
        kf = sorted({int((kk["tick"] - m["first_tick"]) // ds)
                     for kk in kills_cache[stem]
                     if kk.get("round_num") == m["round_num"]
                     and m["first_tick"] <= kk["tick"] <= m["last_tick"]})

        # per-frame nats, stride 1 (alignment matters here)
        s = torch.full((T,), float("nan"))
        ts = list(range(L - 1, T - k))
        for i in range(0, len(ts), args.batch):
            chunk = ts[i:i+args.batch]
            wins = torch.stack([r[t-L+1:t+1] for t in chunk]).to(args.device)
            logp = F.log_softmax(model.heads(wins)["dist_logits"][:, -1].float(), -1).cpu()
            for j, t in enumerate(chunk):
                tc = dist_class((r[t+k] - r[t])[xy])
                nll = -logp[j].gather(1, tc.unsqueeze(1)).squeeze(1)
                alive = r[t][alive_i] > 0.5
                # PER-ALIVE-PLAYER MEAN, not sum: summed nats scale with alive
                # count, so kills mechanically DEPRESS post-kill surprise and
                # confound the boundary test (council audit 2026-06-12).
                s[t] = nll[alive].mean() if alive.any() else float("nan")

        valid = ~s.isnan()
        all_scores.append(s[valid])
        # event-triggered accumulation
        for fk in kf:
            n_kills_used += 1
            for oi, off in enumerate(range(OFF_LO, OFF_HI)):
                t = fk + off
                if 0 <= t < T and valid[t]:
                    curve_sum[oi] += s[t]; curve_n[oi] += 1
        # detection labels
        kset = set(kf)
        for t in range(L - 1, T - k):
            if any((t < f2 <= t + k) for f2 in kset):
                pos_scores.append(s[t]); pos_maps.append(m["map_name"])
            elif not any((t - 16 <= f2 <= t + 24) for f2 in kset):
                neg_scores.append(s[t]); neg_maps.append(m["map_name"])

    allc = torch.cat(all_scores)
    print(f"rounds {rounds-1 if args.max_rounds and rounds>args.max_rounds else rounds}  "
          f"kills aligned {n_kills_used}  frames scored {len(allc)}  "
          f"(mean {allc.mean():.1f}, median {allc.median():.1f} nats)")

    print("\nevent-triggered average (kill at offset 0; s_t looks 1s AHEAD, so the "
          "ramp should start ~-8):")
    curve = curve_sum / curve_n.clamp(min=1)
    cmin, cmax = curve.min().item(), curve.max().item()
    for oi, off in enumerate(range(OFF_LO, OFF_HI)):
        if off % 2: continue
        v = curve[oi].item()
        bar = "#" * int((v - cmin) / max(1e-6, cmax - cmin) * 46)
        mark = " <-- KILL" if off == 0 else ""
        print(f"  {off*0.125:+5.1f}s  {v:5.1f}  {bar}{mark}")

    pos = torch.tensor(pos_scores); neg = torch.tensor(neg_scores)
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    print(f"\ndetection AUC (kill within next 1s vs clean negatives): "
          f"{auc(scores, labels):.3f}   (pos {len(pos)}, neg {len(neg)})")
    print(f"mean nats: kill-imminent {pos.mean():.1f} vs quiet {neg.mean():.1f} "
          f"({pos.mean()/neg.mean():.1f}x)")

    # per-map breakdown (datasheet mandate: per-map, never pooled — mirrors decision_eval)
    print("\nper-map detection:")
    print(f"{'map':12s} {'pos':>6s} {'neg':>7s} {'AUC':>7s} {'imminent':>9s} {'quiet':>6s}")
    for mp in sorted(set(pos_maps) | set(neg_maps)):
        p = torch.tensor([v for v, mm in zip(pos_scores, pos_maps) if mm == mp])
        ng = torch.tensor([v for v, mm in zip(neg_scores, neg_maps) if mm == mp])
        sc = torch.cat([p, ng]); lb = torch.cat([torch.ones(len(p)), torch.zeros(len(ng))])
        pm = p.mean().item() if len(p) else float("nan")
        nm = ng.mean().item() if len(ng) else float("nan")
        print(f"{mp:12s} {len(p):>6d} {len(ng):>7d} {auc(sc, lb):7.3f} {pm:8.1f} {nm:5.1f}")


if __name__ == "__main__":
    main()
