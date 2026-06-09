#!/usr/bin/env python3
"""Inspect the full feature vector for a round/frame — every one of the 687 v3 dims,
labeled and de-normalized. The "what's actually in the state" viewer.

  list schema:   python scripts/inspect_features.py --list
  one frame:     python scripts/inspect_features.py --round 0 --frame 200
  full player:   python scripts/inspect_features.py --round 0 --frame 200 --player 1
"""
from __future__ import annotations
import argparse
import math
import torch

WEAPONS = ["knife","pistol_low","pistol_high","smg","shotgun","rifle_t","rifle_ct",
           "awp","scout","auto_sniper","lmg","smoke","flash","molly","he","decoy","c4","other"]
MAPS = ["de_ancient","de_dust2","de_inferno","de_mirage","de_nuke","de_overpass","de_train"]
PHASE = ["freeze","live","post_plant","end"]
BOMB = ["none","carried","planted_a","planted_b"]
SLOTS = ["T1","T2","T3","T4","T5","CT1","CT2","CT3","CT4","CT5"]

# per-player layout (65 dims): name, denorm-fn (None = raw)
PP = (
    [("x", lambda v: v*3000), ("y", lambda v: v*3000), ("z", lambda v: v*500),
     ("sin_yaw", None), ("cos_yaw", None), ("sin_pitch", None), ("cos_pitch", None),
     ("hp", lambda v: v*100), ("armor", lambda v: v*100), ("has_helmet", None),
     ("has_defuser", None), ("log_balance->$", lambda v: math.expm1(v*10)),
     ("log_equip->$", lambda v: math.expm1(v*10)), ("alive", None), ("has_c4", None)]
    + [(f"primary:{w}", None) for w in WEAPONS]
    + [(f"secondary:{w}", None) for w in WEAPONS]
    + [("util:smoke", None), ("util:flash", None), ("util:molly", None),
       ("util:he", None), ("util:decoy", None)]
    + [("d_enemy(u)", lambda v: v*3000), ("d_mate(u)", lambda v: v*3000),
       ("n_los", lambda v: v*5), ("exposed", None), ("n_fov", lambda v: v*5),
       ("n_aim", lambda v: v*5), ("aim_err(rad)", lambda v: v*math.pi),
       ("d_bomb(u)", lambda v: v*3000), ("t_since_los(fr)", lambda v: v*64)]
)
GLOBAL = (
    [(f"map:{m}", None) for m in MAPS]
    + [(f"phase:{p}", None) for p in PHASE]
    + [("score_t", lambda v: v*16), ("score_ct", lambda v: v*16),
       ("round_num", lambda v: v*30), ("round_time(s)", lambda v: v*115)]
    + [(f"bomb:{b}", None) for b in BOMB]
    + [("bomb_x", lambda v: v*3000), ("bomb_y", lambda v: v*3000),
       ("bomb_age(s)", lambda v: v*40), ("money_diff", lambda v: v*5e4),
       ("equip_diff", lambda v: v*5e4), ("alive_t", lambda v: v*5),
       ("alive_ct", lambda v: v*5), ("alive_diff", lambda v: v*5),
       ("t_since_kill", None), ("t_since_event", None)]
    + [(f"rtime_sincos[{i}]", None) for i in range(8)]
)
NP, PPD = 10, 65


def onehot_name(vals, names):
    i = int(vals.argmax()); return f"{names[i]} ({vals[i]:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--player", type=int, default=None, help="dump ALL 65 dims for this slot (0-9)")
    ap.add_argument("--list", action="store_true", help="print the index->name schema only")
    args = ap.parse_args()

    if args.list:
        print("=== per-player block (x10), dims 0..64 ===")
        for i, (nm, _) in enumerate(PP):
            print(f"  [{i:2d}] {nm}")
        print(f"=== global block, frame dims {NP*PPD}..{NP*PPD+len(GLOBAL)-1} ===")
        for i, (nm, _) in enumerate(GLOBAL):
            print(f"  [{NP*PPD+i:3d}] (g{i:2d}) {nm}")
        print(f"\nfeature_dim = {NP}*{PPD} + {len(GLOBAL)} = {NP*PPD+len(GLOBAL)}")
        return

    blob = torch.load(args.pt, map_location="cpu", weights_only=False)
    r = blob["tensors"][args.round]; meta = blob["metas"][args.round]
    T = r.shape[0]
    f = args.frame if args.frame is not None else T // 2
    frame = r[f].numpy()
    print(f"=== {meta['demo_stem']} round {meta['round_num']} ({meta.get('map_name')}, winner={meta.get('winner')}) "
          f"frame {f}/{T} ===\n")

    g = frame[NP*PPD:]
    print("GLOBAL:")
    print(f"  map={onehot_name(g[0:7], MAPS)}  phase={onehot_name(g[7:11], PHASE)}  "
          f"bomb={onehot_name(g[15:19], BOMB)}")
    print(f"  score T {g[11]*16:.0f} - CT {g[12]*16:.0f}  round {g[13]*30:.0f}  "
          f"time {g[14]*115:.0f}s  alive T{g[24]*5:.0f}/CT{g[25]*5:.0f}")
    print(f"  bomb_xy ({g[19]*3000:.0f},{g[20]*3000:.0f})  money_diff {g[22]*5e4:+.0f}  "
          f"equip_diff {g[23]*5e4:+.0f}\n")

    if args.player is not None:
        p = args.player; blk = frame[p*PPD:(p+1)*PPD]
        print(f"FULL DUMP — slot {SLOTS[p]} (all 65 dims):")
        for i, (nm, fn) in enumerate(PP):
            v = blk[i]; dv = fn(v) if fn else v
            star = " *" if abs(v) > 1e-4 else ""
            print(f"  [{i:2d}] {nm:22s} raw={v:+.4f}  ->{dv:>12.3f}{star}")
        return

    # compact per-player table (key dims)
    print(f"{'slot':5} {'x':>7} {'y':>7} {'z':>6} {'hp':>4} {'aliv':>4} {'primary':>10} "
          f"{'$':>6} {'nLOS':>4} {'expo':>4} {'nFOV':>4} {'aimErr':>6} {'dEnemy':>7}")
    for p in range(NP):
        b = frame[p*PPD:(p+1)*PPD]
        prim = WEAPONS[int(b[15:33].argmax())]
        print(f"{SLOTS[p]:5} {b[0]*3000:7.0f} {b[1]*3000:7.0f} {b[2]*500:6.0f} {b[7]*100:4.0f} "
              f"{b[13]:4.0f} {prim:>10} {math.expm1(b[11]*10):6.0f} {b[58]*5:4.0f} {b[59]:4.0f} "
              f"{b[60]*5:4.0f} {b[62]*math.pi:6.2f} {b[56]*3000:7.0f}")
    print("\n(derived: nLOS=enemies with line; expo=exposed; nFOV=enemies I see; "
          "aimErr=rad to nearest seen enemy; dEnemy=units to nearest enemy)")
    print("Use --player N for the full 65-dim dump of one slot, or --list for the schema.")


if __name__ == "__main__":
    main()
