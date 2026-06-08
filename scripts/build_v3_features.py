#!/usr/bin/env python3
"""Build feature_schema_v3: append 9 derived PERCEPTION dims per player to the v2
tensors (597 -> 687), computed FROM the v2 tensors themselves (no reparse).

Recovers world x/y/z, view direction (cos/sin yaw), alive, side (slot 0-4=T,
5-9=CT), and bomb x/y from the v2 frame, then computes per player:
  0 dist_nearest_enemy      (/3000)
  1 dist_nearest_teammate   (/3000)
  2 n_enemies_los   (/5)    clear line-of-sight enemies (SYMMETRIC exposure)
  3 is_exposed      (0/1)   n_enemies_los > 0
  4 n_enemies_in_fov (/5)   LOS AND enemy in MY view cone  (what I'm looking at)
  5 n_enemies_aim_me (/5)   LOS AND I'm in ENEMY's cone     (threats on me)
  6 min_aim_error   (/pi)   angle from my crosshair to nearest LOS enemy (1=none)
  7 dist_to_bomb    (/3000)
  8 time_since_los  (/64)   frames since I last had any LOS enemy (cap 64)

LOS via awpy VisibilityChecker (.tri, eye+64), distance-gated, validated at 91.2%
kill-agreement. FOV makes the "seeing" dims asymmetric/meaningful (pure LOS is
symmetric). Multiprocessed (fork + COW shared blob), VC cached per map per worker.

Usage:
  validate: python scripts/build_v3_features.py --split val --limit 5 --workers 1
  full:     python scripts/build_v3_features.py --split val  --workers 12
            python scripts/build_v3_features.py --split train --workers 12
"""
from __future__ import annotations
import os
# Cap BLAS/OMP threads BEFORE numpy/torch import. With fork, 12 workers x 32
# intra-op threads = ~384 threads (oversubscription); pin to 1/worker.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, math, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

PPD = 56                      # v2 per-player dim
NP_ = 10
DERIVED = 9
EYE = 64.0
MAX_DIST = 3500.0
FOV_COS = math.cos(math.radians(53))   # ~half-FOV; dot>FOV_COS => in view cone
TIME_CAP = 64                 # frames (8s at 8Hz)
TRIS = "/home/soone/.awpy/tris"

_BLOB = None
_MAPS = None
_VC = {}


def _get_vc(map_name):
    import gc
    from awpy.visibility import VisibilityChecker
    if map_name not in _VC:
        _VC.clear(); gc.collect()   # FREE the prior ~1.5GB BVH BEFORE building the
                                    # next — avoids the double-hold that OOM'd at 4 workers
        _VC[map_name] = VisibilityChecker(path=Path(f"{TRIS}/{map_name}.tri"))
    return _VC[map_name]


def compute_derived(t: np.ndarray, vc) -> np.ndarray:
    """t: [T,597] float32 v2 frame. Returns [T, 10*9] derived (player-major)."""
    T = t.shape[0]
    pl = t[:, :NP_ * PPD].reshape(T, NP_, PPD)
    g = t[:, NP_ * PPD:]
    x = pl[:, :, 0] * 3000.0
    y = pl[:, :, 1] * 3000.0
    z = pl[:, :, 2] * 500.0
    dirx = pl[:, :, 4]                     # cos_yaw
    diry = pl[:, :, 3]                     # sin_yaw
    alive = pl[:, :, 13] > 0.5
    bx = g[:, 19] * 3000.0
    by = g[:, 20] * 3000.0
    out = np.zeros((T, NP_, DERIVED), dtype=np.float32)
    last_los = np.full(NP_, TIME_CAP, dtype=np.float32)   # frames since LOS
    T_idx, CT_idx = range(0, 5), range(5, 10)

    for f in range(T):
        ax, ay, az, al = x[f], y[f], z[f], alive[f]
        dx_, dy_ = dirx[f], diry[f]
        # pairwise distances (all players)
        px = ax[:, None] - ax[None, :]
        py = ay[:, None] - ay[None, :]
        dist = np.sqrt(px * px + py * py)
        # LOS matrix (cross-team, alive, gated)
        los = np.zeros((NP_, NP_), dtype=bool)
        for i in T_idx:
            if not al[i]:
                continue
            for j in CT_idx:
                if not al[j] or dist[i, j] > MAX_DIST:
                    continue
                v = vc.is_visible((ax[i], ay[i], az[i] + EYE),
                                  (ax[j], ay[j], az[j] + EYE))
                los[i, j] = los[j, i] = v
        for p in range(NP_):
            o = out[f, p]
            if not al[p]:
                last_los[p] = TIME_CAP
                continue
            enemies = list(CT_idx) if p < 5 else list(T_idx)
            mates = [m for m in (T_idx if p < 5 else CT_idx) if m != p]
            en_alive = [e for e in enemies if al[e]]
            mt_alive = [m for m in mates if al[m]]
            # distances
            o[0] = min((dist[p, e] for e in en_alive), default=3000.0) / 3000.0
            o[1] = min((dist[p, m] for m in mt_alive), default=3000.0) / 3000.0
            los_en = [e for e in en_alive if los[p, e]]
            o[2] = len(los_en) / 5.0
            o[3] = 1.0 if los_en else 0.0
            # FOV: dir to enemy . my facing
            n_fov = n_aim = 0
            best_dot = -1.0
            for e in los_en:
                ex, ey = ax[e] - ax[p], ay[e] - ay[p]
                norm = math.hypot(ex, ey) or 1.0
                ux, uy = ex / norm, ey / norm
                my_dot = dx_[p] * ux + dy_[p] * uy        # is enemy in my cone
                if my_dot > FOV_COS:
                    n_fov += 1
                en_dot = dx_[e] * (-ux) + dy_[e] * (-uy)  # am I in enemy's cone
                if en_dot > FOV_COS:
                    n_aim += 1
                best_dot = max(best_dot, my_dot)
            o[4] = n_fov / 5.0
            o[5] = n_aim / 5.0
            o[6] = (math.acos(max(-1.0, min(1.0, best_dot))) / math.pi) if los_en else 1.0
            o[7] = min(math.hypot(ax[p] - bx[f], ay[p] - by[f]), 3000.0) / 3000.0
            last_los[p] = 0.0 if los_en else min(TIME_CAP, last_los[p] + 1)
            o[8] = last_los[p] / TIME_CAP
    return out.reshape(T, NP_ * DERIVED)


def _worker(idx):
    t = _BLOB["tensors"][idx].numpy()
    return idx, compute_derived(t, _get_vc(_MAPS[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, default=4,
                    help="each worker holds ~1-1.5GB BVH; keep workers*1.5GB + blob < WSL RAM cap")
    ap.add_argument("--limit", type=int, default=0, help="process only N rounds (validation)")
    ap.add_argument("--data", default="data/processed/tick_sequences")
    args = ap.parse_args()

    global _BLOB, _MAPS
    src = Path(args.data) / f"{args.split}.pt"
    print(f"loading {src} ...")
    _BLOB = torch.load(src, map_location="cpu", weights_only=False)
    n = len(_BLOB["tensors"])
    _MAPS = [m.get("map_name", "de_dust2") for m in _BLOB["metas"]]
    idxs = list(range(n))
    idxs.sort(key=lambda i: _MAPS[i])          # group by map for VC cache hits
    if args.limit:
        idxs = idxs[:args.limit]
    print(f"{args.split}: {n} rounds, processing {len(idxs)} on {args.workers} workers")

    import gc
    t0 = time.time()
    exp_dim = NP_ * (PPD + DERIVED) + (_BLOB["feature_dim"] - NP_ * PPD)   # 687

    def merge_v3(t_np, d):
        """[T,597] v2 frame + [T,90] derived -> [T,687] v3 (insert 9 dims/player)."""
        T = t_np.shape[0]
        pl = t_np[:, :NP_ * PPD].reshape(T, NP_, PPD)
        dr = d.reshape(T, NP_, DERIVED)
        merged = np.concatenate([pl, dr], axis=2).reshape(T, NP_ * (PPD + DERIVED))
        return np.concatenate([merged, t_np[:, NP_ * PPD:]], axis=1).astype(np.float32)

    if args.workers == 1:
        # LOW-MEMORY path: merge each round IN PLACE, freeing the v2 tensor as we go,
        # so we never hold (full old blob + full new-tensor list) at once.
        # Peak ~ blob (3.4->3.9GB as it grows) + ONE BVH (~1.5GB) ~= 5.5GB.
        for k, i in enumerate(idxs):
            t = _BLOB["tensors"][i].numpy()
            d = compute_derived(t, _get_vc(_MAPS[i]))
            _BLOB["tensors"][i] = torch.from_numpy(merge_v3(t, d))   # replace v2 -> v3
            del t, d
            if k % 50 == 0:
                gc.collect()
                print(f"  {k}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        import multiprocessing as mp
        results = {}
        with mp.Pool(args.workers) as pool:
            for k, (i, d) in enumerate(pool.imap_unordered(_worker, idxs,
                                       chunksize=max(1, len(idxs)//(args.workers*4)))):
                results[i] = d
                if k % 100 == 0:
                    print(f"  {k}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)
        for i in idxs:                                  # merge in place, free as we go
            _BLOB["tensors"][i] = torch.from_numpy(merge_v3(_BLOB["tensors"][i].numpy(), results[i]))
            results[i] = None
    print(f"computed derived in {time.time()-t0:.0f}s", flush=True)

    v3 = [_BLOB["tensors"][i] for i in idxs]            # the rounds we processed (v3 now)
    # shape + sanity asserts
    assert v3[0].shape[1] == exp_dim, (v3[0].shape[1], exp_dim)
    assert not torch.isnan(torch.cat(v3[:200], 0)).any(), "NaN in v3 features"
    dc = np.concatenate([x.numpy()[:, :NP_*(PPD+DERIVED)].reshape(-1, NP_, PPD+DERIVED)[:, :, PPD:]
                         .reshape(-1, DERIVED) for x in v3[:200]], 0)
    names = ["d_enemy","d_mate","n_los","exposed","n_fov","n_aim","aim_err","d_bomb","t_since"]
    print("derived feature ranges (min/mean/max):")
    for c, nm in enumerate(names):
        print(f"  {nm:9s} {dc[:,c].min():.3f} / {dc[:,c].mean():.3f} / {dc[:,c].max():.3f}")
    print(f"  exposed rate: mean {dc[:,3].mean():.3f}")

    if args.limit:
        print("[limit mode] not saving."); return
    out = Path(args.data) / f"{args.split}_v3.pt"
    # _BLOB["tensors"] is already v3, in original order (replaced in place)
    _BLOB["feature_dim"] = exp_dim
    _BLOB["per_player_dim"] = PPD + DERIVED
    _BLOB["schema_version"] = "feature_schema_v3"
    torch.save(_BLOB, out)
    print(f"saved {out}  feature_dim={exp_dim} per_player={PPD+DERIVED}", flush=True)


if __name__ == "__main__":
    main()
