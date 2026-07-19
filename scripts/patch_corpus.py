#!/usr/bin/env python3
"""Runbook [1]: patch the merged corpus blobs in place of a re-bake.

Recomputes, from information already inside the tensors, the three defect
families the 2026-07-18 adversarial review confirmed (datasheet §5 D3/D4/D6):
  1. bomb_state one-hot (global 15-18, [none, carried, planted_a, planted_b])
     — carried from any player's has_c4 (per-player dim 14); planted from
     bomb_x/y != 0; SITE from plant position (BOMBSITE_CENTROIDS xy; nuke by
     the z of the nearest alive player at the plant frame, NUKE_Z_SPLIT).
  2. round_time (global 14) + its 8 sinusoids (global 29-36) — re-anchored at
     the in-tensor freeze->live phase transition, clamped >= 0, mirroring the
     v2.1 builder (periods 2/5/20/115 s, [sin x4 | cos x4]).
  3. v3 dim7 dist_to_bomb (per-player derived offset 7) — plant-gated with
     sentinel 1.0, min(hypot, 3000)/3000 in world units.

Writes NEW files (<stem>_p1.pt) — the originals are the pre-patch snapshot.
Loads WITHOUT mmap (this script mutates tensors; see _corpus.load_corpus
docstring). Applies NO exclusions: patched bytes must stay exclusion-free.
Stamps patch_lineage into each blob and writes corpus_manifest.json
(corpus-strategy §4; push_blobs_hf.py contract: blobs[basename]={sha256,bytes}).
Post-write, runs the invariant checks from tests/test_corpus_invariants.py
and refuses to report success on any violation.

Usage: .venv/bin/python scripts/patch_corpus.py [--only val_v2m,val_v3m]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from build_tick_sequences import BOMBSITE_CENTROIDS, NUKE_Z_SPLIT
from test_corpus_invariants import (
    check_bomb_bits_consistent,
    check_dim7_plant_gated,
    check_lineage,
    check_match_ids,
    check_no_nan_inf,
    check_round_time,
)

TS_DIR = ROOT / "data/processed/tick_sequences"
BLOBS = ["val_v2m", "train_v2m", "val_v3m", "train_v3m"]
SPLIT_OF = {"val_v2m": "val", "train_v2m": "train",
            "val_v3m": "val", "train_v3m": "train"}
PERIODS_S = np.array([2.0, 5.0, 20.0, 115.0], dtype=np.float32)
HZ = 8.0
RAW_PPD = 56
HAS_C4 = 14

# global-block offsets (feature_schema v2 layout, verified against
# build_tick_sequences.py:672-700)
G_PHASE = 7          # phase onehot(4): [freeze, live, post_plant, end]
G_RT = 14            # round_time_s / 115
G_BOMB = 15          # bomb_state onehot(4)
G_BX, G_BY = 19, 20  # bomb_x/3000, bomb_y/3000
G_SIN = 29           # 8 sinusoid dims


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_site(map_name: str, bx_w: float, by_w: float, z_w: float | None) -> str:
    """Mirror build_tick_sequences.derive_site_from_plant (no event fallback:
    blobs carry no event strings; centroid table covers every corpus map)."""
    if map_name == "de_nuke" and z_w is not None:
        return "a" if z_w >= NUKE_Z_SPLIT else "b"
    (ax, ay), (bx, by) = BOMBSITE_CENTROIDS[map_name]
    da = (bx_w - ax) ** 2 + (by_w - ay) ** 2
    db = (bx_w - bx) ** 2 + (by_w - by) ** 2
    return "a" if da <= db else "b"


def patch_round(t: torch.Tensor, ppd: int, map_name: str) -> None:
    """In-place patch of one round tensor [T, F]."""
    g0 = 10 * ppd
    g = t[:, g0:]
    T = t.shape[0]

    # --- 2) round clock: re-anchor at the freeze->live transition -----------
    live = g[:, G_PHASE + 1] > 0.5
    f0 = int(np.argmax(live.numpy())) if bool(live.any()) else 0
    rt_s = np.maximum(0.0, (np.arange(T, dtype=np.float32) - f0)) / HZ
    g[:, G_RT] = torch.from_numpy(rt_s / 115.0)
    theta = 2 * np.pi * rt_s[:, None] / PERIODS_S[None, :]
    g[:, G_SIN:G_SIN + 4] = torch.from_numpy(np.sin(theta).astype(np.float32))
    g[:, G_SIN + 4:G_SIN + 8] = torch.from_numpy(np.cos(theta).astype(np.float32))

    # --- 1) bomb_state bits --------------------------------------------------
    c4_cols = torch.stack([t[:, p * ppd + HAS_C4] for p in range(10)], dim=1)
    carried = c4_cols.max(dim=1).values > 0.5
    bx_n, by_n = g[:, G_BX], g[:, G_BY]
    planted = (bx_n.abs() + by_n.abs()) > 1e-9
    g[:, G_BOMB:G_BOMB + 4] = 0.0
    if bool(planted.any()):
        fp = int(np.argmax(planted.numpy()))
        bx_w = float(bx_n[fp]) * 3000.0
        by_w = float(by_n[fp]) * 3000.0
        z_w = None
        if map_name == "de_nuke":
            # planter proxy: z of the nearest alive player to the bomb at fp
            best_d, z_val = None, None
            for p in range(10):
                if t[fp, p * ppd + 13] < 0.5:  # alive dim
                    continue
                dx = float(t[fp, p * ppd + 0]) * 3000.0 - bx_w
                dy = float(t[fp, p * ppd + 1]) * 3000.0 - by_w
                d = dx * dx + dy * dy
                if best_d is None or d < best_d:
                    best_d, z_val = d, float(t[fp, p * ppd + 2]) * 500.0
            z_w = z_val
        site = derive_site(map_name, bx_w, by_w, z_w)
        site_col = G_BOMB + (2 if site == "a" else 3)
        g[planted, site_col] = 1.0
    g[carried & ~planted, G_BOMB + 1] = 1.0
    none_mask = ~carried & ~planted
    g[none_mask, G_BOMB + 0] = 1.0
    # carried frames that are also planted resolve to planted (builder override)
    both = carried & planted
    if bool(both.any()):
        pass  # planted site bit already set above; carried bit not set for these

    # --- 3) v3 dim7: plant-gated dist_to_bomb --------------------------------
    if ppd > RAW_PPD:
        for p in range(10):
            d7 = p * ppd + RAW_PPD + 7
            px = t[:, p * ppd + 0] * 3000.0
            py = t[:, p * ppd + 1] * 3000.0
            dist_w = torch.hypot(px - bx_n * 3000.0, py - by_n * 3000.0)
            t[:, d7] = torch.where(planted,
                                   torch.clamp(dist_w, max=3000.0) / 3000.0,
                                   torch.ones_like(dist_w))


def git_sha(path: Path) -> str:
    out = subprocess.run(["git", "hash-object", str(path)],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma subset of " + ",".join(BLOBS))
    a = ap.parse_args()
    todo = [b for b in (a.only.split(",") if a.only else BLOBS) if b]

    with open(TS_DIR / "split_manifest_v2.json") as f:
        split_manifest = json.load(f)
    manifest_path = TS_DIR / "corpus_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as _mf:
            manifest = json.load(_mf)
    else:
        manifest = {
        "corpus_version": "2.1.0+patch1",
        "blobs": {},
        "patch_lineage": [],
        "split": {"seed": split_manifest.get("seed", 0),
                  "val_frac": split_manifest.get("val_frac", 0.15),
                  "unit": "match",
                  "dedup_key": "(norm_stem, round_num, first_tick, n_ticks)",
                  "split_manifest_sha256": sha256_file(TS_DIR / "split_manifest_v2.json")},
        "parser": {"demoparser2": ">=0.41.3", "awpy": "see uv.lock"},
        "builder": {"schema_version": "v2.1",
                    "builder_sha": git_sha(ROOT / "scripts/build_tick_sequences.py"),
                    "note": "blobs predate v2.1; this patch retrofits D3/D4/D6 fixes"},
        "exclusions": {"load_time_mask": sorted(["de_anubis", "de_train"]),
                       "note": "datasheet §5 D1/D2; NEVER baked into blob bytes"},
        "datasheet": "docs/datasheet.md",
    }

    script_sha = git_sha(Path(__file__))
    date = _dt.date.today().isoformat()
    for name in todo:
        src = TS_DIR / f"{name}.pt"
        dst = TS_DIR / f"{name}_p1.pt"
        print(f"=== {name}: load (non-mmap, mutating) ...", flush=True)
        sha_pre = sha256_file(src)
        blob = torch.load(src, map_location="cpu", weights_only=False)
        ppd = blob.get("per_player_dim", RAW_PPD)
        n_pl = sum(1 for m in blob["metas"] if not m.get("match_id"))
        assert n_pl == 0, f"{name}: {n_pl} metas missing match_id"
        for t, m in zip(blob["tensors"], blob["metas"]):
            patch_round(t, ppd, m.get("map_name", ""))
        entry = {"script": "scripts/patch_corpus.py", "script_sha": script_sha,
                 "transforms": ["bomb_state_bits(D3)", "round_time_reanchor(D4)"]
                 + (["dim7_plant_gate(D6)"] if ppd > RAW_PPD else []),
                 "sha256_pre": sha_pre,
                 "sha256_post": "recorded-in-corpus_manifest.json",
                 "date": date}
        blob["patch_lineage"] = blob.get("patch_lineage", []) + [entry]
        blob["schema_version"] = blob.get("schema_version", "v2") + "+patch1"

        print("    invariant checks (in-memory) ...", flush=True)
        viol = (check_bomb_bits_consistent(blob) + check_round_time(blob)
                + check_no_nan_inf(blob) + check_lineage(blob)
                + check_match_ids(blob, manifest=split_manifest,
                                  split=SPLIT_OF[name]))
        if ppd > RAW_PPD:
            viol += check_dim7_plant_gated(blob)
        if viol:
            print(f"!! {name}: {len(viol)} violations — NOT SAVING:")
            for v in viol[:12]:
                print("   ", v)
            sys.exit(1)

        print(f"    save {dst.name} ...", flush=True)
        torch.save(blob, dst)
        del blob
        gc.collect()
        sha_post = sha256_file(dst)
        manifest["blobs"][dst.name] = {"sha256": sha_post,
                                       "bytes": dst.stat().st_size,
                                       "sha256_source": sha_pre}
        manifest["patch_lineage"].append({**entry, "blob": dst.name,
                                          "sha256_post": sha_post})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"    OK  pre={sha_pre[:12]}  post={sha_post[:12]}")

    # manifest also tracks the split manifest for push_blobs_hf
    manifest["blobs"]["split_manifest_v2.json"] = {
        "sha256": manifest["split"]["split_manifest_sha256"],
        "bytes": (TS_DIR / "split_manifest_v2.json").stat().st_size}
    manifest["blobs"]["corpus_manifest.json"] = {"sha256": "self", "bytes": 0}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nmanifest -> {manifest_path}")
    print("done. next: validation re-bake diff (runbook [1] VALIDATE), then "
          ".venv/bin/python scripts/push_blobs_hf.py")


if __name__ == "__main__":
    main()
