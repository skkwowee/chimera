#!/usr/bin/env python3
"""Export demo rounds to the cs2-demo-viewer WITH line-of-sight overlay, so we
can visually validate the derived-visibility features (the LOS that scored 91.2%
on the kill-agreement test).

Reads the per-tick parquet + kills/rounds JSON + the awpy .tri collision mesh,
computes the LOS matrix per (downsampled) frame, and writes viewer round JSON
with per-frame `los_edges` (cross-team pairs with clear line-of-sight) and a
per-player `sees_enemy` flag. No reparse of the .dem.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
from awpy.visibility import VisibilityChecker

EYE = 64.0
MAX_LOS_DIST = 3500.0   # distance-gate: skip raycasts beyond this (cheap + rare)


def los_matrix(players, vc):
    """players: list of dicts with x,y,z,side,alive. Returns (edges, sees set).
    edge = [i,j] for cross-team alive pairs with clear LOS (symmetric)."""
    n = len(players)
    edges, sees = [], set()
    for i in range(n):
        pi = players[i]
        if not pi["alive"]:
            continue
        for j in range(i + 1, n):
            pj = players[j]
            if not pj["alive"] or pj["side"] == pi["side"]:
                continue
            dx, dy, dz = pi["x"]-pj["x"], pi["y"]-pj["y"], pi["z"]-pj["z"]
            if dx*dx+dy*dy+dz*dz > MAX_LOS_DIST**2:
                continue
            if vc.is_visible((pi["x"],pi["y"],pi["z"]+EYE),(pj["x"],pj["y"],pj["z"]+EYE)):
                edges.append([i, j]); sees.add(i); sees.add(j)
    return edges, sees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="spirit-vs-falcons-m2-dust2")
    ap.add_argument("--map", default="de_dust2")
    ap.add_argument("--out-stem", default="dust2-los")
    ap.add_argument("--viewer", default="/home/soone/cs2-demo-viewer/public/viewer-data")
    ap.add_argument("--n-rounds", type=int, default=3)
    ap.add_argument("--stride", type=int, default=8)   # downsample ticks
    args = ap.parse_args()

    base = Path("data/processed/demos")
    df = pl.read_parquet(base / f"{args.stem}_ticks.parquet").sort("tick")
    rounds = json.load(open(base / f"{args.stem}_rounds.json"))
    kills = json.load(open(base / f"{args.stem}_kills.json"))
    vc = VisibilityChecker(path=Path(f"/home/soone/.awpy/tris/{args.map}.tri"))

    # pick the N most action-heavy rounds (most kills)
    from collections import Counter
    kc = Counter(k["round_num"] for k in kills if k.get("round_num"))
    top = [rn for rn, _ in kc.most_common(args.n_rounds)]
    rounds_sel = sorted([r for r in rounds if r["round_num"] in top], key=lambda r: r["round_num"])

    out = Path(args.viewer) / args.out_stem
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ri, r in enumerate(rounds_sel, 1):
        rn = r["round_num"]
        start, end = r["start"], r.get("official_end", r["end"])
        sub = df.filter((pl.col("round_num") == rn) & (pl.col("tick") >= start) & (pl.col("tick") <= end))
        ticks = sorted(sub["tick"].unique().to_list())[::args.stride]
        frames = []
        for t in ticks:
            rows = sub.filter(pl.col("tick") == t).sort(["side", "steamid"]).to_dicts()
            players = []
            for row in rows[:10]:
                hp = row.get("health") or 0
                players.append({
                    "name": row.get("name", ""), "side": (row.get("side") or "").upper(),
                    "x": float(row["X"]), "y": float(row["Y"]), "z": float(row["Z"]),
                    "yaw": float(row.get("yaw") or 0.0), "hp": int(hp), "alive": hp > 0,
                })
            edges, sees = los_matrix(players, vc)
            for idx, p in enumerate(players):
                p["sees_enemy"] = idx in sees
            frames.append({"tick": int(t), "players": players, "los_edges": edges})
        (out / f"round_{ri:02d}.json").write_text(json.dumps({
            "round_num": rn, "start_tick": start, "end_tick": end,
            "winner": r.get("winner", ""), "reason": r.get("reason", ""),
            "bomb_plant_tick": r.get("bomb_plant"), "bomb_site": r.get("bomb_site"),
            "frames": frames}))
        written.append(rn)
        avg_edges = np.mean([len(f["los_edges"]) for f in frames])
        print(f"round {ri} (demo round {rn}): {len(frames)} frames, avg {avg_edges:.1f} LOS edges/frame")

    rkills = [{"tick": k["tick"], "round_num": k["round_num"],
               "attacker_name": k.get("attacker_name"), "victim_name": k.get("victim_name"),
               "weapon": k.get("weapon"), "attacker_side": k.get("attacker_side"),
               "attacker_X": k.get("attacker_X"), "attacker_Y": k.get("attacker_Y"),
               "victim_X": k.get("victim_X"), "victim_Y": k.get("victim_Y")}
              for k in kills if k.get("round_num") in written]
    (out / "meta.json").write_text(json.dumps({
        "stem": args.out_stem, "map_name": args.map,
        "header": {"map_name": args.map, "server_name": "LOS overlay demo"},
        "rounds": [{"round_num": i+1, "winner": r.get("winner",""), "reason": r.get("reason","")}
                   for i, r in enumerate(rounds_sel)],
        "kills": rkills, "damages": [], "shots": [], "bomb": []}))

    # update index.json
    idx_path = Path(args.viewer) / "index.json"
    idx = json.load(open(idx_path)) if idx_path.exists() else []
    idx = [e for e in idx if e["stem"] != args.out_stem]
    idx.append({"stem": args.out_stem, "map_name": args.map, "rounds": len(rounds_sel)})
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"wrote {len(written)} rounds to {out}; index updated")


if __name__ == "__main__":
    main()
