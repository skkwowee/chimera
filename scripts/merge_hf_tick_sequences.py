#!/usr/bin/env python3
"""Merge ALL per-match tick-sequence blobs from HF (skkwowee/chimera-cs2,
tick_sequences/<match_id>/{train,val}.pt) into pooled train/val blobs with a
MATCH-LEVEL split (fixes the team/style leak where one match's maps straddled
the old demo-level split).

The per-match internal train/val split is IGNORED: every round from every match
is pooled, dedup'd by (demo_stem, round_num, first_tick, n_ticks) — round_num
alone is NOT enough: different matches can produce the SAME demo_stem (e.g.
natus-vincere-vs-vitality-m2-anubis exists in 2394174 AND 2393350 as different
games). HF is canonical, BUT it is NOT a
superset of the local corpus: only ~27 of the 81 local demo stems exist on HF.
So after pooling HF, local train/val rounds whose demo has zero overlap with the
pooled keys are BACK-FILLED as pseudo-matches ("local-<team-pair>", grouping all
of a team pair's demos together — coarser than match-level, still leak-safe).
Disable with --hf-only. Local train_v3/val_v3 are also consulted to decide which
rounds already have 687-d v3 tensors. Blob-level event_*/summaries keys dropped.

Outputs (data/processed/tick_sequences/, '.smoke' suffix in --limit-matches mode):
  train_v2m.pt / val_v2m.pt     597-d blobs: tensors/metas/feature_dim/
                                downsample/schema_version (+match_id per meta)
  split_manifest_v2.json        seed, match_id -> side, per-map round counts
  v3_todo.json                  per split: rounds with NO reusable v3 tensor
                                (index into the v2m blob + demo_stem/round_num)

Then bake the delta + assemble v3m (ONE BVH at a time — do not raise workers):
  python scripts/build_v3_features.py --src data/processed/tick_sequences/train_v2m.pt \
      --out data/processed/tick_sequences/train_v3m.pt \
      --todo data/processed/tick_sequences/v3_todo.json \
      --reuse data/processed/tick_sequences/train_v3.pt,data/processed/tick_sequences/val_v3.pt \
      --workers 1
  (same for val_v2m -> val_v3m)

Usage:
  smoke: python scripts/merge_hf_tick_sequences.py --limit-matches 3
  full:  python scripts/merge_hf_tick_sequences.py
"""
from __future__ import annotations
import argparse, gc, json, random, re, resource, time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "skkwowee/chimera-cs2"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "tick_sequences"
KEY_MAPS = ("de_mirage", "de_dust2", "de_inferno")
FDIM = 597
DOWNSAMPLE = 8
# one real match whose local stems are malformed ('<t1>-vs-<t2>' mangled); keep
# its three demos in ONE pseudo-match or the split could straddle a match
LOCAL_GROUP_FIX = {"ninjas-vs-pyjamas": "ninjas-in-pyjamas"}


def norm_stem(s: str) -> str:
    """Canonical team order: a-vs-b-mX-map == b-vs-a-mX-map. The HF demo pipeline
    and the old local pipeline disagree on team order for the same game (e.g.
    vitality-vs-natus-vincere-m2-anubis vs natus-vincere-vs-vitality-m2-anubis).
    Used ONLY for identity matching; metas keep the original stem."""
    m = re.match(r"^(.+?)-vs-(.+?)-(m\d+-.+)$", s)
    if not m:
        return s
    t1, t2, rest = m.groups()
    return "-vs-".join(sorted((t1, t2))) + "-" + rest


def team_pair(stem: str) -> str:
    """'aurora-vs-furia-m2-inferno' -> 'aurora-vs-furia' (pseudo-match key for
    local-only demos; the old pipeline kept no match ids)."""
    s = re.sub(r"-m\d+-\w+$", "", norm_stem(stem))
    return LOCAL_GROUP_FIX.get(s, s)


def load_blob(path):
    """mmap keeps tensor data file-backed (page cache, evictable) — peak RSS stays low."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def pool_rounds(matches: list[str], fileset: set[str]):
    """Download + pool every round of every match. Returns
    (rounds, seen, versions) — versions maps input source -> schema_version
    (pre-provenance blobs report 'unversioned-pre-v2.1')."""
    rounds, seen, stem_match, collided, versions = [], {}, {}, set(), {}
    t0 = time.time()
    for k, mid in enumerate(matches):
        for part in ("train", "val"):
            rel = f"tick_sequences/{mid}/{part}.pt"
            if rel not in fileset:
                continue
            try:
                b = load_blob(hf_hub_download(REPO_ID, rel, repo_type="dataset"))
                assert b["feature_dim"] == FDIM, f"feature_dim={b['feature_dim']}"
                assert b["downsample"] == DOWNSAMPLE, f"downsample={b['downsample']}"
            except Exception as e:
                print(f"  WARN skipping {rel}: {type(e).__name__}: {e}")
                continue
            versions[mid] = b.get("schema_version", "unversioned-pre-v2.1")
            for t, m in zip(b["tensors"], b["metas"]):
                if t.ndim != 2 or t.shape[1] != FDIM or t.dtype != torch.float32:
                    print(f"  WARN bad tensor {rel} r{m.get('round_num')}: {tuple(t.shape)} {t.dtype}")
                    continue
                if t.shape[0] != m["n_ticks"]:
                    print(f"  WARN n_ticks mismatch {rel} r{m['round_num']}: {t.shape[0]} vs {m['n_ticks']}")
                key = (norm_stem(m["demo_stem"]), m["round_num"], m["first_tick"], m["n_ticks"])
                if key in seen:                       # DEDUP: keep first occurrence
                    print(f"  WARN dup round {key}: match {mid} vs {seen[key]} — keeping first")
                    continue
                seen[key] = mid
                if stem_match.setdefault(m["demo_stem"], mid) != mid:
                    collided.add((m["demo_stem"], stem_match[m["demo_stem"]], mid))
                meta = {k2: v for k2, v in m.items() if not k2.startswith("event_")}
                meta["match_id"] = mid
                rounds.append((mid, meta, t))         # tensor stays mmap-backed
            del b
        if (k + 1) % 10 == 0 or k + 1 == len(matches):
            gc.collect()
            print(f"  pooled {k+1}/{len(matches)} matches, {len(rounds)} rounds  {time.time()-t0:.0f}s", flush=True)
    for stem, m1, m2 in sorted(collided):
        print(f"  NOTE stem collision {stem}: matches {m1} + {m2} (distinct games, both kept)")
    return rounds, seen, versions


def check_schema_versions(versions: dict) -> None:
    """Refuse to merge inputs baked under different schema versions.

    Dim-preserving semantic changes (bomb-state site fix, round_time
    re-anchoring) are invisible to the feature_dim assert — mixing pre-fix
    and post-fix blobs would silently poison the corpus. Fails listing the
    offending inputs per version."""
    by_ver = defaultdict(list)
    for src, ver in versions.items():
        by_ver[ver].append(str(src))
    if len(by_ver) > 1:
        detail = "; ".join(
            f"{v}: {sorted(srcs)[:8]}{' ...' if len(srcs) > 8 else ''}"
            for v, srcs in sorted(by_ver.items()))
        raise SystemExit(
            f"FATAL schema_version mismatch across inputs — re-bake the old "
            f"matches before merging. {detail}")


def backfill_local(rounds, seen, out_dir: Path, versions: dict):
    """Append local rounds whose demo has ZERO key overlap with the pooled HF
    rounds, grouped into 'local-<team-pair>' pseudo-matches. Demos with any
    overlap are HF-canonical: skipped whole (stragglers would imply a parser
    drift we'd rather hear about than silently mix)."""
    by_stem = defaultdict(list)
    for name in ("train.pt", "val.pt"):
        p = out_dir / name
        if not p.exists():
            print(f"  WARN no local {p} — skipping back-fill of that blob")
            continue
        b = load_blob(p)
        assert b["feature_dim"] == FDIM and b["downsample"] == DOWNSAMPLE
        versions[f"local:{name}"] = b.get("schema_version", "unversioned-pre-v2.1")
        for t, m in zip(b["tensors"], b["metas"]):
            by_stem[m["demo_stem"]].append((t, m))     # tensors stay mmap-backed
        del b
    gc.collect()
    added, demos, pseudo = 0, 0, set()
    for stem in sorted(by_stem):
        rs = by_stem[stem]
        dup = sum((norm_stem(m["demo_stem"]), m["round_num"], m["first_tick"], m["n_ticks"]) in seen
                  for _, m in rs)
        if dup:
            if dup < len(rs):
                print(f"  WARN local demo {stem}: {dup}/{len(rs)} rounds match HF — "
                      f"PARTIAL overlap (parser drift?), dropping local copy entirely")
            continue                                   # HF-canonical demo
        mid = "local-" + team_pair(stem)
        for t, m in rs:
            meta = {k: v for k, v in m.items() if not k.startswith("event_")}
            meta["match_id"] = mid
            rounds.append((mid, meta, t))
        added += len(rs)
        demos += 1
        pseudo.add(mid)
    print(f"  back-filled {added} local-only rounds / {demos} demos as {len(pseudo)} "
          f"pseudo-matches (HF is NOT a superset of the old corpus)")
    return rounds


def split_matches(rounds, seed: int, val_frac: float) -> set[str]:
    """Deterministic MATCH-level split, stratified by each match's dominant map so
    val gets a share of every key map. Returns the set of val match_ids."""
    match_maps = defaultdict(Counter)
    for mid, m, _ in rounds:
        match_maps[mid][m["map_name"]] += 1
    groups = defaultdict(list)
    for mid in sorted(match_maps):
        dom = max(sorted(match_maps[mid]), key=match_maps[mid].get)   # sorted = stable tie-break
        groups[dom].append(mid)
    rng, val = random.Random(seed), set()
    for gmap in sorted(groups):
        g = groups[gmap][:]
        rng.shuffle(g)
        n_val = round(val_frac * len(g))
        if gmap in KEY_MAPS:
            n_val = max(1, n_val)
        val.update(g[:n_val])
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-matches", type=int, default=0,
                    help="smoke mode: first N matches; outputs get a .smoke suffix")
    ap.add_argument("--matches", default="",
                    help="smoke mode: comma-sep explicit match ids (also .smoke suffix)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--data", default=str(OUT_DIR))
    ap.add_argument("--hf-only", action="store_true",
                    help="skip back-filling local-only demos (drops ~50 demos not on HF)")
    args = ap.parse_args()
    out_dir = Path(args.data)
    sfx = ".smoke" if (args.limit_matches or args.matches) else ""

    api = HfApi()
    files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    matches = sorted({f.split("/")[1] for f in files
                      if f.startswith("tick_sequences/") and f.endswith(".pt")})
    if args.matches:
        want = set(args.matches.split(","))
        assert want <= set(matches), f"unknown match ids: {want - set(matches)}"
        matches = sorted(want)
    elif args.limit_matches:
        matches = matches[:args.limit_matches]
    print(f"{len(matches)} matches on HF{' (limited)' if sfx else ''}")

    rounds, seen, versions = pool_rounds(matches, files)
    assert rounds, "no rounds pooled"
    if not args.hf_only:
        rounds = backfill_local(rounds, seen, out_dir, versions)
    check_schema_versions(versions)
    schema_version = next(iter(versions.values()), "unversioned-pre-v2.1")

    # ---- match-level split -------------------------------------------------
    val_matches = split_matches(rounds, args.seed, args.val_frac)
    side_idx = {"train": [], "val": []}
    for i, (mid, _, _) in enumerate(rounds):
        side_idx["val" if mid in val_matches else "train"].append(i)
    map_counts = {s: Counter(rounds[i][1]["map_name"] for i in side_idx[s]) for s in side_idx}
    for mp in KEY_MAPS:
        tot = map_counts["train"][mp] + map_counts["val"][mp]
        share = map_counts["val"][mp] / tot if tot else 0.0
        flag = "" if 0.08 <= share <= 0.30 or tot == 0 else "  <-- CHECK: skewed"
        print(f"  {mp}: val share {share:.2f} ({map_counts['val'][mp]}/{tot}){flag}")

    all_mids = sorted({r[0] for r in rounds})
    manifest = {"seed": args.seed, "val_frac": args.val_frac, "n_matches": len(all_mids),
                "n_hf_matches": len(matches),
                "matches": {mid: ("val" if mid in val_matches else "train") for mid in all_mids},
                "round_counts": {s: dict(map_counts[s]) for s in ("train", "val")}}
    mpath = out_dir / f"split_manifest_v2{sfx}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {mpath}")

    # ---- v3 reuse index (existing 687-d rounds) ------------------------------
    # 4-tuple key: stem collisions exist (same stem, different game) and a v2->v3
    # mixup would silently poison features; first_tick+n_ticks disambiguate.
    v3_keys = set()
    for name in ("train_v3.pt", "val_v3.pt"):
        p = out_dir / name
        if not p.exists():
            print(f"  WARN no {p} — all rounds will be in v3_todo")
            continue
        b = load_blob(p)
        assert b.get("per_player_dim") == 65, f"{name}: per_player_dim={b.get('per_player_dim')}"
        for j, m in enumerate(b["metas"]):
            assert b["tensors"][j].shape[0] == m["n_ticks"], (name, j)
            v3_keys.add((norm_stem(m["demo_stem"]), m["round_num"], m["first_tick"], m["n_ticks"]))
        del b
        gc.collect()
    local_stems = {k[0] for k in v3_keys}
    pooled_stems = {norm_stem(r[1]["demo_stem"]) for r in rounds}
    print(f"local v3 index: {len(v3_keys)} rounds / {len(local_stems)} stems; "
          f"{len(local_stems & pooled_stems)} stems in pooled corpus"
          + ("" if sfx else f"; {len(local_stems - pooled_stems)} uncovered (expect 0)"))
    if not sfx and local_stems - pooled_stems:
        print(f"  WARN local stems not in pooled corpus: {sorted(local_stems - pooled_stems)[:5]} ...")

    todo, reused = {"train": [], "val": []}, Counter()
    for side in ("train", "val"):
        for pos, i in enumerate(side_idx[side]):       # pos = index INTO the v2m blob
            m = rounds[i][1]
            if (norm_stem(m["demo_stem"]), m["round_num"], m["first_tick"], m["n_ticks"]) in v3_keys:
                reused[side] += 1
            else:
                todo[side].append({"index": pos, "demo_stem": m["demo_stem"],
                                   "round_num": m["round_num"],
                                   "first_tick": m["first_tick"], "n_ticks": m["n_ticks"]})
    tpath = out_dir / f"v3_todo{sfx}.json"
    tpath.write_text(json.dumps(todo, indent=1))
    print(f"wrote {tpath}  (train: {len(todo['train'])} to bake, val: {len(todo['val'])})")

    # ---- write v2m blobs (one side at a time) --------------------------------
    for side in ("train", "val"):
        blob = {"tensors": [rounds[i][2] for i in side_idx[side]],
                "metas":   [rounds[i][1] for i in side_idx[side]],
                "feature_dim": FDIM, "downsample": DOWNSAMPLE,
                "schema_version": schema_version}
        out = out_dir / f"{side}_v2m{sfx}.pt"
        torch.save(blob, out)
        print(f"wrote {out}  ({len(blob['tensors'])} rounds)", flush=True)
        del blob
        gc.collect()

    # ---- summary -------------------------------------------------------------
    print("\n== summary ==")
    print(f"{'side':6s} {'matches':>7} {'rounds':>7} {'demos':>6} {'v3-reuse':>8} {'v3-bake':>8}")
    for side in ("train", "val"):
        mids = {rounds[i][0] for i in side_idx[side]}
        stems = {rounds[i][1]["demo_stem"] for i in side_idx[side]}
        print(f"{side:6s} {len(mids):>7} {len(side_idx[side]):>7} {len(stems):>6} "
              f"{reused[side]:>8} {len(todo[side]):>8}")
    all_maps = sorted(set(map_counts["train"]) | set(map_counts["val"]))
    print(f"{'map':14s} {'train':>6} {'val':>5}")
    for mp in all_maps:
        print(f"{mp:14s} {map_counts['train'][mp]:>6} {map_counts['val'][mp]:>5}")
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")


if __name__ == "__main__":
    main()
