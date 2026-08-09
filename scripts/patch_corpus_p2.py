#!/usr/bin/env python3
"""Runbook [1b]: derive the P2 corpus from the certified P1 blobs.

P2 is dimension-preserving. It fixes information already recoverable from the
P1 tensors and attaches optional, non-model sidecars from the 81 local parquet
archives:

* persist the dropped C4 xy position from the carrier's last position until a
  pickup or plant;
* clamp normalized bomb age to [0, 1];
* retain at most 7 seconds (57 frames at 8 Hz, including the first end frame)
  of end phase;
* enrich round metadata with the positional-slot steamids/names when a local
  parquet matches the round's full source key;
* attach ``place_ids`` as [T, 10] int16 tensors plus an explicit, shared
  ``place_vocab``. ID 0 is always ``<missing>``.

The source-key check is deliberately strict. Several unrelated matches share a
demo stem; a stem-only join would silently attach the wrong identities/places.
Unmatched rounds receive an all-zero sidecar and a status explaining why.

This writer never applies load-time map exclusions. It reads ``*_p1.pt`` and
writes new ``*_p2.pt`` files via a ``.partial`` file followed by an atomic
rename. Existing outputs are refused unless ``--overwrite`` is explicit.

The repository blob guardrail forbids loading train blobs in agent sessions.
Use ``--only val_v2m`` for local certification, then run the four-blob command
manually on the quiet machine after reviewing the emitted report.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_corpus_invariants import (
    check_bomb_bits_consistent,
    check_dim7_plant_gated,
    check_lineage,
    check_match_ids,
    check_no_nan_inf,
    check_round_time,
)

TS_DIR = ROOT / "data/processed/tick_sequences"
DEMOS_DIR = ROOT / "data/processed/demos"
BLOBS = ["val_v2m", "train_v2m", "val_v3m", "train_v3m"]
P2_BLOB_NAMES = [f"{name}_p2.pt" for name in BLOBS]
SPLIT_OF = {
    "val_v2m": "val",
    "train_v2m": "train",
    "val_v3m": "val",
    "train_v3m": "train",
}

N_PLAYERS = 10
RAW_PPD = 56
HAS_C4 = 14
HZ = 8
MAX_END_SECONDS = 7
MAX_END_FRAMES = MAX_END_SECONDS * HZ + 1
MISSING_PLACE = "<missing>"

# Offsets within the 37-d global block.
G_PHASE = 7          # [freeze, live, post_plant, end]
G_BOMB = 15          # [none, carried, planted_a, planted_b]
G_BX, G_BY = 19, 20
G_BOMB_AGE = 21


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return result.stdout.strip()


def p2_keep_length(tensor: torch.Tensor, ppd: int) -> int:
    """Return the retained length after the round_end + 7 second cap."""
    global_start = N_PLAYERS * ppd
    end = tensor[:, global_start + G_PHASE + 3] > 0.5
    if not bool(end.any()):
        # r24 truncation: some source rounds contain no end-phase frame. This is
        # disclosed in D8 and cannot be repaired from the tensor alone.
        return tensor.shape[0]
    first_end = int(end.nonzero(as_tuple=False)[0, 0])
    return min(tensor.shape[0], first_end + MAX_END_FRAMES)


def patch_round_p2(tensor: torch.Tensor, ppd: int) -> tuple[torch.Tensor, dict[str, int]]:
    """Apply tensor-only P2 fixes and return the (possibly cropped) tensor."""
    global_start = N_PLAYERS * ppd
    global_block = tensor[:, global_start:]
    c4 = torch.stack(
        [tensor[:, player * ppd + HAS_C4] for player in range(N_PLAYERS)],
        dim=1,
    ) > 0.5
    any_c4 = c4.any(dim=1)
    bomb_bits = global_block[:, G_BOMB:G_BOMB + 4]
    planted = (bomb_bits[:, 2] > 0.5) | (bomb_bits[:, 3] > 0.5)

    dropped_frames = 0
    drop_events = 0
    invalid_drop_sources = 0
    drop_xy: tuple[float, float] | None = None
    for frame in range(1, tensor.shape[0]):
        falling_edge = bool(any_c4[frame - 1] and not any_c4[frame])
        if falling_edge and not bool(planted[frame]):
            carrier_slots = c4[frame - 1].nonzero(as_tuple=False).flatten()
            if carrier_slots.numel():
                carrier = int(carrier_slots[0])
                x = float(tensor[frame - 1, carrier * ppd])
                y = float(tensor[frame - 1, carrier * ppd + 1])
                if np.isfinite(x) and np.isfinite(y):
                    drop_xy = (x, y)
                    drop_events += 1
                else:
                    drop_xy = None
                    invalid_drop_sources += 1

        # Pickup and plant both end a dropped interval. Check this after the
        # falling edge so the first no-carrier frame receives the drop position.
        if bool(planted[frame]) or bool(any_c4[frame]):
            drop_xy = None
        elif drop_xy is not None:
            global_block[frame, G_BX] = drop_xy[0]
            global_block[frame, G_BY] = drop_xy[1]
            dropped_frames += 1

    before_clamp = global_block[:, G_BOMB_AGE].clone()
    global_block[:, G_BOMB_AGE].clamp_(0.0, 1.0)
    bomb_age_clamped = int((before_clamp != global_block[:, G_BOMB_AGE]).sum())

    keep = p2_keep_length(tensor, ppd)
    frames_cropped = tensor.shape[0] - keep
    return tensor[:keep], {
        "drop_events": drop_events,
        "dropped_frames": dropped_frames,
        "invalid_drop_sources": invalid_drop_sources,
        "bomb_age_clamped": bomb_age_clamped,
        "frames_cropped": frames_cropped,
        "rounds_cropped": int(frames_cropped > 0),
    }


def build_place_vocab(demos_dir: Path) -> list[str]:
    """Build one deterministic vocabulary shared by every P2 blob."""
    places: set[str] = set()
    for path in sorted(demos_dir.glob("*_ticks.parquet")):
        schema = pl.scan_parquet(path).collect_schema()
        if "place" not in schema.names():
            continue
        values = (
            pl.scan_parquet(path)
            .select(pl.col("place").drop_nulls().unique())
            .collect()["place"]
            .to_list()
        )
        places.update(str(value) for value in values if str(value).strip())
    return [MISSING_PLACE, *sorted(places)]


def _slot_order(round_df: pl.DataFrame) -> list[int | None]:
    first_tick = round_df["tick"].min()
    first = round_df.filter(pl.col("tick") == first_tick).sort("steamid")
    t_ids = first.filter(pl.col("side").str.to_lowercase() == "t")["steamid"].to_list()[:5]
    ct_ids = first.filter(pl.col("side").str.to_lowercase() == "ct")["steamid"].to_list()[:5]
    return [*t_ids, *([None] * (5 - len(t_ids))), *ct_ids, *([None] * (5 - len(ct_ids)))]


def _round_sidecar(
    round_df: pl.DataFrame,
    meta: dict,
    tensor_length: int,
    keep_length: int,
    place_to_id: dict[str, int],
) -> tuple[torch.Tensor, list[int | None], list[str | None], int] | None:
    """Return a verified local sidecar, or None on a source-key mismatch."""
    if round_df.height == 0:
        return None
    ticks = np.sort(round_df["tick"].unique().to_numpy())
    kept_ticks = ticks[:: int(meta.get("downsample", HZ))]
    if len(kept_ticks) != tensor_length:
        return None
    if int(kept_ticks[0]) != int(meta.get("first_tick", -1)):
        return None
    if int(kept_ticks[-1]) != int(meta.get("last_tick", -1)):
        return None

    slots = _slot_order(round_df)
    names_by_sid: dict[int, str] = {}
    for sid in (slot for slot in slots if slot is not None):
        names = (
            round_df.filter(pl.col("steamid") == sid)["name"]
            .drop_nulls()
            .to_list()
        )
        if names:
            names_by_sid[int(sid)] = str(names[0])

    place_ids = torch.zeros((keep_length, N_PLAYERS), dtype=torch.int16)
    lookup = pl.DataFrame({"tick": kept_ticks[:keep_length].astype(np.int32)})
    for slot_index, sid in enumerate(slots):
        if sid is None:
            continue
        player = (
            round_df.filter(pl.col("steamid") == sid)
            .select("tick", "place")
            .unique(subset=["tick"], keep="first")
        )
        aligned = lookup.join(player, on="tick", how="left")["place"].to_list()
        ids = [place_to_id.get(str(place), 0) if place is not None else 0 for place in aligned]
        place_ids[:, slot_index] = torch.tensor(ids, dtype=torch.int16)

    steamids = [int(sid) if sid is not None else None for sid in slots]
    names = [names_by_sid.get(int(sid)) if sid is not None else None for sid in slots]
    return place_ids, steamids, names, int(kept_ticks[keep_length - 1])


def attach_local_sidecars(
    blob: dict,
    demos_dir: Path,
    place_vocab: list[str],
    keep_lengths: list[int],
) -> tuple[list[torch.Tensor], dict[str, int], dict[int, int]]:
    """Attach verified identity/place data without trusting demo stem alone."""
    place_to_id = {place: index for index, place in enumerate(place_vocab)}
    if len(place_to_id) != len(place_vocab):
        raise ValueError("place_vocab contains duplicates")
    if len(place_vocab) >= torch.iinfo(torch.int16).max:
        raise ValueError(f"place_vocab too large for int16: {len(place_vocab)}")

    sidecars = [
        torch.zeros((keep, N_PLAYERS), dtype=torch.int16)
        for keep in keep_lengths
    ]
    exact_last_ticks: dict[int, int] = {}
    by_stem: dict[str, list[int]] = defaultdict(list)
    for index, meta in enumerate(blob["metas"]):
        by_stem[str(meta.get("demo_stem", ""))].append(index)

    report = {
        "verified_rounds": 0,
        "source_unavailable_rounds": 0,
        "source_key_mismatch_rounds": 0,
        "verified_stems": 0,
    }
    columns = ["tick", "round_num", "steamid", "name", "side", "place"]
    for stem, indices in sorted(by_stem.items()):
        path = demos_dir / f"{stem}_ticks.parquet"
        if not path.exists():
            report["source_unavailable_rounds"] += len(indices)
            for index in indices:
                blob["metas"][index]["p2_sidecar_status"] = "source_unavailable"
            continue
        schema_names = set(pl.scan_parquet(path).collect_schema().names())
        if not set(columns) <= schema_names:
            report["source_unavailable_rounds"] += len(indices)
            for index in indices:
                blob["metas"][index]["p2_sidecar_status"] = "source_columns_missing"
            continue
        demo_df = pl.read_parquet(path, columns=columns)
        stem_verified = False
        for index in indices:
            meta = blob["metas"][index]
            round_df = demo_df.filter(pl.col("round_num") == int(meta["round_num"]))
            result = _round_sidecar(
                round_df,
                meta,
                blob["tensors"][index].shape[0],
                keep_lengths[index],
                place_to_id,
            )
            if result is None:
                report["source_key_mismatch_rounds"] += 1
                meta["p2_sidecar_status"] = "source_key_mismatch"
                continue
            places, steamids, names, exact_last = result
            sidecars[index] = places
            exact_last_ticks[index] = exact_last
            meta["slot_steamids"] = steamids
            meta["slot_names"] = names
            meta["slot_order"] = "T0..T4,CT0..CT4"
            meta["p2_sidecar_status"] = "verified_local_parquet"
            report["verified_rounds"] += 1
            stem_verified = True
        report["verified_stems"] += int(stem_verified)
        del demo_df
        gc.collect()
    return sidecars, report, exact_last_ticks


def update_meta_after_crop(
    meta: dict,
    old_length: int,
    new_length: int,
    exact_last_tick: int | None,
) -> None:
    meta["n_ticks"] = new_length
    if new_length == old_length:
        return
    old_last_tick = int(meta["last_tick"])
    meta["p2_original_n_ticks"] = old_length
    meta["p2_original_last_tick"] = old_last_tick
    if exact_last_tick is not None:
        meta["last_tick"] = exact_last_tick
        meta["p2_last_tick_source"] = "verified_local_parquet"
    else:
        removed = old_length - new_length
        meta["last_tick"] = old_last_tick - removed * int(meta.get("downsample", HZ))
        meta["p2_last_tick_source"] = "downsample_estimate_no_tick_sidecar"


def check_p2_blob(blob: dict) -> list[str]:
    """P2-specific structural invariants, independent of real blob size."""
    messages: list[str] = []
    n_rounds = len(blob.get("metas", []))
    places = blob.get("place_ids")
    vocab = blob.get("place_vocab")
    if not isinstance(vocab, list) or not vocab or vocab[0] != MISSING_PLACE:
        messages.append("check_p2_blob: place_vocab must start with <missing>")
    if not isinstance(places, list) or len(places) != n_rounds:
        messages.append(
            f"check_p2_blob: place_ids length {len(places) if isinstance(places, list) else 'missing'} "
            f"!= metas length {n_rounds}"
        )
        return messages
    vocab_size = len(vocab) if isinstance(vocab, list) else 0
    for index, (tensor, meta, place) in enumerate(zip(blob["tensors"], blob["metas"], places)):
        if int(meta.get("n_ticks", -1)) != tensor.shape[0]:
            messages.append(f"check_p2_blob: round {index} meta n_ticks != tensor length")
        if place.shape != (tensor.shape[0], N_PLAYERS) or place.dtype != torch.int16:
            messages.append(
                f"check_p2_blob: round {index} place_ids {tuple(place.shape)}/{place.dtype} "
                f"!= ({tensor.shape[0]}, {N_PLAYERS})/int16"
            )
        elif place.numel() and (int(place.min()) < 0 or int(place.max()) >= vocab_size):
            messages.append(f"check_p2_blob: round {index} place id outside vocabulary")
        ppd = int(blob.get("per_player_dim", RAW_PPD))
        global_start = N_PLAYERS * ppd
        end_count = int((tensor[:, global_start + G_PHASE + 3] > 0.5).sum())
        if end_count > MAX_END_FRAMES:
            messages.append(
                f"check_p2_blob: round {index} has {end_count} end frames > {MAX_END_FRAMES}"
            )
        age = tensor[:, global_start + G_BOMB_AGE]
        if bool(((age < 0) | (age > 1)).any()):
            messages.append(f"check_p2_blob: round {index} bomb_age outside [0,1]")
    return messages[:20]


def _sum_stats(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + int(value)


def patch_blob(
    blob: dict,
    demos_dir: Path,
    place_vocab: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Apply P2 to an in-memory blob. Split out for fixture certification."""
    ppd = int(blob.get("per_player_dim", RAW_PPD))
    old_lengths = [tensor.shape[0] for tensor in blob["tensors"]]
    keep_lengths = [p2_keep_length(tensor, ppd) for tensor in blob["tensors"]]
    place_ids, sidecar_report, exact_last_ticks = attach_local_sidecars(
        blob, demos_dir, place_vocab, keep_lengths
    )

    stats: dict[str, int] = {}
    patched_tensors = []
    for index, tensor in enumerate(blob["tensors"]):
        patched, round_stats = patch_round_p2(tensor, ppd)
        if patched.shape[0] != keep_lengths[index]:
            raise AssertionError("P2 crop length changed between planning and mutation")
        patched_tensors.append(patched)
        update_meta_after_crop(
            blob["metas"][index],
            old_lengths[index],
            patched.shape[0],
            exact_last_ticks.get(index),
        )
        _sum_stats(stats, round_stats)
    blob["tensors"] = patched_tensors
    blob["place_ids"] = place_ids
    blob["place_vocab"] = place_vocab
    blob["place_sidecar_schema"] = {
        "dtype": "int16",
        "shape": "[round_frames,10]",
        "slot_order": "T0..T4,CT0..CT4",
        "missing_id": 0,
        "source_join": "(demo_stem,round_num,first_tick,n_ticks,last_tick)",
    }
    return stats, sidecar_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma subset of " + ",".join(BLOBS))
    parser.add_argument("--demos-dir", type=Path, default=DEMOS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    todo = [name for name in (args.only.split(",") if args.only else BLOBS) if name]
    unknown = set(todo) - set(BLOBS)
    if unknown:
        raise SystemExit(f"unknown blob names: {sorted(unknown)}")

    split_manifest_path = TS_DIR / "split_manifest_v2.json"
    split_manifest = json.loads(split_manifest_path.read_text())
    manifest_path = TS_DIR / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    place_vocab = build_place_vocab(args.demos_dir)
    print(f"place vocab: {len(place_vocab)} entries from {args.demos_dir}")

    script_sha = git_sha(Path(__file__))
    date = _dt.date.today().isoformat()
    for name in todo:
        src = TS_DIR / f"{name}_p1.pt"
        dst = TS_DIR / f"{name}_p2.pt"
        partial = dst.with_suffix(dst.suffix + ".partial")
        if not src.exists():
            raise SystemExit(f"missing P1 source: {src}")
        if (dst.exists() or partial.exists()) and not args.overwrite:
            raise SystemExit(
                f"refusing existing output/partial for {name}; inspect it, then use --overwrite"
            )
        if args.overwrite:
            # Targets are resolved constants beneath TS_DIR, never user-derived
            # directories. Files only; no recursive operation.
            dst.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)

        print(f"=== {name}: hashing + non-mmap load ...", flush=True)
        sha_pre = sha256_file(src)
        blob = torch.load(src, map_location="cpu", weights_only=False)
        if "patch1" not in str(blob.get("schema_version", "")):
            raise SystemExit(f"{src.name} is not a certified P1 blob")

        stats, sidecar_report = patch_blob(blob, args.demos_dir, place_vocab)
        transforms = [
            "dropped_bomb_xy_from_c4_falling_edge",
            "metadata_identity_enrichment_source_keyed",
            "place_ids_sidecar_source_keyed",
            "end_phase_cap_7s(D8)",
            "bomb_age_clamp_1(D8)",
        ]
        entry = {
            "script": "scripts/patch_corpus_p2.py",
            "script_sha": script_sha,
            "transforms": transforms,
            "sha256_pre": sha_pre,
            "sha256_post": "recorded-in-corpus_manifest.json",
            "date": date,
        }
        blob["patch_lineage"] = blob.get("patch_lineage", []) + [entry]
        blob["schema_version"] = str(blob.get("schema_version", "v2+patch1")) + "+patch2"
        blob["p2_report"] = {"tensor": stats, "sidecar": sidecar_report}

        violations = (
            check_bomb_bits_consistent(blob)
            + check_round_time(blob)
            + check_no_nan_inf(blob)
            + check_lineage(blob)
            + check_match_ids(blob, manifest=split_manifest, split=SPLIT_OF[name])
            + check_p2_blob(blob)
        )
        if int(blob.get("per_player_dim", RAW_PPD)) > RAW_PPD:
            violations += check_dim7_plant_gated(blob)
        if violations:
            print(f"!! {name}: {len(violations)} violations; not saving")
            for violation in violations[:20]:
                print("   ", violation)
            raise SystemExit(1)

        print(f"    report tensor={stats} sidecar={sidecar_report}", flush=True)
        print(f"    save {partial.name} ...", flush=True)
        torch.save(blob, partial)
        del blob
        gc.collect()
        sha_post = sha256_file(partial)
        os.replace(partial, dst)

        manifest["blobs"][dst.name] = {
            "sha256": sha_post,
            "bytes": dst.stat().st_size,
            "sha256_source": sha_pre,
        }
        # --overwrite replaces one generation; it must not accumulate stale
        # lineage rows for the same destination blob.
        manifest["patch_lineage"] = [
            prior for prior in manifest["patch_lineage"]
            if prior.get("blob") != dst.name
        ]
        manifest["patch_lineage"].append(
            {**entry, "blob": dst.name, "sha256_post": sha_post,
             "report": {"tensor": stats, "sidecar": sidecar_report}}
        )
        completed = sorted(name for name in P2_BLOB_NAMES if name in manifest["blobs"])
        p2_complete = len(completed) == len(P2_BLOB_NAMES)
        manifest["p2_status"] = {
            "required_blobs": P2_BLOB_NAMES,
            "completed_blobs": completed,
            "complete": p2_complete,
            "canonical": False,
            "note": "canonical only after all blobs and fresh-builder validation diff pass",
        }
        manifest["corpus_version"] = (
            "2.2.0+patch2" if p2_complete else "2.1.0+patch1"
        )
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"    OK pre={sha_pre[:12]} post={sha_post[:12]}")

    manifest["blobs"]["corpus_manifest.json"] = {"sha256": "self", "bytes": 0}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest -> {manifest_path}")
    print("P2 bytes written. Still required before canonical promotion: fresh-builder diff report.")


if __name__ == "__main__":
    main()
