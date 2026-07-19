"""Single source of truth for corpus-level round exclusions (datasheet §5).

These are DEFECT exclusions, applied by default everywhere a blob is loaded — do
not confuse with the opt-in `--maps` subset selector in train_world_model.py.

D1  de_anubis  — not in MAP_VOCAB, so its map one-hot is all-zeros
                 (build_tick_sequences.py:256,575). 216 rounds with no map identity.
D2  de_train   — 16 train / 0 val rounds: too few to learn, impossible to evaluate.

Reversible: this only filters lists at load time; the .pt blobs are untouched. To
un-exclude, drop the map from EXCLUDED_MAPS (and, for anubis, add it to MAP_VOCAB +
rebuild so it actually carries identity).
"""
from __future__ import annotations

import os

import torch

EXCLUDED_MAPS = frozenset({"de_anubis", "de_train"})


def load_corpus(path, *, maps=None, tag=None):
    """THE corpus reader — every script that reads a tick-sequence blob goes
    through here (infra-plan §1 item 1).

    torch.load(..., mmap=True): tensor storages stay ON DISK (page cache,
    evictable) until actually touched, so loading a multi-GB blob costs ~MBs
    of RSS instead of the full file. clean_blob and the `maps` keep-set filter
    only rebuild the blob's parallel python LISTS — they never touch tensor
    storage — so both are mmap-safe.

    Consumers must NOT assume the returned tensors are writable: mmap'd
    storages are shared/file-backed — .clone() before any in-place mutation.
    Corpus WRITERS (patch/bake/merge scripts) must not use this: it applies
    the datasheet §5 defect exclusions, which must never leak into bytes
    written back to disk.

    maps: optional keep-set — comma-separated string or iterable of map_names;
          filters ALL parallel per-round lists in lockstep (value_probe's
          _mfilter pattern). None/empty = keep all maps.
    tag:  label for log lines; defaults to the file's basename.
    """
    tag = tag or os.path.basename(str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    clean_blob(blob, tag=tag)  # datasheet §5 D1/D2
    if maps:
        keep = set(maps.split(",")) if isinstance(maps, str) else set(maps)
        n0 = len(blob["metas"])
        idx = [i for i, m in enumerate(blob["metas"]) if m.get("map_name") in keep]
        for k, v in list(blob.items()):
            if isinstance(v, list) and len(v) == n0:
                blob[k] = [v[i] for i in idx]
        print(f"[corpus {tag}] maps filter {sorted(keep)}: kept {len(idx)}/{n0} rounds")
    return blob


def clean_blob(blob: dict, *, verbose: bool = True, tag: str = "") -> int:
    """Drop rounds on defective maps in-place. Returns number of rounds kept.

    Expects a tick-sequence blob: {"tensors": [...], "metas": [...], ...}. Any
    parallel per-round lists (event labels/times, summaries) are filtered in lockstep
    if present, so indices stay aligned.
    """
    metas = blob["metas"]
    keep = [i for i, m in enumerate(metas) if m.get("map_name") not in EXCLUDED_MAPS]
    n_before = len(metas)
    if len(keep) != n_before:
        for k, v in list(blob.items()):
            if isinstance(v, list) and len(v) == n_before:
                blob[k] = [v[i] for i in keep]
    if verbose:
        dropped = n_before - len(keep)
        pfx = f"[corpus {tag}]" if tag else "[corpus]"
        print(f"{pfx} kept {len(keep)}/{n_before} rounds (dropped {dropped} on {sorted(EXCLUDED_MAPS)})")
    return len(keep)
