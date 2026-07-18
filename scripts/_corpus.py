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

EXCLUDED_MAPS = frozenset({"de_anubis", "de_train"})


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
