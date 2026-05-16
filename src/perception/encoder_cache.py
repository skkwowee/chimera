"""Lookup adapter for precomputed Level-2 encoder embeddings.

Built by scripts/build_encoder_embedding_cache.py. At GRPO training/eval
time, looks up the v6 encoder's embedding at a given (demo, round, tick)
without paying any forward-pass cost.

Tick lookup picks the nearest downsampled position (encoder runs at 8 Hz
from 64 Hz raw). If the requested tick is outside any round's range, the
caller decides what to do — `lookup()` returns None.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class EncoderEmbeddingCache:
    """Pre-computed (demo_stem, round_num, tick) → embedding lookup.

    Accepts either:
      - a single .pt file written by build_encoder_embedding_cache.py
        (legacy / small-d_model layout), OR
      - a directory containing embedding_cache_train.pt +
        embedding_cache_val.pt (split layout used for d_model≥1024 to
        avoid OOM during build), OR
      - a path to one of the *_train.pt / *_val.pt split files (the
        sibling file is auto-loaded if present).
    """

    def __init__(self, cache_path: str | Path):
        p = Path(cache_path)
        payloads: list[dict] = []
        if p.is_dir():
            # Discover all cache files in this directory: legacy single-file,
            # split-by-split, or chunked-by-N-rounds (any of these forms).
            paths = sorted(p.glob("embedding_cache*.pt"))
            for fp in paths:
                payloads.append(torch.load(str(fp), weights_only=False,
                                             map_location="cpu"))
        else:
            payloads.append(torch.load(str(p), weights_only=False,
                                         map_location="cpu"))
            # If the user passed embedding_cache_train.pt or a chunk file,
            # auto-load its siblings from the same directory.
            if "embedding_cache" in p.name and p.name != "embedding_cache.pt":
                for sib in sorted(p.parent.glob("embedding_cache*.pt")):
                    if sib == p:
                        continue
                    payloads.append(torch.load(str(sib), weights_only=False,
                                                 map_location="cpu"))

        if not payloads:
            raise FileNotFoundError(f"no embedding cache files at {p}")

        first = payloads[0]
        self.encoder_ckpt: str = first.get("encoder_ckpt", "")
        self.d_model: int = int(first["d_model"])
        self.downsample: int = int(first["downsample"])
        # Merge entries (split layout) or just take the single dict
        self.entries: dict[tuple[str, int], dict] = {}
        for pl in payloads:
            self.entries.update(pl["entries"])

    @property
    def n_rounds(self) -> int:
        return len(self.entries)

    @property
    def n_ticks(self) -> int:
        return sum(int(e["embeddings"].shape[0]) for e in self.entries.values())

    def has(self, demo_stem: str, round_num: int) -> bool:
        return (str(demo_stem), int(round_num)) in self.entries

    def lookup(
        self,
        demo_stem: str,
        round_num: int,
        tick: int,
    ) -> np.ndarray | None:
        """Return the embedding at the downsampled tick closest to `tick`.

        Returns None if (demo, round) is missing from the cache. Out-of-range
        ticks clamp to the first/last available position in the round.
        """
        key = (str(demo_stem), int(round_num))
        e = self.entries.get(key)
        if e is None:
            return None
        ticks = e["ticks"]
        idx = int(np.searchsorted(ticks, int(tick)))
        if idx <= 0:
            idx = 0
        elif idx >= len(ticks):
            idx = len(ticks) - 1
        else:
            if abs(int(ticks[idx - 1]) - int(tick)) <= abs(int(ticks[idx]) - int(tick)):
                idx -= 1
        return e["embeddings"][idx].astype(np.float32)

    def event_embedding(
        self,
        demo_stem: str,
        round_num: int,
        event_tick: int,
        window_ticks: int = 128,
    ) -> np.ndarray | None:
        """Causal mean-pool of encoder embeddings over the window of
        `window_ticks` raw ticks ENDING AT event_tick. At the default 8 Hz
        downsample, 128 raw ticks = 16 downsampled positions = 2 seconds of
        buildup ending at the event moment.

        Strictly causal: never includes positions past event_tick. Critical
        for use with the v4+ causal encoder — otherwise event embeddings
        would peek at outcomes that haven't happened yet.

        Returns None if (demo, round) missing or no positions fall in window.
        """
        key = (str(demo_stem), int(round_num))
        e = self.entries.get(key)
        if e is None:
            return None
        ticks = e["ticks"]
        et = int(event_tick)
        # Positions in [et - window_ticks, et]
        end = int(np.searchsorted(ticks, et, side="right"))  # exclusive
        start = int(np.searchsorted(ticks, et - int(window_ticks), side="left"))
        if end <= start:
            # Event tick is before any cached position — fall back to nearest single tick
            return self.lookup(demo_stem, round_num, event_tick)
        slab = e["embeddings"][start:end].astype(np.float32)
        return slab.mean(axis=0)

    def lookup_batch(
        self,
        keys: list[tuple[str, int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch lookup. Returns (embeddings (N, d) float32, mask (N,) bool)
        where mask[i] is False for missing (demo, round) entries.
        For missing entries the embedding row is zeros."""
        n = len(keys)
        out = np.zeros((n, self.d_model), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, (demo, rn, tick) in enumerate(keys):
            emb = self.lookup(demo, rn, tick)
            if emb is not None:
                out[i] = emb
                mask[i] = True
        return out, mask


def load_default(repo_root: str | Path | None = None) -> EncoderEmbeddingCache:
    """Convenience loader for the canonical v6 cache."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    return EncoderEmbeddingCache(
        Path(repo_root) / "outputs" / "round_encoder" / "v6_81demos" / "embedding_cache.pt"
    )
