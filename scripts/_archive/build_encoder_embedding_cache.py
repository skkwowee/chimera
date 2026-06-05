#!/usr/bin/env python3
"""Precompute Level-2 encoder embeddings for every (demo, round, tick) in
the tick-sequence dataset.

Per docs/round-encoder-design.md §8, the integration point is RECALLIndex
in src/training/recall.py — a `(demo_stem, round_num, tick)` lookup that
returns a fixed-dim event embedding. This script builds that lookup table
once so GRPO training doesn't pay encoder forward-pass cost per row.

Output:
    outputs/round_encoder/<run>/embedding_cache.pt
      {
        "encoder_ckpt": str,
        "d_model": int,
        "downsample": int,
        "entries": {
          (demo_stem, round_num): {
            "ticks": int64 array (T,),       # raw 64Hz ticks
            "embeddings": float16 array (T, d),
          },
          ...
        },
      }

Storage estimate: ~1700 rounds × ~1000 ticks × 512 dim × 2 bytes =
~1.7 GB at fp16. Acceptable for one-shot offline.

Usage:
    python scripts/build_encoder_embedding_cache.py
    python scripts/build_encoder_embedding_cache.py --ckpt outputs/round_encoder/<run>/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v6_81demos" / "best.pt"


def load_encoder(ckpt_path: Path, device):
    sys.path.insert(0, str(REPO / "scripts"))
    from train_round_encoder import RoundEncoder, TrainConfig
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in TrainConfig.__dataclass_fields__})
    enc = RoundEncoder(cfg).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    return enc, cfg, ckpt


def encode_blob_chunked(blob, encoder, device, downsample, chunk_size,
                          out_path_prefix: Path, split_name: str,
                          encoder_ckpt: str, d_model: int) -> list[Path]:
    """Encode every round, saving every `chunk_size` rounds to its own file.

    Why chunking: at d_model=1024, the input blob is ~3GB AND the accumulating
    output dict grows to ~3GB. With only 16GB total RAM we can't hold both
    plus torch.save's serialization buffer. Saving in chunks keeps the live
    output dict small.

    Each chunk file is a self-contained payload that EncoderEmbeddingCache
    knows how to merge at load time.
    """
    import gc
    tensors = blob["tensors"]
    metas = blob["metas"]
    n = len(tensors)
    out: dict[tuple[str, int], dict] = {}
    chunk_paths: list[Path] = []
    chunk_i = 0

    def flush_chunk():
        nonlocal out, chunk_i
        if not out:
            return
        path = out_path_prefix.parent / (
            f"embedding_cache_{split_name}_chunk{chunk_i:03d}.pt")
        torch.save({
            "encoder_ckpt": encoder_ckpt,
            "d_model": d_model,
            "downsample": downsample,
            "entries": out,
        }, path)
        sz = path.stat().st_size / 1e6
        print(f"    flushed chunk {chunk_i}: {len(out)} rounds, {sz:.0f} MB", flush=True)
        chunk_paths.append(path)
        out = {}
        chunk_i += 1
        gc.collect()

    with torch.no_grad():
        for i in range(n):
            tensor = tensors[i]
            meta = metas[i]
            T = tensor.shape[0]
            if T < 2:
                tensors[i] = None
                continue
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
            h = h.squeeze(0).float().cpu().numpy().astype(np.float16)
            first_raw = int(meta["first_tick"])
            raw_ticks = first_raw + np.arange(T, dtype=np.int64) * downsample
            key = (meta["demo_stem"], int(meta["round_num"]))
            out[key] = {"ticks": raw_ticks, "embeddings": h}
            tensors[i] = None
            del tensor, x
            if len(out) >= chunk_size:
                flush_chunk()
    flush_chunk()
    return chunk_paths


def run(args):
    """Write per-split cache files to avoid peaking on both blobs in RAM
    simultaneously. EncoderEmbeddingCache reads either single-file or
    split-file layout transparently."""
    import gc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}")

    encoder, cfg, ckpt = load_encoder(args.ckpt, device)
    print(f"Encoder: d_model={cfg.d_model}, n_layers={cfg.n_layers}")
    print()

    out_root = args.ckpt.parent
    written_files: list[Path] = []
    last_downsample = 8

    for split_name in ("train", "val"):
        path = DATA_DIR / f"{split_name}.pt"
        if not path.exists():
            print(f"  WARN: {path} missing, skipping")
            continue
        print(f"Loading {split_name}.pt...")
        blob = torch.load(path, weights_only=False)
        downsample = int(blob.get("downsample", 8))
        last_downsample = downsample
        t0 = time.time()
        chunk_paths = encode_blob_chunked(
            blob, encoder, device, downsample,
            chunk_size=args.chunk_size,
            out_path_prefix=out_root / f"embedding_cache_{split_name}",
            split_name=split_name,
            encoder_ckpt=str(args.ckpt),
            d_model=cfg.d_model,
        )
        dt = time.time() - t0
        print(f"  Encoded → {len(chunk_paths)} chunk file(s) in {dt:.1f}s")
        del blob
        gc.collect()
        written_files.extend(chunk_paths)

    if not written_files:
        print("ERROR: no entries encoded — bail.")
        return

    print()
    print(f"Done. {len(written_files)} chunk file(s) under {out_root}/")
    print(f"  → EncoderEmbeddingCache(path={out_root}) auto-loads all chunks.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--chunk-size", type=int, default=300,
                    help="rounds per chunk file (default 300 ≈ 600MB per chunk "
                         "at d=1024)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
