#!/usr/bin/env python3
"""σ_s diagnostic for the trained Level-2 round encoder.

The methodology gate (docs/methodology.md axis 1): for each query state, find
k nearest neighbors in embedding space; compute the std of `round_won` across
those neighbors. A useful representation lands in the **Goldilocks band
median ∈ [0.15, 0.45]**:
  - too low  → outcome collapse (encoder maps everything to outcome axis)
  - too high → random / no signal at all
  - in band  → neighbors share tactical structure but outcome remains uncertain

Crucially, our v4 encoder was trained C1-clean: it never saw round_won. So a
collapse here would NOT be F2 — it would be that the SSL forward-prediction
objectives accidentally aligned the embedding with outcome anyway. Random
would mean the embeddings carry no decision-relevant information at all.

Methodology:
  1. Reconstruct the encoder from a ckpt (best.pt or last.pt)
  2. Encode every round in train.pt and val.pt; sample query positions per round
  3. Build FAISS index over train query embeddings (cosine)
  4. For each val query: kNN with k=32 against train; compute σ_s = std of
     train neighbors' round_won (binary {0=ct, 1=t})
  5. Aggregate σ_s distribution; verdict against the band

Train/val split is demo-disjoint (set up by build_tick_sequences.py), so the
F1 same-round-leak failure mode can't apply here — train and val demos don't
share players or rounds.

Usage:
    python scripts/encoder_sigma_s_diagnostic.py
    python scripts/encoder_sigma_s_diagnostic.py --ckpt outputs/round_encoder/<run>/best.pt
    python scripts/encoder_sigma_s_diagnostic.py --queries-per-round 16 --k 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v4_default_50ep" / "best.pt"

# Pass thresholds from docs/methodology.md axis 1
GOLDILOCKS_LO = 0.15
GOLDILOCKS_HI = 0.45


def load_encoder(ckpt_path: Path, device: torch.device):
    """Reconstruct the encoder from a checkpoint."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from train_round_encoder import RoundEncoder, TrainConfig

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in TrainConfig.__dataclass_fields__})
    encoder = RoundEncoder(cfg).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, cfg, ckpt


def sample_query_positions(T: int, n: int) -> np.ndarray:
    """Pick `n` evenly-spaced positions from [0, T-1]. Skips the very first
    few ticks (still freeze time for many rounds — uninformative)."""
    skip = min(20, max(0, T // 10))
    if T - skip <= n:
        return np.arange(skip, T)
    return np.linspace(skip, T - 1, n, dtype=np.int64)


def encode_split(
    split_path: Path,
    encoder,
    device: torch.device,
    queries_per_round: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode every round, return (embeddings (N, d), labels (N,), demo_stems (N,))
    where N = total query positions across all rounds in the split."""
    blob = torch.load(split_path, weights_only=False)
    tensors: list[torch.Tensor] = blob["tensors"]
    metas: list[dict] = blob["metas"]

    embs = []
    labels = []
    stems = []

    with torch.no_grad():
        for tensor, meta in zip(tensors, metas):
            T = tensor.shape[0]
            if T < 2:
                continue
            x = tensor.unsqueeze(0).to(device)  # (1, T, F)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)  # (1, T, d_model)
            h = h.squeeze(0).float().cpu().numpy()  # (T, d_model)

            qpos = sample_query_positions(T, queries_per_round)
            label = 1 if meta.get("winner") == "t" else 0  # binary round_won
            for p in qpos:
                embs.append(h[p])
                labels.append(label)
                stems.append(meta.get("demo_stem", ""))

    return np.stack(embs), np.array(labels, dtype=np.int32), stems


def percentile(vals: np.ndarray, p: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, p))


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}")

    encoder, cfg, ckpt = load_encoder(args.ckpt, device)
    print(f"Encoder: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"feature_dim={cfg.feature_dim}")
    print(f"Trained for {ckpt['epoch'] + 1} epochs, val_total={ckpt['val_metrics'].get('val_total', '?')}")
    print()

    # Encode train
    print(f"Encoding train queries (n_queries/round = {args.queries_per_round})...")
    train_embs, train_labels, train_stems = encode_split(
        DATA_DIR / "train.pt", encoder, device, args.queries_per_round,
    )
    print(f"  train queries: {train_embs.shape[0]}, "
          f"label balance: t={int(train_labels.sum())} / "
          f"ct={int((1 - train_labels).sum())}")

    print("Encoding val queries...")
    val_embs, val_labels, val_stems = encode_split(
        DATA_DIR / "val.pt", encoder, device, args.queries_per_round,
    )
    print(f"  val queries: {val_embs.shape[0]}, "
          f"label balance: t={int(val_labels.sum())} / "
          f"ct={int((1 - val_labels).sum())}")
    print()

    # Confirm demo-disjoint split (no F1 risk)
    train_demos = set(train_stems)
    val_demos = set(val_stems)
    overlap = train_demos & val_demos
    print(f"Train demos: {len(train_demos)}, val demos: {len(val_demos)}, "
          f"overlap: {len(overlap)} (should be 0)")
    print()

    # Build FAISS index over train embeddings (cosine = inner product on
    # L2-normalized vectors)
    d = train_embs.shape[1]
    train_n = train_embs.astype(np.float32)
    faiss.normalize_L2(train_n)
    index = faiss.IndexFlatIP(d)  # exact, simple — train_n is small enough
    index.add(train_n)

    val_n = val_embs.astype(np.float32)
    faiss.normalize_L2(val_n)
    print(f"Searching kNN (k={args.k}, exact cosine)...")
    sims, idxs = index.search(val_n, args.k)  # (n_val, k)
    # Per-query σ_s: std of round_won across the k retrieved train neighbors
    neighbor_labels = train_labels[idxs]  # (n_val, k)
    sigma_s = neighbor_labels.std(axis=1)  # (n_val,)
    mean_sim = sims.mean(axis=1)
    print()

    # Aggregate
    median = float(np.median(sigma_s))
    p25 = percentile(sigma_s, 25)
    p75 = percentile(sigma_s, 75)
    in_band_frac = float(((sigma_s >= GOLDILOCKS_LO) & (sigma_s <= GOLDILOCKS_HI)).mean())
    below_frac = float((sigma_s < GOLDILOCKS_LO).mean())
    above_frac = float((sigma_s > GOLDILOCKS_HI).mean())
    zero_spread_frac = float((sigma_s < 0.05).mean())  # near-collapse

    # Theoretical max σ_s for binary labels at our base rate
    p = float(train_labels.mean())
    max_possible = float(np.sqrt(p * (1 - p)))

    # Verdict
    if median < GOLDILOCKS_LO:
        verdict = "FAIL: below band — outcome collapse (despite C1)"
    elif median > GOLDILOCKS_HI:
        verdict = "FAIL: above band — embeddings near-random / no clustering"
    else:
        verdict = "PASS"

    print("=" * 64)
    print(f"σ_s diagnostic — encoder {args.ckpt.name}")
    print("=" * 64)
    print(f"  Goldilocks band:       [{GOLDILOCKS_LO}, {GOLDILOCKS_HI}]")
    print(f"  Theoretical max σ_s:    {max_possible:.4f}  (binary p={p:.3f})")
    print()
    print(f"  median σ_s:             {median:.4f}")
    print(f"  p25 / p75:              {p25:.4f} / {p75:.4f}")
    print(f"  fraction in band:       {in_band_frac:.1%}")
    print(f"  fraction below band:    {below_frac:.1%}  (collapse risk)")
    print(f"  fraction above band:    {above_frac:.1%}  (random risk)")
    print(f"  near-zero (<0.05):      {zero_spread_frac:.1%}")
    print(f"  mean kNN cosine sim:    {mean_sim.mean():.4f}")
    print()
    print(f"  VERDICT: {verdict}")
    print()

    # Save full numbers
    out = args.ckpt.parent / f"sigma_s_diag_{args.ckpt.stem}.json"
    out.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "k": args.k,
        "queries_per_round": args.queries_per_round,
        "n_train_queries": int(train_embs.shape[0]),
        "n_val_queries": int(val_embs.shape[0]),
        "label_base_rate_t": p,
        "max_possible_sigma_s": max_possible,
        "median_sigma_s": median,
        "p25_sigma_s": p25,
        "p75_sigma_s": p75,
        "in_band_frac": in_band_frac,
        "below_band_frac": below_frac,
        "above_band_frac": above_frac,
        "near_zero_frac": zero_spread_frac,
        "mean_neighbor_cosine": float(mean_sim.mean()),
        "verdict": verdict,
        "goldilocks_band": [GOLDILOCKS_LO, GOLDILOCKS_HI],
    }, indent=2))
    print(f"  → wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--k", type=int, default=32, help="kNN neighbors")
    ap.add_argument("--queries-per-round", type=int, default=12,
                    help="positions to sample per round (evenly spaced)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
