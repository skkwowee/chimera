#!/usr/bin/env python3
"""Level-2 gate L2-G3: qualitative 2D-projection sanity check.

Per docs/alignment-delta.md §8: "2-D projection (UMAP/t-SNE/PCA) colored
by pro-action category vs. round_won should show category structure, not
outcome collapse. Subjective but required."

We don't have hand-labeled pro-action categories yet, so we use the
next-event-type label that build_tick_sequences.py emits as a proxy
(kill_t / kill_ct / bomb_planted / bomb_defused / bomb_exploded /
round_end / none). It's a finer-grained "what's about to happen" signal
than pro-action and is what the encoder explicitly trains on.

PASS = event-type panel shows distinct clusters (encoder learned the
task) AND round_won panel is diffuse, not split cleanly along an axis
(encoder did NOT collapse to outcome).

Writes outputs/round_encoder/<run>/clustering_<ckpt>.png.

Usage:
    python scripts/probe_clustering.py
    python scripts/probe_clustering.py --ckpt outputs/round_encoder/<run>/best.pt
    python scripts/probe_clustering.py --max-queries 4000   # for speed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v6_81demos" / "best.pt"

EVENT_COLORS = {
    "kill_t":        "#d62728",  # red — T kills
    "kill_ct":       "#1f77b4",  # blue — CT kills
    "bomb_planted":  "#ff7f0e",  # orange
    "bomb_defused":  "#17becf",  # cyan
    "bomb_exploded": "#bcbd22",  # olive
    "round_end":     "#7f7f7f",  # gray
    "none":          "#cccccc",  # light gray — fade into background
}
WIN_COLORS = {0: "#1f77b4", 1: "#d62728"}  # ct=blue, t=red


def load_encoder(ckpt_path: Path, device):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from train_round_encoder import RoundEncoder, TrainConfig
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in TrainConfig.__dataclass_fields__})
    encoder = RoundEncoder(cfg).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, cfg


def encode_with_event_labels(split_path: Path, encoder, device,
                              queries_per_round: int, max_queries: int):
    """For each round, sample queries_per_round positions and pull
    (embedding, round_won, event_label_at_position)."""
    blob = torch.load(split_path, weights_only=False)
    tensors = blob["tensors"]
    metas = blob["metas"]
    ev_labels_all = blob["event_labels"]
    vocab = blob["event_vocab"]

    embs, wins, evs = [], [], []
    with torch.no_grad():
        for tensor, meta, ev_lbl in zip(tensors, metas, ev_labels_all):
            T = tensor.shape[0]
            if T < 30:
                continue
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
            h = h.squeeze(0).float().cpu().numpy()
            skip = min(20, max(0, T // 10))
            if T - skip <= queries_per_round:
                qpos = np.arange(skip, T)
            else:
                qpos = np.linspace(skip, T - 1, queries_per_round,
                                    dtype=np.int64)
            win = 1 if meta.get("winner") == "t" else 0
            ev_arr = ev_lbl.numpy()
            for p in qpos:
                embs.append(h[p])
                wins.append(win)
                evs.append(int(ev_arr[p]))
    embs = np.stack(embs).astype(np.float32)
    wins = np.array(wins, dtype=np.int8)
    evs = np.array(evs, dtype=np.int8)
    if max_queries and embs.shape[0] > max_queries:
        rng = np.random.default_rng(0)
        idx = rng.choice(embs.shape[0], size=max_queries, replace=False)
        embs, wins, evs = embs[idx], wins[idx], evs[idx]
    return embs, wins, evs, vocab


def project_2d(x: np.ndarray, method: str):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(x)
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1,
                             metric="cosine", random_state=0)
        return reducer.fit_transform(x)
    raise ValueError(f"unknown method {method}")


def plot_2x2(p_pca, p_umap, wins, evs, vocab, out_png: Path,
              title_prefix: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, proj, name in [(axes[0, 0], p_pca, "PCA"),
                            (axes[1, 0], p_umap, "UMAP")]:
        # Plot "none" first (background), then events on top so they pop
        none_idx = vocab.index("none")
        bg = evs == none_idx
        fg = ~bg
        ax.scatter(proj[bg, 0], proj[bg, 1], s=2, alpha=0.15,
                    c=EVENT_COLORS["none"], label="none")
        for i, name_e in enumerate(vocab):
            if name_e == "none":
                continue
            sel = (evs == i) & fg
            if sel.any():
                ax.scatter(proj[sel, 0], proj[sel, 1], s=6, alpha=0.7,
                            c=EVENT_COLORS[name_e], label=f"{name_e} ({int(sel.sum())})")
        ax.set_title(f"{name} — colored by next-event-type")
        ax.legend(fontsize=8, loc="best", markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    for ax, proj, name in [(axes[0, 1], p_pca, "PCA"),
                            (axes[1, 1], p_umap, "UMAP")]:
        for w in [0, 1]:
            sel = wins == w
            label = "ct win" if w == 0 else "t win"
            ax.scatter(proj[sel, 0], proj[sel, 1], s=3, alpha=0.5,
                        c=WIN_COLORS[w], label=f"{label} ({int(sel.sum())})")
        ax.set_title(f"{name} — colored by round_won")
        ax.legend(fontsize=10, loc="best", markerscale=3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {out_png}")


def quantify(p: np.ndarray, labels: np.ndarray, vocab: list[str] | None = None):
    """Return mean within-class distance / mean across-class distance.
    Smaller ratio = tighter clusters relative to global spread."""
    overall_mean = p.mean(axis=0)
    global_spread = float(np.linalg.norm(p - overall_mean, axis=1).mean())
    if global_spread == 0:
        return float("nan")
    classes = np.unique(labels)
    within = []
    for c in classes:
        sel = labels == c
        if sel.sum() < 3:
            continue
        cmean = p[sel].mean(axis=0)
        within.append(float(np.linalg.norm(p[sel] - cmean, axis=1).mean()))
    return float(np.mean(within) / global_spread) if within else float("nan")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}")

    encoder, cfg = load_encoder(args.ckpt, device)
    print(f"Encoder: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"feature_dim={cfg.feature_dim}")
    print()

    print(f"Encoding train queries (n_queries/round={args.queries_per_round}, "
          f"cap {args.max_queries})...")
    embs, wins, evs, vocab = encode_with_event_labels(
        DATA_DIR / "train.pt", encoder, device,
        args.queries_per_round, args.max_queries,
    )
    print(f"  {embs.shape[0]} queries, d={embs.shape[1]}")
    print(f"  event distribution: " + ", ".join(
        f"{v}={int((evs == i).sum())}" for i, v in enumerate(vocab)))
    print(f"  win balance: t={int(wins.sum())} / ct={int((1 - wins).sum())}")
    print()

    print("Computing PCA...")
    p_pca = project_2d(embs, "pca")
    print("Computing UMAP (cosine, n_neighbors=30)...")
    p_umap = project_2d(embs, "umap")
    print()

    # Quantify: tightness of event-type clusters vs round_won clusters
    # within each projection
    pca_event_ratio = quantify(p_pca, evs)
    pca_win_ratio = quantify(p_pca, wins)
    umap_event_ratio = quantify(p_umap, evs)
    umap_win_ratio = quantify(p_umap, wins)
    print(f"PCA  within/global  event={pca_event_ratio:.3f}  win={pca_win_ratio:.3f}")
    print(f"UMAP within/global  event={umap_event_ratio:.3f}  win={umap_win_ratio:.3f}")
    # PASS heuristic: event-cluster tightness < win-cluster tightness in UMAP
    # AND ratios < 1.0 (clusters tighter than global)
    pass_pca = pca_event_ratio < pca_win_ratio and pca_event_ratio < 1.0
    pass_umap = umap_event_ratio < umap_win_ratio and umap_event_ratio < 1.0
    verdict = ("PASS" if pass_umap else
                "MAYBE" if pass_pca else
                "FAIL")
    print(f"\nVerdict: {verdict}  (event tighter than win in PCA={pass_pca}, UMAP={pass_umap})")
    print()

    out_dir = args.ckpt.parent
    title = f"L2-G3 clustering — {args.ckpt.name}"
    plot_2x2(p_pca, p_umap, wins, evs, vocab,
              out_dir / f"clustering_{args.ckpt.stem}.png", title)

    (out_dir / f"clustering_{args.ckpt.stem}.json").write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "n_queries": int(embs.shape[0]),
        "pca_event_within_global_ratio": pca_event_ratio,
        "pca_win_within_global_ratio": pca_win_ratio,
        "umap_event_within_global_ratio": umap_event_ratio,
        "umap_win_within_global_ratio": umap_win_ratio,
        "verdict": verdict,
        "interpretation": (
            "PASS = event-type clusters tighter than round_won clusters "
            "in UMAP — encoder learned tactical structure, did not collapse "
            "to outcome axis."
        ),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--queries-per-round", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=8000,
                    help="cap total queries for projection speed (default 8000)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
