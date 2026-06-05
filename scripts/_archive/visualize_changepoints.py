#!/usr/bin/env python3
"""Visualize learned event boundaries from a v9 change-point encoder.

For each chosen val round, plots:
  1. b_prob over time (the model's per-tick boundary probability)
  2. Discrete boundaries chosen at eval (top-K via target density)
  3. Actual events (kills, plants, defuses, etc.) as vertical lines

If the learned boundaries align with the actual event ticks, the
segmentation mechanism is doing what we want. If they're uniform, the
density penalty just locks the rate without picking meaningful positions.

Output: outputs/round_encoder/<run>/changepoints.png

Usage:
    python scripts/visualize_changepoints.py
    python scripts/visualize_changepoints.py --ckpt outputs/round_encoder/<run>/best.pt
    python scripts/visualize_changepoints.py --n-rounds 6 --val-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_round_encoder import RoundEncoder, TrainConfig, ChangePointHead

DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v9_changepoint" / "best.pt"

EVENT_VOCAB = ["kill_t", "kill_ct", "bomb_planted", "bomb_defused",
                "bomb_exploded", "round_end", "none"]
NONE_IDX = 6
EVENT_COLORS = {
    "kill_t":        "#d62728",
    "kill_ct":       "#1f77b4",
    "bomb_planted":  "#ff7f0e",
    "bomb_defused":  "#17becf",
    "bomb_exploded": "#bcbd22",
    "round_end":     "#7f7f7f",
}


def load_components(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in TrainConfig.__dataclass_fields__})
    encoder = RoundEncoder(cfg).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    cp = ChangePointHead(
        cfg.d_model, max_segments=cfg.change_point_max_segments,
        init_target_density=cfg.change_point_target_density,
    ).to(device).eval()
    cp.load_state_dict(ckpt["change_point_head"])
    return encoder, cp, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--n-rounds", type=int, default=6,
                    help="how many val rounds to plot (default 6)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}")
    encoder, cp_head, cfg = load_components(args.ckpt, device)
    target_density = cfg.change_point_target_density
    print(f"  target density: {target_density}")
    print()

    print("Loading val.pt...")
    blob = torch.load(DATA_DIR / "val.pt", weights_only=False)
    tensors = blob["tensors"]
    metas = blob["metas"]
    ev_labels = blob["event_labels"]
    ev_times = blob["event_times"]
    downsample = int(blob.get("downsample", 8))

    # Pick variety: aim for rounds of different lengths + outcomes
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(tensors), size=min(args.n_rounds, len(tensors)), replace=False)
    print(f"Plotting {len(idxs)} rounds: {idxs.tolist()}")
    print()

    fig, axes = plt.subplots(len(idxs), 1, figsize=(13, 2.0 * len(idxs)),
                               sharex=False)
    if len(idxs) == 1:
        axes = [axes]

    with torch.no_grad():
        for plot_i, sample_i in enumerate(idxs):
            tensor = tensors[sample_i]
            meta = metas[sample_i]
            lbl = ev_labels[sample_i].numpy()
            tm = ev_times[sample_i].numpy()
            T = tensor.shape[0]
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
                cp_out = cp_head(h)
            b_prob = cp_out["b_prob"].squeeze(0).float().cpu().numpy()  # (T,)
            seg_id = cp_out["seg_id"].squeeze(0).cpu().numpy()

            # Identify discrete boundary positions: top-K where K matches target
            k = max(1, int(round(target_density * T)))
            boundary_ticks = np.argsort(-b_prob)[:k]
            boundary_ticks.sort()

            # Identify actual events: ticks where label != none and time<downsample
            event_ticks_by_type = {name: [] for name in EVENT_VOCAB[:-1]}
            for t_idx in range(T):
                if lbl[t_idx] != NONE_IDX and tm[t_idx] < downsample:
                    event_ticks_by_type[EVENT_VOCAB[lbl[t_idx]]].append(t_idx)

            ax = axes[plot_i]
            xs = np.arange(T)
            ax.plot(xs, b_prob, color="#444444", linewidth=1.0, alpha=0.85,
                     label=f"b_prob (mean={b_prob.mean():.3f})")
            ax.axhline(target_density, color="#888888", linestyle="--",
                       linewidth=0.7, alpha=0.6,
                       label=f"target density ({target_density})")
            # Discrete boundaries
            for bt in boundary_ticks:
                ax.axvline(bt, color="#2ca02c", alpha=0.5, linewidth=1.2)
            # Actual events
            for name, ticks in event_ticks_by_type.items():
                for t_e in ticks:
                    ax.axvline(t_e, color=EVENT_COLORS[name], alpha=0.95,
                                linewidth=1.6, linestyle=":")
            ax.set_ylim(0.0, max(0.5, b_prob.max() * 1.1))
            ax.set_ylabel(f"b_prob\nround {meta['round_num']}\n{meta.get('winner', '?')} won",
                            fontsize=8)
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.2)
            if plot_i == 0:
                # Legend with event colors
                from matplotlib.lines import Line2D
                legend_lines = [
                    Line2D([0], [0], color="#444444", linewidth=1.5, label="b_prob"),
                    Line2D([0], [0], color="#888888", linestyle="--", label="target"),
                    Line2D([0], [0], color="#2ca02c", alpha=0.7, label="picked boundary"),
                ] + [
                    Line2D([0], [0], color=c, linestyle=":", linewidth=2,
                            label=name.replace("_", " "))
                    for name, c in EVENT_COLORS.items()
                ]
                ax.legend(handles=legend_lines, loc="upper right", fontsize=7,
                            ncol=2)

    axes[-1].set_xlabel("Downsampled tick (8 Hz)")
    fig.suptitle(f"Learned change-points vs actual events — {args.ckpt.parent.name}",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out or args.ckpt.parent / "changepoints.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Quantitative diagnostic: for each round, what fraction of actual events
    # lie within ±W ticks of a chosen boundary? (Higher = boundaries align
    # with events.)
    print()
    print("Boundary-vs-event alignment (proximity diagnostic):")
    for window_w in (2, 4, 8):
        hits = 0
        total = 0
        for sample_i in idxs:
            T = tensors[sample_i].shape[0]
            x = tensors[sample_i].unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    h = encoder(x)
                    cp_out = cp_head(h)
            b_prob = cp_out["b_prob"].squeeze(0).float().cpu().numpy()
            k = max(1, int(round(target_density * T)))
            boundary_ticks = np.sort(np.argsort(-b_prob)[:k])
            lbl = ev_labels[sample_i].numpy()
            tm = ev_times[sample_i].numpy()
            for t_idx in range(T):
                if lbl[t_idx] != NONE_IDX and tm[t_idx] < downsample:
                    total += 1
                    # nearest boundary
                    dists = np.abs(boundary_ticks - t_idx)
                    if dists.min() <= window_w:
                        hits += 1
        if total > 0:
            print(f"  events within ±{window_w} ticks of a boundary: "
                  f"{hits}/{total} = {hits/total:.1%}")


if __name__ == "__main__":
    main()
