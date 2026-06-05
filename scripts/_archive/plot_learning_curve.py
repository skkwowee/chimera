#!/usr/bin/env python3
"""Render the v6 encoder learning-curve as a publication figure.

Reads outputs/round_encoder/learning_curve/curve_epochs15.json and writes
both PNG (for previews) and PDF (for the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO / "outputs" / "round_encoder" / "learning_curve" / "curve_epochs15.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out-stem", type=Path,
                    default=REPO / "outputs" / "round_encoder" / "learning_curve" / "learning_curve")
    ap.add_argument("--gate", type=float, default=0.65,
                    help="L2-G2 probe-acc gate line (default 0.65)")
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    runs = data["results"]
    n = np.array([r["n_demos"] for r in runs])
    rounds = np.array([r["n_train_rounds"] for r in runs])
    probe = np.array([r["probe_acc"] for r in runs])
    event = np.array([r["best_val_acc_event_only"] for r in runs])
    val_demos = data["val_demos"]
    epochs = data["epochs_per_run"]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Probe accuracy — primary axis
    ax1.plot(n, probe, "o-", color="#1f4e79", linewidth=2.0, markersize=8,
              label="probe_outcome val_acc (L2-G2)")
    ax1.axhline(args.gate, color="#1f4e79", linestyle="--", alpha=0.45,
                 linewidth=1, label=f"L2-G2 gate (≥ {args.gate})")
    ax1.set_xlabel("Training demos (val split fixed: 12 demos)")
    ax1.set_ylabel("probe_outcome val_acc", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_ylim(0.5, 0.85)
    ax1.grid(True, alpha=0.25)

    # Event-acc — secondary axis
    ax2 = ax1.twinx()
    ax2.plot(n, event, "s--", color="#c44e52", linewidth=1.5, markersize=6,
              alpha=0.85, label="val_acc_event_only")
    ax2.axhline(1.0 / 7, color="#c44e52", linestyle=":", alpha=0.4,
                 linewidth=1, label="random (7-class)")
    ax2.set_ylabel("val_acc_event_only (non-none classes)",
                    color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.set_ylim(0.0, 0.75)

    # Annotate
    for x, y, r in zip(n, probe, rounds):
        ax1.annotate(f"  {r}r", (x, y), fontsize=8,
                      color="#555555", verticalalignment="bottom")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9, framealpha=0.95)

    title = (f"Level-2 encoder learning curve "
              f"(v6 config, {epochs}ep, val={val_demos} demos)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = args.out_stem.with_suffix(".png")
    pdf_path = args.out_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
