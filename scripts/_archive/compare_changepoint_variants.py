#!/usr/bin/env python3
"""Compare v9 (forecast) baseline vs v10 variants (contrastive/variance/outcome)
on the boundary-event alignment diagnostic.

For each checkpoint:
  1. Load encoder + ChangePointHead
  2. Run inference on val rounds → b_prob per tick
  3. Pick top-K boundaries (K matches target density)
  4. Measure: of N actual events in val, what fraction are within ±W ticks
     of a picked boundary, for W in {2, 4, 8}
  5. Report alongside random baseline = (2W+1)*K/T  (probability a random
     window covers a uniformly-placed boundary)

Output: table to stdout, also written to outputs/round_encoder/v10_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_round_encoder import RoundEncoder, TrainConfig, ChangePointHead

DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
EVENT_VOCAB = ["kill_t", "kill_ct", "bomb_planted", "bomb_defused",
                "bomb_exploded", "round_end", "none"]
NONE_IDX = 6

DEFAULT_RUNS = [
    ("v9_forecast",    REPO / "outputs" / "round_encoder" / "v9_changepoint" / "best.pt"),
    ("v10a_contrastive", REPO / "outputs" / "round_encoder" / "v10a_contrastive" / "best.pt"),
    ("v10b_variance",  REPO / "outputs" / "round_encoder" / "v10b_variance" / "best.pt"),
    ("v10c_outcome",   REPO / "outputs" / "round_encoder" / "v10c_outcome" / "best.pt"),
]


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


def evaluate_alignment(encoder, cp_head, target_density, blob, device, windows=(2, 4, 8)):
    """Returns dict with per-W alignment %, random baseline %, and aggregate stats."""
    tensors = blob["tensors"]
    metas = blob["metas"]
    ev_labels = blob["event_labels"]
    ev_times = blob["event_times"]
    downsample = int(blob.get("downsample", 8))

    hits = {w: 0 for w in windows}
    rand_hits = {w: 0.0 for w in windows}
    total = 0
    n_boundaries = 0
    n_ticks = 0
    seg_count_sum = 0.0
    seg_count_n = 0

    with torch.no_grad():
        for sample_i in range(len(tensors)):
            tensor = tensors[sample_i]
            T = tensor.shape[0]
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
                cp_out = cp_head(h)
            b_prob = cp_out["b_prob"].squeeze(0).float().cpu().numpy()
            k = max(1, int(round(target_density * T)))
            boundary_ticks = np.sort(np.argsort(-b_prob)[:k])
            n_boundaries += len(boundary_ticks)
            n_ticks += T
            seg_count_sum += float((cp_out["seg_mask"].squeeze(0).float().sum().item()))
            seg_count_n += 1

            lbl = ev_labels[sample_i].numpy()
            tm = ev_times[sample_i].numpy()

            # Event positions
            event_ticks = []
            for t_idx in range(T):
                if lbl[t_idx] != NONE_IDX and tm[t_idx] < downsample:
                    event_ticks.append(t_idx)
            total += len(event_ticks)

            for w in windows:
                # Real alignment
                for t_e in event_ticks:
                    if len(boundary_ticks) > 0:
                        if np.min(np.abs(boundary_ticks - t_e)) <= w:
                            hits[w] += 1
                # Random baseline: probability that any of k uniformly-placed
                # boundaries falls within ±w of a fixed event
                #   = 1 - (1 - (2w+1)/T)^k     (with replacement, fine approx)
                k_eff = len(boundary_ticks)
                T_eff = T
                p_single = min(1.0, (2 * w + 1) / T_eff)
                p_hit = 1.0 - (1.0 - p_single) ** k_eff
                rand_hits[w] += p_hit * len(event_ticks)

    out = {"n_events": total, "n_rounds": len(tensors),
           "n_boundaries_total": n_boundaries,
           "n_ticks_total": n_ticks,
           "avg_boundaries_per_round": n_boundaries / max(1, len(tensors)),
           "avg_segments_per_round": seg_count_sum / max(1, seg_count_n)}
    for w in windows:
        out[f"align_w{w}"] = hits[w] / max(1, total)
        out[f"random_w{w}"] = rand_hits[w] / max(1, total)
        out[f"lift_w{w}"] = out[f"align_w{w}"] - out[f"random_w{w}"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", default=None,
                    help="optional list of name=ckpt_path; defaults to v9 + v10a/b/c")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs" / "round_encoder" / "v10_compare.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    runs = DEFAULT_RUNS
    if args.runs:
        runs = []
        for spec in args.runs:
            name, ckpt = spec.split("=", 1)
            runs.append((name, Path(ckpt)))

    print("Loading val.pt...")
    blob = torch.load(DATA_DIR / "val.pt", weights_only=False)
    print(f"  {len(blob['tensors'])} val rounds\n")

    all_results = {}
    print("=" * 96)
    print(f"{'run':22s}  {'target_d':>9s}  {'avg_segs':>8s}  "
          f"{'W=2':>14s}  {'W=4':>14s}  {'W=8':>14s}")
    print(f"{'':22s}  {'':>9s}  {'':>8s}  "
          f"{'real/rand→lift':>14s}  {'real/rand→lift':>14s}  {'real/rand→lift':>14s}")
    print("=" * 96)
    for name, ckpt in runs:
        if not ckpt.exists():
            print(f"  {name:22s}  (skipped — ckpt not found: {ckpt})")
            continue
        encoder, cp_head, cfg = load_components(ckpt, device)
        td = cfg.change_point_target_density
        res = evaluate_alignment(encoder, cp_head, td, blob, device)
        res["loss_variant"] = getattr(cfg, "change_point_loss", "forecast")
        res["aux_align_weight"] = getattr(cfg, "change_point_aux_align_weight", 0.0)
        all_results[name] = res
        def fmt(w):
            return (f"{res[f'align_w{w}']:.1%}/{res[f'random_w{w}']:.1%}"
                    f"→{res[f'lift_w{w}']:+.1%}")
        print(f"  {name:22s}  {td:>9.3f}  {res['avg_segments_per_round']:>8.1f}  "
              f"{fmt(2):>14s}  {fmt(4):>14s}  {fmt(8):>14s}")
    print()
    print(f"n_events_total per run: {res['n_events']}")
    print(f"Interpretation: lift > 0 = boundaries align with events better than random.")
    print(f"  v9 was anti-correlated (lift < 0). v10 variants test alignment fixes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
