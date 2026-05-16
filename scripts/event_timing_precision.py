#!/usr/bin/env python3
"""Event-timing precision: when the encoder predicts the next event, how
many ticks off is it?

We already measure type accuracy (val_acc_event_only ≈ 0.53 on v6 / 0.54
on v7 — well above 7-class random 0.143). But "type right" with wrong
timing tells the downstream policy the wrong thing: a kill predicted at
t+8 vs t+200 is a different tactical claim. This script measures the
timing error directly.

Pipeline:
  1. Re-encode all val rounds with the frozen v6 encoder
  2. Run time_to_event head per tick → predicted ticks-to-next-event
     (in log space, then un-normalized back to raw 64Hz ticks)
  3. Compare to ground-truth time_to_event labels from train.pt/val.pt
  4. Report median absolute error in seconds, per event type, on
     positions where the type prediction was ALSO correct

Output: outputs/round_encoder/<run>/event_timing_precision.json

Usage:
    python scripts/event_timing_precision.py
    python scripts/event_timing_precision.py --ckpt outputs/round_encoder/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from train_round_encoder import RoundEncoder, TrainConfig, NextEventHead, TimeToEventHead

DATA_DIR = REPO / "data" / "processed" / "tick_sequences"
DEFAULT_CKPT = REPO / "outputs" / "round_encoder" / "v6_81demos" / "best.pt"

# Match EVENT_VOCAB from build_tick_sequences.py
EVENT_VOCAB = ["kill_t", "kill_ct", "bomb_planted", "bomb_defused",
                "bomb_exploded", "round_end", "none"]
NONE_IDX = 6


def load_encoder_and_heads(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in TrainConfig.__dataclass_fields__})
    encoder = RoundEncoder(cfg).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])

    event_head = None
    time_head = None
    if ckpt.get("event_head") is not None:
        event_head = NextEventHead(cfg.d_model, cfg.n_event_classes).to(device).eval()
        event_head.load_state_dict(ckpt["event_head"])
    if ckpt.get("time_head") is not None:
        time_head = TimeToEventHead(cfg.d_model).to(device).eval()
        time_head.load_state_dict(ckpt["time_head"])
    return encoder, event_head, time_head, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ckpt: {args.ckpt}\n")

    encoder, event_head, time_head, cfg = load_encoder_and_heads(args.ckpt, device)
    if event_head is None or time_head is None:
        sys.exit("ckpt missing event_head/time_head — was this trained with event objectives?")
    horizon = float(cfg.event_horizon_ticks)
    log_horizon = math.log1p(horizon)
    print(f"Encoder: d_model={cfg.d_model}, event_horizon={horizon:.0f} raw ticks "
          f"= {horizon/64:.1f}s\n")

    blob = torch.load(DATA_DIR / "val.pt", weights_only=False)

    # Collect (true_event, pred_event, true_time, pred_time) for each non-pad
    # non-none position
    true_evs, pred_evs, true_times, pred_times = [], [], [], []
    with torch.no_grad():
        for tensor, ev_lbl, ev_t in zip(blob["tensors"], blob["event_labels"], blob["event_times"]):
            T = tensor.shape[0]
            if T < 2: continue
            x = tensor.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                h = encoder(x)
                logits = event_head(h)
                t_pred_log = time_head(h)  # normalized log space
            pred_type = logits.squeeze(0).argmax(dim=-1).cpu().numpy()
            true_type = ev_lbl.numpy()
            true_t = ev_t.numpy()
            # Convert pred log back to ticks: pred_log * log_horizon → log(1+ticks) → ticks
            pred_log_unnorm = t_pred_log.squeeze(0).float().cpu().numpy() * log_horizon
            pred_t = np.expm1(np.clip(pred_log_unnorm, 0, log_horizon + 1))  # raw ticks
            # Filter: keep non-pad, non-none positions only
            valid = (true_type != NONE_IDX) & (true_type >= 0)
            for i in np.where(valid)[0]:
                true_evs.append(int(true_type[i]))
                pred_evs.append(int(pred_type[i]))
                true_times.append(float(true_t[i]))
                pred_times.append(float(pred_t[i]))

    true_evs = np.array(true_evs)
    pred_evs = np.array(pred_evs)
    true_times = np.array(true_times)
    pred_times = np.array(pred_times)
    abs_err = np.abs(true_times - pred_times)
    type_correct = true_evs == pred_evs

    print(f"Non-'none' val positions: {len(true_evs)}")
    print(f"Type-correct fraction:    {type_correct.mean():.4f}")
    print(f"  (val_acc_event_only equivalent — confirms ckpt loaded right)")
    print()

    def stats(mask: np.ndarray, label: str):
        if not mask.any():
            return
        err = abs_err[mask]
        print(f"  {label:30s}  n={int(mask.sum()):>5d}  "
              f"median={np.median(err):>6.1f} ticks ({np.median(err)/64:.2f}s)  "
              f"p25={np.percentile(err,25):>5.1f}  p75={np.percentile(err,75):>5.1f}  "
              f"mean={err.mean():>5.1f}")

    print("=" * 76)
    print("Timing error |pred_ticks - true_ticks| on type-correct positions:")
    print("=" * 76)
    stats(type_correct, "ALL type-correct events")
    print()
    for i, name in enumerate(EVENT_VOCAB[:-1]):  # skip "none"
        mask = type_correct & (true_evs == i)
        stats(mask, name)
    print()
    print("Same, on type-WRONG positions (encoder predicted wrong event type):")
    stats(~type_correct, "ALL type-wrong events")
    print()

    out_path = args.ckpt.parent / "event_timing_precision.json"
    results = {
        "ckpt": str(args.ckpt),
        "horizon_ticks": horizon,
        "n_non_none_positions": int(len(true_evs)),
        "type_acc_event_only": float(type_correct.mean()),
        "overall": {
            "median_abs_err_ticks": float(np.median(abs_err[type_correct])),
            "median_abs_err_seconds": float(np.median(abs_err[type_correct]) / 64),
            "p25_ticks": float(np.percentile(abs_err[type_correct], 25)),
            "p75_ticks": float(np.percentile(abs_err[type_correct], 75)),
        },
        "per_event_type": {},
    }
    for i, name in enumerate(EVENT_VOCAB[:-1]):
        mask = type_correct & (true_evs == i)
        if mask.any():
            err = abs_err[mask]
            results["per_event_type"][name] = {
                "n": int(mask.sum()),
                "median_ticks": float(np.median(err)),
                "median_seconds": float(np.median(err) / 64),
                "p25_ticks": float(np.percentile(err, 25)),
                "p75_ticks": float(np.percentile(err, 75)),
            }
    out_path.write_text(json.dumps(results, indent=2))
    print(f"→ wrote {out_path}")


if __name__ == "__main__":
    main()
