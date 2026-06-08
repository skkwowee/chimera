#!/usr/bin/env python3
"""Closed-loop ROLLOUT test for the world model — the real "can it simulate?" test.

Single-step eval (eval_world_model.py) feeds REAL history and predicts one frame.
This instead does autoregressive rollout: predict t+k, append the prediction,
slide the window, predict t+2k from the (now partly-predicted) buffer, etc. Error
COMPOUNDS — this is what matters for using the model as a simulator (planning,
value rollouts). We track POSITION error per rollout step vs the ground-truth
future, against a const-velocity rollout (extrapolate the last real velocity).

Caveat: the model has a uniform head, so rolling the full 597-d frame back feeds
its (imperfect) predictions for static/categorical dims too — that drift can
corrupt the input and inflate later-step error. So this is a CONSERVATIVE read of
position-rollout quality; a categorical-aware head would only help.

Usage: python scripts/rollout_eval.py --ckpt outputs/world_model/h8/best.pt --steps 5
"""
from __future__ import annotations

import argparse, shutil, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS, PER_PLAYER_DIM  # noqa

POS_IDX = [p * PER_PLAYER_DIM + d for p in range(N_PLAYERS) for d in (0, 1, 2)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/world_model/h8/best.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val.pt")
    ap.add_argument("--steps", type=int, default=5, help="rollout steps (each = horizon k)")
    ap.add_argument("--n", type=int, default=400, help="rollout samples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(args.ckpt, safe)
        ck = torch.load(safe, map_location="cpu", weights_only=False)
    a = ck["args"]; L = a["window"]; k = a["horizon"]
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"], a["heads"])
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    R = args.steps
    need = L + R * k + 1
    print(f"ckpt step {ck.get('step')}  window={L}  horizon k={k} ({k*125}ms/step)  "
          f"rollout {R} steps = {R*k*125}ms")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    rounds = [t for t in blob["tensors"] if t.shape[0] >= need]

    pos_idx = torch.tensor(POS_IDX)
    m_err = torch.zeros(R); cv_err = torch.zeros(R); cnt = 0
    # deterministic spread of start points across rounds
    for ri, r in enumerate(rounds):
        if cnt >= args.n:
            break
        start = (ri * 37) % (r.shape[0] - need) + 1
        real = r[start:start + L + R * k].to(args.device)   # [L+R*k, F]
        buf = real[:L].clone().unsqueeze(0)                  # [1, L, F]
        last = real[L - 1]                                   # last real frame
        vel = real[L - 1] - real[L - 2]                      # last real per-tick velocity
        for step in range(1, R + 1):
            res = model(buf)[:, -1, :]                        # predicted residual for newest frame
            pred = buf[0, -1] + res[0]                        # predicted frame at +k
            true = real[L - 1 + step * k]
            cv = last + (step * k) * vel                      # const-velocity extrapolation
            m_err[step - 1] += F.l1_loss(pred[pos_idx], true[pos_idx]).item()
            cv_err[step - 1] += F.l1_loss(cv[pos_idx], true[pos_idx]).item()
            buf = torch.cat([buf[:, 1:, :], pred.view(1, 1, -1)], dim=1)  # slide window
        cnt += 1

    m_err /= cnt; cv_err /= cnt
    print(f"\nposition L1 error over {cnt} rollouts (normalized coords; lower=better):\n")
    print(f"{'step':>4s} {'t(ms)':>6s} {'model':>9s} {'const-vel':>10s} {'model better by':>15s}")
    for s in range(R):
        better = (cv_err[s] - m_err[s]) / cv_err[s] * 100 if cv_err[s] > 0 else 0
        print(f"{s+1:>4d} {(s+1)*k*125:>6d} {m_err[s]:9.5f} {cv_err[s]:10.5f} {better:14.1f}%")
    print("\nWhat to look for: model should stay BELOW const-velocity at every step. The "
          "gap shrinking (or flipping) as steps grow = compounding error / where the "
          "simulator stops being trustworthy for planning.")


if __name__ == "__main__":
    main()
