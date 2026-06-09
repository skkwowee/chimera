#!/usr/bin/env python3
"""Closed-loop ROLLOUT test for the world model — the real "can it simulate?" test.

Single-step eval (eval_world_model.py) feeds REAL history and predicts one frame.
This instead does autoregressive rollout: predict t+k, append the prediction,
slide the window, predict t+2k from the (now partly-predicted) buffer, etc. Error
COMPOUNDS — this is what matters for using the model as a simulator (planning,
value rollouts). We track POSITION error per rollout step vs the ground-truth
future, against a const-velocity rollout (extrapolate the last real velocity).

For --cv-residual checkpoints the head predicts a CORRECTION over a const-velocity
prior, so the rolled frame is cur + head_residual + k*(cur - prev) — omitting the
cv base term would silently turn the eval into garbage (the model would look like
a regression).

VALUE-THROUGH-ROLLOUT (--value-auc): the GRPO go/no-go evidence. For each anchor
we compute the value-head logit on the real window (depth 0), then keep computing
it as generated frames replace real ones (depth d = d rollout steps = d*k*125 ms
into the imagined future), and report rank-AUC of logit vs round winner PER DEPTH.
AUC holding near the depth-0 value at 4-8 s of imagined future means world-model
rollouts carry value signal — "generate the group" GRPO is viable. AUC decaying
to ~0.5 means rollouts lose value info — shrink horizons.

Caveats (both modes are CONSERVATIVE reads):
  - The model has a uniform head, so rolling the full frame back feeds its
    (imperfect) predictions for static/categorical dims too — that drift can
    corrupt the input and inflate later-step error.
  - The derived perception dims (v3: last 9 of each player block) are INPUT-ONLY,
    never predicted; as generated frames are appended they drift/go stale, so the
    value head reads degraded perception features at depth > 0. A simulator that
    re-derived them from predicted state would only help.

Usage:
  python scripts/rollout_eval.py --ckpt outputs/wm_3map/h8_mt/best_ns.pt --steps 5
  python scripts/rollout_eval.py --ckpt outputs/wm_3map/h8_mt/best.pt --value-auc
"""
from __future__ import annotations

import argparse, shutil, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_world_model import build_model, N_PLAYERS, auc  # noqa

XY_SCALE = 3000.0   # normalized coord -> game units (see featurization)


def load_ckpt(path):
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"; shutil.copy(path, safe)
        return torch.load(safe, map_location="cpu", weights_only=False)


def roll_step(model, buf, k, cv_residual, gap=1, want_value=False):
    """One AR step: predict the next frame at +k from buf and slide the window.

    Returns (new_buf, pred_frame, value_logit_on_new_buf or None). For cv_residual
    checkpoints the head is a correction over the const-velocity prior, so
    pred = cur + head_residual + k * per_tick_velocity. On the first step the last
    two buffer frames are adjacent real frames (gap=1 tick), so that's
    k*(cur - prev) exactly as in training; once a prediction is appended the last
    two frames are k ticks apart (gap=k), so the per-tick velocity is
    (cur - prev)/gap — without the /gap the prior extrapolates k× too far and the
    rollout silently explodes.
    """
    res = model.gen_residual(buf)[:, -1, :]   # dist decode when present, forward() otherwise
    pred = buf[0, -1] + res[0]
    if cv_residual:
        pred = pred + (k / gap) * (buf[0, -1] - buf[0, -2])
    buf = torch.cat([buf[:, 1:, :], pred.view(1, 1, -1)], dim=1)
    if want_value:
        return buf, pred, model.heads(buf)["value"][:, -1]
    return buf, pred, None


@torch.no_grad()
def drift_eval(model, rounds, L, k, cv_residual, args, pos_idx):
    """Position-drift table: model rollout vs const-velocity, per rollout step."""
    R = args.steps
    need = L + R * k + 1
    keep = [t for t in rounds if t.shape[0] >= need]
    m_err = torch.zeros(R); cv_err = torch.zeros(R)
    m_xy = torch.zeros(R); cv_xy = torch.zeros(R)
    cnt = 0
    # deterministic spread of start points across rounds
    for ri, r in enumerate(keep):
        if cnt >= args.n:
            break
        start = (ri * 37) % (r.shape[0] - need) + 1
        real = r[start:start + L + R * k].to(args.device)   # [L+R*k, F]
        buf = real[:L].clone().unsqueeze(0)                  # [1, L, F]
        last = real[L - 1]                                   # last real frame
        vel = real[L - 1] - real[L - 2]                      # last real per-tick velocity
        for step in range(1, R + 1):
            buf, pred, _ = roll_step(model, buf, k, cv_residual, gap=1 if step == 1 else k)
            true = real[L - 1 + step * k]
            cv = last + (step * k) * vel                      # const-velocity extrapolation
            m_err[step - 1] += F.l1_loss(pred[pos_idx], true[pos_idx]).item()
            cv_err[step - 1] += F.l1_loss(cv[pos_idx], true[pos_idx]).item()
            # mean xy euclidean error in GAME UNITS, averaged over ALL 10 players
            # (alive or not — simple; dead players are static so this under-states
            # the per-alive-player number a bit)
            tp = true[pos_idx].view(N_PLAYERS, 3)[:, :2]
            m_xy[step - 1] += ((pred[pos_idx].view(N_PLAYERS, 3)[:, :2] - tp)
                               * XY_SCALE).norm(dim=1).mean().item()
            cv_xy[step - 1] += ((cv[pos_idx].view(N_PLAYERS, 3)[:, :2] - tp)
                                * XY_SCALE).norm(dim=1).mean().item()
        cnt += 1

    m_err /= cnt; cv_err /= cnt; m_xy /= cnt; cv_xy /= cnt
    print(f"\nposition L1 error over {cnt} rollouts (normalized coords; lower=better);"
          f"\nxy(u) = mean per-player xy euclidean error in game units (all 10 players):\n")
    print(f"{'step':>4s} {'t(ms)':>6s} {'model':>9s} {'const-vel':>10s} "
          f"{'m_xy(u)':>8s} {'cv_xy(u)':>9s} {'model better by':>15s}")
    for s in range(R):
        better = (cv_err[s] - m_err[s]) / cv_err[s] * 100 if cv_err[s] > 0 else 0
        print(f"{s+1:>4d} {(s+1)*k*125:>6d} {m_err[s]:9.5f} {cv_err[s]:10.5f} "
              f"{m_xy[s]:8.1f} {cv_xy[s]:9.1f} {better:14.1f}%")
    print("\nWhat to look for: model should stay BELOW const-velocity at every step. The "
          "gap shrinking (or flipping) as steps grow = compounding error / where the "
          "simulator stops being trustworthy for planning.")


@torch.no_grad()
def value_auc_eval(model, blob, L, k, cv_residual, args):
    """Value-through-rollout: rank-AUC of the value-head logit vs round winner, at
    depth 0 (real window) and after each AR rollout step (imagined future)."""
    assert hasattr(model, "heads"), "--value-auc needs the multi-task (player) arch"
    depths = sorted(int(d) for d in args.depths.split(","))
    max_d = depths[-1]
    need = L + max_d * k + 1
    # keep tensors and metas ZIPPED when filtering by length — labels come from metas
    keep = [(t, m) for t, m in zip(blob["tensors"], blob["metas"]) if t.shape[0] >= need]
    print(f"value-through-rollout: depths {depths}  max imagined future "
          f"{max_d*k*125}ms  rounds with len>= {need}: {len(keep)}")

    logits = {d: [] for d in depths}
    labels = []
    cnt, pass_ = 0, 0
    # deterministic spread; multiple passes (offset start) if rounds < --n anchors
    while cnt < args.n and pass_ < 8:
        for ri, (r, m) in enumerate(keep):
            if cnt >= args.n:
                break
            start = (ri * 37 + pass_ * 61) % (r.shape[0] - need) + 1
            real = r[start:start + L + max_d * k].to(args.device)
            buf = real[:L].clone().unsqueeze(0)              # real window ending at anchor
            v = model.heads(buf)["value"][:, -1]             # depth 0: real frames only
            if 0 in logits:
                logits[0].append(v.float().cpu())
            for d in range(1, max_d + 1):
                buf, _, v = roll_step(model, buf, k, cv_residual,
                                      gap=1 if d == 1 else k, want_value=True)
                if d in logits:
                    logits[d].append(v.float().cpu())
            labels.append(1.0 if m.get("winner") == "ct" else 0.0)
            cnt += 1
        pass_ += 1

    lab = torch.tensor(labels)
    print(f"\nvalue AUC vs rollout depth ({cnt} anchors; depth 0 = real window, "
          f"depth d = d generated frames in the buffer):\n")
    print(f"{'depth':>5s} {'t(ms)':>6s} {'n':>5s} {'AUC':>7s}")
    for d in depths:
        a = auc(torch.cat(logits[d]), lab)
        print(f"{d:>5d} {d*k*125:>6d} {cnt:>5d} {a:7.3f}")
    print("\nAUC holding near depth-0 value at 4-8s -> world-model rollouts carry value "
          "signal -> 'generate the group' GRPO plan viable. AUC decaying to ~0.5 -> "
          "rollouts lose value info -> shrink horizons.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/world_model/h8/best.pt")
    ap.add_argument("--val-pt", default="data/processed/tick_sequences/val_v3.pt")
    ap.add_argument("--steps", type=int, default=5, help="rollout steps (each = horizon k)")
    ap.add_argument("--n", type=int, default=400, help="rollout samples / anchors")
    ap.add_argument("--value-auc", action="store_true",
                    help="value-through-rollout mode: AUC of value logit vs winner per depth")
    ap.add_argument("--depths", default="0,1,2,4,8",
                    help="comma-sep rollout depths for --value-auc (0 = real window)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.value_auc and args.n == 400:
        args.n = 300

    ck = load_ckpt(args.ckpt)
    a = ck["args"]; L = a["window"]; k = a["horizon"]
    ppd = ck["per_player_dim"]
    cv_residual = a.get("cv_residual", False)
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"],
                        a["heads"], per_player_dim=ppd, dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"]); model.to(args.device).eval()
    print(f"ckpt step {ck.get('step')}  window={L}  horizon k={k} ({k*125}ms/step)  "
          f"per_player_dim={ppd}  cv_residual={cv_residual}")

    blob = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    assert blob["feature_dim"] == ck["feature_dim"], (blob["feature_dim"], ck["feature_dim"])

    if args.value_auc:
        value_auc_eval(model, blob, L, k, cv_residual, args)
    else:
        print(f"rollout {args.steps} steps = {args.steps*k*125}ms")
        pos_idx = torch.tensor([p * ppd + d for p in range(N_PLAYERS) for d in (0, 1, 2)],
                               device=args.device)
        drift_eval(model, blob["tensors"], L, k, cv_residual, args, pos_idx)


if __name__ == "__main__":
    main()
