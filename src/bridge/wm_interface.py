"""World-model interface — extract the frozen latent + predictive channels.

bridge-design.md §1 (Interface A): the bridge consumes the world model's
contextualized **token grid** `[B, 11, 512]` (10 players + global) for the LAST
frame of a window, NOT the pooled `latent()`. Per-player structure is preserved
so the LLM can attend to individual players.

§1 also says to AUGMENT the latent with PREDICTIVE-head outputs — the part that
is NOT in raw features. This is what makes the grounding non-circular (§5): text
that is a pure function of the current frame is exactly what ablate-the-latent
must punish; text that requires the FORWARD model is what it must reward. The
channels here are the cheap, real foresight signals; the multi-step rollout
summary (value mean/spread at +2s/+4s) is the documented extension for the SFT
generator (step 2), which reuses scripts/rollout_eval.py's stepping.

The world model lives in scripts/train_world_model.py; we add it to sys.path the
same way scripts/nla_capacity_probe.py does, keeping src/bridge import-clean.
"""
from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path
import torch

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

N_PRED_CHANNELS = 3  # [current value logit, 1-step-ahead value logit, per-player top dist class]


def load_world_model(ckpt_path: str, device: str = "cpu"):
    """Load + freeze the world model. Returns (model, meta-dict)."""
    from train_world_model import build_model  # noqa: late import (sys.path set above)
    with tempfile.TemporaryDirectory() as td:
        safe = Path(td) / "c.pt"
        shutil.copy(ckpt_path, safe)
        ck = torch.load(safe, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_model(a["arch"], ck["feature_dim"], a["d_model"], a["layers"],
                        a["heads"], per_player_dim=ck["per_player_dim"],
                        dist=a.get("dist_head", False))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {"window": a["window"], "d_model": a["d_model"], "ppd": ck["per_player_dim"],
            "feature_dim": ck["feature_dim"], "dist": a.get("dist_head", False)}
    return model, meta


@torch.no_grad()
def extract(model, x: torch.Tensor):
    """x: [B, L, F] raw frame windows. Returns, for the LAST frame:
        grid     : [B, 11, 512]  contextualized per-token latent (featurizer input)
        z_pooled : [B, 512]      h.mean(dim=2) — the value-head input / recon floor
        channels : [B, 11, N_PRED_CHANNELS]  predictive-head augmentation (§1)
    """
    h, _ = model._grid(x)              # [B, L, 11, 512]
    grid = h[:, -1]                    # [B, 11, 512] last frame
    z_pooled = grid.mean(dim=1)        # == model.latent(x)[:, -1]
    P = model.n_players                # 10

    out = model.heads(x)
    v_now = out["value"][:, -1]                                # [B] current P(CT win) logit
    v_next = _one_step_value(model, x)                         # [B] 1-step-ahead (foresight)
    if model.dist:
        top_cls = out["dist_logits"][:, -1].argmax(-1).float() / 96.0   # [B, P]
    else:
        top_cls = torch.zeros(x.shape[0], P, device=x.device)

    B = x.shape[0]
    channels = torch.zeros(B, P + 1, N_PRED_CHANNELS, device=x.device)
    channels[:, :, 0] = v_now.unsqueeze(1)                     # broadcast to all tokens
    channels[:, :, 1] = v_next.unsqueeze(1)
    channels[:, :P, 2] = top_cls                               # player tokens; global stays 0
    return grid, z_pooled, channels


@torch.no_grad()
def _one_step_value(model, x: torch.Tensor) -> torch.Tensor:
    """Foresight channel: slide the window forward by the model's own predicted
    next frame and re-read the value head. Real, cheap, and non-circular (requires
    the forward model). Returns [B] logit."""
    res = model.gen_residual(x)                     # [B, L, F]
    x_pred = x[:, -1] + res[:, -1]                  # predicted frame at t+horizon
    x_next = torch.cat([x[:, 1:], x_pred[:, None]], dim=1)
    return model.heads(x_next)["value"][:, -1]
