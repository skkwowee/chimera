"""Model smoke test — the regression net under runbook [3]'s detach/SS surgery.

Builds the factored model at real feature dims (tiny width), runs one
multi-head forward on [2, 8, 597], assembles a scalar loss touching the
residual + value + distributional heads, backprops, and asserts every gradient
produced is finite. CPU, seconds."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from train_world_model import build_model


def test_model_smoke_forward_backward():
    torch.manual_seed(0)
    model = build_model("player", feature_dim=597, d_model=64, layers=2, heads=2,
                        per_player_dim=56, dist=True)
    x = torch.randn(2, 8, 597)

    out = model.heads(x)
    assert set(out) >= {"residual", "value", "dist_logits", "dist_off"}
    assert out["residual"].shape == (2, 8, 597)
    assert out["value"].shape == (2, 8)
    assert out["dist_logits"].shape[:3] == (2, 8, 10)
    assert out["dist_off"].shape[-1] == 2

    # scalar loss touching residual + value + dist heads
    loss = (
        out["residual"].pow(2).mean()
        + F.binary_cross_entropy_with_logits(out["value"],
                                             torch.zeros_like(out["value"]))
        + F.cross_entropy(
            out["dist_logits"].reshape(-1, out["dist_logits"].shape[-1]),
            torch.zeros(out["dist_logits"].reshape(-1, out["dist_logits"].shape[-1]).shape[0],
                        dtype=torch.long))
        + out["dist_off"].pow(2).mean()
    )
    assert torch.isfinite(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()

    got_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert got_grad, "no parameter received a gradient"
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    assert not bad, f"non-finite grads: {bad[:10]}"

    # every head actually trained by this loss
    for head in ("player_head", "global_head", "value_head", "dist_head"):
        assert any(n.startswith(head) for n in got_grad), f"{head} got no gradient"
