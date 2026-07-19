"""Acceptance test for runbook [3a] (Knob 5): the value head must be DETACHED.

The trunk must be gradient-identical to value_weight=0 — that is what makes the
C1 probe-transfer claim exogenous (outcome information is read out of the
representation, never injected into it). A string-grep cannot certify this
(rename the head and it passes); the canonical guarantee is gradient-based:
backprop the value loss ALONE and assert that no parameter outside the value
head receives a gradient.

EXPECTED RED until the stop-grad detach lands in train_world_model.py ([3a]).
Ported from the closed chimera2 clean-room experiment's rail test; adapted to
this repo's build_model/heads() API.

Run: .venv/bin/python -m pytest tests/test_no_value_leak.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_world_model import build_model  # noqa: E402


def test_no_value_leak():
    torch.manual_seed(0)
    model = build_model("player", feature_dim=597, d_model=64, layers=2, heads=2,
                        per_player_dim=56, dist=True)
    x = torch.randn(2, 8, 597)
    out = model.heads(x)
    assert isinstance(out, dict) and "value" in out, "heads() must return dict with 'value'"

    v = out["value"].float()
    v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        v, torch.zeros_like(v))
    model.zero_grad(set_to_none=True)
    v_loss.backward()

    head_params = {id(p) for p in model.value_head.parameters()}
    leaks = [n for n, p in model.named_parameters()
             if id(p) not in head_params and p.grad is not None]
    assert not leaks, (
        "outcome gradient reached non-value-head params (trunk is NOT "
        f"gradient-identical to value_weight=0): {leaks[:10]}")
