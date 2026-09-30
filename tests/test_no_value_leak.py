"""Acceptance test for runbook [3a] (Knob 5): the value head must be DETACHED.

The trunk must be gradient-identical to value_weight=0 — that is what makes the
C1 probe-transfer claim exogenous (outcome information is read out of the
representation, never injected into it). A string-grep cannot certify this
(rename the head and it passes); the canonical guarantee is gradient-based:
backprop the value loss ALONE and assert that no parameter outside the value
head receives a gradient.

Run: .venv/bin/python -m pytest tests/test_no_value_leak.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_world_model import (
    build_model,
    prediction_and_value_parameters,
    step_prediction_and_value,
)


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


def test_outcome_labels_cannot_change_prediction_optimizer_updates():
    """Includes clipping and AdamW state, not only direct backpropagation."""
    torch.manual_seed(91)
    template = build_model("player", feature_dim=597, d_model=16, layers=1,
                           heads=2, per_player_dim=56, dist=False).eval()
    # Saturation makes the two label assignments produce very different value
    # gradient norms; a shared global clipping norm fails this regression test.
    with torch.no_grad():
        template.value_head[-1].bias.fill_(15.0)
    models = [copy.deepcopy(template), copy.deepcopy(template)]
    x = torch.randn(1, 3, 597)
    for label, model in zip((0.0, 1.0), models):
        params = prediction_and_value_parameters(model)
        optimizers = tuple(torch.optim.AdamW(group, lr=0.01) for group in params)
        scalers = tuple(torch.amp.GradScaler("cpu", enabled=False) for _ in params)
        for _ in range(4):
            output = model.heads(x)
            ns_loss = 10 * output["residual"].square().mean()
            value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["value"], torch.full_like(output["value"], label))
            step_prediction_and_value(ns_loss, value_loss, optimizers, scalers, params)
    for left, right in zip(*(prediction_and_value_parameters(m)[0] for m in models)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert any(not torch.equal(left, right) for left, right in zip(
        *(prediction_and_value_parameters(m)[1] for m in models)))


def test_value_overflow_does_not_skip_prediction_step():
    """CPU AMP exercises independent overflow/scale state without a GPU run."""
    left = torch.nn.Parameter(torch.tensor(2.0))
    right = torch.nn.Parameter(torch.tensor(2.0))
    optimizers = tuple(torch.optim.SGD([p], lr=0.1) for p in (left, right))
    scalers = tuple(torch.amp.GradScaler("cpu", init_scale=8.0) for _ in (left, right))
    step_prediction_and_value(left.square(), right * float("inf"),
                              optimizers, scalers, ([left], [right]))
    assert left.item() < 2.0
    assert right.item() == 2.0
    assert scalers[0].get_scale() == 8.0
    assert scalers[1].get_scale() == 4.0
