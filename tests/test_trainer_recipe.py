from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _corpus import load_corpus
from train_world_model import (
    ALIVE_DIM,
    N_PLAYERS,
    build_model,
    dist_loss_mask,
    sample_and_swap_context,
    scheduled_sampling_probability,
)


class _ResidualStub:
    def __init__(self, residual: float):
        self.residual = residual
        self.calls = 0

    def gen_residual(self, x, **kwargs):
        self.calls += 1
        assert kwargs["sample"] is True
        assert kwargs["temperature"] == 1.0
        return torch.full_like(x, self.residual)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_scheduled_sampling_ramp_is_locked():
    assert scheduled_sampling_probability(0) == 0.0
    assert scheduled_sampling_probability(1999) == 0.0
    assert scheduled_sampling_probability(2000) == 0.0
    assert scheduled_sampling_probability(8500) == pytest.approx(0.25)
    assert scheduled_sampling_probability(15000) == 0.5
    assert scheduled_sampling_probability(25000) == 0.5
    assert scheduled_sampling_probability(25000, 0.0) == 0.0


def test_sample_and_swap_executes_positive_probability_path():
    model = _ResidualStub(10.0)
    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    swapped, count = sample_and_swap_context(
        model,
        x,
        horizon=2,
        probability=1.0,
        swap_generator=_generator(1),
        decode_generator=_generator(2),
    )

    assert model.calls == 1
    assert count == 2 * (6 - 2)
    torch.testing.assert_close(swapped[:, :2], x[:, :2])
    torch.testing.assert_close(swapped[:, 2:], x[:, :-2] + 10.0)
    torch.testing.assert_close(x, torch.arange(2 * 6 * 3).reshape(2, 6, 3).float())


def test_sample_and_swap_skips_decode_when_disabled():
    model = _ResidualStub(10.0)
    x = torch.randn(2, 6, 3)
    out, count = sample_and_swap_context(
        model,
        x,
        horizon=2,
        probability=0.0,
        swap_generator=_generator(1),
        decode_generator=_generator(2),
    )
    assert out is x
    assert count == 0
    assert model.calls == 0


def test_dist_loss_mask_uses_real_alive_ends_and_freeze():
    ppd = 56
    freeze_col = N_PLAYERS * ppd + 7
    feature_dim = freeze_col + 4
    x = torch.zeros(1, 2, feature_dim)
    y = torch.zeros_like(x)
    xp = x[..., : N_PLAYERS * ppd].reshape(1, 2, N_PLAYERS, ppd)
    yp = y[..., : N_PLAYERS * ppd].reshape(1, 2, N_PLAYERS, ppd)
    xp[..., ALIVE_DIM] = 1.0
    yp[..., ALIVE_DIM] = 1.0
    xp[:, 0, 1, ALIVE_DIM] = 0.0
    yp[:, 0, 2, ALIVE_DIM] = 0.0
    x[:, 1, freeze_col] = 1.0

    mask = dist_loss_mask(x, y, ppd, freeze_col)
    assert mask.shape == (1, 2, N_PLAYERS)
    assert mask[0, 0].sum().item() == N_PLAYERS - 2
    assert not mask[0, 0, 1]
    assert not mask[0, 0, 2]
    assert not mask[0, 1].any()


@pytest.mark.parametrize("dist", [False, True])
def test_decode_zeroes_dead_player_xy(dist):
    torch.manual_seed(0)
    model = build_model(
        "player",
        feature_dim=597,
        d_model=32,
        layers=1,
        heads=2,
        per_player_dim=56,
        dist=dist,
    ).eval()
    x = torch.randn(1, 4, 597)
    players = x[..., : N_PLAYERS * 56].reshape(1, 4, N_PLAYERS, 56)
    players[..., ALIVE_DIM] = 1.0
    players[..., 3, ALIVE_DIM] = 0.0

    residual = model.gen_residual(x, sample=dist, generator=_generator(7))
    decoded = residual[..., : N_PLAYERS * 56].reshape(1, 4, N_PLAYERS, 56)
    assert torch.equal(decoded[..., 3, 0:2], torch.zeros_like(decoded[..., 3, 0:2]))


def test_no_clean_is_explicit_and_loud(tmp_path, capsys):
    path = tmp_path / "fixture.pt"
    blob = {
        "tensors": [torch.zeros(2, 3), torch.ones(2, 3)],
        "metas": [{"map_name": "de_anubis"}, {"map_name": "de_mirage"}],
    }
    torch.save(blob, path)

    loaded = load_corpus(path, clean=False, tag="fixture")
    assert len(loaded["metas"]) == 2
    assert "WARNING: defect exclusions DISABLED" in capsys.readouterr().out

