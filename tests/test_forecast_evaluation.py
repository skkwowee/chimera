"""Two regressions: joint score arithmetic and the saved-forecast workflow."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_forecasts import generate, report, scores
from train_world_model import build_model


def test_joint_score_and_empty_mask():
    # Each draw gets a different player right: selecting per-player would cheat.
    draws = np.array([[[0., 0.], [10., 0.]], [[10., 0.], [0., 0.]]])
    np.testing.assert_allclose(scores(draws, np.zeros((2, 2)), np.ones(2, bool)),
                               [5, 5, 10 / np.sqrt(2) - 5])
    assert np.isnan(scores(draws, np.zeros((2, 2)), np.zeros(2, bool))).all()


def test_saved_forecasts_replay(tmp_path):
    torch.set_num_threads(1)
    config = dict(arch="player", window=5, horizon=4, d_model=16, layers=1, heads=2, dist_head=True)
    model = build_model("player", 597, 16, 1, 2, per_player_dim=56, dist=True)
    checkpoint, corpus, out = (tmp_path / name for name in ("model.pt", "val.pt", "forecast.npz"))
    torch.save(dict(model=model.state_dict(), args=config, per_player_dim=56, feature_dim=597), checkpoint)
    r = torch.zeros(12, 597)
    p = r[:, :560].reshape(12, 10, 56)
    p[..., 13] = 1
    p[..., 0] = torch.arange(12)[:, None] / 3000
    meta = dict(match_id="fixture", demo_stem="demo", round_num=1, first_tick=0, map_name="de_mirage")
    torch.save(dict(tensors=[r], metas=[meta], feature_dim=597, per_player_dim=56, downsample=8), corpus)
    args = argparse.Namespace(checkpoint=checkpoint, corpus=corpus, out=out, stride=4, samples=2, seed=0, device="cpu")
    generate(args)
    result = report(out)
    assert result["methods"]["cv"]["mean_error"] < 1e-5
    assert result["locked_gates"] == "not_adjudicated"
    with pytest.raises(FileExistsError):
        generate(args)
    args.out = tmp_path / "repeat.npz"
    generate(args)
    with np.load(out, allow_pickle=False) as a, np.load(args.out, allow_pickle=False) as b:
        np.testing.assert_array_equal(a["model"], b["model"])
        assert json.loads(str(a["metadata"])) == json.loads(str(b["metadata"]))
    ck = torch.load(checkpoint, weights_only=False)
    ck["schema_version"] = "different-same-width-schema"
    torch.save(ck, checkpoint)
    args.out = tmp_path / "mismatch.npz"
    with pytest.raises(ValueError, match="semantic schema"):
        generate(args)
