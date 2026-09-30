"""Adversarial contracts: no real corpus/checkpoint or GPU required."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.artifacts import (
    digest,
    file_hash,
    load_bundle,
    require_compatible,
    save_bundle,
    validate_manifest,
)
from src.evaluation.metrics import binary_auc, forecast_scores, paired_cluster_interval
from src.evaluation.runner import generate, make_manifest, summarize


def fixture_blob(prefix="eval"):
    tensors, metas = [], []
    for i in range(3):
        r = torch.zeros(20, 597)
        p = r[:, :560].reshape(20, 10, 56)
        p[..., 13] = 1
        p[..., 0] = torch.arange(20)[:, None] * (i + 1) / 3000
        r[:, 560 + 8] = 1  # live
        tensors.append(r)
        metas.append(
            {
                "match_id": f"{prefix}-{i}",
                "demo_stem": f"demo-{i}",
                "round_num": 1,
                "first_tick": 100,
                "map_name": "de_mirage",
            }
        )
    return {"tensors": tensors, "metas": metas, "feature_dim": 597, "per_player_dim": 56, "downsample": 8}


def fixture_bundle(prefix="eval", horizon=1, steps=3):
    blob = fixture_blob(prefix)
    manifest = make_manifest(blob, "a" * 64, window=5, horizon=horizon, steps=steps, stride=5)
    return generate(blob, manifest, samples=4)


def test_auc_ties_and_invalid_inputs():
    assert binary_auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert binary_auc([1, 1, 1, 1], [1, 1, 0, 0]) == 0.5
    assert binary_auc([0, 1, 2, 3], [0, 0, 1, 1]) == 1
    assert np.isnan(binary_auc([0, 1], [1, 1]))
    with pytest.raises(ValueError):
        binary_auc([np.nan], [1])


def test_path_metrics_cannot_switch_members_at_each_time():
    truth = np.zeros((1, 2, 1, 2))
    samples = np.zeros((1, 2, 2, 1, 2))
    samples[0, 0, 1, 0, 0] = 10
    samples[0, 1, 0, 0, 0] = 10
    result = forecast_scores(samples, truth, np.ones((1, 2, 1), bool), sampling="iid")
    assert result["minade_joint"][0] == 5
    assert result["minade_player"][0] == 5
    assert result["energy_joint"][0] == pytest.approx(10 / np.sqrt(2) - 5)


def test_joint_minade_cannot_switch_members_between_players():
    truth = np.zeros((1, 1, 2, 2))
    samples = np.zeros((1, 2, 1, 2, 2))
    samples[0, 0, 0, 1, 0] = 10
    samples[0, 1, 0, 0, 0] = 10
    result = forecast_scores(samples, truth, np.ones((1, 1, 2), bool), sampling="iid")
    assert result["minade_player"][0] == 0
    assert result["minade_joint"][0] == 5


def test_energy_point_score_and_empty_support():
    y = np.zeros((2, 1, 1, 2))
    x = np.full((2, 4, 1, 1, 2), 3.0)
    mask = np.array([[[True]], [[False]]])
    result = forecast_scores(x, y, mask, sampling="deterministic")
    assert result["energy_joint"][0] == pytest.approx(np.sqrt(18))
    assert np.isnan(result["energy_joint"][1])
    with pytest.raises(ValueError, match="two iid"):
        forecast_scores(x[:, :1], y, mask, sampling="iid")
    x[0, 1, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="members differ"):
        forecast_scores(x, y, mask, sampling="deterministic")


def test_cluster_bootstrap_keeps_pairing_and_weights():
    result = paired_cluster_interval([2, 4, 8], [1, 3, 7], ["a", "a", "b"], replicates=100)
    assert result["ci95"] == [1, 1]
    assert result["delta"] == 1
    assert paired_cluster_interval([1, 2], [0, 0], ["a", "a"], replicates=100)["ci95"] is None
    assert result == paired_cluster_interval([2, 4, 8], [1, 3, 7], ["a", "a", "b"], replicates=100)


def test_manifest_is_stable_and_self_authenticates():
    blob = fixture_blob()
    a = make_manifest(blob, "a" * 64, window=5, horizon=1, steps=3)
    blob["tensors"].reverse()
    blob["metas"].reverse()
    b = make_manifest(blob, "a" * 64, window=5, horizon=1, steps=3)
    assert a == b
    assert all(row["source_tick"] is None for row in a["anchors"])
    a["anchors"][0]["frame"] += 1
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(a)


def test_duplicate_source_round_is_rejected():
    blob = fixture_blob()
    blob["metas"][1] = blob["metas"][0]
    with pytest.raises(ValueError, match="duplicate"):
        make_manifest(blob, "a" * 64, window=5)


def test_bundle_roundtrip_integrity_and_refuse_overwrite(tmp_path):
    meta, arrays = fixture_bundle()
    path = tmp_path / "forecasts.npz"
    save_bundle(path, meta, arrays)
    loaded_meta, loaded_arrays = load_bundle(path)
    require_compatible(meta, arrays, loaded_meta, loaded_arrays)
    with pytest.raises(FileExistsError):
        save_bundle(path, meta, arrays)
    altered = dict(loaded_arrays)
    altered["truth"] = altered["truth"] + 1
    bad = tmp_path / "bad.npz"
    np.savez(bad, metadata=np.array(json.dumps(loaded_meta)), **altered)
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(bad)


def test_comparison_rejects_different_truth_seed_or_anchor_order():
    meta, arrays = fixture_bundle()
    other = copy.deepcopy(meta)
    other["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        require_compatible(meta, arrays, other, arrays)
    other = copy.deepcopy(meta)
    other["manifest"]["anchors"].reverse()
    with pytest.raises(ValueError, match="manifest"):
        require_compatible(meta, arrays, other, arrays)
    with pytest.raises(ValueError, match="truth"):
        require_compatible(meta, arrays, meta, {**arrays, "truth": arrays["truth"] + 1})


def save_checkpoint(path, horizon=1):
    from train_world_model import build_model

    model = build_model("player", 597, 16, 1, 2, per_player_dim=56, dist=True)
    args = {
        "arch": "player",
        "window": 5,
        "horizon": horizon,
        "d_model": 16,
        "layers": 1,
        "heads": 2,
        "dist_head": True,
    }
    torch.save({"model": model.state_dict(), "args": args, "feature_dim": 597, "per_player_dim": 56}, path)


def test_model_native_step_replay_and_mixed_cadence_refusal(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, horizon=4)
    blob = fixture_blob()
    manifest = make_manifest(blob, "a" * 64, window=5, horizon=4, steps=1, stride=20)
    meta, a = generate(blob, manifest, checkpoint=checkpoint, samples=2)
    _, b = generate(blob, manifest, checkpoint=checkpoint, samples=2)
    np.testing.assert_array_equal(a["pred_model"], b["pred_model"])
    assert meta["methods"]["model"]["checkpoint_sha256"] == file_hash(checkpoint)
    manifest["anchors"].reverse()
    manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    _, b = generate(blob, manifest, checkpoint=checkpoint, samples=2)
    np.testing.assert_array_equal(a["pred_model"], b["pred_model"][::-1])
    manifest = make_manifest(blob, "a" * 64, window=5, horizon=4, steps=2)
    with pytest.raises(ValueError, match="mixed cadence"):
        generate(blob, manifest, checkpoint=checkpoint)


def test_report_has_no_false_pass_and_records_missing_truth():
    meta, arrays = fixture_bundle()
    report = summarize(meta, arrays, replicates=100)
    assert report["locked_gates"]["status"] == "not_adjudicated"
    assert report["truth_64hz"] == "unavailable"
    assert report["methods"]["smooth_cv"]["overall"]["ade_mean"]["mean"] < 1e-5
    json.dumps(report, allow_nan=False)


def test_cli_end_to_end_and_changed_corpus_refusal(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/eval_forecasts.py"
    corpus = tmp_path / "fixture.pt"
    torch.save(fixture_blob(), corpus)
    manifest, bundle, report = (tmp_path / name for name in ("anchors.json", "forecasts.npz", "report.json"))

    def run(*args):
        return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=root, capture_output=True, text=True)

    r = run("manifest", "--corpus", corpus, "--window", 5, "--horizon", 1, "--steps", 3, "--out", manifest)
    assert r.returncode == 0, r.stderr
    r = run("generate", "--corpus", corpus, "--manifest", manifest, "--out", bundle)
    assert r.returncode == 0, r.stderr
    r = run("score", "--bundle", bundle, "--replicates", 100, "--out", report)
    assert r.returncode == 0, r.stderr
    assert json.loads(report.read_text())["status"] == "exploratory_only"
    torch.save(fixture_blob("changed"), corpus)
    r = run("generate", "--corpus", corpus, "--manifest", manifest, "--out", tmp_path / "other.npz")
    assert r.returncode != 0 and "corpus bytes differ" in r.stderr
