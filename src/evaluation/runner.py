"""Native-frame forecasting adapter and exploratory report builder.

This intentionally refuses mixed-cadence autoregression, does not interpolate
missing ticks, and cannot adjudicate the locked canonical experiment gates.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch

from .artifacts import VERSION, digest, file_hash, validate_manifest
from .metrics import forecast_scores, paired_cluster_interval


def implementation_provenance():
    """Content hashes bind results to the actual code, including uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        "scripts/eval_forecasts.py",
        "scripts/train_world_model.py",
        "scripts/_corpus.py",
        "src/evaluation/runner.py",
        "src/evaluation/artifacts.py",
        "src/evaluation/metrics.py",
    ]
    return {path: file_hash(root / path) for path in paths}


def round_key(meta):
    return {k: meta[k] for k in ("match_id", "demo_stem", "round_num", "first_tick")}


def make_manifest(blob, corpus_hash, *, window=96, horizon=4, steps=1, stride=32, maps=None):
    if min(horizon, steps, stride) < 1 or window < 5:
        raise ValueError("positive horizon/steps/stride and window >= 5 required")
    if blob.get("downsample") != 8:
        raise ValueError("adapter only supports nominal 8 Hz corpora")
    ppd = blob.get("per_player_dim", 56)
    if ppd not in (56, 65) or blob["feature_dim"] != 10 * ppd + 37:
        raise ValueError("unsupported corpus layout")
    if len(blob["tensors"]) != len(blob["metas"]):
        raise ValueError("unaligned corpus lists")
    anchors, seen = [], set()
    for tensor, meta in zip(blob["tensors"], blob["metas"]):
        if maps and meta["map_name"] not in maps:
            continue
        key = round_key(meta)
        if not key["match_id"] or digest(key) in seen:
            raise ValueError("missing match identity or duplicate source round")
        seen.add(digest(key))
        for frame in range(window - 1, len(tensor) - horizon * steps, stride):
            # Only the anchor phase is filtered. Target survival is a scoring mask.
            if tensor[frame, 10 * ppd + 7] > 0.5 or tensor[frame, 10 * ppd + 10] > 0.5:
                continue
            anchor = {
                **key,
                "map_name": meta["map_name"],
                "frame": frame,
                "source_tick": None,
            }  # D7: first_tick + frame*8 is NOT a real timestamp.
            anchors.append({**anchor, "id": digest(anchor)})
    anchors.sort(key=lambda a: a["id"])
    body = {
        "version": VERSION,
        "corpus_sha256": corpus_hash,
        "spec": {
            "window": window,
            "horizon": horizon,
            "steps": steps,
            "nominal_hz": 8,
            "units": "game_xy",
            "mask": "alive_at_anchor_and_target",
            "timing": "nominal_frame_offsets_not_verified_source_ticks",
        },
        "selection": {"stride": stride, "maps": sorted(maps) if maps else None, "anchor_phase": "not_freeze_not_end"},
        "anchors": anchors,
    }
    result = {**body, "manifest_hash": digest(body)}
    validate_manifest(result)
    return result


def anchor_seed(seed, anchor_id, method):
    return int(digest([seed, anchor_id, method])[:15], 16)


def load_checkpoint(path, spec, ppd, device):
    # Import works with the CLI's scripts path and the repository test setup.
    from train_world_model import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    if (args["window"], args["horizon"], ck["per_player_dim"]) != (spec["window"], spec["horizon"], ppd):
        raise ValueError("checkpoint window/horizon/schema differs from manifest")
    if args["arch"] != "player":
        raise ValueError("only player checkpoints supported")
    if args.get("cv_residual", False) and args.get("dist_head", False):
        raise ValueError("ambiguous combined CV/distributional checkpoint is unsupported")
    if spec["steps"] > 1 and (spec["horizon"] != 1 or ppd != 56):
        raise ValueError("multi-step generation requires k=1 raw v2; mixed cadence/stale v3 is unsupported")
    model = build_model(
        args["arch"],
        ck["feature_dim"],
        args["d_model"],
        args["layers"],
        args["heads"],
        per_player_dim=ppd,
        dist=args.get("dist_head", False),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, args


def generate(blob, manifest, *, samples=16, seed=0, temperature=1.0, checkpoint=None, device="cpu"):
    validate_manifest(manifest)
    if samples < 2 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("K >= 2 and finite positive temperature required")
    spec = manifest["spec"]
    ppd, h, steps = blob.get("per_player_dim", 56), spec["horizon"], spec["steps"]
    window = spec["window"]
    model = None
    methods = {"copy": {"sampling": "deterministic"}, "smooth_cv": {"sampling": "deterministic"}}
    if checkpoint:
        model, model_args = load_checkpoint(checkpoint, spec, ppd, device)
        methods["model"] = {
            "sampling": "iid" if model.dist else "deterministic",
            "checkpoint_sha256": file_hash(checkpoint),
            "checkpoint_args": model_args,
        }
    index = {}
    for tensor, meta in zip(blob["tensors"], blob["metas"]):
        key = digest(round_key(meta))
        if key in index:
            raise ValueError("duplicate corpus round")
        index[key] = (tensor, meta)
    n = len(manifest["anchors"])
    truth = np.empty((n, steps, 10, 2), dtype=np.float32)
    mask = np.empty((n, steps, 10), dtype=bool)
    origin = np.empty((n, 10, 2), dtype=np.float32)
    predictions = {name: np.empty((n, samples, steps, 10, 2), dtype=np.float32) for name in methods}
    offsets = np.arange(1, steps + 1) * h
    for i, a in enumerate(manifest["anchors"]):
        r, meta = index[digest(round_key(a))]
        t = a["frame"]
        if meta["map_name"] != a["map_name"] or t < window - 1 or t + offsets[-1] >= len(r):
            raise ValueError("anchor does not match corpus")
        players = r[:, : 10 * ppd].reshape(len(r), 10, ppd)
        current = players[t, :, :2].numpy() * 3000
        velocity = (players[t, :, :2] - players[t - 4, :, :2]).numpy() * (3000 / 4)
        target = players[t + offsets, :, :2].numpy() * 3000
        truth[i], origin[i] = target, current
        mask[i] = ((players[t, :, 13] > 0.5) & (players[t + offsets, :, 13] > 0.5)).numpy()
        cv = current + offsets[:, None, None] * velocity
        predictions["copy"][i] = np.broadcast_to(current, (samples, steps, 10, 2))
        predictions["smooth_cv"][i] = cv
        if model:
            with torch.inference_mode():
                buf = r[t - window + 1 : t + 1].to(device).unsqueeze(0).repeat(samples, 1, 1)
                generator = torch.Generator(device=device).manual_seed(anchor_seed(seed, a["id"], "model"))
                for j in range(steps):
                    residual = model.gen_residual(buf, sample=model.dist, temperature=temperature, generator=generator)[
                        :, -1
                    ]
                    pred = buf[:, -1] + residual
                    if model_args.get("cv_residual", False):
                        pred += h * (buf[:, -1] - buf[:, -2])
                    predictions["model"][i, :, j] = (
                        pred[:, : 10 * ppd].reshape(samples, 10, ppd)[..., :2].cpu().numpy() * 3000
                    )
                    buf = torch.cat([buf[:, 1:], pred[:, None]], dim=1)
    metadata = {
        "version": VERSION,
        "manifest": manifest,
        "samples": samples,
        "seed": seed,
        "temperature": temperature,
        "methods": methods,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "status": "exploratory_only",
        "truth_64hz": "unavailable",
        "timing": spec["timing"],
        "implementation": implementation_provenance(),
    }
    arrays = {"truth": truth, "mask": mask, "origin": origin, **{"pred_" + k: v for k, v in predictions.items()}}
    return metadata, arrays


def summarize(meta, arrays, *, baseline="smooth_cv", replicates=2000, seed=0):
    methods = meta["methods"]
    if baseline not in methods:
        raise ValueError("requested comparator is absent")
    ids = [a["match_id"] for a in meta["manifest"]["anchors"]]
    maps = np.array([a["map_name"] for a in meta["manifest"]["anchors"]])
    scores = {
        name: forecast_scores(arrays["pred_" + name], arrays["truth"], arrays["mask"], sampling=m["sampling"])
        for name, m in methods.items()
    }

    def aggregate(values):
        valid = np.isfinite(values)
        return {
            "mean": float(values[valid].mean()) if valid.any() else None,
            "anchors": int(valid.sum()),
            "excluded_anchors": int((~valid).sum()),
        }

    report = {
        "version": VERSION,
        "status": "exploratory_only",
        "manifest_hash": meta["manifest"]["manifest_hash"],
        "samples": meta["samples"],
        "methods": {},
        "paired_deltas": {},
        "forecast_implementation": meta["implementation"],
        "scoring_implementation": implementation_provenance(),
        "method_provenance": meta["methods"],
        "runtime": meta["runtime"],
        "anchors": meta["manifest"]["anchors"],
        "mask": meta["manifest"]["spec"]["mask"],
        "estimand": "pooled_anchor_mean",
        "energy_norm": "masked_joint_path_L2_div_sqrt_observed_player_times",
        "energy_estimator": "fair_MC_for_iid_exact_for_deterministic",
        "scope": "survivor_conditioned_xy_not_full_state_probability_or_causal_planning",
        "timing": meta["timing"],
        "truth_64hz": "unavailable",
        "locked_gates": {
            "status": "not_adjudicated",
            "reason": "Missing certified 64Hz, rollout/coherence controls and locked inference protocol",
        },
    }
    for name, metrics in scores.items():
        report["methods"][name] = {
            "overall": {k: aggregate(v) for k, v in metrics.items()},
            "per_map": {str(mp): {k: aggregate(v[maps == mp]) for k, v in metrics.items()} for mp in sorted(set(maps))},
            "per_anchor": {k: [float(x) if np.isfinite(x) else None for x in v] for k, v in metrics.items()},
        }
        if name != baseline:
            report["paired_deltas"][name] = {
                k: paired_cluster_interval(v, scores[baseline][k], ids, seed=seed, replicates=replicates)
                for k, v in metrics.items()
            }
    report["baseline"] = baseline
    return report


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
