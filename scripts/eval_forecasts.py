#!/usr/bin/env python3
"""Native-step XY sanity check, not a canonical gate. See docs/evaluation-system.md."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from _corpus import load_corpus
from train_world_model import CANONICAL_MAPS, build_model

FORMAT = "chimera-native-step-v1"


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def scores(samples, truth, alive):
    """[K,P,2], [P,2], [P]; fair energy score for IID draws, exact for K=1."""
    x, y, mask = np.asarray(samples, float), np.asarray(truth, float), np.asarray(alive)
    if x.ndim != 3 or x.shape[1:] != y.shape or y.shape != (len(mask), 2) or mask.dtype != bool:
        raise ValueError("unaligned XY samples/truth/mask")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("empty or nonfinite forecasts")
    if not mask.any():
        return np.full(3, np.nan)
    errors = np.linalg.norm(x[:, mask] - y[mask], axis=-1).mean(axis=1)
    # One ensemble member represents ALL players, never the best draw per player.
    joint = x[:, mask].reshape(len(x), -1) / np.sqrt(mask.sum())
    target = y[mask].reshape(-1) / np.sqrt(mask.sum())
    pairs = sum(np.linalg.norm(joint[j + 1:] - joint[j], axis=1).sum() for j in range(len(x) - 1))
    energy = np.linalg.norm(joint - target, axis=1).mean()
    if len(x) > 1:
        energy -= pairs / (len(x) * (len(x) - 1))
    return np.array([errors.mean(), errors.min(), energy])


def generate(args):
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.stride < 1 or args.samples < 2:
        raise ValueError("stride >= 1 and samples >= 2 required")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck["args"]
    ppd, window, horizon = ck["per_player_dim"], config["window"], config["horizon"]
    if config["arch"] != "player" or ppd not in (56, 65) or window < 5 or horizon < 1:
        raise ValueError("requires player checkpoint, v2/v3 schema, window >= 5 and positive horizon")
    if config.get("dist_head", False) and config.get("cv_residual", False):
        raise ValueError("ambiguous combined distributional/CV checkpoint")
    model = build_model("player", ck["feature_dim"], config["d_model"], config["layers"],
                        config["heads"], per_player_dim=ppd, dist=config.get("dist_head", False))
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    blob = load_corpus(args.corpus, maps=CANONICAL_MAPS)
    if (blob.get("downsample"), blob.get("per_player_dim", 56), blob["feature_dim"]) != (8, ppd, 10 * ppd + 37):
        raise ValueError("corpus/checkpoint schema mismatch or unsupported cadence")
    arrays = {name: [] for name in ("truth", "alive", "copy", "cv", "model")}
    anchors, seen = [], set()
    for r, meta in zip(blob["tensors"], blob["metas"], strict=True):
        key = [meta[k] for k in ("match_id", "demo_stem", "round_num", "first_tick")]
        identity = json.dumps(key)
        if not key[0] or identity in seen:
            raise ValueError("missing match ID or duplicate source round")
        seen.add(identity)
        p = r[:, :10 * ppd].reshape(len(r), 10, ppd)
        for t in range(window - 1, len(r) - horizon, args.stride):
            if r[t, 10 * ppd + 7] > .5 or r[t, 10 * ppd + 10] > .5:
                continue  # exclude freeze/end anchors, not future outcomes
            anchor = [*key, t]
            seed = int(hashlib.sha256(json.dumps([args.seed, anchor]).encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            with torch.inference_mode():
                x = r[t - window + 1:t + 1].to(args.device).unsqueeze(0).repeat(args.samples, 1, 1)
                pred = x[:, -1] + model.gen_residual(x, sample=model.dist, generator=generator)[:, -1]
                if config.get("cv_residual", False):
                    pred += horizon * (x[:, -1] - x[:, -2])
            current = p[t, :, :2].numpy() * 3000
            arrays["truth"].append(p[t + horizon, :, :2].numpy() * 3000)
            arrays["alive"].append(((p[t, :, 13] > .5) & (p[t + horizon, :, 13] > .5)).numpy())
            arrays["copy"].append(current)
            arrays["cv"].append(current + horizon * (p[t, :, :2] - p[t - 4, :, :2]).numpy() * 750)
            arrays["model"].append(pred[:, :10 * ppd].reshape(args.samples, 10, ppd)[..., :2].cpu().numpy() * 3000)
            anchors.append({"source": anchor, "map": meta["map_name"]})
    if not anchors:
        raise ValueError("no eligible anchors")
    metadata = {"format": FORMAT, "checkpoint": file_hash(args.checkpoint), "corpus": file_hash(args.corpus),
                "config": config, "seed": args.seed, "samples": args.samples, "stride": args.stride,
                "temperature": 1.0, "anchors": anchors, "torch": torch.__version__, "device": args.device,
                "code": {name: file_hash(Path(__file__).with_name(name))
                         for name in ("eval_forecasts.py", "train_world_model.py", "_corpus.py")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "xb") as f:
        np.savez_compressed(f, metadata=np.array(json.dumps(metadata)), **{k: np.stack(v) for k, v in arrays.items()})


def report(path):
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
        if meta["format"] != FORMAT:
            raise ValueError("unsupported bundle")
        result = {"status": "exploratory_only", "locked_gates": "not_adjudicated", "bundle_sha256": file_hash(path),
                  "scope": "survivor-conditioned XY; nominal frame offsets; no 64Hz or rollout certification",
                  "methods": {}}
        truth, alive = bundle["truth"], bundle["alive"]
        for name in ("copy", "cv", "model"):
            predictions = bundle[name] if name == "model" else bundle[name][:, None]
            values = np.array([scores(x, y, mask) for x, y, mask in zip(predictions, truth, alive, strict=True)])
            valid = np.isfinite(values).all(axis=1)
            result["methods"][name] = dict(zip(("mean_error", "best_joint_error", "energy"),
                values[valid].mean(axis=0).tolist() if valid.any() else [None] * 3))
            result["methods"][name].update(anchors=int(valid.sum()), excluded=int((~valid).sum()))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="save native-step forecasts on fixed-stride validation examples")
    for flag in ("corpus", "checkpoint", "out"):
        run.add_argument("--" + flag, required=True)
    run.add_argument("--stride", type=int, default=32)
    run.add_argument("--samples", type=int, default=16)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cpu")
    sub.add_parser("score", help="score a saved bundle without loading the model").add_argument("bundle")
    args = parser.parse_args()
    if args.command == "run":
        generate(args)
        print(f"Saved {args.out}; use score to inspect it.")
    else:
        print(json.dumps(report(args.bundle), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
