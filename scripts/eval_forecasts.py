#!/usr/bin/env python3
"""Freeze anchors, generate saved forecasts, and score them without a model rerun.

See docs/evaluation-system.md. No canonical pass/fail decision is emitted.
Only load trusted local PyTorch corpora/checkpoints. NPZ bundles use no pickle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _corpus import load_corpus

from src.evaluation.artifacts import file_hash, load_bundle, require_compatible, save_bundle, save_json
from src.evaluation.runner import generate, make_manifest, read_json, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest", help="freeze source-keyed anchor selection")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--maps", default="de_ancient,de_dust2,de_inferno,de_mirage,de_nuke")
    p = sub.add_parser("generate", help="native-horizon predictions and cheap baselines")
    p.add_argument("--corpus", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score", help="replay metrics from immutable forecast files")
    p.add_argument("--bundle", required=True)
    p.add_argument("--compare", help="optional identical-protocol bundle, methods prefixed other_")
    p.add_argument("--baseline", default="smooth_cv")
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error("output exists; use a new path to preserve prior evidence")
    if args.command == "manifest":
        sha = file_hash(args.corpus)
        blob = load_corpus(args.corpus)
        result = make_manifest(
            blob,
            sha,
            window=args.window,
            horizon=args.horizon,
            steps=args.steps,
            stride=args.stride,
            maps=set(args.maps.split(",")),
        )
        save_json(args.out, result)
    elif args.command == "generate":
        manifest = read_json(args.manifest)
        if file_hash(args.corpus) != manifest["corpus_sha256"]:
            parser.error("corpus bytes differ from frozen manifest")
        blob = load_corpus(args.corpus)
        metadata, arrays = generate(
            blob,
            manifest,
            samples=args.samples,
            seed=args.seed,
            temperature=args.temperature,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        save_bundle(args.out, metadata, arrays)
    else:
        meta, arrays = load_bundle(args.bundle)
        sources = {"bundle_sha256": file_hash(args.bundle)}
        if args.compare:
            other_meta, other_arrays = load_bundle(args.compare)
            require_compatible(meta, arrays, other_meta, other_arrays)
            for name, method in other_meta["methods"].items():
                key = "other_" + name
                if key in meta["methods"]:
                    parser.error("comparison method name collision")
                meta["methods"][key] = {
                    **method,
                    "implementation": other_meta["implementation"],
                    "runtime": other_meta["runtime"],
                }
                arrays["pred_" + key] = other_arrays["pred_" + name]
            sources["comparison_sha256"] = file_hash(args.compare)
            sources["comparison_implementation"] = other_meta["implementation"]
            sources["comparison_runtime"] = other_meta["runtime"]
        report = summarize(meta, arrays, baseline=args.baseline, replicates=args.replicates, seed=args.seed)
        save_json(args.out, {**report, "sources": sources})
    print(f"{args.command}: wrote {args.out} (exploratory; locked gates not adjudicated)")


if __name__ == "__main__":
    main()
