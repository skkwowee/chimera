#!/usr/bin/env python3
"""
Compare zero-shot Claude vs Qwen 3.5-27B on CS2 screenshot analysis.

Finds matching label/screenshot pairs, runs inference with selected models,
and compares per-field accuracy against ground truth.

Usage:
    python scripts/compare_models.py --samples 5
    python scripts/compare_models.py --samples 5 --models claude
    python scripts/compare_models.py --samples 5 --models qwen
    python scripts/compare_models.py --samples 5 --models claude qwen
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fields to compare (same as scripts/evaluate.py)
COMPARE_FIELDS = [
    "map_name",
    "round_phase",
    "player_side",
    "player_health",
    "player_armor",
    "player_money",
    "weapon_primary",
    "weapon_secondary",
    "alive_teammates",
    "alive_enemies",
    "bomb_status",
]


def find_pairs(labeled_dir: Path, captures_dir: Path) -> list[dict]:
    """Find label/screenshot pairs where both files exist."""
    pairs = []
    for label_path in sorted(labeled_dir.glob("*.json")):
        stem = label_path.stem
        # Search across all capture subdirectories
        for jpg in captures_dir.glob(f"*/raw/{stem}.jpg"):
            pairs.append({
                "id": stem,
                "label_path": label_path,
                "image_path": jpg,
            })
            break  # one match is enough
    return pairs


def compare_fields(label: dict, prediction: dict) -> dict:
    """Compare game_state fields between label and prediction."""
    label_state = label.get("game_state", {})
    pred_state = prediction.get("game_state", {})

    details = {}
    matches = 0
    total = 0

    for field in COMPARE_FIELDS:
        label_val = label_state.get(field)
        pred_val = pred_state.get(field)
        if label_val is not None:
            total += 1
            match = label_val == pred_val
            if match:
                matches += 1
            details[field] = {
                "label": label_val,
                "prediction": pred_val,
                "match": match,
            }

    return {"matches": matches, "total": total, "details": details}


def format_val(val) -> str:
    """Format a value for table display."""
    if val is None:
        return "null"
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "[]"
    return str(val)


def print_sample_table(sample_id: str, label: dict, predictions: dict[str, dict]):
    """Print a per-sample comparison table."""
    print(f"\n=== {sample_id} ===\n")

    model_names = list(predictions.keys())
    comparisons = {
        name: compare_fields(label, pred)
        for name, pred in predictions.items()
    }

    # Header
    header = f"| {'Field':<20} | {'Label':<14} |"
    sep = f"|{'-'*22}|{'-'*16}|"
    for name in model_names:
        header += f" {name:<14} |"
        sep += f"{'-'*16}|"
    print(header)
    print(sep)

    # Rows
    label_state = label.get("game_state", {})
    for field in COMPARE_FIELDS:
        label_val = label_state.get(field)
        if label_val is None:
            continue
        row = f"| {field:<20} | {format_val(label_val):<14} |"
        for name in model_names:
            detail = comparisons[name]["details"].get(field, {})
            pred_val = detail.get("prediction")
            match = detail.get("match", False)
            display = format_val(pred_val)
            if not match:
                display = f"*{display}"
            row += f" {display:<14} |"
        print(row)

    # Accuracy row
    row = f"| {'Accuracy':<20} | {'':<14} |"
    for name in model_names:
        c = comparisons[name]
        row += f" {c['matches']}/{c['total']:<13} |"
    print(row)

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Compare Claude vs Qwen on CS2 screenshots")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--models", nargs="+", default=["claude", "qwen"],
                        choices=["claude", "qwen"], help="Models to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--labeled-dir", type=str, default="data/labeled")
    parser.add_argument("--captures-dir", type=str, default="data/captures")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    captures_dir = Path(args.captures_dir)

    # Find all matching pairs
    pairs = find_pairs(labeled_dir, captures_dir)
    print(f"Found {len(pairs)} label/screenshot pairs")

    if not pairs:
        print("No matching pairs found. Run `python scripts/data.py pull` first.")
        sys.exit(1)

    # Sample
    random.seed(args.seed)
    samples = random.sample(pairs, min(args.samples, len(pairs)))
    print(f"Sampled {len(samples)} for evaluation (seed={args.seed})")
    print(f"Models: {', '.join(args.models)}")

    # Initialize models
    models = {}
    if "claude" in args.models:
        from src.inference import ClaudeVLMInference
        models["Claude"] = ClaudeVLMInference()
        print("Claude client ready")
    if "qwen" in args.models:
        from src.inference import Qwen3VLInference
        models["Qwen"] = Qwen3VLInference()
        print("Loading Qwen model...")

    # Run inference and compare
    all_results = []
    aggregate = {name: {"matches": 0, "total": 0} for name in models}

    for i, pair in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)}: {pair['id']} ---")

        # Load label
        with open(pair["label_path"]) as f:
            label = json.load(f)

        # Run each model
        predictions = {}
        for name, model in models.items():
            print(f"  Running {name}...")
            pred = model.analyze(pair["image_path"])
            predictions[name] = pred

        # Compare and print table
        comparisons = print_sample_table(pair["id"], label, predictions)

        # Accumulate
        sample_result = {
            "id": pair["id"],
            "image": str(pair["image_path"]),
            "label": label,
            "predictions": {},
        }
        for name in models:
            c = comparisons[name]
            aggregate[name]["matches"] += c["matches"]
            aggregate[name]["total"] += c["total"]
            sample_result["predictions"][name] = {
                "output": predictions[name],
                "matches": c["matches"],
                "total": c["total"],
            }
        all_results.append(sample_result)

    # Summary
    print(f"\n{'='*50}")
    print("=== Summary ===\n")
    header = f"| {'Model':<10} | {'Accuracy':<20} |"
    sep = f"|{'-'*12}|{'-'*22}|"
    print(header)
    print(sep)
    for name in models:
        a = aggregate[name]
        pct = a["matches"] / a["total"] * 100 if a["total"] > 0 else 0
        print(f"| {name:<10} | {a['matches']}/{a['total']} ({pct:.1f}%){'':<8} |")

    # Save results
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"comparison_{timestamp}.json"

    save_data = {
        "timestamp": timestamp,
        "seed": args.seed,
        "models": args.models,
        "num_samples": len(samples),
        "aggregate": {
            name: {
                "matches": a["matches"],
                "total": a["total"],
                "accuracy": a["matches"] / a["total"] if a["total"] > 0 else 0,
            }
            for name, a in aggregate.items()
        },
        "samples": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
