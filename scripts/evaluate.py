#!/usr/bin/env python3
"""
Evaluate VLM predictions against Claude-generated labels.

Compares local model outputs to ground truth labels to measure accuracy.

Usage:
    python scripts/evaluate.py --labels data/labeled --predictions data/predictions
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json_files(directory: Path) -> dict[str, dict]:
    """Load all JSON files from a directory, keyed by stem."""
    results = {}
    for path in directory.glob("*.json"):
        with open(path) as f:
            results[path.stem] = json.load(f)
    return results


def compare_game_state(label: dict, prediction: dict) -> dict:
    """Compare game state fields between label and prediction."""
    label_state = label.get("game_state", {})
    pred_state = prediction.get("game_state", {})

    fields = [
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

    matches = 0
    total = 0
    details = {}

    for field in fields:
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

    return {
        "accuracy": matches / total if total > 0 else 0,
        "matches": matches,
        "total": total,
        "details": details,
    }


def evaluate(labels_dir: str, predictions_dir: str, verbose: bool = False):
    """Evaluate predictions against labels."""
    labels_path = Path(labels_dir)
    preds_path = Path(predictions_dir)

    labels = load_json_files(labels_path)
    predictions = load_json_files(preds_path)

    # Handle prediction file naming (might have _analysis suffix)
    pred_mapping = {}
    for key in predictions:
        clean_key = key.replace("_analysis", "")
        pred_mapping[clean_key] = predictions[key]
    predictions = pred_mapping

    # Find common keys
    common = set(labels.keys()) & set(predictions.keys())
    print(f"Found {len(common)} screenshots with both labels and predictions")
    print(f"  Labels only: {len(labels) - len(common)}")
    print(f"  Predictions only: {len(predictions) - len(common)}")

    if not common:
        print("No matching files to evaluate!")
        return

    # Evaluate each
    results = []
    for key in sorted(common):
        comparison = compare_game_state(labels[key], predictions[key])
        comparison["screenshot"] = key
        results.append(comparison)

        if verbose:
            print(f"\n{key}: {comparison['accuracy']:.1%} ({comparison['matches']}/{comparison['total']})")
            for field, detail in comparison["details"].items():
                status = "match" if detail["match"] else "MISMATCH"
                print(f"  {field}: {detail['label']} vs {detail['prediction']} [{status}]")

    # Summary statistics
    overall_matches = sum(r["matches"] for r in results)
    overall_total = sum(r["total"] for r in results)
    overall_accuracy = overall_matches / overall_total if overall_total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Overall accuracy: {overall_accuracy:.1%} ({overall_matches}/{overall_total})")

    # Per-field accuracy
    field_stats = {}
    for r in results:
        for field, detail in r["details"].items():
            if field not in field_stats:
                field_stats[field] = {"matches": 0, "total": 0}
            field_stats[field]["total"] += 1
            if detail["match"]:
                field_stats[field]["matches"] += 1

    print("\nPer-field accuracy:")
    for field, stats in sorted(field_stats.items(), key=lambda x: x[1]["matches"]/x[1]["total"] if x[1]["total"] > 0 else 0):
        acc = stats["matches"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {field}: {acc:.1%} ({stats['matches']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions against labels")
    parser.add_argument("--labels", type=str, default="data/labeled", help="Directory with ground truth labels")
    parser.add_argument("--predictions", type=str, default="data/predictions", help="Directory with model predictions")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-screenshot details")

    args = parser.parse_args()
    evaluate(args.labels, args.predictions, args.verbose)


if __name__ == "__main__":
    main()
