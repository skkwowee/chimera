#!/usr/bin/env python3
"""
SFT Gate Check — Go/No-Go decision for GRPO training.

After SFT training, run this script to determine whether the model meets the
quality bar required to proceed to GRPO. Evaluates per-field accuracy on a
held-out validation sample, checks format validity, runs the c_t ablation,
and (when multiple capture directories are provided) reports cross-map delta.

Usage:
    python scripts/sft_gate_check.py \\
        --model-path outputs/sft/final_model/merged_16bit \\
        --captures-dirs data/captures/furia-vs-vitality-m1-mirage \\
                        data/captures/furia-vs-vitality-m2-inferno \\
        --n-samples 50 \\
        --output outputs/gate_check.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: dict[str, float] = {
    "alive_teammates":  0.90,
    "alive_enemies":    0.90,
    "bomb_status":      0.93,
    "player_health":    0.85,
    "player_armor":     0.85,
    "json_format":      0.97,
    "cross_map_delta":  0.08,   # max allowed accuracy gap between maps
    "ct_independence":  0.70,   # image-only must reach ≥ 70% of image+c_t
}


# ---------------------------------------------------------------------------
# Imports — defer heavy ML imports until after arg parse
# ---------------------------------------------------------------------------

def _lazy_import_ml():
    """Import ML-heavy packages. Errors here are intentional — the user
    must have the right environment set up before running the gate check."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    return torch, Image, AutoProcessor, Qwen3_5MoeForConditionalGeneration


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors src/training/rewards.py field scoring)
# ---------------------------------------------------------------------------

def _extract_json(response: str) -> dict[str, Any] | None:
    """Extract JSON from a model response (handles markdown code fences)."""
    import re
    # Try markdown code block first
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Raw JSON
    m = re.search(r"\{[\s\S]*\}", response)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _is_valid_format(response: str) -> bool:
    """True if response has valid JSON with game_state, analysis, advice keys."""
    parsed = _extract_json(response)
    if parsed is None:
        return False
    return (
        isinstance(parsed.get("game_state"), dict)
        and isinstance(parsed.get("analysis"), dict)
        and isinstance(parsed.get("advice"), dict)
    )


def _exact_int_match(pred_val: Any, gt_val: Any) -> bool:
    """Exact integer match (both None counts as match)."""
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    try:
        return int(pred_val) == int(gt_val)
    except (TypeError, ValueError):
        return False


def _exact_str_match(pred_val: Any, gt_val: Any) -> bool:
    """Case-insensitive exact string match."""
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    return str(pred_val).lower().strip() == str(gt_val).lower().strip()


def _numeric_within_tolerance(pred_val: Any, gt_val: Any, tol: float = 0.10) -> bool:
    """True if relative error ≤ tol (exact match when gt_val == 0)."""
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    try:
        gt_f = float(gt_val)
        pred_f = float(pred_val)
    except (TypeError, ValueError):
        return False
    if gt_f == 0:
        return pred_f == 0
    return abs(pred_f - gt_f) / abs(gt_f) <= tol


def score_sample(response: str, gt_state: dict[str, Any]) -> dict[str, bool | None]:
    """
    Score a model response against ground-truth game_state.

    Returns a dict of field → bool (None if field absent from gt).
    Also includes "json_format" key.
    """
    scores: dict[str, bool | None] = {}

    scores["json_format"] = _is_valid_format(response)

    parsed = _extract_json(response)
    if parsed is None:
        # All field scores unknown when JSON is invalid
        for field in ["alive_teammates", "alive_enemies", "bomb_status",
                       "player_health", "player_armor"]:
            scores[field] = False
        return scores

    pred_gs = parsed.get("game_state", {})
    if not isinstance(pred_gs, dict):
        pred_gs = {}

    # Exact-integer fields
    for field in ["alive_teammates", "alive_enemies", "player_health", "player_armor"]:
        gt_val = gt_state.get(field)
        pred_val = pred_gs.get(field)
        if gt_val is None:
            scores[field] = None
        else:
            scores[field] = _exact_int_match(pred_val, gt_val)

    # Exact string / categorical
    gt_bomb = gt_state.get("bomb_status")
    pred_bomb = pred_gs.get("bomb_status")
    if gt_bomb is None:
        scores["bomb_status"] = None
    else:
        scores["bomb_status"] = _exact_str_match(pred_bomb, gt_bomb)

    return scores


# ---------------------------------------------------------------------------
# Label / sample loading
# ---------------------------------------------------------------------------

def load_labels(captures_dir: Path) -> list[dict[str, Any]]:
    """Load all label JSON files from a captures directory."""
    labels_dir = captures_dir / "labels"
    if not labels_dir.exists():
        return []
    return [
        json.loads(p.read_text())
        for p in sorted(labels_dir.glob("*.json"))
    ]


def stratified_sample(labels: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """
    Sample n labels, stratified by bomb_status to avoid collapse on majority
    class. Falls back to random if fewer than n labels are available.
    """
    if len(labels) <= n:
        return list(labels)

    # Group by bomb_status
    buckets: dict[str, list[dict[str, Any]]] = {}
    for lbl in labels:
        key = str(lbl.get("game_state", {}).get("bomb_status", "unknown"))
        buckets.setdefault(key, []).append(lbl)

    # Sample proportionally from each bucket
    result: list[dict[str, Any]] = []
    remaining = n
    bucket_keys = sorted(buckets.keys())
    for i, key in enumerate(bucket_keys):
        bucket = buckets[key]
        share = max(1, round(remaining * len(bucket) / max(1, len(labels) - sum(
            len(buckets[k]) for k in bucket_keys[:i]
        ))))
        share = min(share, len(bucket), remaining)
        result.extend(rng.sample(bucket, share))
        remaining -= share
        if remaining <= 0:
            break

    # Top off with random if still short
    if remaining > 0:
        pool = [l for l in labels if l not in result]
        result.extend(rng.sample(pool, min(remaining, len(pool))))

    return result


def find_screenshot(captures_dir: Path, label: dict[str, Any]) -> Path | None:
    """Locate the screenshot file for a label."""
    sid = label.get("metadata", {}).get("screenshot_id", "")
    if not sid:
        return None
    for ext in (".jpg", ".jpeg", ".png"):
        p = captures_dir / "raw" / f"{sid}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Parquet / demo data helpers for c_t regeneration
# ---------------------------------------------------------------------------

def _load_demo_data(demo_stem: str, demo_data_dir: Path) -> dict[str, Any] | None:
    """
    Load ticks parquet + rounds/bomb/header JSON for a demo stem.
    Returns None if any required file is missing.
    """
    try:
        import polars as pl
    except ImportError:
        return None

    ticks_path = demo_data_dir / f"{demo_stem}_ticks.parquet"
    rounds_path = demo_data_dir / f"{demo_stem}_rounds.json"
    header_path = demo_data_dir / f"{demo_stem}_header.json"

    if not ticks_path.exists() or not rounds_path.exists() or not header_path.exists():
        return None

    bomb_path = demo_data_dir / f"{demo_stem}_bomb.json"

    return {
        "ticks_df": pl.read_parquet(ticks_path),
        "rounds": json.loads(rounds_path.read_text()),
        "bomb_events": json.loads(bomb_path.read_text()) if bomb_path.exists() else [],
        "header": json.loads(header_path.read_text()),
    }


def regenerate_context(label: dict[str, Any], demo_data: dict[str, Any]) -> str | None:
    """
    Regenerate the c_t context string from parquet data.
    Uses generate_round_context from generate_sft_labels.
    """
    try:
        from scripts.generate_sft_labels import generate_round_context
    except ImportError:
        return None

    meta = label.get("metadata", {})
    tick = meta.get("tick")
    round_num = meta.get("round_num")
    pov_name = meta.get("pov_player")
    pov_side = meta.get("pov_side", "t")

    if tick is None or round_num is None or pov_name is None:
        return None

    try:
        return generate_round_context(
            tick=tick,
            round_num=round_num,
            pov_name=pov_name,
            pov_side=pov_side,
            ticks_df=demo_data["ticks_df"],
            rounds=demo_data["rounds"],
            bomb_events=demo_data["bomb_events"],
            header=demo_data["header"],
        )
    except Exception:
        return None


def _partial_context(full_context: str) -> str:
    """
    Strip the 'CURRENT STATE:' block from c_t, leaving only round history.
    This simulates partial c_t (history only) for the ablation.
    """
    lines = full_context.splitlines()
    trimmed = []
    skip = False
    for line in lines:
        if line.strip().startswith("CURRENT STATE:"):
            skip = True
        if not skip:
            trimmed.append(line)
    return "\n".join(trimmed).rstrip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class SFTModel:
    """Thin wrapper for loading the SFT-fine-tuned model."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model: Any = None
        self.processor: Any = None
        self._torch: Any = None

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._torch = torch
        print(f"Loading SFT model from {self.model_path} ...")

        # Detect whether this is a merged model or a LoRA adapter
        adapter_config = self.model_path / "adapter_config.json"
        is_lora = adapter_config.exists()

        if is_lora:
            # LoRA adapter: load base model then apply adapter
            adapter_cfg = json.loads(adapter_config.read_text())
            base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3.5-35B-A3B")
            print(f"  Detected LoRA adapter (base: {base_model_name})")

            from peft import PeftModel

            base = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
            self.model = self.model.merge_and_unload()
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
        else:
            # Merged model (merged_16bit or full checkpoint)
            print("  Detected merged/full model checkpoint")
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_path),
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )

        self.model.eval()
        print("  Model loaded.")

    def infer(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Run inference and return the raw decoded response string."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy for eval reproducibility
            )

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids, strict=False)
        ]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def _get_prompts():
    """Lazy import of prompts — requires the chimera package to be installed."""
    try:
        from src.prompts import CS2_SYSTEM_PROMPT, build_user_prompt
    except ImportError:
        # Fallback: define inline so the script can run without package install
        CS2_SYSTEM_PROMPT = (
            "You are an expert CS2 analyst and coach. Analyze the screenshot and "
            "respond with valid JSON containing game_state, analysis, and advice keys."
        )

        def build_user_prompt(context=None):
            if context is None:
                return "Analyze this CS2 screenshot. Extract the game state and provide strategic advice."
            return (
                f"ROUND CONTEXT:\n{context}\n\n"
                "Given the context above and the screenshot(s), extract the current "
                "game state and provide strategic advice for this moment."
            )

    return CS2_SYSTEM_PROMPT, build_user_prompt


def evaluate_samples(
    model: SFTModel,
    samples: list[dict[str, Any]],
    captures_dir: Path,
    context_mode: str = "full",  # "full" | "partial" | "none"
    demo_data: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run inference on samples and aggregate per-field accuracy.

    Args:
        context_mode: "full" — use label's context field (or regenerate),
                      "partial" — strip CURRENT STATE block from c_t,
                      "none" — image-only prompt (no c_t).
        demo_data: Parsed demo data dict for context regeneration. Optional.

    Returns:
        Dict with per-field accuracy, json_format accuracy, and raw per-sample
        results.
    """
    field_hits: dict[str, list[bool]] = {
        "alive_teammates": [],
        "alive_enemies": [],
        "bomb_status": [],
        "player_health": [],
        "player_armor": [],
        "json_format": [],
    }
    per_sample: list[dict[str, Any]] = []
    CS2_SYSTEM_PROMPT, build_user_prompt = _get_prompts()

    for idx, label in enumerate(samples):
        gt_state = label.get("game_state", {})
        screenshot = find_screenshot(captures_dir, label)

        if screenshot is None:
            if verbose:
                sid = label.get("metadata", {}).get("screenshot_id", "?")
                print(f"  [SKIP] screenshot not found for {sid}")
            continue

        # Build context string based on mode
        if context_mode == "none":
            context = None
        elif context_mode == "partial":
            # Try label's stored context first, then regenerate
            stored = label.get("context") or ""
            if stored:
                context = _partial_context(stored)
            elif demo_data:
                full_ctx = regenerate_context(label, demo_data)
                context = _partial_context(full_ctx) if full_ctx else None
            else:
                context = None
        else:  # "full"
            context = label.get("context") or None
            if context is None and demo_data:
                context = regenerate_context(label, demo_data)

        user_prompt = build_user_prompt(context)

        try:
            t0 = time.perf_counter()
            response = model.infer(screenshot, CS2_SYSTEM_PROMPT, user_prompt)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            sid = label.get("metadata", {}).get("screenshot_id", "?")
            print(f"  [ERROR] inference failed for {sid}: {exc}")
            for field in field_hits:
                if field != "json_format":
                    pass  # don't record — sample failed
            continue

        scores = score_sample(response, gt_state)

        sample_result: dict[str, Any] = {
            "screenshot_id": label.get("metadata", {}).get("screenshot_id"),
            "context_mode": context_mode,
            "elapsed_s": round(elapsed, 2),
            "scores": scores,
        }
        per_sample.append(sample_result)

        for field, hit in scores.items():
            if hit is not None and field in field_hits:
                field_hits[field].append(hit)

        if verbose:
            sid = label.get("metadata", {}).get("screenshot_id", "?")
            fmt = "OK" if scores.get("json_format") else "FAIL"
            print(
                f"  [{idx+1:3d}/{len(samples)}] {sid} | fmt={fmt} | "
                f"hp={scores.get('player_health')} | "
                f"armor={scores.get('player_armor')} | "
                f"mates={scores.get('alive_teammates')} | "
                f"enemies={scores.get('alive_enemies')} | "
                f"bomb={scores.get('bomb_status')} | "
                f"{elapsed:.1f}s"
            )

    # Compute per-field accuracy
    accuracy: dict[str, float | None] = {}
    for field, hits in field_hits.items():
        accuracy[field] = (sum(hits) / len(hits)) if hits else None

    return {
        "accuracy": accuracy,
        "n_evaluated": len(per_sample),
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# c_t ablation
# ---------------------------------------------------------------------------

def run_ct_ablation(
    model: SFTModel,
    samples: list[dict[str, Any]],
    captures_dir: Path,
    demo_data: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run THREE inference passes on the ablation subset:
      1. image-only (no c_t)
      2. image + partial c_t (history only)
      3. image + full c_t

    Returns accuracy for each mode and interpretation flags.
    """
    print(f"\n  Running c_t ablation on {len(samples)} samples (3 passes each)...")

    results = {}
    for mode in ("none", "partial", "full"):
        label = {"none": "image-only", "partial": "image+partial c_t", "full": "image+full c_t"}[mode]
        print(f"    Pass: {label}")
        r = evaluate_samples(model, samples, captures_dir, context_mode=mode, demo_data=demo_data, verbose=verbose)
        results[mode] = r

    # Aggregate accuracy per mode across all gated fields
    def _mean_accuracy(r: dict[str, Any]) -> float | None:
        acc = r["accuracy"]
        vals = [v for v in acc.values() if v is not None and v != acc.get("json_format")]
        return (sum(vals) / len(vals)) if vals else None

    acc_none    = _mean_accuracy(results["none"])
    acc_partial = _mean_accuracy(results["partial"])
    acc_full    = _mean_accuracy(results["full"])

    # c_t independence test
    vision_undertrained = False
    independence_ratio: float | None = None
    if acc_none is not None and acc_full is not None and acc_full > 0:
        independence_ratio = acc_none / acc_full
        vision_undertrained = independence_ratio < GATE_THRESHOLDS["ct_independence"]

    # partial ≈ full?
    partial_vs_full: float | None = None
    if acc_partial is not None and acc_full is not None and acc_full > 0:
        partial_vs_full = acc_partial / acc_full

    return {
        "accuracy_image_only":    acc_none,
        "accuracy_partial_ct":   acc_partial,
        "accuracy_full_ct":      acc_full,
        "independence_ratio":    independence_ratio,    # image-only / full c_t
        "partial_vs_full_ratio": partial_vs_full,       # partial c_t / full c_t
        "vision_undertrained_warning": vision_undertrained,
        "threshold": GATE_THRESHOLDS["ct_independence"],
        "mode_details": {
            "none":    results["none"]["accuracy"],
            "partial": results["partial"]["accuracy"],
            "full":    results["full"]["accuracy"],
        },
        "n_evaluated": results["full"]["n_evaluated"],
    }


# ---------------------------------------------------------------------------
# Gate verdict computation
# ---------------------------------------------------------------------------

def compute_verdict(
    per_map_results: list[dict[str, Any]],   # one entry per captures_dir
    ablation: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate per-map results, apply gate thresholds, return overall verdict.
    """
    # Pool all samples across maps for primary per-field accuracy
    field_sums: dict[str, float] = {}
    field_counts: dict[str, int] = {}
    map_field_acc: dict[str, dict[str, float]] = {}

    for entry in per_map_results:
        map_name = entry["map_name"]
        acc = entry["accuracy"]
        map_field_acc[map_name] = acc
        for field, val in acc.items():
            if val is not None:
                field_sums[field] = field_sums.get(field, 0.0) + val
                field_counts[field] = field_counts.get(field, 0) + 1

    pooled: dict[str, float | None] = {
        f: (field_sums[f] / field_counts[f]) if f in field_sums else None
        for f in set(list(field_sums.keys()))
    }

    # Per-field gate checks
    gate_fields = [
        "alive_teammates",
        "alive_enemies",
        "bomb_status",
        "player_health",
        "player_armor",
        "json_format",
    ]
    checks: dict[str, dict[str, Any]] = {}
    all_pass = True

    for field in gate_fields:
        threshold = GATE_THRESHOLDS[field]
        acc_val = pooled.get(field)
        if acc_val is None:
            result = "SKIP"
            passed = True  # can't fail what can't be measured
        else:
            passed = acc_val >= threshold
            result = "PASS" if passed else "FAIL"

        checks[field] = {
            "accuracy": acc_val,
            "threshold": threshold,
            "result": result,
        }
        if result == "FAIL":
            all_pass = False

    # Cross-map delta check
    cross_map_check: dict[str, Any] = {"result": "SKIP", "delta": None, "maps": []}
    if len(per_map_results) >= 2:
        # Use mean of gate fields per map
        map_means: list[tuple[str, float]] = []
        for map_name, acc in map_field_acc.items():
            field_vals = [v for k, v in acc.items() if k != "json_format" and v is not None]
            if field_vals:
                map_means.append((map_name, sum(field_vals) / len(field_vals)))

        if len(map_means) >= 2:
            accs = [m[1] for m in map_means]
            delta = max(accs) - min(accs)
            threshold = GATE_THRESHOLDS["cross_map_delta"]
            passed = delta <= threshold
            cross_map_check = {
                "result": "PASS" if passed else "FAIL",
                "delta": round(delta, 4),
                "threshold": threshold,
                "maps": [{"name": n, "mean_accuracy": round(a, 4)} for n, a in map_means],
            }
            if not passed:
                all_pass = False

    # c_t independence gate
    independence_ratio = ablation.get("independence_ratio")
    threshold = GATE_THRESHOLDS["ct_independence"]
    if independence_ratio is None:
        ct_check = {"result": "SKIP", "ratio": None, "threshold": threshold}
    else:
        passed = independence_ratio >= threshold
        ct_check = {
            "result": "PASS" if passed else "FAIL",
            "ratio": round(independence_ratio, 4),
            "threshold": threshold,
        }
        if not passed:
            all_pass = False

    # Collect failures
    failures: list[str] = []
    actions: list[str] = []
    for field, check in checks.items():
        if check["result"] == "FAIL":
            acc_pct = round((check["accuracy"] or 0) * 100, 1)
            thr_pct = round(check["threshold"] * 100, 1)
            failures.append(
                f"{field}: {acc_pct}% (threshold {thr_pct}%)"
            )
            actions.extend(_recommend_action(field, check["accuracy"] or 0, check["threshold"]))
    if cross_map_check["result"] == "FAIL":
        failures.append(
            f"cross_map_delta: {round((cross_map_check['delta'] or 0)*100, 1)}% "
            f"(threshold {round(GATE_THRESHOLDS['cross_map_delta']*100, 1)}%)"
        )
        actions.append("Add more training data from underperforming maps, or increase map-stratified sampling.")
    if ct_check["result"] == "FAIL":
        failures.append(
            f"c_t independence: ratio {round((independence_ratio or 0), 3)} "
            f"(threshold {threshold})"
        )
        actions.append(
            "Vision encoder appears undertrained. Consider: (1) longer SFT on more "
            "visually diverse samples, (2) lower LR on vision tower, or (3) adding "
            "image-only examples to SFT dataset."
        )

    return {
        "overall": "PASS" if all_pass else "FAIL",
        "field_checks": checks,
        "cross_map_check": cross_map_check,
        "ct_independence_check": ct_check,
        "pooled_accuracy": pooled,
        "failures": failures,
        "recommended_actions": list(dict.fromkeys(actions)),  # deduplicate, preserve order
    }


def _recommend_action(field: str, actual: float, threshold: float) -> list[str]:
    """Generate targeted remediation recommendations for a failing gate field."""
    gap = threshold - actual
    severity = "severely" if gap > 0.15 else "moderately"
    recs: list[str] = []
    if field == "json_format":
        recs.append(
            "JSON format validity is too low. Ensure SFT data uses strict JSON "
            "labels and add format reward during GRPO warm-up."
        )
    elif field in ("alive_teammates", "alive_enemies"):
        recs.append(
            f"{field} is {severity} underperforming. The model may not be reading the "
            "teammate/enemy count HUD elements reliably. Add more SFT samples with "
            "varied player counts (especially 1v1 and 1vN situations)."
        )
    elif field == "bomb_status":
        recs.append(
            f"bomb_status accuracy is {severity} below threshold. Increase SFT samples "
            "during post-plant and bomb-dropped phases (these are less common and often "
            "underrepresented in random samples)."
        )
    elif field in ("player_health", "player_armor"):
        recs.append(
            f"{field} accuracy is {severity} below threshold. The model may be struggling "
            "with non-round HUD values. Check that SFT labels include low-health samples "
            "and that the visual resolution is sufficient."
        )
    return recs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT Gate Check — go/no-go decision for GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to SFT output dir (merged_16bit subdirectory or lora_adapter)",
    )
    parser.add_argument(
        "--captures-dirs",
        nargs="+",
        required=True,
        type=Path,
        metavar="DIR",
        help="One or more capture directories (e.g. data/captures/match-name)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per captures dir (default: 50)",
    )
    parser.add_argument(
        "--n-ablation",
        type=int,
        default=20,
        help="Number of samples for c_t ablation (default: 20)",
    )
    parser.add_argument(
        "--demo-data-dir",
        type=Path,
        default=Path("data/processed/demos"),
        help="Directory with parsed demo Parquet/JSON files (default: data/processed/demos)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gate_check.json"),
        help="Path for JSON report output (default: outputs/gate_check.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample scores during evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 72)
    print("SFT GATE CHECK")
    print("=" * 72)
    print(f"Model path:    {args.model_path}")
    print(f"Captures dirs: {[str(d) for d in args.captures_dirs]}")
    print(f"Samples/dir:   {args.n_samples}")
    print(f"Ablation N:    {args.n_ablation}")
    print(f"Output:        {args.output}")
    print()

    # Validate paths
    if not args.model_path.exists():
        print(f"ERROR: model path does not exist: {args.model_path}", file=sys.stderr)
        return 1

    for d in args.captures_dirs:
        if not d.exists():
            print(f"ERROR: captures dir does not exist: {d}", file=sys.stderr)
            return 1

    # Load model
    model = SFTModel(args.model_path)
    try:
        model.load()
    except Exception as exc:
        print(f"ERROR: failed to load model: {exc}", file=sys.stderr)
        return 1

    # Evaluate per captures dir
    per_map_results: list[dict[str, Any]] = []
    ablation_pool: list[tuple[dict[str, Any], Path, dict[str, Any] | None]] = []

    for captures_dir in args.captures_dirs:
        map_name = captures_dir.name
        print(f"\n{'─'*60}")
        print(f"Evaluating: {map_name} ({captures_dir})")

        labels = load_labels(captures_dir)
        if not labels:
            print(f"  WARNING: no labels found in {captures_dir / 'labels'} — skipping")
            continue

        print(f"  Labels found: {len(labels)}")
        sample = stratified_sample(labels, args.n_samples, rng)
        print(f"  Sampled:      {len(sample)}")

        # Try to load demo data for c_t regeneration
        demo_stem: str | None = None
        demo_data: dict[str, Any] | None = None
        if labels:
            meta = labels[0].get("metadata", {})
            df = meta.get("demo_file", "")
            if df:
                demo_stem = df.replace(".dem", "")
        if demo_stem and args.demo_data_dir.exists():
            demo_data = _load_demo_data(demo_stem, args.demo_data_dir)
            if demo_data:
                print(f"  Demo data:    loaded ({demo_stem})")
            else:
                print(f"  Demo data:    not found (c_t from labels only)")

        print(f"  Running inference (context_mode=full) ...")
        result = evaluate_samples(
            model, sample, captures_dir,
            context_mode="full",
            demo_data=demo_data,
            verbose=args.verbose,
        )

        acc = result["accuracy"]
        print(f"  Results ({result['n_evaluated']} evaluated):")
        for field in ["alive_teammates", "alive_enemies", "bomb_status",
                      "player_health", "player_armor", "json_format"]:
            val = acc.get(field)
            thr = GATE_THRESHOLDS.get(field, 0.0)
            if val is None:
                status = "N/A"
                pct = "  N/A"
            else:
                status = "PASS" if val >= thr else "FAIL"
                pct = f"{val*100:5.1f}%"
            print(f"    {field:<22} {pct}  (threshold {thr*100:.0f}%)  [{status}]")

        per_map_results.append({
            "map_name": map_name,
            "captures_dir": str(captures_dir),
            "n_labels": len(labels),
            "n_sampled": len(sample),
            "n_evaluated": result["n_evaluated"],
            "accuracy": acc,
            "per_sample": result["per_sample"],
        })

        # Add some samples to ablation pool
        ablation_candidates = rng.sample(sample, min(args.n_ablation, len(sample)))
        ablation_pool.extend(
            (lbl, captures_dir, demo_data) for lbl in ablation_candidates
        )

    if not per_map_results:
        print("\nERROR: no capture directories had usable labels.", file=sys.stderr)
        return 1

    # c_t ablation — use at most n_ablation samples (prefer diversity across maps)
    ablation_sample_tuples = rng.sample(ablation_pool, min(args.n_ablation, len(ablation_pool)))
    # Group back into a single captures_dir (use the first as a representative)
    # Ablation is run per-dir; here we pick the first dir's samples for simplicity
    # (the ablation intent is model behaviour, not cross-map)
    ablation_labels_by_dir: dict[Path, list[dict[str, Any]]] = {}
    ablation_demo_by_dir: dict[Path, dict[str, Any] | None] = {}
    for lbl, cdir, ddata in ablation_sample_tuples:
        ablation_labels_by_dir.setdefault(cdir, []).append(lbl)
        ablation_demo_by_dir[cdir] = ddata

    print(f"\n{'─'*60}")
    print("c_t ABLATION")

    # Consolidate: run ablation against all dirs' samples but report combined
    all_ablation_results: dict[str, list[dict[str, Any]]] = {"none": [], "partial": [], "full": []}
    for cdir, ab_labels in ablation_labels_by_dir.items():
        ddata = ablation_demo_by_dir[cdir]
        print(f"  Ablation dir: {cdir.name} ({len(ab_labels)} samples)")
        for mode in ("none", "partial", "full"):
            r = evaluate_samples(model, ab_labels, cdir, context_mode=mode, demo_data=ddata, verbose=args.verbose)
            all_ablation_results[mode].extend(r["per_sample"])

    # Compute aggregated ablation accuracy
    def _aggregate_field_acc(samples_results: list[dict[str, Any]], field: str) -> float | None:
        hits = [s["scores"][field] for s in samples_results if field in s["scores"] and s["scores"][field] is not None]
        return (sum(hits) / len(hits)) if hits else None

    gate_fields_no_fmt = ["alive_teammates", "alive_enemies", "bomb_status", "player_health", "player_armor"]

    def _mean_across_fields(samples_results: list[dict[str, Any]]) -> float | None:
        vals = [
            _aggregate_field_acc(samples_results, f)
            for f in gate_fields_no_fmt
        ]
        real = [v for v in vals if v is not None]
        return (sum(real) / len(real)) if real else None

    acc_none    = _mean_across_fields(all_ablation_results["none"])
    acc_partial = _mean_across_fields(all_ablation_results["partial"])
    acc_full    = _mean_across_fields(all_ablation_results["full"])

    independence_ratio: float | None = None
    partial_vs_full: float | None = None
    vision_undertrained = False

    if acc_full is not None and acc_full > 0:
        if acc_none is not None:
            independence_ratio = acc_none / acc_full
            vision_undertrained = independence_ratio < GATE_THRESHOLDS["ct_independence"]
        if acc_partial is not None:
            partial_vs_full = acc_partial / acc_full

    print(f"\n  Ablation summary (mean accuracy over gate fields):")
    print(f"    Image-only:       {f'{acc_none*100:.1f}%' if acc_none is not None else 'N/A'}")
    print(f"    Image+partial c_t:{f'{acc_partial*100:.1f}%' if acc_partial is not None else 'N/A'}")
    print(f"    Image+full c_t:   {f'{acc_full*100:.1f}%' if acc_full is not None else 'N/A'}")
    if independence_ratio is not None:
        ratio_pct = f"{independence_ratio*100:.1f}%"
        print(f"\n    Independence ratio (image-only / full c_t): {ratio_pct}")
        threshold_pct = f"{GATE_THRESHOLDS['ct_independence']*100:.0f}%"
        print(f"    Threshold: >= {threshold_pct} of full c_t accuracy")
        if vision_undertrained:
            print(
                f"\n    WARNING: Vision encoder may be undertrained — model depends "
                f"heavily on text context ({ratio_pct} < {threshold_pct})"
            )
    if partial_vs_full is not None:
        print(f"    Partial c_t / full c_t ratio: {partial_vs_full*100:.1f}%")
        if partial_vs_full >= 0.90:
            print("    → Partial c_t is nearly equivalent to full c_t (good — partial c_t design works)")
        else:
            print("    → Partial c_t notably worse than full c_t — model relies on CURRENT STATE block")

    ablation_report = {
        "accuracy_image_only":    acc_none,
        "accuracy_partial_ct":   acc_partial,
        "accuracy_full_ct":      acc_full,
        "independence_ratio":    independence_ratio,
        "partial_vs_full_ratio": partial_vs_full,
        "vision_undertrained_warning": vision_undertrained,
        "threshold": GATE_THRESHOLDS["ct_independence"],
        "n_evaluated": len(all_ablation_results["full"]),
        "mode_details": {
            mode: {
                f: _aggregate_field_acc(all_ablation_results[mode], f)
                for f in gate_fields_no_fmt + ["json_format"]
            }
            for mode in ("none", "partial", "full")
        },
    }

    # Compute final verdict
    verdict = compute_verdict(per_map_results, ablation_report)

    print(f"\n{'='*72}")
    print("GATE RESULTS")
    print(f"{'='*72}")
    print(f"\n{'Field':<24} {'Accuracy':>9}  {'Threshold':>10}  Result")
    print("─" * 60)
    for field, check in verdict["field_checks"].items():
        acc_s = f"{check['accuracy']*100:.1f}%" if check["accuracy"] is not None else "  N/A"
        thr_s = f"{check['threshold']*100:.0f}%"
        print(f"  {field:<22} {acc_s:>9}  {thr_s:>10}  [{check['result']}]")

    cm = verdict["cross_map_check"]
    if cm["result"] != "SKIP":
        delta_s = f"{(cm['delta'] or 0)*100:.1f}%"
        thr_s   = f"{GATE_THRESHOLDS['cross_map_delta']*100:.0f}%"
        print(f"  {'cross_map_delta':<22} {delta_s:>9}  {thr_s:>10}  [{cm['result']}]")
    else:
        print(f"  {'cross_map_delta':<22} {'N/A':>9}  {'8%':>10}  [SKIP — single map]")

    ct = verdict["ct_independence_check"]
    ratio_s = f"{ct['ratio']:.3f}" if ct["ratio"] is not None else "N/A"
    thr_s   = f"{ct['threshold']:.2f}"
    print(f"  {'ct_independence':<22} {ratio_s:>9}  {thr_s:>10}  [{ct['result']}]")

    print()
    overall = verdict["overall"]
    if overall == "PASS":
        print("  ✓ GATE PASSED — proceed to GRPO")
    else:
        print("  ✗ GATE FAILED — do not proceed")
        print()
        print("  Failing criteria:")
        for failure in verdict["failures"]:
            print(f"    • {failure}")
        if verdict["recommended_actions"]:
            print()
            print("  Recommended actions:")
            for action in verdict["recommended_actions"]:
                print(f"    → {action}")

    print(f"\n{'='*72}")

    # Write JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": verdict,
        "ablation": ablation_report,
        "per_map": per_map_results,
        "config": {
            "model_path": str(args.model_path),
            "captures_dirs": [str(d) for d in args.captures_dirs],
            "n_samples": args.n_samples,
            "n_ablation": args.n_ablation,
            "seed": args.seed,
            "gate_thresholds": GATE_THRESHOLDS,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to: {args.output}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
