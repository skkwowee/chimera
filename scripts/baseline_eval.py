#!/usr/bin/env python3
"""
Baseline evaluation of VLM zero-shot (or SFT) performance on CS2 HUD screenshot reading.

Evaluates how well the model reads structured game state from CS2 screenshots
without any round context (image only). This establishes baseline numbers that
drive the SFT curriculum — which fields need the most improvement, and whether
the model clears the gate thresholds required for production.

Usage:
    # Default: sample 50 screenshots across all captures
    uv run python scripts/baseline_eval.py

    # All screenshots from a specific match
    uv run python scripts/baseline_eval.py \\
        --screenshots data/captures/furia-vs-vitality-m1-mirage \\
        --n-samples 0

    # Custom model + larger sample
    uv run python scripts/baseline_eval.py \\
        --model-name Qwen/Qwen3.5-35B-A3B \\
        --n-samples 100 \\
        --output outputs/baseline_35b.json

    # Few-shot evaluation (3 examples in prompt)
    uv run python scripts/baseline_eval.py --few-shot 3
"""

import torch._dynamo

torch._dynamo.config.disable = True

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from src.prompts import CS2_PERCEPTION_SYSTEM_PROMPT, CS2_PERCEPTION_USER_PROMPT
from src.utils.config import DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------
# Gate thresholds — pass/fail criteria for the model to be "eval-ready"
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: dict[str, float] = {
    "format_valid": 0.97,
    "alive_teammates_exact": 0.90,
    "alive_enemies_exact": 0.90,
    "bomb_status_exact": 0.93,
    "player_health_exact": 0.85,
}


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json(response: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from raw model output.

    Handles:
      - Markdown code blocks (```json ... ```)
      - Bare JSON objects embedded in reasoning text
      - Partial/truncated JSON (best-effort)
    """
    # 1. Try markdown code block first
    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try the largest JSON-looking chunk in the response
    # Find the last '{' that opens a top-level object (greedy match)
    brace_match = re.search(r"\{[\s\S]*\}", response)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Find the last complete top-level object by scanning braces
    depth = 0
    start = None
    for i, ch in enumerate(response):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(response[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


# ---------------------------------------------------------------------------
# Field normalization helpers
# ---------------------------------------------------------------------------

def _normalize_weapon(w: str | None) -> str | None:
    """Normalize weapon name for fuzzy comparison."""
    if w is None:
        return None
    return re.sub(r"[\s\-_]", "", str(w).lower())


def _fuzzy_weapon_match(pred: str | None, gt: str | None) -> float:
    """
    Fuzzy weapon match: normalize then check substring containment.

    'ak47' matches 'AK-47', 'ak-47', 'AK47' etc.
    Returns 1.0 on match, 0.0 otherwise.
    """
    p = _normalize_weapon(pred)
    g = _normalize_weapon(gt)
    if p is None and g is None:
        return 1.0
    if p is None or g is None:
        return 0.0
    return 1.0 if (p in g or g in p) else 0.0


def _normalize_map(name: str | None) -> str | None:
    """Strip 'de_' prefix, lowercase."""
    if name is None:
        return None
    s = str(name).lower().strip()
    if s.startswith("de_"):
        s = s[3:]
    return s


def _utility_jaccard(pred_list: list | None, gt_list: list | None) -> float:
    """Jaccard similarity between two utility lists (case-insensitive)."""
    if not isinstance(pred_list, list):
        pred_list = []
    if not isinstance(gt_list, list):
        gt_list = []
    pred_set = {re.sub(r"[\s\-_]", "", x.lower()) for x in pred_list}
    gt_set   = {re.sub(r"[\s\-_]", "", x.lower()) for x in gt_list}
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    return len(pred_set & gt_set) / len(pred_set | gt_set)


def _utility_exact(pred_list: list | None, gt_list: list | None) -> float:
    """Exact match for utility lists (order-independent, case-insensitive)."""
    if not isinstance(pred_list, list):
        pred_list = []
    if not isinstance(gt_list, list):
        gt_list = []
    pred_set = {re.sub(r"[\s\-_]", "", x.lower()) for x in pred_list}
    gt_set   = {re.sub(r"[\s\-_]", "", x.lower()) for x in gt_list}
    return 1.0 if pred_set == gt_set else 0.0


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

def score_prediction(
    pred_state: dict[str, Any] | None,
    gt_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare predicted game_state against ground-truth game_state.

    Returns a dict with a score (0.0–1.0) for every evaluated field,
    plus combined metrics like health_tol5, money_tol500, etc.
    """
    scores: dict[str, Any] = {}

    if pred_state is None:
        # Model produced no valid game_state — zero everything out
        zero_fields = [
            "player_health_exact", "player_health_tol5",
            "player_armor_exact", "player_armor_tol5",
            "player_side", "weapon_primary", "weapon_secondary",
            "utility_jaccard", "utility_exact",
            "alive_teammates_exact", "alive_enemies_exact",
            "bomb_status", "map_name",
            "round_phase", "score_t", "score_ct",
            "player_has_helmet", "has_defuser",
            "player_money_tol500",
        ]
        return {f: 0.0 for f in zero_fields}

    gs = pred_state

    # --- player_health ---
    gt_hp = gt_state.get("player_health")
    pred_hp = gs.get("player_health")
    if gt_hp is not None:
        try:
            exact = 1.0 if int(pred_hp) == int(gt_hp) else 0.0
            tol   = 1.0 if abs(int(pred_hp) - int(gt_hp)) <= 5 else 0.0
        except (TypeError, ValueError):
            exact = tol = 0.0
        scores["player_health_exact"] = exact
        scores["player_health_tol5"]  = tol

    # --- player_armor ---
    gt_ar = gt_state.get("player_armor")
    pred_ar = gs.get("player_armor")
    if gt_ar is not None:
        try:
            exact = 1.0 if int(pred_ar) == int(gt_ar) else 0.0
            tol   = 1.0 if abs(int(pred_ar) - int(gt_ar)) <= 5 else 0.0
        except (TypeError, ValueError):
            exact = tol = 0.0
        scores["player_armor_exact"] = exact
        scores["player_armor_tol5"]  = tol

    # --- player_side (normalize to uppercase T/CT) ---
    gt_side   = str(gt_state.get("player_side", "")).upper().strip() if gt_state.get("player_side") else None
    pred_side = str(gs.get("player_side", "")).upper().strip() if gs.get("player_side") else None
    if gt_side is not None:
        scores["player_side"] = 1.0 if pred_side == gt_side else 0.0

    # --- weapon_primary / weapon_secondary (fuzzy) ---
    for wf in ("weapon_primary", "weapon_secondary"):
        gt_w   = gt_state.get(wf)
        pred_w = gs.get(wf)
        scores[wf] = _fuzzy_weapon_match(pred_w, gt_w)

    # --- utility (Jaccard + exact) ---
    gt_util   = gt_state.get("utility", [])
    pred_util = gs.get("utility", [])
    scores["utility_jaccard"] = _utility_jaccard(pred_util, gt_util)
    scores["utility_exact"]   = _utility_exact(pred_util, gt_util)

    # --- alive_teammates / alive_enemies (exact — CRITICAL) ---
    for cnt_field in ("alive_teammates", "alive_enemies"):
        gt_cnt   = gt_state.get(cnt_field)
        pred_cnt = gs.get(cnt_field)
        if gt_cnt is not None:
            try:
                scores[f"{cnt_field}_exact"] = 1.0 if int(pred_cnt) == int(gt_cnt) else 0.0
            except (TypeError, ValueError):
                scores[f"{cnt_field}_exact"] = 0.0

    # --- bomb_status (normalize to lowercase) ---
    gt_bomb   = str(gt_state.get("bomb_status", "")).lower().strip() if gt_state.get("bomb_status") else None
    pred_bomb = str(gs.get("bomb_status", "")).lower().strip() if gs.get("bomb_status") else None
    if gt_bomb is not None:
        scores["bomb_status"] = 1.0 if pred_bomb == gt_bomb else 0.0

    # --- map_name (fuzzy: strip de_, lowercase) ---
    gt_map   = _normalize_map(gt_state.get("map_name"))
    pred_map = _normalize_map(gs.get("map_name"))
    if gt_map is not None:
        if pred_map is None:
            scores["map_name"] = 0.0
        else:
            scores["map_name"] = 1.0 if (pred_map in gt_map or gt_map in pred_map) else 0.0

    # --- round_phase (lowercase exact) ---
    gt_rp   = str(gt_state.get("round_phase", "")).lower().strip() if gt_state.get("round_phase") else None
    pred_rp = str(gs.get("round_phase", "")).lower().strip() if gs.get("round_phase") else None
    if gt_rp is not None:
        scores["round_phase"] = 1.0 if pred_rp == gt_rp else 0.0

    # --- score_t / score_ct (exact) ---
    for sf in ("score_t", "score_ct"):
        gt_s   = gt_state.get(sf)
        pred_s = gs.get(sf)
        if gt_s is not None:
            try:
                scores[sf] = 1.0 if int(pred_s) == int(gt_s) else 0.0
            except (TypeError, ValueError):
                scores[sf] = 0.0

    # --- player_has_helmet / has_defuser (boolean exact) ---
    for bf in ("player_has_helmet", "has_defuser"):
        gt_b   = gt_state.get(bf)
        pred_b = gs.get(bf)
        if gt_b is not None:
            if isinstance(pred_b, str):
                pred_b = pred_b.lower() in ("true", "1", "yes")
            scores[bf] = 1.0 if bool(pred_b) == bool(gt_b) else 0.0

    # --- player_money (±500 tolerance) ---
    gt_money   = gt_state.get("player_money")
    pred_money = gs.get("player_money")
    if gt_money is not None:
        try:
            scores["player_money_tol500"] = 1.0 if abs(int(pred_money) - int(gt_money)) <= 500 else 0.0
        except (TypeError, ValueError):
            scores["player_money_tol500"] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def load_all_samples(captures_dir: Path, labels_dir: Path | None) -> list[dict[str, Any]]:
    """
    Discover all (screenshot, label) pairs under captures_dir.

    Searches both captures_dir/*/raw/ and captures_dir/*/screenshots/ layouts.
    If labels_dir is given explicitly, use that; otherwise look for a sibling
    labels/ dir next to each raw/ or screenshots/ directory.
    """
    samples: list[dict[str, Any]] = []

    if labels_dir is not None:
        # Explicit labels dir: pair each label with its screenshot
        label_files = sorted(labels_dir.glob("*.json"))
        for lf in label_files:
            stem = lf.stem
            # Find corresponding image in captures_dir
            image = None
            for ext in (".jpg", ".jpeg", ".png"):
                for sub in ("raw", "screenshots", ""):
                    candidate = (captures_dir / sub / (stem + ext)) if sub else (captures_dir / (stem + ext))
                    if candidate.exists():
                        image = candidate
                        break
                if image:
                    break
            if image is None:
                continue
            label_data = json.loads(lf.read_text())
            samples.append({"image": image, "label": label_data, "stem": stem})
        return samples

    # Auto-discovery across all match directories
    search_root = captures_dir
    match_dirs = [d for d in search_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not match_dirs:
        match_dirs = [search_root]

    for match_dir in sorted(match_dirs):
        # Find raw/screenshots subdir
        raw_dir: Path | None = None
        for sub in ("raw", "screenshots"):
            candidate = match_dir / sub
            if candidate.is_dir():
                raw_dir = candidate
                break
        if raw_dir is None:
            # Images may sit directly in match_dir
            raw_dir = match_dir

        # Find labels dir
        ldir = match_dir / "labels"
        if not ldir.is_dir():
            continue

        for lf in sorted(ldir.glob("*.json")):
            stem = lf.stem
            image = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = raw_dir / (stem + ext)
                if candidate.exists():
                    image = candidate
                    break
            if image is None:
                continue
            label_data = json.loads(lf.read_text())
            samples.append({"image": image, "label": label_data, "stem": stem, "match": match_dir.name})

    return samples


def stratified_sample(
    samples: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Stratified sample by alive_enemies count (0–5).

    Ensures at least 3 samples per group if available. Fills remaining quota
    proportionally from larger groups.
    """
    if n == 0 or n >= len(samples):
        return list(samples)

    # Group by alive_enemies
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        gs = s["label"].get("game_state", {})
        ae = gs.get("alive_enemies", -1)
        try:
            ae = int(ae)
        except (TypeError, ValueError):
            ae = -1
        groups[ae].append(s)

    # Guarantee at least 3 per group (up to n budget)
    MIN_PER_GROUP = 3
    chosen: list[dict[str, Any]] = []
    group_order = sorted(groups.keys())

    guaranteed: dict[int, list] = {}
    for key in group_order:
        take = min(MIN_PER_GROUP, len(groups[key]))
        guaranteed[key] = rng.sample(groups[key], take)
        chosen.extend(guaranteed[key])

    remaining_budget = n - len(chosen)

    if remaining_budget > 0:
        # Collect leftover samples (not already chosen)
        chosen_stems = {s["stem"] for s in chosen}
        leftover: list[dict[str, Any]] = []
        for key in group_order:
            leftover.extend([s for s in groups[key] if s["stem"] not in chosen_stems])
        rng.shuffle(leftover)
        chosen.extend(leftover[:remaining_budget])

    rng.shuffle(chosen)
    return chosen[:n]


# ---------------------------------------------------------------------------
# Message formatting helpers
# ---------------------------------------------------------------------------

def build_messages(
    image_path: Path,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the messages list for the processor.

    Format:
      system: CS2_PERCEPTION_SYSTEM_PROMPT
      user (few-shot demo 1): image + prompt + label answer
      assistant (demo 1): JSON answer
      ...
      user (actual query): image + prompt
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CS2_PERCEPTION_SYSTEM_PROMPT},
    ]

    if few_shot_examples:
        for ex in few_shot_examples:
            ex_image = ex["image"]
            ex_label = ex["label"]
            gt_state = ex_label.get("game_state", {})
            # Keep only the game_state fields the perception model should output
            gs_clean = {k: gt_state.get(k) for k in (
                "map_name", "round_phase", "player_side", "player_health",
                "player_armor", "player_money", "team_money_total",
                "weapon_primary", "weapon_secondary", "utility",
                "alive_teammates", "alive_enemies", "bomb_status",
                "site", "visible_enemies",
            )}
            answer = json.dumps({"game_state": gs_clean}, indent=2)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": str(ex_image)},
                    {"type": "text", "text": CS2_PERCEPTION_USER_PROMPT},
                ],
            })
            messages.append({"role": "assistant", "content": answer})

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": CS2_PERCEPTION_USER_PROMPT},
        ],
    })

    return messages


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_scores(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Average per-field scores across all samples."""
    field_totals: dict[str, float] = defaultdict(float)
    field_counts: dict[str, int] = defaultdict(int)

    for r in results:
        s = r.get("field_scores", {})
        for field, val in s.items():
            if val is not None:
                field_totals[field] += float(val)
                field_counts[field] += 1

    return {
        field: (field_totals[field] / field_counts[field]) if field_counts[field] > 0 else 0.0
        for field in field_totals
    }


def compute_gates(
    per_field: dict[str, float],
    n_format_valid: int,
    n_total: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate gate thresholds."""
    format_acc = n_format_valid / n_total if n_total > 0 else 0.0

    gates: dict[str, dict[str, Any]] = {
        "format_valid": {
            "accuracy": format_acc,
            "threshold": GATE_THRESHOLDS["format_valid"],
            "pass": format_acc >= GATE_THRESHOLDS["format_valid"],
        },
    }
    for gate_key, field_key in (
        ("alive_teammates_exact", "alive_teammates_exact"),
        ("alive_enemies_exact", "alive_enemies_exact"),
        ("bomb_status_exact", "bomb_status"),
        ("player_health_exact", "player_health_exact"),
    ):
        acc = per_field.get(field_key, 0.0)
        gates[gate_key] = {
            "accuracy": acc,
            "threshold": GATE_THRESHOLDS[gate_key],
            "pass": acc >= GATE_THRESHOLDS[gate_key],
        }

    return gates


def print_report(
    per_field: dict[str, float],
    gates: dict[str, dict[str, Any]],
    n_total: int,
    n_format_valid: int,
    elapsed: float,
) -> None:
    """Print a human-readable evaluation report to stdout."""
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  CS2 HUD BASELINE EVALUATION RESULTS{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  Samples: {n_total}  |  Format valid: {n_format_valid}/{n_total}  |  "
          f"Elapsed: {elapsed:.1f}s  |  {elapsed/n_total:.1f}s/sample")
    print()

    # Gate section
    print(f"{BOLD}  GATE THRESHOLDS (pass/fail){RESET}")
    print(f"  {'-' * 60}")
    for gate_name, info in gates.items():
        status = PASS if info["pass"] else FAIL
        print(
            f"  {gate_name:<30}  {info['accuracy']:>6.1%}  "
            f"(threshold {info['threshold']:.0%})  {status}"
        )

    print()
    print(f"{BOLD}  PER-FIELD ACCURACY{RESET}")
    print(f"  {'-' * 60}")

    # Ordered field display
    field_display_order = [
        # Critical: alive counts
        ("alive_enemies_exact",    "alive_enemies (exact)       [CRITICAL]"),
        ("alive_teammates_exact",  "alive_teammates (exact)     [CRITICAL]"),
        # Health / armor
        ("player_health_exact",    "player_health (exact)"),
        ("player_health_tol5",     "player_health (±5 tol)"),
        ("player_armor_exact",     "player_armor (exact)"),
        ("player_armor_tol5",      "player_armor (±5 tol)"),
        # Side / phase
        ("player_side",            "player_side (exact)"),
        ("round_phase",            "round_phase (exact)"),
        # Weapons
        ("weapon_primary",         "weapon_primary (fuzzy)"),
        ("weapon_secondary",       "weapon_secondary (fuzzy)"),
        # Utility
        ("utility_jaccard",        "utility (Jaccard)"),
        ("utility_exact",          "utility (exact set)"),
        # Bomb / map
        ("bomb_status",            "bomb_status (exact)"),
        ("map_name",               "map_name (fuzzy)"),
        # Economy
        ("player_money_tol500",    "player_money (±500 tol)"),
        # Scores
        ("score_t",                "score_t (exact)"),
        ("score_ct",               "score_ct (exact)"),
        # Booleans
        ("player_has_helmet",      "player_has_helmet (exact)"),
        ("has_defuser",            "has_defuser (exact)"),
    ]

    for key, label in field_display_order:
        if key in per_field:
            acc = per_field[key]
            bar_len = int(acc * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {label:<35}  {acc:>6.1%}  [{bar}]")

    # Any remaining fields not in the display order
    shown = {k for k, _ in field_display_order}
    extras = {k: v for k, v in per_field.items() if k not in shown}
    if extras:
        print(f"\n  {'-' * 60}")
        print(f"  Other fields:")
        for k, v in sorted(extras.items()):
            print(f"  {k:<35}  {v:>6.1%}")

    print()
    all_passed = all(info["pass"] for info in gates.values())
    if all_passed:
        print(f"  {BOLD}[{PASS}] All gates passed — model is eval-ready.{RESET}")
    else:
        n_passed = sum(1 for info in gates.values() if info["pass"])
        n_gates  = len(gates)
        print(f"  {BOLD}[{FAIL}] {n_passed}/{n_gates} gates passed — "
              f"SFT needed before GRPO.{RESET}")
    print(f"{'=' * 70}")
    print()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    rng = random.Random(42)

    # ----- Resolve captures / labels directories -----
    if args.screenshots:
        captures_path = Path(args.screenshots)
        # Check for raw/ subdir
        if (captures_path / "raw").is_dir():
            print(f"[info] Using captures: {captures_path} (raw/ subdir found)")
        elif (captures_path / "screenshots").is_dir():
            print(f"[info] Using captures: {captures_path} (screenshots/ subdir found)")
        else:
            print(f"[info] Using captures: {captures_path} (images directly in dir)")
    else:
        captures_path = Path("data/captures")
        print(f"[info] Using captures root: {captures_path}")

    labels_path = Path(args.labels) if args.labels else None

    # ----- Discover samples -----
    print("[info] Discovering (screenshot, label) pairs...")
    all_samples = load_all_samples(captures_path, labels_path)
    print(f"[info] Found {len(all_samples)} labeled screenshot pairs")

    if len(all_samples) == 0:
        print("[error] No samples found. Check --screenshots and --labels paths.")
        return

    # ----- Stratified sampling -----
    samples = stratified_sample(all_samples, args.n_samples, rng)
    print(f"[info] Selected {len(samples)} samples (stratified by alive_enemies)")

    # alive_enemies distribution summary
    dist: dict[int, int] = defaultdict(int)
    for s in samples:
        ae = s["label"].get("game_state", {}).get("alive_enemies", -1)
        try:
            ae = int(ae)
        except (TypeError, ValueError):
            ae = -1
        dist[ae] += 1
    print(f"[info] alive_enemies distribution: { {k: dist[k] for k in sorted(dist)} }")

    # ----- Resolve few-shot examples -----
    few_shot_examples: list[dict[str, Any]] = []
    if args.few_shot > 0:
        # Pick few-shot demos from the full pool (not in evaluation set)
        eval_stems = {s["stem"] for s in samples}
        demo_pool  = [s for s in all_samples if s["stem"] not in eval_stems]
        if len(demo_pool) < args.few_shot:
            print(f"[warn] Requested {args.few_shot} few-shot examples but only "
                  f"{len(demo_pool)} non-eval samples available. Using {len(demo_pool)}.")
        few_shot_examples = rng.sample(demo_pool, min(args.few_shot, len(demo_pool)))
        print(f"[info] Using {len(few_shot_examples)} few-shot examples in prompt")

    # ----- Load model -----
    model_name = args.model_name
    print(f"\n[info] Loading model: {model_name}")
    t_load = time.time()

    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16

    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
        model.eval()
    except Exception as e:
        print(f"[warn] Qwen3_5ForConditionalGeneration failed ({e}), "
              "falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    load_elapsed = time.time() - t_load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[info] Model loaded in {load_elapsed:.1f}s | GPU: {allocated:.2f} GB allocated")
    else:
        print(f"[info] Model loaded in {load_elapsed:.1f}s (CPU)")

    has_vision = hasattr(model, "model") and hasattr(model.model, "visual")
    if not has_vision:
        print("[warn] Vision encoder not detected — model may be text-only")

    # ----- Inference loop -----
    print(f"\n[info] Running inference on {len(samples)} samples...\n")

    results: list[dict[str, Any]] = []
    n_format_valid = 0
    t_eval_start = time.time()

    for idx, sample in enumerate(samples):
        image_path: Path = sample["image"]
        label: dict[str, Any] = sample["label"]
        gt_state: dict[str, Any] = label.get("game_state", {})

        # Build messages
        messages = build_messages(image_path, few_shot_examples if few_shot_examples else None)

        # Tokenize
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception as e:
            print(f"  [{idx+1}/{len(samples)}] {image_path.name}  TOKENIZE ERROR: {e}")
            results.append({
                "stem": sample["stem"],
                "image": str(image_path),
                "gt_state": gt_state,
                "raw_output": None,
                "parsed_state": None,
                "format_valid": False,
                "field_scores": score_prediction(None, gt_state),
                "error": str(e),
            })
            continue

        # Generate
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_elapsed = time.time() - t0

        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True)

        # Parse JSON
        parsed = extract_json(raw_output)
        format_valid = parsed is not None and "game_state" in parsed
        pred_state = parsed.get("game_state") if (parsed and "game_state" in parsed) else None

        if format_valid:
            n_format_valid += 1

        # Score
        field_scores = score_prediction(pred_state, gt_state)

        # Quick per-sample status line
        ae_exact = field_scores.get("alive_enemies_exact", 0.0)
        hp_exact = field_scores.get("player_health_exact", 0.0)
        n_toks   = len(generated_ids)
        fmt_mark = "OK" if format_valid else "NO-JSON"
        ae_mark  = f"AE={'ok' if ae_exact else 'X'}"
        hp_mark  = f"HP={'ok' if hp_exact else 'X'}"
        print(
            f"  [{idx+1:>3}/{len(samples)}]  {image_path.name:<45}  "
            f"{fmt_mark}  {ae_mark}  {hp_mark}  "
            f"{gen_elapsed:.1f}s  {n_toks}tok"
        )

        results.append({
            "stem": sample["stem"],
            "image": str(image_path),
            "match": sample.get("match", ""),
            "gt_state": gt_state,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "parsed_state": pred_state,
            "format_valid": format_valid,
            "field_scores": field_scores,
            "gen_elapsed": gen_elapsed,
            "n_tokens": int(n_toks),
        })

    total_elapsed = time.time() - t_eval_start

    # ----- Aggregate -----
    per_field = aggregate_scores(results)
    gates = compute_gates(per_field, n_format_valid, len(samples))

    # ----- Print report -----
    print_report(per_field, gates, len(samples), n_format_valid, total_elapsed)

    # ----- Save JSON output -----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": {
            "model_name": model_name,
            "n_samples": len(samples),
            "n_format_valid": n_format_valid,
            "few_shot": args.few_shot,
            "dtype": args.dtype,
            "elapsed_seconds": total_elapsed,
            "seconds_per_sample": total_elapsed / len(samples) if samples else 0.0,
        },
        "gates": gates,
        "per_field_accuracy": per_field,
        "alive_enemies_distribution": {str(k): dist[k] for k in sorted(dist)},
        "samples": results,
    }

    output_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[info] Detailed results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline eval: zero-shot VLM accuracy on CS2 HUD screenshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--screenshots",
        default=None,
        help="Path to captures directory (default: auto-discover data/captures/*). "
             "Can be a match dir (e.g. data/captures/furia-vs-vitality-m1-mirage) "
             "or the root data/captures dir.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to labels directory (default: auto-detect sibling labels/ dirs). "
             "Only needed if labels are stored separately from captures.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50, 0 = use all)",
    )
    parser.add_argument(
        "--output",
        default="outputs/baseline_eval.json",
        help="Output JSON path (default: outputs/baseline_eval.json)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        metavar="N",
        help="Number of few-shot example screenshot-label pairs to prepend (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
