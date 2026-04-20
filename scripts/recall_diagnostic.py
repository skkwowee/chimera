#!/usr/bin/env python3
"""RECALL signal-quality diagnostic.

Runs a saved LoRA over N val samples and, for each generated advice
block, dumps the full RECALL state -- extracted action dict, 5-dim
action vector, top-k neighbor states / actions / outcomes / cosine-
similarities, matched-mask, Q_hat, V_hat, advantage, confident-flag.

One JSON object per generation in <output>/recall_diag.jsonl. Use it
to answer "is the RECALL signal actually informative?" before
investing in a new state/action representation.

Usage (on pod with venv + SFT + trained LoRA):
    /workspace/venv/bin/python scripts/recall_diagnostic.py \
        --lora /workspace/outputs/grpo/f08v4_resumed/final_model/lora_adapter \
        --sft /workspace/checkpoints/sft-r4-checkpoint-304 \
        --train-data data/training/grpo/smoke_test.jsonl \
        --n-samples 20 \
        --output /workspace/outputs/diag/recall_diag.jsonl

Doesn't train anything; doesn't mutate RECALL. Read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.training.recall import (
    RECALLIndex,
    tactical_embedding,
    action_embedding,
    _extract_action_from_text,
    _ACTION_SIM_THRESHOLD,
)


def build_system_prompt(perception_only: bool) -> str:
    if perception_only:
        schema_request = "containing a game_state key whose value is a JSON object"
    else:
        schema_request = (
            "containing game_state, analysis, and advice keys, where "
            "each value is a JSON object (NOT a string)"
        )
    return (
        "You are a CS2 game analyst. Respond ONLY with a single-line "
        f"compact JSON object (no whitespace, no newlines) {schema_request}. "
        "No other text."
    )


def extract_user_text(prompt_content: Any) -> str:
    if isinstance(prompt_content, list):
        return "\n".join(
            b["text"] for b in prompt_content
            if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        )
    return str(prompt_content)


def diagnose_one(
    advice_text: str,
    ground_truth: dict[str, Any],
    recall: RECALLIndex,
    k: int = 32,
    k_min: int = 5,
) -> dict[str, Any]:
    """Rebuild RECALLIndex.query step by step and record every intermediate."""
    game_state = ground_truth.get("game_state", {})
    if not isinstance(game_state, dict):
        game_state = {}

    action = _extract_action_from_text(advice_text)
    action_vec = action_embedding(action)

    if recall._n == 0 or recall._state_index is None:
        return {
            "advice_text": advice_text[:500],
            "extracted_action": action,
            "action_vec": action_vec.tolist(),
            "error": "empty_recall_index",
        }

    state_vec = tactical_embedding(game_state).reshape(1, -1)
    k_actual = min(k, recall._n)
    distances, indices = recall._state_index.search(state_vec, k_actual)
    indices = indices[0]
    valid = indices[indices >= 0]

    if len(valid) == 0:
        return {
            "advice_text": advice_text[:500],
            "extracted_action": action,
            "action_vec": action_vec.tolist(),
            "error": "no_valid_neighbors",
        }

    neighbor_outcomes = recall._outcomes[valid]
    v_hat = float(np.mean(neighbor_outcomes))

    q_vec = action_vec.reshape(1, -1)
    neighbor_actions = recall._action_embeddings[valid]
    q_norm = float(np.linalg.norm(q_vec))
    neighbor_norms = np.linalg.norm(neighbor_actions, axis=1)

    if q_norm < 1e-8:
        # Fallback path in the real code: movement_direction match.
        q_move = float(action.get("movement_direction", 0))
        matched_mask = np.abs(neighbor_actions[:, 0] - q_move) < 0.5
        cos_sim = np.full(len(valid), np.nan, dtype=np.float32)
        fallback = "movement_direction_match"
    else:
        safe_norms = np.maximum(neighbor_norms, 1e-8)
        cos_sim = (neighbor_actions @ q_vec.T).flatten() / (safe_norms * q_norm)
        matched_mask = cos_sim > _ACTION_SIM_THRESHOLD
        fallback = None

    matched_idx = np.where(matched_mask)[0]
    if len(matched_idx) == 0:
        q_hat = v_hat
        confident = False
    else:
        q_hat = float(np.mean(neighbor_outcomes[matched_idx]))
        confident = len(matched_idx) >= k_min

    advantage = q_hat - v_hat if confident else 0.0

    cos_sim_list = [None if np.isnan(v) else float(v) for v in cos_sim.tolist()]

    return {
        "advice_text": advice_text[:500],
        "extracted_action": action,
        "action_vec": action_vec.tolist(),
        "action_vec_norm": q_norm,
        "fallback_mode": fallback,
        "state_vec_norm": float(np.linalg.norm(state_vec)),
        "k_used": int(k_actual),
        "k_valid_neighbors": int(len(valid)),
        "v_hat": v_hat,
        "q_hat": q_hat,
        "advantage": advantage,
        "confident": confident,
        "matched_count": int(len(matched_idx)),
        "cos_sim_min": float(np.nanmin(cos_sim)) if q_norm >= 1e-8 else None,
        "cos_sim_max": float(np.nanmax(cos_sim)) if q_norm >= 1e-8 else None,
        "cos_sim_mean": float(np.nanmean(cos_sim)) if q_norm >= 1e-8 else None,
        "cos_sim_above_threshold_frac": float(matched_mask.mean()),
        "neighbor_outcomes_mean": v_hat,
        "neighbor_outcomes_std": float(np.std(neighbor_outcomes)),
        "top5_cos_sim": cos_sim_list[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--sft", type=str, default=None, help="Optional SFT LoRA to merge first")
    parser.add_argument("--train-data", type=str, required=True, help="JSONL used to build RECALL index")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--perception-only", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading base {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.sft:
        print(f"Merging SFT adapter from {args.sft}...")
        model = PeftModel.from_pretrained(model, args.sft)
        model = model.merge_and_unload()

    print(f"Loading trained LoRA from {args.lora}...")
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    # Build RECALL index same way the trainer does. build_from_samples walks
    # sample["ground_truth"] itself; pass the full samples through.
    print(f"Building RECALL index from {args.train_data}...")
    samples = [json.loads(l) for l in open(args.train_data)]
    recall = RECALLIndex()
    # Training split was first 90% (mirror trainer; val is last 10%).
    train_cut = int(0.9 * len(samples))
    recall.build_from_samples(samples[:train_cut])
    print(f"RECALL built: {recall._n} samples indexed")

    # Eval split (last 10%, deterministic) -- take first n_samples
    val_samples = samples[train_cut:][: args.n_samples]
    print(f"Diagnosing {len(val_samples)} val samples...")

    sys_prompt = build_system_prompt(args.perception_only)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "w")

    for i, sample in enumerate(val_samples):
        prompt_content = sample.get("prompt", "")
        ground_truth = sample.get("ground_truth", {})
        user_text = extract_user_text(prompt_content)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=args.n_generations,
            )

        prompt_len = inputs.input_ids.shape[1]
        for g_idx in range(output.shape[0]):
            gen_ids = output[g_idx, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Try to pull advice text from the parsed JSON; fall back to the whole response.
            from src.training.rewards import _extract_json_from_response
            parsed = _extract_json_from_response(response)
            if parsed and isinstance(parsed.get("advice"), dict):
                advice = parsed["advice"]
                advice_text = " ".join(
                    str(v) for v in advice.values() if isinstance(v, str)
                )
            else:
                advice_text = response

            diag = diagnose_one(advice_text, ground_truth, recall)
            diag["sample_i"] = i
            diag["generation_i"] = g_idx
            diag["response_first500"] = response[:500]
            diag["gt_game_state_keys"] = list(ground_truth.get("game_state", {}).keys()) if isinstance(ground_truth.get("game_state"), dict) else []
            out_f.write(json.dumps(diag) + "\n")
            out_f.flush()

        print(f"  [{i+1}/{len(val_samples)}] wrote {args.n_generations} generations")

    out_f.close()
    print(f"\nWrote {len(val_samples) * args.n_generations} rows to {output_path}")
    print("\nSuggested slices:")
    print(f"  cat {output_path} | jq -c '{{confident, matched_count, advantage, cos_sim_mean}}' | head")
    print(f"  cat {output_path} | jq '.action_vec_norm' | sort | uniq -c")


if __name__ == "__main__":
    main()
