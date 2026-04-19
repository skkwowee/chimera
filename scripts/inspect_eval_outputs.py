#!/usr/bin/env python3
"""Quick eval-output inspector. Loads the smoke-trained LoRA and dumps
the raw text of the first few eval samples so we can see why the
format_gate is scoring 0.0."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

from src.training.rewards import format_gate_reward, perceptual_accuracy_reward, _extract_json_from_response

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
LORA = "/workspace/outputs/grpo/smoke/final_model/lora_adapter"
DATA = REPO / "data/training/grpo/smoke_test.jsonl"
N_SAMPLES = 3

print(f"Loading {MODEL_NAME} + LoRA from {LORA}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA)
model.eval()
print("Model loaded.")

# Load data, take last N_SAMPLES (the val split was the last 10% of the file).
with open(DATA) as f:
    samples = [json.loads(line) for line in f]
print(f"Total samples: {len(samples)}, inspecting last {N_SAMPLES} (= eval split)")
samples = samples[-N_SAMPLES:]

for i, sample in enumerate(samples):
    print("\n" + "=" * 80)
    print(f"SAMPLE {i+1}/{N_SAMPLES}")
    print("=" * 80)

    prompt_content = sample["prompt"]
    if isinstance(prompt_content, list) and prompt_content and isinstance(prompt_content[0], dict):
        text_parts = [b["text"] for b in prompt_content if b.get("type") == "text"]
        user_text = "\n".join(text_parts)
    else:
        user_text = str(prompt_content)

    messages = [
        {"role": "system", "content": (
            "You are a CS2 game analyst. Respond ONLY with a JSON object "
            "containing a game_state key. No other text."
        )},
        {"role": "user", "content": user_text},
    ]

    print(f"\n[prompt sent, last 300 chars]:\n...{user_text[-300:]}")

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
        )
    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n[model response, raw]:\n{gen}")

    parsed = _extract_json_from_response(gen)
    print(f"\n[_extract_json result]: {parsed}")

    gate = format_gate_reward(gen, perception_only=True)
    gt = sample.get("ground_truth", {})
    pa = gate * perceptual_accuracy_reward(gen, ground_truth=gt)
    print(f"[format_gate]: {gate}")
    print(f"[perceptual_accuracy (gated)]: {pa}")
