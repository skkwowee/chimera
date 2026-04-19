#!/usr/bin/env python3
"""Reproduce the trainer.evaluate() path on 2 samples and print every step."""

from __future__ import annotations
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.training.grpo_trainer import CS2GRPOTrainer, CS2GRPOConfig
from src.training.rewards import format_gate_reward, _extract_json_from_response

# Match smoke launcher
config = CS2GRPOConfig(
    model_name="Qwen/Qwen3.5-35B-A3B",
    output_dir="/tmp/dbg",
    max_new_tokens=512,
    use_lora=True,
    perception_only=True,
    attn_implementation="flash_attention_2",
)

trainer = CS2GRPOTrainer(config)
trainer.load_model()

# Apply the trained LoRA on top
from peft import PeftModel
print("Loading trained LoRA from /workspace/outputs/grpo/smoke/final_model/lora_adapter")
trainer.model = PeftModel.from_pretrained(
    trainer.model.base_model.model if hasattr(trainer.model, 'base_model') else trainer.model,
    "/workspace/outputs/grpo/smoke/final_model/lora_adapter",
)

# Load val data the way the trainer would
samples = [json.loads(l) for l in open("/workspace/chimera/data/training/grpo/smoke_test.jsonl")]
val = samples[1811:]  # 90/10 split → val is last 202
eval_data = val[:2]  # first 2 of val

print(f"\nTotal: {len(samples)}, val: {len(val)}, evaluating: {len(eval_data)}")

# Now hack: monkey-patch evaluate to print response per sample
from src.training import grpo_trainer as gt
orig_decode = trainer.processor.decode
def loud_decode(*a, **kw):
    r = orig_decode(*a, **kw)
    print(f"\n[eval response, repr 300 chars]: {repr(r[:300])}")
    print(f"[format_gate]: {format_gate_reward(r, perception_only=True)}")
    return r
trainer.processor.decode = loud_decode

print("\nCalling trainer.evaluate() with 2 samples...")
results = trainer.evaluate(eval_data=eval_data)
print(f"\n[final results]: {results}")
