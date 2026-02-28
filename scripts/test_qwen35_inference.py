#!/usr/bin/env python3
"""
Test Qwen3.5-27B inference on RTX 4090 (24GB).

Dense 27B model — BnB 4-bit quantization (~15GB VRAM).
Requires ~20GB+ system RAM for weight deserialization during loading.

Usage:
    uv run python scripts/test_qwen35_inference.py
"""
import torch._dynamo
torch._dynamo.config.disable = True

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastVisionModel
from transformers import AutoProcessor
from src.prompts import CS2_SYSTEM_PROMPT
from PIL import Image


def main():
    print("=" * 60)
    print("Qwen3.5-27B Inference Test")
    print("=" * 60)

    # Load model
    print("\nLoading model with BnB 4-bit quantization...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/Qwen3.5-27B",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-27B")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # Find test screenshots
    captures_dir = Path("data/captures")
    screenshots = sorted(captures_dir.glob("*/raw/*.jpg"))[:3]
    if not screenshots:
        print("No screenshots found in data/captures/*/raw/")
        print("Run: python scripts/data.py pull --all")
        return

    print(f"\nTesting on {len(screenshots)} screenshots...\n")

    for path in screenshots:
        print(f"{'='*60}")
        print(f"Screenshot: {path.name}")
        print(f"{'='*60}")

        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": CS2_SYSTEM_PROMPT},
        ]}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        print("Generating...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
            )

        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Try to parse as JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response[start:end])
                print(json.dumps(parsed, indent=2))
            else:
                print(response[:2000])
        except json.JSONDecodeError:
            print(response[:2000])
        print()

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"Final GPU memory: {allocated:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
