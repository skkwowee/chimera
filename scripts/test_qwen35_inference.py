#!/usr/bin/env python3
"""
Test Qwen3.5-27B VLM inference on RTX 4090 (24GB).

Loads the local pre-quantized BnB NF4 checkpoint (full VLM with vision encoder).
Tests with screenshots if available, otherwise falls back to text-only.

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

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from src.prompts import CS2_SYSTEM_PROMPT
from PIL import Image

MODEL_DIR = Path("models/Qwen3.5-27B-bnb-4bit")


def main():
    print("=" * 60)
    print("Qwen3.5-27B VLM Inference Test")
    print("=" * 60)

    if not MODEL_DIR.exists():
        print(f"ERROR: {MODEL_DIR} not found.")
        print("Run: uv run python scripts/quantize_base_model.py")
        sys.exit(1)

    # Verify this is the full VLM, not text-only
    meta_path = MODEL_DIR / "quantize_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("includes_vision_encoder", False):
            print("ERROR: Checkpoint is text-only (missing vision encoder).")
            print("Need the full VLM checkpoint. See scripts/cloud_quantize.py.")
            sys.exit(1)
        print(f"Checkpoint: {meta.get('parameter_count_B', '?')} params, vision={meta.get('includes_vision_encoder')}")

    # Load model
    print(f"\nLoading VLM from {MODEL_DIR}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved\n")

    # Find test screenshots
    captures_dir = Path("data/captures")
    screenshots = sorted(captures_dir.glob("*/raw/*.jpg"))[:3]

    if screenshots:
        print(f"Testing on {len(screenshots)} screenshots...\n")

        for path in screenshots:
            print(f"{'=' * 60}")
            print(f"Screenshot: {path.name}")
            print(f"{'=' * 60}")

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
            response = processor.decode(generated_ids, skip_special_tokens=True)

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
    else:
        print("No screenshots found in data/captures/*/raw/")
        print("Run: python scripts/data.py pull --all")
        print("\nRunning text-only test instead...\n")

        messages = [{"role": "user", "content": [
            {"type": "text", "text": CS2_SYSTEM_PROMPT + "\n\n"
                "You are watching a CS2 match on de_inferno. "
                "The CT player has $4700, an M4A4, full armor+helmet, "
                "one smoke and one flashbang. Score is 8-7 in favor of T side. "
                "3 CTs alive, 4 Ts alive. Bomb has not been spotted yet. "
                "Provide your analysis as JSON."
            },
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
        response = processor.decode(generated_ids, skip_special_tokens=True)

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
