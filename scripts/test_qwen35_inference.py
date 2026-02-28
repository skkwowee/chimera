#!/usr/bin/env python3
"""
Test Qwen3.5-27B VLM inference on RTX 4090 (24GB).

Loads the NF4-quantized full VLM from HuggingFace Hub.
Tests with screenshots if available, otherwise falls back to text-only.

Usage:
    uv run python scripts/test_qwen35_inference.py
    uv run python scripts/test_qwen35_inference.py --clear-cache
"""
import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import sys
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from src.prompts import CS2_SYSTEM_PROMPT
from PIL import Image

HUB_REPO = "skkwowee/Qwen3.5-27B-bnb-4bit"


def clear_hf_cache(repo_id: str):
    """Delete cached revisions for a specific HF Hub repo."""
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            revisions = [rev.commit_hash for rev in repo.revisions]
            if not revisions:
                print(f"No cached revisions found for {repo_id}")
                return
            strategy = cache_info.delete_revisions(*revisions)
            print(f"Freeing {strategy.expected_freed_size_str} from {repo_id} cache...")
            strategy.execute()
            print("Cache cleared.")
            return
    print(f"No cache entry found for {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3.5-27B VLM inference")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete HF Hub cache for the model after inference")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3.5-27B VLM Inference Test")
    print("=" * 60)

    # Load model from Hub
    print(f"\nLoading VLM from {HUB_REPO}...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        HUB_REPO,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(HUB_REPO)

    load_time = time.time() - t0
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    has_vision = hasattr(model, "visual")
    print(f"Loaded in {load_time:.1f}s | GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    print(f"Vision encoder: {'yes' if has_vision else 'NO — something is wrong'}\n")

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
            t_start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                )
            t_elapsed = time.time() - t_start

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

            n_tokens = len(generated_ids)
            print(f"\n  [{t_elapsed:.1f}s | {n_tokens} tokens | {n_tokens/t_elapsed:.1f} tok/s]")
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
        t_start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
            )
        t_elapsed = time.time() - t_start

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

        n_tokens = len(generated_ids)
        print(f"\n  [{t_elapsed:.1f}s | {n_tokens} tokens | {n_tokens/t_elapsed:.1f} tok/s]")

    print()
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"Final GPU memory: {allocated:.2f} GB")
    print("Done!")

    if args.clear_cache:
        print(f"\nClearing HF cache for {HUB_REPO}...")
        clear_hf_cache(HUB_REPO)


if __name__ == "__main__":
    main()
