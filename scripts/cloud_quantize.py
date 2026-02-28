#!/usr/bin/env python3
"""
Quantize Qwen3.5-27B full VLM to BnB NF4 and push to HuggingFace Hub.

Run on a cloud GPU (A100 80GB recommended). The full bf16 model is ~55GB,
so this requires more VRAM than a consumer card.

CRITICAL: Uses Qwen3_5ForConditionalGeneration, NOT AutoModelForCausalLM.
AutoModelForCausalLM silently strips the vision encoder, producing a
text-only checkpoint. We need the full VLM for screenshot-based training.

Prerequisites:
    - GPU with >= 48GB VRAM (A100 80GB, A6000 48GB)
    - pip install transformers bitsandbytes accelerate huggingface-hub psutil
    - huggingface-cli login (with write access to HUB_REPO)

Usage:
    python cloud_quantize.py
    python cloud_quantize.py --hub-repo your-name/your-repo
    python cloud_quantize.py --no-push  # save locally only

Output:
    Pushes quantized checkpoint to HuggingFace Hub.
    Local copy saved to ./Qwen3.5-27B-VLM-bnb-4bit/
"""
import argparse
import json
import sys
import time

import psutil
import torch

SOURCE_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_HUB_REPO = "skkwowee/Qwen3.5-27B-bnb-4bit"
LOCAL_DIR = "Qwen3.5-27B-VLM-bnb-4bit"


def log(msg: str):
    mem = psutil.virtual_memory()
    gpu_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"RAM {mem.used / 1024**3:.1f}/{mem.total / 1024**3:.1f}GB | "
          f"GPU {gpu_alloc:.1f}GB | {msg}")


def check_prerequisites():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.")
        sys.exit(1)

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    mem = psutil.virtual_memory()

    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.0f}GB)")
    print(f"RAM: {mem.total / 1024**3:.0f}GB")
    print(f"Source: {SOURCE_MODEL}")
    print()

    if gpu_mem < 40:
        print(f"WARNING: {gpu_mem:.0f}GB VRAM may not be enough.")
        print("Recommended: A100 80GB or A6000 48GB.")
        print("Continuing anyway...\n")


def quantize_and_push(hub_repo: str, push: bool):
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

    log(f"Loading {SOURCE_MODEL} as full VLM with BnB NF4...")
    log("Using Qwen3_5ForConditionalGeneration (NOT AutoModelForCausalLM — preserves vision encoder)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        SOURCE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    log("Model loaded. Verifying vision encoder is present...")

    # Verify we got the full VLM, not just the text decoder
    has_vision = hasattr(model, 'visual') or hasattr(model, 'model') and hasattr(model.model, 'visual')
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {param_count / 1e9:.2f}B (stored; ~27B logical before NF4 packing)")

    if not has_vision:
        print("\nERROR: Vision encoder not found on model.")
        print("Make sure you're using Qwen3_5ForConditionalGeneration, not AutoModelForCausalLM.")
        sys.exit(1)

    log(f"Vision encoder present. Full VLM loaded.")

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    processor = AutoProcessor.from_pretrained(SOURCE_MODEL)

    # Save locally
    log(f"Saving to {LOCAL_DIR}/...")
    model.save_pretrained(LOCAL_DIR)
    tokenizer.save_pretrained(LOCAL_DIR)
    processor.save_pretrained(LOCAL_DIR)

    # Write metadata
    meta = {
        "source_model": SOURCE_MODEL,
        "quantization": "bnb-4bit-nf4",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "parameter_count": param_count,
        "parameter_count_B": f"{param_count / 1e9:.2f}B",
        "includes_vision_encoder": True,
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(f"{LOCAL_DIR}/quantize_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_size = sum(f.stat().st_size for f in __import__('pathlib').Path(LOCAL_DIR).rglob("*") if f.is_file())
    log(f"Local checkpoint size: {total_size / 1024**3:.1f}GB")

    # Push to Hub
    if push:
        log(f"Pushing to {hub_repo}...")
        model.push_to_hub(hub_repo, commit_message="Qwen3.5-27B VLM BnB NF4 (full VLM with vision encoder)")
        tokenizer.push_to_hub(hub_repo)
        processor.push_to_hub(hub_repo)

        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"{LOCAL_DIR}/quantize_meta.json",
            path_in_repo="quantize_meta.json",
            repo_id=hub_repo,
        )
        log(f"Pushed to https://huggingface.co/{hub_repo}")
    else:
        log("Skipping Hub push (--no-push). Checkpoint saved locally only.")

    log("Done!")


def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5-27B VLM to BnB NF4")
    parser.add_argument("--hub-repo", default=DEFAULT_HUB_REPO, help=f"HF Hub repo (default: {DEFAULT_HUB_REPO})")
    parser.add_argument("--no-push", action="store_true", help="Save locally only, don't push to Hub")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3.5-27B VLM — Cloud BnB NF4 Quantization")
    print("=" * 60)
    print()

    check_prerequisites()
    quantize_and_push(args.hub_repo, push=not args.no_push)

    print()
    print("=" * 60)
    print("Done! Pull the checkpoint locally with:")
    print(f"  python scripts/data.py pull --model")
    print("=" * 60)


if __name__ == "__main__":
    main()
