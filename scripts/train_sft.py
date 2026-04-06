#!/usr/bin/env python3
"""
SFT training script for CS2 VLM fine-tuning.

Uses transformers + peft for LoRA supervised fine-tuning of Qwen3.5-35B-A3B MoE (bf16).
SFT teaches the model output format and CS2 domain knowledge before GRPO
refinement.

Usage:
    # Basic training (saves merged model for GRPO handoff)
    python scripts/train_sft.py --screenshots data/raw --labels data/labeled

    # Custom settings
    python scripts/train_sft.py --epochs 5 --lr 1e-5 --max-seq-length 4096

    # Resume from checkpoint
    python scripts/train_sft.py --resume outputs/sft/checkpoint-500

    # Dry run (load model and check VRAM, no training)
    python scripts/train_sft.py --dry-run

    # Benchmark: time 3 steps and estimate total training time
    python scripts/train_sft.py --benchmark

    # Then use the merged output as GRPO base:
    python scripts/train_grpo.py --model-name outputs/sft/final_model/merged_16bit
"""

import argparse
import sys
from pathlib import Path

from src.training import (
    CS2SFTConfig,
    CS2SFTTrainer,
    convert_labeled_to_sft_format,
    create_sft_dataset,
)
from src.utils.config import DEFAULT_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-35B-A3B MoE on CS2 screenshots using SFT (supervised fine-tuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--screenshots",
        type=str,
        default="data/raw",
        help="Directory containing screenshot images",
    )
    data_group.add_argument(
        "--labels",
        type=str,
        default="data/labeled",
        help="Directory containing JSON label files",
    )
    data_group.add_argument(
        "--output",
        type=str,
        default="outputs/sft",
        help="Output directory for checkpoints and model",
    )
    data_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (rest is validation)",
    )

    # Model settings
    model_group = parser.add_argument_group("Model settings")
    model_group.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name or path to load",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model precision",
    )

    # LoRA settings
    lora_group = parser.add_argument_group("LoRA settings")
    lora_group.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning, requires more VRAM)",
    )
    lora_group.add_argument(
        "--lora-r",
        type=int,
        default=4,
        help="LoRA rank",
    )
    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
        help="LoRA alpha",
    )
    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training parameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    train_group.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    train_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )

    # SFT-specific settings
    sft_group = parser.add_argument_group("SFT settings")
    sft_group.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for SFT",
    )
    sft_group.add_argument(
        "--no-finetune-vision",
        action="store_true",
        help="Freeze vision layers (default: finetune vision for SFT)",
    )

    # Checkpointing
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    checkpoint_group.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    checkpoint_group.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--save-merged",
        action="store_true",
        default=True,
        help="Save merged model for GRPO handoff (default: True)",
    )
    output_group.add_argument(
        "--no-save-merged",
        action="store_true",
        help="Save LoRA adapter only (no merge)",
    )

    # Other options
    parser.add_argument(
        "--benchmark",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Time N forward+backward steps (default: 3) and estimate total training time",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and check VRAM usage without training",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (no training)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve save_merged: --no-save-merged overrides the default
    save_merged = not args.no_save_merged

    # Create config from args
    config = CS2SFTConfig(
        model_name=args.model_name,
        torch_dtype=args.dtype,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        finetune_vision_layers=not args.no_finetune_vision,
        output_dir=args.output,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb else "none",
    )

    print("=" * 60)
    print("CS2 SFT Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  - rank: {config.lora_r}")
        print(f"  - alpha: {config.lora_alpha}")
    print(f"Vision layers: {'trainable' if config.finetune_vision_layers else 'frozen'}")
    print(f"Max seq length: {config.max_seq_length}")
    print(f"Save merged: {save_merged}")
    print(f"Output: {config.output_dir}")
    print()

    # Create trainer
    trainer = CS2SFTTrainer(config)

    # Dry run: just load model and check memory
    if args.dry_run:
        print("Dry run: loading model to check VRAM usage...")
        trainer.load_model()
        print("\nDry run complete. Model loaded successfully.")
        return

    # Benchmark: time forward+backward steps and estimate total training time
    if args.benchmark is not None:
        import time

        import torch

        n_steps = args.benchmark

        print("Loading data for benchmark...")
        screenshots_dir = Path(args.screenshots)
        labels_dir = Path(args.labels)
        sft_items = convert_labeled_to_sft_format(
            screenshots_dir=screenshots_dir, labels_dir=labels_dir,
        )
        if len(sft_items) == 0:
            print("Error: No labeled data found!")
            sys.exit(1)
        train_data, _ = create_sft_dataset(sft_items, train_ratio=args.train_ratio, seed=args.seed)

        print(f"Benchmark: {len(sft_items)} samples, {n_steps} timed steps")
        print()

        trainer.load_model()
        trainer.prepare_data(train_data)
        assert trainer.model is not None, "model must be loaded"
        assert trainer.processor is not None, "processor must be loaded"
        assert trainer.train_dataset is not None, "train_dataset must be prepared"

        # Build a batch from the first sample
        batch = trainer._vision_data_collator([trainer.train_dataset[0]])
        batch = {k: v.to(trainer.model.device) for k, v in batch.items()}

        seq_len = batch["input_ids"].shape[1]
        image_token_id = trainer.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        n_vision = (batch["input_ids"] == image_token_id).sum().item()
        n_text = seq_len - n_vision

        print(f"Sequence: {seq_len} tokens ({n_vision} vision, {n_text} text)")
        trainer._print_memory_usage()
        print()

        # Warmup
        print("Warmup step...")
        trainer.model.train()
        out = trainer.model(**batch)
        out.loss.backward()
        trainer.model.zero_grad()
        torch.cuda.synchronize()

        vram_after = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM after first step: {vram_after:.2f}GB allocated, {vram_reserved:.2f}GB reserved")
        print()

        # Timed steps
        times = []
        for i in range(n_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = trainer.model(**batch)
            out.loss.backward()
            trainer.model.zero_grad()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  Step {i+1}/{n_steps}: {elapsed:.2f}s (loss={out.loss.item():.4f})")

        avg = sum(times) / len(times)
        total_steps = len(train_data) * config.num_epochs
        # With gradient accumulation, optimizer steps = total_steps / grad_accum
        # but forward+backward happens every step
        est_hours = (avg * total_steps) / 3600

        print()
        print("=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Avg step time:    {avg:.2f}s")
        print(f"Steps per epoch:  {len(train_data)}")
        print(f"Total steps:      {total_steps} ({config.num_epochs} epochs)")
        print(f"Estimated total:  {est_hours:.1f} hours")
        print(f"VRAM peak:        {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
        print(f"Inference speed:  ~{1/avg:.2f} samples/sec")
        return

    # Load and prepare data
    print("Loading data...")
    screenshots_dir = Path(args.screenshots)
    labels_dir = Path(args.labels)

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Convert labeled data to SFT format
    sft_items = convert_labeled_to_sft_format(
        screenshots_dir=screenshots_dir,
        labels_dir=labels_dir,
    )

    if len(sft_items) == 0:
        print("Error: No labeled data found!")
        print(f"  Screenshots: {screenshots_dir}")
        print(f"  Labels: {labels_dir}")
        sys.exit(1)

    print(f"Found {len(sft_items)} labeled samples")

    # Create train/val split
    train_data, val_data = create_sft_dataset(
        sft_items,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    print()

    # Prepare data for trainer
    trainer.prepare_data(train_data, val_data)

    # Evaluation only
    if args.eval_only:
        print("Evaluation only mode...")
        trainer.load_model()
        results = trainer.evaluate()
        print("\nEvaluation complete.")
        return

    # Set seeds for reproducibility
    import random
    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Train
    print("Starting training...")
    trainer.train(resume_from=args.resume)

    # Save model
    output_path = Path(args.output) / "final_model"
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(
        output_path,
        save_merged=save_merged,
    )

    # Final evaluation
    print("\nRunning final evaluation...")
    results = trainer.evaluate()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path}")
    if save_merged:
        print(f"Merged model at: {output_path / 'merged_16bit'}")
        print("\nTo use as GRPO base:")
        print(f"  python scripts/train_grpo.py --model-name {output_path / 'merged_16bit'}")
    print(f"\nFinal mean weighted total: {results.get('mean_weighted_total', 'N/A'):.4f}")
    print(f"  Format gate pass rate: {results.get('mean_format_gate', 'N/A'):.4f}")
    for signal in ("perceptual_accuracy", "decision_alignment", "outcome"):
        val = results.get(f"mean_{signal}", "N/A")
        label = signal.replace("_", " ").title()
        if isinstance(val, float):
            print(f"  {label}: {val:.4f}")
        else:
            print(f"  {label}: {val}")


if __name__ == "__main__":
    main()
