#!/usr/bin/env python3
"""
GRPO training script for CS2 VLM fine-tuning.

Uses Unsloth for memory-efficient training of Qwen3-VL on 24GB VRAM.

Revised reward architecture (D013):
  - Multiplicative format gate (invalid JSON → zero total reward)
  - 3 weighted signals: R_percept (0.20), R_decision (0.30), R_outcome (0.50)
  - KL regularization (λ=0.02) prevents mode collapse

Usage:
    # Basic training (SFT→GRPO handoff)
    python scripts/train_grpo.py \\
        --model-name outputs/sft/final_model/merged_16bit \\
        --screenshots data/raw --labels data/labeled

    # Custom reward weights
    python scripts/train_grpo.py --reward-weights 0.20 0.30 0.50

    # Adjust KL penalty
    python scripts/train_grpo.py --kl-coef 0.05

    # Resume from checkpoint
    python scripts/train_grpo.py --resume outputs/grpo/checkpoint-500

    # Dry run (load model and check VRAM, no training)
    python scripts/train_grpo.py --dry-run
"""

# Import unsloth first to ensure all optimizations are applied
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    CS2GRPOConfig,
    CS2GRPOTrainer,
    convert_labeled_to_grpo_format,
    create_grpo_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL on CS2 screenshots using GRPO",
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
        default="outputs/grpo",
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
        default="Qwen/Qwen3.5-27B",
        help="Model name or path (use SFT merged output for SFT→GRPO handoff)",
    )
    model_group.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more VRAM)",
    )
    model_group.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM fast inference",
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
        default=16,
        help="LoRA rank",
    )
    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training parameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=3,
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
        default=2e-5,
        help="Learning rate",
    )
    train_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )

    # GRPO settings
    grpo_group = parser.add_argument_group("GRPO settings")
    grpo_group.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Group size for relative ranking (more = better advantage estimates)",
    )
    grpo_group.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    grpo_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    grpo_group.add_argument(
        "--kl-coef",
        type=float,
        default=0.02,
        help="KL divergence penalty coefficient (prevents mode collapse)",
    )

    # Reward weights: [R_percept, R_decision, R_outcome]
    # Format gate is multiplicative (not a weighted signal)
    reward_group = parser.add_argument_group("Reward weights")
    reward_group.add_argument(
        "--reward-weights",
        type=float,
        nargs=3,
        default=[0.20, 0.30, 0.50],
        metavar=("PERCEPT", "DECISION", "OUTCOME"),
        help="Weights for the 3 reward signals (format gate is multiplicative)",
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
        help="Save merged model (LoRA weights merged into base model)",
    )
    output_group.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Quantization method for merged GGUF export",
    )

    # Other options
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from args
    config = CS2GRPOConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        use_vllm=not args.no_vllm,
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
        num_generations=args.num_generations,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        reward_weights=args.reward_weights,
        output_dir=args.output,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    print("=" * 60)
    print("CS2 GRPO Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"4-bit quantization: {config.use_4bit}")
    print(f"LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  - rank: {config.lora_r}")
        print(f"  - alpha: {config.lora_alpha}")
    print(f"KL coefficient: {config.kl_coef}")
    print(f"Reward weights: percept={config.reward_weights[0]}, "
          f"decision={config.reward_weights[1]}, "
          f"outcome={config.reward_weights[2]}")
    print(f"Output: {config.output_dir}")
    print()

    # Create trainer
    trainer = CS2GRPOTrainer(config)

    # Dry run: just load model and check memory
    if args.dry_run:
        print("Dry run: loading model to check VRAM usage...")
        trainer.load_model()
        print("\nDry run complete. Model loaded successfully.")
        return

    # Load and prepare data
    print("Loading data...")
    screenshots_dir = Path(args.screenshots)
    labels_dir = Path(args.labels)

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Convert labeled data to GRPO format
    grpo_items = convert_labeled_to_grpo_format(
        screenshots_dir=screenshots_dir,
        labels_dir=labels_dir,
    )

    if len(grpo_items) == 0:
        print("Error: No labeled data found!")
        print(f"  Screenshots: {screenshots_dir}")
        print(f"  Labels: {labels_dir}")
        sys.exit(1)

    print(f"Found {len(grpo_items)} labeled samples")

    # Create train/val split
    train_data, val_data = create_grpo_dataset(
        grpo_items,
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

    # Train
    print("Starting training...")
    trainer.train(resume_from=args.resume)

    # Save model
    output_path = Path(args.output) / "final_model"
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(
        output_path,
        save_merged=args.save_merged,
        quantization_method=args.quantization if args.save_merged else None,
    )

    # Final evaluation
    print("\nRunning final evaluation...")
    results = trainer.evaluate()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path}")
    print(f"Final mean weighted total: {results.get('mean_weighted_total', 'N/A'):.4f}")
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
