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
import logging
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.training import (
    CS2SFTConfig,
    CS2SFTTrainer,
    convert_labeled_to_sft_format,
    create_sft_dataset,
)
from src.utils.config import DEFAULT_MODEL_NAME

log = logging.getLogger("sft")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-35B-A3B MoE on CS2 screenshots using SFT (supervised fine-tuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to pre-built SFT dataset JSON (from build_sft_dataset.py). "
             "If provided, --screenshots and --labels are ignored.",
    )
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


def _setup_logging(output_dir: str) -> Path:
    """Configure file + console logging. Returns log file path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_file = output / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def main():
    args = parse_args()

    log_file = _setup_logging(args.output)
    log.info("Log file: %s", log_file)
    log.info("Args: %s", vars(args))

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

    if args.dataset:
        # Load pre-built dataset from build_sft_dataset.py
        import json as _json
        import random as _random
        from src.training.data_utils import (
            prepare_conversation_format,
            format_ground_truth_as_json,
        )

        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)

        with open(dataset_path) as f:
            records = _json.load(f)
        print(f"Loaded {len(records)} records from {dataset_path}")

        # Shuffle and split
        _random.seed(args.seed)
        _random.shuffle(records)
        split_idx = int(len(records) * args.train_ratio)

        train_data = []
        val_data = []
        for i, rec in enumerate(records):
            gs = rec["game_state"]
            analysis = rec.get("analysis", {})
            advice = rec.get("advice", {})

            # Build response JSON (same format as SFT target)
            response = _json.dumps(
                {"game_state": gs, "analysis": analysis, "advice": advice},
                separators=(",", ":"),
            )

            # Find the screenshot image
            img_dir = rec.get("image_dir", "")
            screenshot_id = rec.get("screenshot_id", "")
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = Path(img_dir) / "raw" / f"{screenshot_id}{ext}"
                if candidate.exists():
                    img_path = str(candidate)
                    break
            if img_path is None:
                continue

            messages = prepare_conversation_format(
                image_path=img_path,
                prompt="Read this CS2 screenshot. Extract the game state from the HUD.",
                response=response,
            )

            if i < split_idx:
                train_data.append({"messages": messages})
            else:
                val_data.append({"messages": messages})
    else:
        screenshots_dir = Path(args.screenshots)
        labels_dir = Path(args.labels)

        if not labels_dir.exists():
            print(f"Error: Labels directory not found: {labels_dir}")
            sys.exit(1)

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

    # Signal handler: save checkpoint on SIGTERM/SIGHUP (e.g. SSH disconnect)
    _interrupted = False

    def _signal_handler(signum, frame):
        nonlocal _interrupted
        sig_name = signal.Signals(signum).name
        log.warning("Received %s — will save checkpoint after current step", sig_name)
        _interrupted = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)

    # Train with error handling
    output_path = Path(args.output) / "final_model"
    try:
        log.info("Starting training...")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                log.info("  GPU %d: %s (%.1f GB)", i, torch.cuda.get_device_name(i), mem)

        trainer.train(resume_from=args.resume)

        log.info("Training complete — saving model...")
        trainer.save_model(output_path, save_merged=save_merged)

    except torch.cuda.OutOfMemoryError:
        log.error("CUDA OOM! Saving emergency checkpoint...")
        log.error("GPU memory: allocated=%.2f GB, reserved=%.2f GB",
                  torch.cuda.memory_allocated() / 1024**3,
                  torch.cuda.memory_reserved() / 1024**3)
        try:
            emergency_path = Path(args.output) / "emergency_checkpoint"
            trainer.save_model(emergency_path, save_merged=False)
            log.info("Emergency checkpoint saved to %s", emergency_path)
        except Exception:
            log.error("Failed to save emergency checkpoint: %s", traceback.format_exc())
        sys.exit(1)

    except KeyboardInterrupt:
        log.warning("Interrupted by user — saving checkpoint...")
        try:
            interrupt_path = Path(args.output) / "interrupted_checkpoint"
            trainer.save_model(interrupt_path, save_merged=False)
            log.info("Checkpoint saved to %s", interrupt_path)
        except Exception:
            log.error("Failed to save checkpoint: %s", traceback.format_exc())
        sys.exit(130)

    except Exception:
        log.error("Training failed:\n%s", traceback.format_exc())
        try:
            crash_path = Path(args.output) / "crash_checkpoint"
            if trainer.model is not None:
                trainer.save_model(crash_path, save_merged=False)
                log.info("Crash checkpoint saved to %s", crash_path)
        except Exception:
            log.error("Failed to save crash checkpoint: %s", traceback.format_exc())
        sys.exit(1)

    # Final summary
    log.info("=" * 60)
    log.info("Training complete!")
    log.info("=" * 60)
    log.info("Model saved to: %s", output_path)
    if save_merged:
        log.info("Merged model at: %s", output_path / "merged_16bit")
        log.info("To use as GRPO base:")
        log.info("  python scripts/train_grpo.py --model-name %s", output_path / "merged_16bit")


if __name__ == "__main__":
    main()
