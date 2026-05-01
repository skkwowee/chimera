#!/usr/bin/env python3
"""
GRPO training script for CS2 VLM fine-tuning.

Uses transformers + peft for LoRA training of Qwen3.5-35B-A3B MoE (bf16).

Revised reward architecture (D024):
  - Multiplicative format gate (invalid JSON -> zero total reward)
  - 2 simplified signals: R_percept + R_strategy (D024 — replaces legacy 3-signal D013)
  - R_decision and R_outcome are ablation baselines only
  - KL regularization (lambda=0.02) prevents mode collapse

Usage:
    # Basic training (SFT->GRPO handoff)
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

import argparse
import sys
from pathlib import Path

import torch

from src.training import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_FUNCTIONS,
    SIMPLIFIED_REWARD_FUNCTIONS,
    SIMPLIFIED_REWARD_WEIGHTS,
    CS2GRPOConfig,
    CS2GRPOTrainer,
    convert_labeled_to_grpo_format,
    create_grpo_dataset,
    judge_reward,
    perceptual_accuracy_reward,
    recall_reward,
)
from src.training.bt_reward import bt_reward
from src.utils.config import DEFAULT_MODEL_NAME

# ---------------------------------------------------------------------------
# Reward mode presets
# ---------------------------------------------------------------------------
REWARD_MODES = {
    "simplified": {
        "functions": SIMPLIFIED_REWARD_FUNCTIONS,
        "weights": SIMPLIFIED_REWARD_WEIGHTS,
        "description": "2-signal: R_percept (0.20) + R_outcome_simple (0.80)",
    },
    "recall": {
        "functions": [perceptual_accuracy_reward, recall_reward],
        "weights": [0.20, 0.80],
        "description": "2-signal: R_percept (0.20) + R_RECALL (0.80)",
    },
    "judge": {
        "functions": [perceptual_accuracy_reward, judge_reward],
        "weights": [0.20, 0.80],
        "description": "2-signal: R_percept (0.20) + R_judge (0.80) -- "
                       "Claude judge ranks the G completions per step. "
                       "Replaces RECALL (see claude-progress.txt 2026-04-23).",
    },
    "bt_head": {
        "functions": [perceptual_accuracy_reward, bt_reward],
        "weights": [0.20, 0.80],
        "description": "2-signal: R_percept (0.20) + R_BT_head (0.80) -- "
                       "neural reward head trained on expert human preferences. "
                       "No per-step API call. Set CHIMERA_BT_HEAD_PATH to the "
                       "trained head dir (e.g., /workspace/outputs/bt_head/v1).",
    },
    "legacy": {
        "functions": REWARD_FUNCTIONS,
        "weights": DEFAULT_REWARD_WEIGHTS,
        "description": "3-signal: R_percept (0.20) + R_decision (0.30) + R_outcome (0.50)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-35B-A3B MoE on CS2 screenshots using GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to JSONL file with pre-built samples (prompt + ground_truth). "
             "If set, --screenshots and --labels are ignored.",
    )
    data_group.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional JSONL with per-sample {idx, demo_stem, round_num, ...} "
             "source metadata. If provided, RECALL builds with source keys "
             "and excludes same-round neighbors at query time. Without it, "
             "RECALL silently includes same-round neighbors which leak "
             "round_won and collapse V̂ — see scripts/recall_variance_diagnostic.py.",
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
        default=DEFAULT_MODEL_NAME,
        help="Model name or path (use SFT merged output for SFT->GRPO handoff)",
    )
    model_group.add_argument(
        "--sft-adapter",
        action="append",
        default=None,
        help="Path to a PEFT LoRA adapter to merge into the base before adding "
             "the GRPO LoRA. Repeat the flag to merge multiple adapters IN "
             "ORDER (useful for resuming: original SFT first, then the GRPO "
             "checkpoint you want to continue from).",
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
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (overrides epochs if > 0)",
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
    grpo_group.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap on number of val samples for the final evaluation. "
             "Default: use the full val split.",
    )

    # Reward configuration
    reward_group = parser.add_argument_group("Reward settings")
    reward_group.add_argument(
        "--reward-mode",
        type=str,
        default="simplified",
        choices=list(REWARD_MODES.keys()),
        help="Reward architecture: simplified (2-signal, default), "
             + "recall (2-signal w/ recall, pending), legacy (original 3-signal)",
    )
    reward_group.add_argument(
        "--reward-weights",
        type=float,
        nargs="+",
        default=None,
        help="Override reward weights (must match number of signals in selected mode). "
             + "If not set, uses the mode's default weights.",
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

    # Other options
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual GRPO loop (bypasses TRL — required for multimodal Qwen3-VL)",
    )
    parser.add_argument(
        "--allow-slow-fallback",
        action="store_true",
        help="Skip the kernel preflight. Without causal_conv1d + flash_linear_attention "
             + "the run is 5-10x slower (last attempt: 14h/40 steps). Do not use in prod.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend. flash_attention_2 is wheel-installable and unrelated "
             + "to the causal_conv1d install — use it. sdpa is a fine fallback.",
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Relax format gate to only require game_state key (for text-only smoke tests)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve reward mode
    mode = REWARD_MODES[args.reward_mode]
    reward_fns = mode["functions"]
    reward_weights = args.reward_weights if args.reward_weights is not None else list(mode["weights"])

    if len(reward_weights) != len(reward_fns):
        print(f"Error: --reward-weights expects {len(reward_fns)} values for "
              + f"mode '{args.reward_mode}', got {len(reward_weights)}")
        sys.exit(1)

    # Create config from args
    config = CS2GRPOConfig(
        model_name=args.model_name,
        sft_adapter=args.sft_adapter or [],
        use_vllm=not args.no_vllm,
        torch_dtype=args.dtype,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        num_generations=args.num_generations,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        reward_weights=reward_weights,
        output_dir=args.output,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        perception_only=args.perception_only,
        allow_slow_fallback=args.allow_slow_fallback,
        attn_implementation=args.attn_impl,
    )

    print("=" * 60)
    print("CS2 GRPO Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Reward mode: {args.reward_mode} — {mode['description']}")
    print(f"LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"  - rank: {config.lora_r}")
        print(f"  - alpha: {config.lora_alpha}")
    print(f"KL coefficient: {config.kl_coef}")
    print(f"Reward weights: {config.reward_weights}")
    print(f"Output: {config.output_dir}")
    print()

    # Create trainer
    trainer = CS2GRPOTrainer(config)
    trainer.set_reward_functions(reward_fns)

    # Dry run: just load model and check memory
    if args.dry_run:
        print("Dry run: loading model to check VRAM usage...")
        trainer.load_model()
        print("\nDry run complete. Model loaded successfully.")
        return

    # Load and prepare data
    print("Loading data...")

    if args.data:
        # Load from pre-built JSONL file
        import json as json_mod
        import random as random_mod

        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            sys.exit(1)

        all_samples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json_mod.loads(line))

        print(f"Loaded {len(all_samples)} samples from {data_path}")

        # Split into train/val
        random_mod.seed(args.seed)
        random_mod.shuffle(all_samples)
        split_idx = int(len(all_samples) * args.train_ratio)
        train_data = all_samples[:split_idx]
        val_data = all_samples[split_idx:]
    else:
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

    # Merge source metadata into samples for same-round exclusion in RECALL.
    # The source JSONL is keyed by `idx` into the ORIGINAL dataset; after
    # train/val shuffle we no longer know each sample's original idx, so we
    # match on (prompt, ground_truth.player_health) — the prompt header is
    # unique per (demo, round, tick, player) and players in different demos
    # render different prompts even if hp matches. Cheap and reliable enough
    # given n=2013 samples. If --source is omitted, RECALL falls back to
    # current behavior (no same-round masking).
    if args.source:
        import json as _json
        src_path = Path(args.source)
        if not src_path.exists():
            print(f"Error: --source file not found: {src_path}")
            sys.exit(1)
        # Load and key source records by their `idx` (= line number in the
        # ORIGINAL dataset). We re-link by re-reading the dataset in original
        # order to get prompt → idx, then sample → prompt → idx → source.
        # This works whether train_data was loaded from --data (shuffled) or
        # built from labels (different ordering entirely).
        if args.data:
            originals = []
            with open(args.data) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        originals.append(_json.loads(line))
            prompt_to_idx: dict[str, int] = {}
            for i, s in enumerate(originals):
                key = _json.dumps(s.get("prompt"), sort_keys=True)
                prompt_to_idx[key] = i
        else:
            prompt_to_idx = {}

        sources_by_idx: dict[int, dict] = {}
        with open(src_path) as f:
            for line in f:
                rec = _json.loads(line)
                sources_by_idx[int(rec["idx"])] = rec

        n_merged = 0
        for split in (train_data, val_data):
            for sample in split:
                key = _json.dumps(sample.get("prompt"), sort_keys=True)
                idx = prompt_to_idx.get(key)
                if idx is None:
                    continue
                src = sources_by_idx.get(idx)
                if src is None:
                    continue
                gt = sample.setdefault("ground_truth", {})
                gt["source"] = {
                    "demo_stem": src.get("demo_stem"),
                    "round_num": src.get("round_num"),
                    "tick": src.get("tick"),
                    "player_name": src.get("player_name"),
                    "player_side": src.get("player_side"),
                    "map_name": src.get("map_name"),
                }
                n_merged += 1
        total = len(train_data) + len(val_data)
        print(f"Merged source metadata into {n_merged}/{total} samples "
              f"(missing entries get no same-round masking)")

    # Prepare data for trainer
    trainer.prepare_data(train_data, val_data)

    # Build RECALL index if using recall reward mode
    if args.reward_mode == "recall":
        trainer.build_recall_index(train_data)

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Train
    print("Starting training...")
    if args.manual:
        trainer.train_manual(resume_from=args.resume)
    else:
        trainer.train(resume_from=args.resume)

    # Save model
    output_path = Path(args.output) / "final_model"
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(
        output_path,
        save_merged=args.save_merged,
    )

    # Final evaluation
    eval_data = None
    if args.max_eval_samples is not None and trainer.val_dataset is not None:
        eval_data = list(trainer.val_dataset)[: args.max_eval_samples]
        print(f"\nRunning final evaluation on {len(eval_data)} samples (capped from {len(trainer.val_dataset)})...")
    else:
        print("\nRunning final evaluation...")
    results = trainer.evaluate(eval_data=eval_data)

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
