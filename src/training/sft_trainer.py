"""
SFT (Supervised Fine-Tuning) trainer for CS2 VLM fine-tuning.

Uses Unsloth for memory-efficient training of Qwen3.5-27B on 24GB VRAM.
Teaches the model output format (valid JSON with game_state/analysis/advice)
and CS2 domain knowledge through supervised learning on demo ground truth data.

SFT should run before GRPO — it outputs a merged 16-bit model that GRPO
loads as its base for reinforcement learning refinement.
"""

# Import unsloth first to ensure all optimizations are applied
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .rewards import (
    DEFAULT_REWARD_WEIGHTS,
    format_gate_reward,
    perceptual_accuracy_reward,
    decision_alignment_reward,
    outcome_reward,
)


@dataclass
class CS2SFTConfig:
    """Configuration for SFT training."""

    # Model settings
    model_name: str = "Qwen/Qwen3.5-27B"
    use_4bit: bool = True
    device: str = "cuda"
    torch_dtype: str = "bfloat16"

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Training settings
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    max_new_tokens: int = 1024

    # SFT-specific settings
    max_seq_length: int = 2048
    finetune_vision_layers: bool = True  # SFT trains vision; GRPO freezes it for vLLM

    # Output settings
    output_dir: str = "outputs/sft"
    save_steps: int = 100
    logging_steps: int = 10

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v if not callable(v) else str(v)
            for k, v in self.__dict__.items()
        }


class CS2SFTTrainer:
    """
    SFT trainer for CS2 screenshot analysis using Unsloth.

    Uses 4-bit quantization and LoRA for memory efficiency on 24GB VRAM.
    Trains the model to produce valid JSON with game_state/analysis/advice
    through supervised learning on demo ground truth data.
    """

    def __init__(self, config: CS2SFTConfig | None = None):
        self.config = config or CS2SFTConfig()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.train_dataset = None
        self.val_dataset = None

    def load_model(self):
        """Load Qwen3.5-27B with Unsloth optimizations."""
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError(
                "Unsloth is required for SFT training. Install with:\n"
                "  pip install unsloth"
            )

        print(f"Loading {self.config.model_name} with Unsloth...")
        print(f"  4-bit quantization: {self.config.use_4bit}")
        print(f"  Finetune vision layers: {self.config.finetune_vision_layers}")
        print(f"  LoRA: {self.config.use_lora}")

        # Load model with Unsloth optimizations
        dtype = getattr(torch, self.config.torch_dtype)

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.use_4bit,
            use_gradient_checkpointing="unsloth",  # Memory efficient
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=self.config.finetune_vision_layers,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                random_state=42,
                use_rslora=True,  # Rank-stabilized LoRA for better performance
                loftq_config=None,
            )

        # Get processor for image handling
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

        print("Model loaded successfully")
        self._print_memory_usage()

    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def prepare_data(
        self,
        train_data: list[dict],
        val_data: list[dict] | None = None,
    ):
        """
        Prepare datasets for SFT training.

        Args:
            train_data: List of training samples with 'messages'
            val_data: Optional validation samples (with 'ground_truth' for evaluate())
        """
        from datasets import Dataset

        self.train_dataset = Dataset.from_list(train_data)

        if val_data:
            self.val_dataset = Dataset.from_list(val_data)

        print(f"Prepared {len(self.train_dataset)} training samples")
        if self.val_dataset:
            print(f"Prepared {len(self.val_dataset)} validation samples")

    def train(self, resume_from: Optional[str] = None):
        """
        Run SFT training.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if self.model is None:
            self.load_model()

        if self.train_dataset is None:
            raise ValueError("No training data prepared. Call prepare_data() first.")

        try:
            from trl import SFTConfig, SFTTrainer
        except ImportError:
            raise ImportError(
                "TRL is required for SFT training. Install with:\n"
                "  pip install trl>=0.12.0"
            )

        print("Configuring SFT trainer...")
        print(f"  Max sequence length: {self.config.max_seq_length}")
        print(f"  Vision layers: {'trainable' if self.config.finetune_vision_layers else 'frozen'}")

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # SFT training configuration
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            bf16=self.config.torch_dtype == "bfloat16",
            fp16=self.config.torch_dtype == "float16",
            # SFT-specific settings
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="",  # We use messages format
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            # Reporting
            report_to="wandb",
        )

        # Create vision data collator for multimodal tokenization
        from unsloth import UnslothVisionDataCollator
        data_collator = UnslothVisionDataCollator(self.model, self.tokenizer)

        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=sft_config,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        print("Starting SFT training...")
        self._print_memory_usage()

        # Train
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()

        print("Training complete!")
        self._print_memory_usage()

        return trainer

    def save_model(
        self,
        output_path: str | Path,
        save_merged: bool = True,
        quantization_method: str = "q4_k_m",
    ):
        """
        Save the trained model.

        Args:
            output_path: Path to save the model
            save_merged: If True, merge LoRA weights and save full model
                         (default True for SFT→GRPO handoff)
            quantization_method: Quantization for merged model (q4_k_m, q8_0, etc.)
        """
        if self.model is None:
            raise ValueError("No model loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError("Unsloth is required. Install with: pip install unsloth")

        if save_merged and self.config.use_lora:
            print(f"Saving merged model to {output_path}...")
            # Save merged 16-bit model
            self.model.save_pretrained_merged(
                output_path / "merged_16bit",
                self.tokenizer,
                save_method="merged_16bit",
            )

            # Optionally save quantized GGUF
            if quantization_method:
                print(f"Saving GGUF with {quantization_method} quantization...")
                self.model.save_pretrained_gguf(
                    output_path / "gguf",
                    self.tokenizer,
                    quantization_method=quantization_method,
                )
        else:
            print(f"Saving LoRA adapter to {output_path}...")
            self.model.save_pretrained(output_path / "lora_adapter")
            self.tokenizer.save_pretrained(output_path / "lora_adapter")

        print(f"Model saved to {output_path}")

    def evaluate(self, eval_data: list[dict] | None = None) -> dict:
        """
        Evaluate the model on validation data.

        Uses the D013 reward architecture: format gate + 3 weighted signals
        (perceptual_accuracy, decision_alignment, outcome) for consistent
        metrics across SFT and GRPO evaluation.

        Args:
            eval_data: Evaluation samples (uses val_dataset if not provided)

        Returns:
            Dictionary with per-signal and aggregate evaluation metrics
        """
        if self.model is None:
            self.load_model()

        eval_dataset = eval_data or self.val_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation data available")

        from tqdm import tqdm

        print("Evaluating model...")

        signal_names = [
            "perceptual_accuracy",
            "decision_alignment",
            "outcome",
        ]
        metrics = {name: [] for name in signal_names}
        metrics["format_gate"] = []
        metrics["weighted_total"] = []

        try:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(self.model)
        except Exception:
            pass

        weights = DEFAULT_REWARD_WEIGHTS
        reward_fns = [
            perceptual_accuracy_reward,
            decision_alignment_reward,
            outcome_reward,
        ]

        for sample in tqdm(eval_dataset, desc="Evaluating"):
            # Generate response
            messages = sample.get("messages", [])
            ground_truth = sample.get("ground_truth", {})

            # Format for generation
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )

            # Decode response
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            # Format gate (multiplicative)
            gate = format_gate_reward(response)
            metrics["format_gate"].append(gate)

            # Compute each reward signal (gated)
            scores = [
                gate * fn(response, ground_truth=ground_truth)
                for fn in reward_fns
            ]

            for name, score in zip(signal_names, scores):
                metrics[name].append(score)

            # Weighted total (already gated)
            weighted = sum(w * s for w, s in zip(weights, scores))
            metrics["weighted_total"].append(weighted)

        # Compute summary statistics
        results = {}
        for key, values in metrics.items():
            if values:
                results[f"mean_{key}"] = sum(values) / len(values)
                results[f"min_{key}"] = min(values)
                results[f"max_{key}"] = max(values)

        results["num_samples"] = len(eval_dataset)

        print("Evaluation results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        return results
