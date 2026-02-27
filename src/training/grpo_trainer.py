"""
GRPO (Group Relative Policy Optimization) trainer for CS2 VLM fine-tuning.

Uses Unsloth for memory-efficient training of Qwen3-VL on 24GB VRAM.

Revised reward architecture (D011):
  - Multiplicative format gate (invalid JSON → zero total reward)
  - 3 weighted reward signals passed to TRL's GRPOTrainer:
      1. R_percept  (α=0.20) — merged hard+soft field accuracy
      2. R_decision (β=0.30) — behavioral feature alignment
      3. R_outcome  (γ=0.50) — outcome-modulated decision reward
  - KL regularization (λ_KL=0.02) prevents mode collapse
"""

# Import unsloth first to ensure all optimizations are applied
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from .rewards import (
    REWARD_FUNCTIONS,
    DEFAULT_REWARD_WEIGHTS,
    format_gate_reward,
    perceptual_accuracy_reward,
    decision_alignment_reward,
    outcome_reward,
)


@dataclass
class CS2GRPOConfig:
    """Configuration for GRPO training."""

    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_4bit: bool = True
    use_vllm: bool = True
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

    # GRPO settings
    num_generations: int = 16  # Group size for relative ranking (G in paper)
    max_new_tokens: int = 1024
    temperature: float = 0.7
    importance_sampling_level: str = "sequence"  # GSPO variant for stability

    # KL regularization — prevents mode collapse onto narrow "safe" advice
    # that scores well across diverse game states (D011 Issue 4)
    kl_coef: float = 0.02

    # Reward weights: [R_percept, R_decision, R_outcome]
    # Format gate is multiplicative (applied inside reward wrappers), not here.
    reward_weights: list[float] = field(
        default_factory=lambda: list(DEFAULT_REWARD_WEIGHTS)
    )

    # Output settings
    output_dir: str = "outputs/grpo"
    save_steps: int = 100
    logging_steps: int = 10

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v if not callable(v) else str(v)
            for k, v in self.__dict__.items()
        }


class CS2GRPOTrainer:
    """
    GRPO trainer for CS2 screenshot analysis using Unsloth.

    Uses 4-bit quantization and LoRA for memory efficiency on 24GB VRAM.

    Revised reward architecture (D011):
      - Multiplicative format gate: invalid JSON → all signals return 0.0
      - 3 reward signals passed to TRL for per-signal advantage computation:
          1. R_percept  (0.20) — perceptual accuracy (merged hard+soft fields)
          2. R_decision (0.30) — behavioral feature alignment with pro play
          3. R_outcome  (0.50) — outcome-modulated decision reward (Ω function)
      - KL penalty (λ=0.02) regularizes against SFT reference
    """

    def __init__(self, config: CS2GRPOConfig | None = None):
        self.config = config or CS2GRPOConfig()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.reward_fns: list[Callable] = list(REWARD_FUNCTIONS)
        self.train_dataset = None
        self.val_dataset = None

    def load_model(self):
        """Load Qwen3-VL with Unsloth optimizations."""
        try:
            from unsloth import FastVisionModel
        except ImportError:
            raise ImportError(
                "Unsloth is required for GRPO training. Install with:\n"
                "  pip install unsloth"
            )

        print(f"Loading {self.config.model_name} with Unsloth...")
        print(f"  4-bit quantization: {self.config.use_4bit}")
        print(f"  vLLM fast inference: {self.config.use_vllm}")
        print(f"  LoRA: {self.config.use_lora}")

        # Load model with Unsloth optimizations
        dtype = getattr(torch, self.config.torch_dtype)

        # Note: finetune_vision_layers=False is required for vLLM support
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.use_4bit,
            use_gradient_checkpointing="unsloth",  # Memory efficient
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=not self.config.use_vllm,  # False for vLLM
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
        Prepare datasets for GRPO training.

        Args:
            train_data: List of training samples with 'prompt' and 'ground_truth'
            val_data: Optional validation samples
        """
        from datasets import Dataset

        def format_sample(sample: dict) -> dict:
            """Format a sample for Unsloth GRPO training."""
            prompt_content = sample["prompt"]
            if isinstance(prompt_content, list):
                messages = [{"role": "user", "content": prompt_content}]
            else:
                messages = [{"role": "user", "content": prompt_content}]

            return {
                "messages": messages,
                "ground_truth": sample.get("ground_truth", {}),
                "image_path": sample.get("image_path", ""),
            }

        # Create HuggingFace datasets
        train_formatted = [format_sample(s) for s in train_data]
        self.train_dataset = Dataset.from_list(train_formatted)

        if val_data:
            val_formatted = [format_sample(s) for s in val_data]
            self.val_dataset = Dataset.from_list(val_formatted)

        print(f"Prepared {len(self.train_dataset)} training samples")
        if self.val_dataset:
            print(f"Prepared {len(self.val_dataset)} validation samples")

    def set_reward_functions(
        self,
        reward_fns: list[Callable] | None = None,
    ):
        """
        Override the default reward functions.

        Args:
            reward_fns: List of callables with signature
                        (response: str, ground_truth: dict | None, **kwargs) -> float
        """
        if reward_fns is not None:
            self.reward_fns = reward_fns
        print(f"Reward functions configured: {len(self.reward_fns)} signals")

    def _create_reward_wrappers(self) -> list[Callable]:
        """
        Wrap each reward function for TRL's GRPOTrainer batch interface.

        Each wrapper applies the multiplicative format gate before computing
        the signal reward. This ensures invalid JSON → 0 for ALL signals,
        not just the format signal.

        Signature: (completions: list[str], **kwargs) -> list[float]
        """
        wrappers = []

        for fn in self.reward_fns:
            def make_wrapper(reward_fn):
                def wrapper(completions: list[str], **kwargs) -> list[float]:
                    ground_truths = kwargs.get("ground_truth", [None] * len(completions))
                    results = []
                    for completion, gt in zip(completions, ground_truths):
                        # Multiplicative format gate: invalid JSON → 0.0 for all signals
                        gate = format_gate_reward(completion)
                        if gate == 0.0:
                            results.append(0.0)
                        else:
                            results.append(reward_fn(completion, ground_truth=gt))
                    return results
                return wrapper
            wrappers.append(make_wrapper(fn))

        return wrappers

    def train(self, resume_from: Optional[str] = None):
        """
        Run GRPO training.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if self.model is None:
            self.load_model()

        if self.train_dataset is None:
            raise ValueError("No training data prepared. Call prepare_data() first.")

        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError:
            raise ImportError(
                "TRL is required for GRPO training. Install with:\n"
                "  pip install trl>=0.12.0"
            )

        print("Configuring GRPO trainer...")
        print(f"  Reward signals: {len(self.reward_fns)}")
        print(f"  Reward weights: {self.config.reward_weights}")
        print(f"  KL coefficient: {self.config.kl_coef}")
        print(f"  Group size (num_generations): {self.config.num_generations}")

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # GRPO training configuration
        grpo_config = GRPOConfig(
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
            # GRPO-specific settings
            num_generations=self.config.num_generations,
            max_completion_length=self.config.max_new_tokens,
            temperature=self.config.temperature,
            # GSPO variant for stability
            importance_sampling_level=self.config.importance_sampling_level,
            # KL regularization against SFT reference (D011)
            kl_coef=self.config.kl_coef,
            # Reward weights for the 3 separate reward functions
            reward_weights=self.config.reward_weights,
            # Reporting
            report_to="wandb",
        )

        # Create GRPO trainer with separate reward functions
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            reward_funcs=self._create_reward_wrappers(),
        )

        print("Starting GRPO training...")
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
        save_merged: bool = False,
        quantization_method: str = "q4_k_m",
    ):
        """
        Save the trained model.

        Args:
            output_path: Path to save the model
            save_merged: If True, merge LoRA weights and save full model
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

        Reports both the 3 active reward signals and the multiplicative
        format gate pass rate.

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

        weights = self.config.reward_weights
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
