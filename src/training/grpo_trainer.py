"""
GRPO (Group Relative Policy Optimization) trainer for CS2 VLM fine-tuning.

Uses transformers + peft for LoRA training of Qwen3.5-35B-A3B MoE (bf16).

Reward architecture (D024 — simplified 2-signal, primary):
  - Multiplicative format gate (invalid JSON -> zero total reward)
  - 2 weighted reward signals passed to TRL's GRPOTrainer:
      1. R_percept   (alpha=0.20) -- merged hard+soft field accuracy
      2. R_strategy   (1-alpha=0.80) -- simplified outcome: a*w + (1-a)*(1-w)
  - KL regularization (lambda_KL=0.02) prevents mode collapse

Legacy/ablation mode (D013 — 3-signal):
  - 3 weighted reward signals: R_percept (0.20), R_decision (0.30), R_outcome (0.50)
  - Selectable via REWARD_FUNCTIONS / DEFAULT_REWARD_WEIGHTS from rewards.py
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..utils.config import DEFAULT_MODEL_NAME
from .rewards import (
    SIMPLIFIED_REWARD_FUNCTIONS,
    SIMPLIFIED_REWARD_WEIGHTS,
    format_gate_reward,
)


@dataclass
class CS2GRPOConfig:
    """Configuration for GRPO training."""

    # Model settings
    model_name: str = DEFAULT_MODEL_NAME
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
    max_steps: int = -1
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

    # KL regularization -- prevents mode collapse onto narrow "safe" advice
    # that scores well across diverse game states (D013)
    kl_coef: float = 0.02

    # Reward weights: [R_percept, R_strategy] (D024 simplified 2-signal)
    # Legacy 3-signal (D013): [0.20, 0.30, 0.50] for [R_percept, R_decision, R_outcome]
    # Format gate is multiplicative (applied inside reward wrappers), not here.
    reward_weights: list[float] = field(
        default_factory=lambda: list(SIMPLIFIED_REWARD_WEIGHTS)
    )

    # Format gate
    perception_only: bool = False  # If True, format gate only requires game_state (not analysis+advice)

    # Output settings
    output_dir: str = "outputs/grpo"
    save_steps: int = 100
    logging_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v if not callable(v) else str(v)
            for k, v in self.__dict__.items()
        }


class CS2GRPOTrainer:
    """
    GRPO trainer for CS2 screenshot analysis.

    Uses LoRA via peft (bf16 base, no quantization).

    Reward architecture (D024 — simplified 2-signal, primary):
      - Multiplicative format gate: invalid JSON -> all signals return 0.0
      - 2 reward signals passed to TRL for per-signal advantage computation:
          1. R_percept  (0.20) -- perceptual accuracy (merged hard+soft fields)
          2. R_strategy (0.80) -- simplified outcome: a*w + (1-a)*(1-w)
      - KL penalty (lambda=0.02) regularizes against SFT reference

    Legacy/ablation mode (D013 — 3-signal) available via set_reward_functions()
    with REWARD_FUNCTIONS / DEFAULT_REWARD_WEIGHTS from rewards.py.
    """

    def __init__(self, config: CS2GRPOConfig | None = None):
        self.config: CS2GRPOConfig = config or CS2GRPOConfig()
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None  # For text-only generation (avoids processor multimodal issues)
        self.reward_fns: list[Callable[..., Any]] = list(SIMPLIFIED_REWARD_FUNCTIONS)
        self.train_dataset: Any = None
        self.val_dataset: Any = None

    def load_model(self):
        """Load Qwen3.5-35B-A3B MoE (bf16) with LoRA."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5MoeForConditionalGeneration

        dtype = getattr(torch, self.config.torch_dtype)

        print(f"Loading {self.config.model_name}...")
        print(f"  vLLM fast inference: {self.config.use_vllm}")
        print(f"  LoRA: {self.config.use_lora}")

        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=dtype,
        )

        self.model.gradient_checkpointing_enable()

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Apply LoRA if enabled
        # GRPO freezes vision layers when using vLLM
        has_vision = hasattr(self.model.model, "visual")
        if self.config.use_lora:
            target_modules = list(self.config.lora_target_modules)
            if not self.config.use_vllm:
                target_modules = [
                    *target_modules,
                    "visual.*q_proj", "visual.*k_proj", "visual.*v_proj",
                    "visual.*o_proj",
                ]

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_rslora=True,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        print(f"Model loaded | vision encoder: {has_vision}")
        self._print_memory_usage()

    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def prepare_data(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]] | None = None,
    ):
        """
        Prepare datasets for GRPO training.

        Args:
            train_data: List of training samples with 'prompt' and 'ground_truth'
            val_data: Optional validation samples
        """
        from datasets import Dataset

        def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
            """Format a sample for GRPO training (TRL 1.0 expects 'prompt' key)."""
            prompt_content = sample["prompt"]
            if isinstance(prompt_content, list):
                # Multimodal content list — wrap as chat messages for TRL
                prompt = [{"role": "user", "content": prompt_content}]
            elif isinstance(prompt_content, str):
                prompt = [{"role": "user", "content": prompt_content}]
            else:
                prompt = prompt_content

            return {
                "prompt": prompt,
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
        reward_fns: list[Callable[..., Any]] | None = None,
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

    @staticmethod
    def _extract_completion_text(completion: Any) -> str:
        """Extract plain text from a TRL completion (str or message list)."""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            # Conversational format: [{"role": "assistant", "content": "..."}]
            texts = []
            for msg in completion:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        # Multimodal content blocks
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                elif isinstance(msg, str):
                    texts.append(msg)
            return "\n".join(texts)
        return str(completion)

    def _create_reward_wrappers(self) -> list[Callable[..., Any]]:
        """
        Wrap each reward function for TRL 1.0's GRPOTrainer batch interface.

        TRL 1.0 calls: reward_func(prompts=..., completions=..., completion_ids=..., **kwargs)
        Where completions are message lists in conversational mode.

        Each wrapper applies the multiplicative format gate before computing
        the signal reward. This ensures invalid JSON -> 0 for ALL signals.
        """
        wrappers = []

        for fn in self.reward_fns:
            def make_wrapper(reward_fn: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(**kwargs: Any) -> list[float]:
                    completions = kwargs.get("completions", [])
                    ground_truths = kwargs.get("ground_truth", [None] * len(completions))
                    results = []
                    for completion, gt in zip(completions, ground_truths, strict=False):
                        text = CS2GRPOTrainer._extract_completion_text(completion)
                        # Multiplicative format gate: invalid JSON -> 0.0 for all signals
                        gate = format_gate_reward(text)
                        if gate == 0.0:
                            results.append(0.0)
                        else:
                            results.append(reward_fn(text, ground_truth=gt))
                    return results
                return wrapper
            wrappers.append(make_wrapper(fn))

        return wrappers

    def train(self, resume_from: str | None = None):
        """
        Run GRPO training.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if self.model is None:
            self.load_model()
        assert self.model is not None
        assert self.processor is not None

        if self.train_dataset is None:
            raise ValueError("No training data prepared. Call prepare_data() first.")

        try:
            from trl.trainer.grpo_config import GRPOConfig
            from trl.trainer.grpo_trainer import GRPOTrainer
        except ImportError as err:
            raise ImportError(
                "TRL is required for GRPO training. Install with: pip install trl>=0.12.0"
            ) from err

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
            max_steps=self.config.max_steps,
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
            # KL regularization against SFT reference
            beta=self.config.kl_coef,
            # Reward weights for the separate reward functions (2 or 3)
            reward_weights=self.config.reward_weights,
            # Reporting — use wandb if available, otherwise none
            report_to="none",
        )

        # Create GRPO trainer with separate reward functions
        trainer = GRPOTrainer(
            model=self.model,  # type: ignore[reportArgumentType]
            processing_class=self.processor,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            reward_funcs=self._create_reward_wrappers(),  # type: ignore[reportArgumentType]
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

    @staticmethod
    def _compute_sequence_log_probs(
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        completion_start: int,
        **model_kwargs: Any,
    ) -> Any:
        """
        Compute per-token log probs for the completion portion of a sequence.

        Args:
            model: The language model.
            input_ids: Full sequence (prompt + completion), shape [1, seq_len].
            attention_mask: Attention mask, shape [1, seq_len].
            completion_start: Index where completion tokens begin.
            **model_kwargs: Extra inputs (pixel_values, image_grid_thw, etc.).

        Returns:
            Sum of log probs over completion tokens (scalar tensor).
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        # logits shape: [1, seq_len, vocab_size]
        logits = outputs.logits
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, completion_start - 1 : -1, :]
        shift_labels = input_ids[:, completion_start:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather log probs at the actual token positions
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum()

    def train_manual(self, resume_from: str | None = None):
        """
        Manual GRPO training loop — bypasses TRL to avoid the multimodal
        prompt repetition bug (TRL #5120).

        For each sample:
          1. Process multimodal input once (handles images correctly)
          2. Generate G completions sequentially (no prompt repetition)
          3. Score each completion with reward functions
          4. Compute group-normalized advantages
          5. Compute policy gradient loss and backprop

        This is equivalent to TRL's GRPOTrainer but without the broken
        prompt duplication that crashes Qwen3VLProcessor.
        """
        if self.model is None:
            self.load_model()
        assert self.model is not None
        assert self.processor is not None

        if self.train_dataset is None:
            raise ValueError("No training data prepared. Call prepare_data() first.")

        from transformers import GenerationConfig

        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler with warmup
        total_steps = config.max_steps if config.max_steps > 0 else (
            len(self.train_dataset) * config.num_epochs
        )
        warmup_steps = int(total_steps * config.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=0.95,
        )

        reward_weights = config.reward_weights

        print("=" * 60)
        print("Manual GRPO Training (TRL bypass)")
        print("=" * 60)
        print(f"  Samples: {len(self.train_dataset)}")
        print(f"  Max steps: {total_steps}")
        print(f"  Group size (G): {config.num_generations}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  LR: {config.learning_rate}")
        print(f"  Reward signals: {len(self.reward_fns)}")
        print(f"  Reward weights: {reward_weights}")
        print()

        # Training log
        log_path = output_dir / "training_log.jsonl"
        log_f = open(log_path, "a")

        global_step = 0
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_reward = 0.0
        accum_reward_std = 0.0
        accum_format_pass = 0.0
        accum_count = 0

        epoch = 0
        while True:
            epoch += 1
            # Shuffle dataset indices each epoch
            import random as _rng
            indices = list(range(len(self.train_dataset)))
            _rng.shuffle(indices)

            for sample_idx in indices:
                sample = self.train_dataset[sample_idx]
                prompt_content = sample["prompt"]
                ground_truth = sample.get("ground_truth", {})

                # Build chat messages — handle content block lists and plain strings
                has_images = False
                if isinstance(prompt_content, list):
                    if prompt_content and isinstance(prompt_content[0], dict) and "type" in prompt_content[0]:
                        has_images = any(b.get("type") == "image" for b in prompt_content)
                        if has_images:
                            messages = [{"role": "user", "content": prompt_content}]
                        else:
                            text_parts = [
                                b["text"] for b in prompt_content
                                if b.get("type") == "text" and "text" in b
                            ]
                            messages = [{"role": "user", "content": "\n".join(text_parts)}]
                    else:
                        messages = prompt_content
                elif isinstance(prompt_content, str):
                    messages = [{"role": "user", "content": prompt_content}]
                else:
                    messages = prompt_content

                # Tokenize — use processor for multimodal, tokenizer for text-only
                try:
                    if has_images:
                        inputs = self.processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                    else:
                        input_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        inputs = self.tokenizer(input_text, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                              for k, v in inputs.items()}
                except Exception as e:
                    print(f"  [step {global_step}] Skipping sample: {e}")
                    continue

                prompt_len = inputs["input_ids"].shape[1]

                # Separate generation kwargs (pixel_values, image_grid_thw, etc.)
                gen_input_keys = {"input_ids", "attention_mask"}
                model_extra_kwargs = {
                    k: v for k, v in inputs.items() if k not in gen_input_keys
                }

                # --- Step 1: Generate G completions ---
                completions_text: list[str] = []
                completions_ids: list[Any] = []

                # Disable gradient checkpointing for generation (enables KV cache)
                self.model.eval()
                self.model.gradient_checkpointing_disable()
                with torch.no_grad():
                    for _g in range(config.num_generations):
                        output = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            generation_config=gen_config,
                            **model_extra_kwargs,
                        )
                        gen_ids = output[0, prompt_len:]
                        decode_fn = self.tokenizer if not has_images else self.processor
                        text = decode_fn.decode(gen_ids, skip_special_tokens=True)
                        completions_text.append(text)
                        completions_ids.append(gen_ids)

                # --- Step 2: Score completions ---
                rewards = []
                format_passes = 0
                for comp_text in completions_text:
                    gate = format_gate_reward(comp_text, perception_only=config.perception_only)
                    if gate == 0.0:
                        rewards.append(0.0)
                    else:
                        format_passes += 1
                        r = sum(
                            w * fn(comp_text, ground_truth=ground_truth)
                            for w, fn in zip(reward_weights, self.reward_fns, strict=False)
                        )
                        rewards.append(gate * r)

                rewards_t = torch.tensor(rewards, device=self.model.device)

                # --- Step 3: Compute advantages ---
                reward_std = rewards_t.std()
                if reward_std < 1e-8:
                    # All rewards identical — no learning signal, skip
                    continue

                advantages = (rewards_t - rewards_t.mean()) / (reward_std + 1e-8)

                # --- Step 4: Compute policy gradient loss ---
                self.model.train()
                self.model.gradient_checkpointing_enable()
                sample_loss = torch.tensor(0.0, device=self.model.device)

                for g_idx in range(config.num_generations):
                    gen_ids = completions_ids[g_idx]
                    if len(gen_ids) == 0:
                        continue

                    # Build full sequence: prompt + completion
                    full_ids = torch.cat([
                        inputs["input_ids"][0], gen_ids
                    ]).unsqueeze(0)
                    full_mask = torch.ones_like(full_ids)

                    log_prob = self._compute_sequence_log_probs(
                        self.model,
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        completion_start=prompt_len,
                        **model_extra_kwargs,
                    )

                    # REINFORCE loss: -advantage * log_prob
                    sample_loss = sample_loss + (-advantages[g_idx] * log_prob)

                # Average over group
                sample_loss = sample_loss / config.num_generations
                # Scale by gradient accumulation
                scaled_loss = sample_loss / config.gradient_accumulation_steps
                scaled_loss.backward()

                # Accumulate metrics
                accum_loss += sample_loss.item()
                accum_reward += rewards_t.mean().item()
                accum_reward_std += reward_std.item()
                accum_format_pass += format_passes / config.num_generations
                accum_count += 1

                # --- Step 5: Optimizer step ---
                if accum_count % config.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log
                    if global_step % config.logging_steps == 0:
                        avg_loss = accum_loss / accum_count
                        avg_reward = accum_reward / accum_count
                        avg_reward_std = accum_reward_std / accum_count
                        avg_format = accum_format_pass / accum_count

                        log_entry = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(avg_loss, 6),
                            "mean_reward": round(avg_reward, 4),
                            "reward_std": round(avg_reward_std, 4),
                            "format_pass_rate": round(avg_format, 4),
                            "grad_norm": round(grad_norm.item(), 4),
                            "lr": round(scheduler.get_last_lr()[0], 8),
                        }
                        print(
                            f"  step {global_step:4d} | "
                            f"loss {avg_loss:.4f} | "
                            f"reward {avg_reward:.3f}±{avg_reward_std:.3f} | "
                            f"fmt {avg_format:.0%} | "
                            f"grad {grad_norm.item():.3f} | "
                            f"lr {scheduler.get_last_lr()[0]:.2e}"
                        )
                        log_f.write(json.dumps(log_entry) + "\n")
                        log_f.flush()

                        accum_loss = 0.0
                        accum_reward = 0.0
                        accum_reward_std = 0.0
                        accum_format_pass = 0.0
                        accum_count = 0

                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(str(ckpt_dir))
                        self.processor.save_pretrained(str(ckpt_dir))
                        print(f"  Saved checkpoint: {ckpt_dir}")

                    if config.max_steps > 0 and global_step >= config.max_steps:
                        break

            if config.max_steps > 0 and global_step >= config.max_steps:
                break
            if config.max_steps <= 0 and epoch >= config.num_epochs:
                break

        log_f.close()
        print(f"\nManual GRPO training complete — {global_step} steps")
        print(f"Log: {log_path}")
        self._print_memory_usage()

    def save_model(
        self,
        output_path: str | Path,
        save_merged: bool = False,
    ):
        """
        Save the trained model.

        Args:
            output_path: Path to save the model
            save_merged: If True, merge LoRA weights and save full model
        """
        if self.model is None:
            raise ValueError("No model loaded")
        if self.processor is None:
            raise ValueError("No processor loaded")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_merged and self.config.use_lora:
            print(f"Saving merged model to {output_path / 'merged_16bit'}...")
            merged = self.model.merge_and_unload()  # type: ignore[reportCallIssue]
            merged.save_pretrained(str(output_path / "merged_16bit"))  # type: ignore[reportCallIssue]
            self.processor.save_pretrained(str(output_path / "merged_16bit"))
        else:
            print(f"Saving LoRA adapter to {output_path / 'lora_adapter'}...")
            self.model.save_pretrained(str(output_path / "lora_adapter"))
            self.processor.save_pretrained(str(output_path / "lora_adapter"))

        print(f"Model saved to {output_path}")

    def evaluate(self, eval_data: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """
        Evaluate the model on validation data.

        Reports the active reward signals (2 for simplified, 3 for legacy)
        and the multiplicative format gate pass rate.

        Args:
            eval_data: Evaluation samples (uses val_dataset if not provided)

        Returns:
            Dictionary with per-signal and aggregate evaluation metrics
        """
        if self.model is None:
            self.load_model()
        assert self.model is not None
        assert self.processor is not None

        eval_dataset = eval_data or self.val_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation data available")

        from tqdm import tqdm

        print("Evaluating model...")

        # Derive signal names from configured reward functions
        # Default (D024): 2 signals [R_percept, R_strategy]
        # Legacy (D013):  3 signals [R_percept, R_decision, R_outcome]
        reward_fns = list(self.reward_fns)
        weights = self.config.reward_weights
        if len(reward_fns) == 2:
            signal_names = ["perceptual_accuracy", "strategy"]
        else:
            signal_names = [
                "perceptual_accuracy",
                "decision_alignment",
                "outcome",
            ]
        metrics: dict[str, list[float]] = {name: [] for name in signal_names}
        metrics["format_gate"] = []
        metrics["weighted_total"] = []

        self.model.eval()  # type: ignore[reportOptionalMemberAccess]

        for sample in tqdm(eval_dataset, desc="Evaluating"):
            # Generate response (sample is a dict from either list[dict] or HF Dataset)
            sample_dict: dict[str, Any] = dict(sample) if not isinstance(sample, dict) else sample
            messages = sample_dict.get("messages", [])
            ground_truth = sample_dict.get("ground_truth", {})

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
            response = self.processor.decode(
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

            for name, score in zip(signal_names, scores, strict=False):
                metrics[name].append(score)

            # Weighted total (already gated)
            weighted = sum(w * s for w, s in zip(weights, scores, strict=False))
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
