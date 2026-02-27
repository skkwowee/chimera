from .rewards import (
    # Active reward functions (D013 revised architecture)
    format_gate_reward,
    perceptual_accuracy_reward,
    decision_alignment_reward,
    outcome_reward,
    REWARD_FUNCTIONS,
    DEFAULT_REWARD_WEIGHTS,
    # Reward math components (exposed for testing / paper figures)
    compute_player_contribution,
    compute_outcome_modulation,
    # Legacy (backward compat — these are thin wrappers or no-ops)
    hard_field_accuracy_reward,
    soft_field_accuracy_reward,
    consistency_reward,
    reasoning_quality_reward,
)
from .data_utils import (
    GRPODataItem,
    convert_labeled_to_grpo_format,
    create_grpo_dataset,
    GRPODataLoader,
)
from .data_utils import (
    convert_labeled_to_sft_format,
    create_sft_dataset,
)
from .grpo_trainer import CS2GRPOConfig, CS2GRPOTrainer
from .sft_trainer import CS2SFTConfig, CS2SFTTrainer

__all__ = [
    # Active rewards (D013)
    "format_gate_reward",
    "perceptual_accuracy_reward",
    "decision_alignment_reward",
    "outcome_reward",
    "REWARD_FUNCTIONS",
    "DEFAULT_REWARD_WEIGHTS",
    # Reward math components
    "compute_player_contribution",
    "compute_outcome_modulation",
    # Legacy rewards (backward compat)
    "hard_field_accuracy_reward",
    "soft_field_accuracy_reward",
    "consistency_reward",
    "reasoning_quality_reward",
    # Data utilities
    "GRPODataItem",
    "convert_labeled_to_grpo_format",
    "create_grpo_dataset",
    "GRPODataLoader",
    "convert_labeled_to_sft_format",
    "create_sft_dataset",
    # Trainers
    "CS2GRPOConfig",
    "CS2GRPOTrainer",
    "CS2SFTConfig",
    "CS2SFTTrainer",
]
