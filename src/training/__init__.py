from .data_utils import (
    GRPODataItem,
    GRPODataLoader,
    convert_labeled_to_grpo_format,
    convert_labeled_to_sft_format,
    create_grpo_dataset,
    create_sft_dataset,
)
from .grpo_trainer import CS2GRPOConfig, CS2GRPOTrainer
from .recall import (
    RECALLIndex,
    action_embedding,
    recall_reward,
    tactical_embedding,
)
from .rewards import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_FUNCTIONS,
    SIMPLIFIED_REWARD_FUNCTIONS,
    SIMPLIFIED_REWARD_WEIGHTS,
    compute_outcome_modulation,
    # Reward math components (exposed for testing / paper figures)
    compute_player_contribution,
    decision_alignment_reward,
    # Active reward functions (D013 revised architecture)
    format_gate_reward,
    outcome_reward,
    perceptual_accuracy_reward,
    simplified_outcome_reward,
)
from .sft_trainer import CS2SFTConfig, CS2SFTTrainer

__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "REWARD_FUNCTIONS",
    "SIMPLIFIED_REWARD_FUNCTIONS",
    "SIMPLIFIED_REWARD_WEIGHTS",
    "CS2GRPOConfig",
    "CS2GRPOTrainer",
    "CS2SFTConfig",
    "CS2SFTTrainer",
    "GRPODataItem",
    "GRPODataLoader",
    "RECALLIndex",
    "action_embedding",
    "compute_outcome_modulation",
    "compute_player_contribution",
    "convert_labeled_to_grpo_format",
    "convert_labeled_to_sft_format",
    "create_grpo_dataset",
    "create_sft_dataset",
    "decision_alignment_reward",
    "format_gate_reward",
    "outcome_reward",
    "perceptual_accuracy_reward",
    "recall_reward",
    "simplified_outcome_reward",
    "tactical_embedding",
]
