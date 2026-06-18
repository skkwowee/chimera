"""Phase-2 language bridge (bridge-design.md). Frozen world-model latent ->
trainable featurizer + Perceiver resampler -> soft tokens -> Qwen (QLoRA), with a
separate text-only NLA decoder as the label-free faithfulness metric.

The world model and the Qwen base are FROZEN; only the featurizer, resampler,
QLoRA adapters, and (separately) the NLA decoder train.
"""
from .featurizer import LatentFeaturizer
from .resampler import PerceiverResampler
from .nla_decoder import NLADecoder, recon_loss, fraction_variance_explained
from .bridge import LanguageBridge
from .llm_stub import TinyLLMStub

__all__ = ["LatentFeaturizer", "PerceiverResampler", "NLADecoder", "recon_loss",
           "fraction_variance_explained", "LanguageBridge", "TinyLLMStub"]
