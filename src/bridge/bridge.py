"""LanguageBridge — the trainable encoder half wired to the LLM (bridge-design.md §2).

featurizer (§2.1) -> resampler (§2.2) -> soft tokens -> injected into the LLM.

Injection v1 (default, §2.3): LLaVA-style soft-prompt PREFIX — prepend the M soft
tokens to the text embedding sequence. Simplest, fastest to a result, and makes
ablate-the-latent trivial (zero the soft tokens). v2 (gated cross-attention) is an
upgrade only if v1's grounding is weak.

The LLM (stub here, Qwen3.6-35B-A3B QLoRA on the pod) is passed in and stays
frozen except for its LoRA adapters; this module owns only the featurizer +
resampler. The world model is frozen upstream (see wm_interface).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurizer import LatentFeaturizer
from .resampler import PerceiverResampler


class LanguageBridge(nn.Module):
    def __init__(self, llm, latent_dim: int = 512, n_pred_channels: int = 0,
                 d_proj: int = 512, n_soft_tokens: int = 32, resampler_depth: int = 3):
        super().__init__()
        d_llm = llm.config.hidden_size
        self.featurizer = LatentFeaturizer(latent_dim, n_pred_channels, d_proj)
        self.resampler = PerceiverResampler(d_proj, d_llm, n_latents=n_soft_tokens,
                                            depth=resampler_depth)
        self.n_soft_tokens = n_soft_tokens
        self.d_llm = d_llm

    def soft_tokens(self, grid: torch.Tensor, channels: torch.Tensor | None = None,
                    ablate: bool = False) -> torch.Tensor:
        """grid: [B, 11, latent_dim] -> soft tokens [B, M, d_llm].
        ablate=True zeros the soft tokens (the §3 ablate-the-latent control)."""
        soft = self.resampler(self.featurizer(grid, channels))
        if ablate:
            soft = torch.zeros_like(soft)
        return soft

    def lm_forward(self, llm, text_ids: torch.Tensor, grid: torch.Tensor,
                   channels: torch.Tensor | None = None, ablate: bool = False):
        """Prefix soft tokens to the text embeddings and run the LLM. Returns the
        teacher-forced LM loss on the TEXT span only (soft positions are not
        prediction targets) plus the LLM output object."""
        soft = self.soft_tokens(grid, channels, ablate=ablate)            # [B, M, d_llm]
        text_embeds = llm.get_input_embeddings()(text_ids)                # [B, T, d_llm]
        inputs_embeds = torch.cat([soft, text_embeds], dim=1)
        B, M = soft.shape[0], soft.shape[1]
        attn = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
        out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)

        # next-token CE over the text span only: logits at position (M+t) predict text_ids[t+1]
        text_logits = out.logits[:, M:]                                   # [B, T, vocab]
        shift_logits = text_logits[:, :-1].reshape(-1, text_logits.shape[-1])
        shift_labels = text_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels)
        return loss, out
