"""Latent featurizer — bridge-design.md §2.1.

Per world-model token (10 players + 1 global = 11): project the frozen latent
`512 (+ predictive channels) -> d_proj`, LayerNorm, and add a *learned role
embedding* so the resampler keeps agent identity (slot k is always player k,
index 10 is global — identity-fixed, no Hungarian matching; see §2b).

This is the first trainable piece. The world model stays frozen; the featurizer
only re-bases the latent into the bridge's own space and tags slot identity.
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_TOKENS = 11  # 10 players + 1 global, fixed by the world-model grid


class LatentFeaturizer(nn.Module):
    def __init__(self, latent_dim: int = 512, n_pred_channels: int = 0,
                 d_proj: int = 512, n_tokens: int = N_TOKENS):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_pred_channels = n_pred_channels
        self.d_proj = d_proj
        self.n_tokens = n_tokens
        self.proj = nn.Linear(latent_dim + n_pred_channels, d_proj)
        self.norm = nn.LayerNorm(d_proj)
        self.role_emb = nn.Embedding(n_tokens, d_proj)  # slot identity, index 10 = global

    def forward(self, grid: torch.Tensor, channels: torch.Tensor | None = None) -> torch.Tensor:
        """grid: [B, n_tokens, latent_dim]  (one frame's contextualized tokens).
        channels: [B, n_tokens, n_pred_channels] predictive-head outputs, or None.
        returns: [B, n_tokens, d_proj]."""
        B, T, D = grid.shape
        assert T == self.n_tokens and D == self.latent_dim, (grid.shape, self.n_tokens, self.latent_dim)
        if self.n_pred_channels:
            assert channels is not None and channels.shape[-1] == self.n_pred_channels, \
                ("featurizer built with n_pred_channels but channels missing/mismatched", self.n_pred_channels)
            grid = torch.cat([grid, channels], dim=-1)
        x = self.norm(self.proj(grid))
        roles = self.role_emb(torch.arange(T, device=grid.device)).unsqueeze(0)  # [1,T,d_proj]
        return x + roles
