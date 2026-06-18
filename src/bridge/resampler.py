"""Perceiver resampler — bridge-design.md §2.2.

M learned queries cross-attend to the variable-length latent-token sequence and
emit **M fixed soft tokens** in the LLM hidden width. Handles both single-moment
(11 tokens in) and event-sequence (K x 11 in) with the same M out — this fixed-M
compression is what lets a whole round fit in sentence-length LLM input.

Standard Flamingo/IDEFICS PerceiverResampler: per layer the queries cross-attend
to [latents ; queries] (keys/values include the queries themselves), then FFN.
2-4 layers. Output projected to d_llm (read from the LLM config at build time).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class _ResamplerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                                nn.Linear(dim * ff_mult, dim))

    def forward(self, q: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # keys/values = [latents ; queries]  (Flamingo variant)
        kv = torch.cat([self.kv_norm(latents), self.q_norm(q)], dim=1)
        attn_out, _ = self.attn(self.q_norm(q), kv, kv, need_weights=False)
        q = q + attn_out
        q = q + self.ff(self.ff_norm(q))
        return q


class PerceiverResampler(nn.Module):
    def __init__(self, d_in: int, d_llm: int, n_latents: int = 32, depth: int = 3,
                 n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_latents = n_latents
        self.d_llm = d_llm
        self.queries = nn.Parameter(torch.randn(1, n_latents, d_in) * 0.02)
        self.layers = nn.ModuleList(
            [_ResamplerLayer(d_in, n_heads, ff_mult, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_in)
        self.out_proj = nn.Linear(d_in, d_llm)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """latents: [B, T, d_in]  (T = 11 single-moment, or K*11 event-sequence).
        returns soft tokens: [B, n_latents, d_llm]."""
        B = latents.shape[0]
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, latents)
        return self.out_proj(self.out_norm(q))
