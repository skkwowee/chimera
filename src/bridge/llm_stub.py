"""Tiny causal-LM stub — bridge-design.md §7 ("local CPU smoke on a tiny LLM stub").

A ~hundred-K-param stand-in for Qwen3.6-35B-A3B that mimics the slice of the HF
CausalLM interface the bridge touches, so the scaffold can be exercised on CPU
with zero pod hours and the real model swaps in without code changes:
  - .config.hidden_size / .config.vocab_size
  - .get_input_embeddings() -> nn.Embedding
  - forward(inputs_embeds=..., attention_mask=...) -> obj with .logits, .last_hidden_state

This is NOT a language model in any useful sense — it exists only to make the
soft-prompt injection, ablate hook, and gradient plumbing testable locally.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class _Config:
    hidden_size: int
    vocab_size: int


@dataclass
class _Output:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class TinyLLMStub(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 128, layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.config = _Config(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Parameter(torch.randn(1, 512, hidden_size) * 0.02)
        enc = nn.TransformerEncoderLayer(hidden_size, n_heads, hidden_size * 4,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> _Output:
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        T = inputs_embeds.shape[1]
        h = inputs_embeds + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        h = self.norm(self.encoder(h, mask=mask, src_key_padding_mask=kpm))
        return _Output(logits=self.lm_head(h), last_hidden_state=h)
