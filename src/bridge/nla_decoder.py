"""NLA reconstructor / decoder half — bridge-design.md §2b.

The bridge (featurizer + resampler) is the *encoder* of a Natural-Language
Autoencoder: latent -> text (NLA antecedent: Anthropic, *Natural Language
Autoencoders*). This module is the *decoder*: text -> latent. Its reconstruction
fidelity is the label-free faithfulness metric — does the generated text render
the world-model latent, or embellish past it?

THE TEXT-ONLY FIREWALL (non-negotiable, §2b). `R` consumes ONLY the decoded
answer string, re-tokenized fresh and re-embedded by a *frozen* text encoder.
There must be ZERO tensor path from the latent / soft tokens / the Qwen residual
stream that ever saw soft tokens. Enforced three ways here:
  1. type assertion — forward() takes integer `y_ids`, never float embeddings;
  2. `.detach()` on the (frozen) re-embedding — no grad escapes the decoder input;
  3. a unit test (scripts/bridge_smoke.py): recon_from_shuffled_text and
     recon_from_empty_text must collapse to the latent-mean floor, and the recon
     loss must produce NO gradient on the featurizer/resampler.
If the firewall is broken, every fidelity number is a lie.

NOT weight-tied with the forward resampler — reuse-in-reverse leaks latent-side
structure. Separate module, trained separately on frozen verbalizer outputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _XAttnLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                                nn.Linear(dim * ff_mult, dim))

    def forward(self, q, kv, key_padding_mask=None):
        a, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv),
                         key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.ff(self.ff_norm(q))
        return q


class NLADecoder(nn.Module):
    """Reverse-Perceiver: M_r learned queries cross-attend to the re-embedded
    answer text H_y, then an MLP head emits the reconstructed latent.

    target_tokens / target_dim set the recon target:
      - pooled-512 floor:  target_tokens=1,  target_dim=512  (z = h.mean(dim=2))
      - per-token grid:    target_tokens=11, target_dim=512  (the real target per
        §7 caveat — holds the per-player structure pooling discards)
    """

    def __init__(self, vocab_size: int, d_txt: int, target_tokens: int = 11,
                 target_dim: int = 512, depth: int = 3, n_heads: int = 8,
                 frozen_embedding: torch.Tensor | None = None):
        super().__init__()
        self.target_tokens = target_tokens
        self.target_dim = target_dim
        # frozen text encoder: its OWN embedding (copy of the LLM's, frozen). In the
        # real pipeline this is the frozen base-LLM text embedding run on the string.
        self.text_emb = nn.Embedding(vocab_size, d_txt)
        if frozen_embedding is not None:
            with torch.no_grad():
                self.text_emb.weight.copy_(frozen_embedding)
        self.text_emb.weight.requires_grad_(False)  # frozen
        self.queries = nn.Parameter(torch.randn(1, target_tokens, d_txt) * 0.02)
        self.layers = nn.ModuleList(
            [_XAttnLayer(d_txt, n_heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d_txt),
                                  nn.GELU(), nn.Linear(d_txt, target_dim))

    def forward(self, y_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """y_ids: [B, T_y] LongTensor token ids of the GENERATED string ONLY.
        returns z_hat: [B, target_tokens, target_dim]."""
        # FIREWALL #1: ids only — a float tensor here means someone piped embeddings
        # (and therefore a gradient path from the soft tokens) straight in.
        assert not torch.is_floating_point(y_ids), \
            "FIREWALL VIOLATION: NLADecoder takes token IDs only, never embeddings/soft-tokens"
        # FIREWALL #2: detach the (frozen) re-embedding so no gradient can escape the
        # decoder's input boundary back toward anything that saw the latent.
        H_y = self.text_emb(y_ids).detach()
        B = H_y.shape[0]
        q = self.queries.expand(B, -1, -1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            q = layer(q, H_y, key_padding_mask=key_padding_mask)
        return self.head(q)  # [B, target_tokens, target_dim]


def recon_loss(z_hat: torch.Tensor, z: torch.Tensor, beta: float = 0.1):
    """L_recon = (1 - cos(z_hat, z)) + beta * MSE(z_hat, z)  (§2b).
    Cosine is the headline (scale-invariant, matches how heads use direction);
    MSE secondary, on standardized targets. Shapes broadcast over tokens."""
    cos = F.cosine_similarity(z_hat, z, dim=-1).mean()
    mse = F.mse_loss(z_hat, z)
    return (1.0 - cos) + beta * mse, cos.item(), mse.item()


def fraction_variance_explained(z_hat: torch.Tensor, z: torch.Tensor, z_mean: torch.Tensor) -> float:
    """R^2 over the latent-mean floor (§3 control c): 1 - SSE / SS_tot. The
    latent-mean baseline is 0 by construction; report THIS, not raw cosine, which
    is deceptively high on the low-rank post-LayerNorm manifold."""
    sse = ((z_hat - z) ** 2).sum().item()
    sstot = ((z - z_mean) ** 2).sum().item()
    return 1.0 - sse / sstot if sstot > 0 else float("nan")
