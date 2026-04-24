"""Adapter that wraps a fine-tuned sentence-transformer into a callable
compatible with RECALLIndex(state_embedder=...).

Why: RECALLIndex was built around `tactical_embedding: dict -> np.ndarray`.
The learned encoder from scripts/train_state_embedding.py is a
SentenceTransformer that takes text. This adapter bridges the two by
serializing the game_state to JSON and encoding it, matching how the
learned encoder was trained.

Lazy-loaded so `import src.training.recall` doesn't pull in
sentence_transformers when the RECALL path isn't exercised.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class LearnedStateEmbedder:
    """Callable(gs: dict) -> np.ndarray wrapping a SentenceTransformer."""

    def __init__(self, encoder_path: str, redact: bool = True):
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(encoder_path)
        self._redact = redact
        self._encoder_path = encoder_path

    def _state_text(self, gs: dict[str, Any]) -> str:
        # Mirror scripts/train_state_embedding.py — the encoder was trained on
        # the same serialization, so inference must match. Redaction default
        # matches what learned_v3_alive was trained with (--redact).
        data = gs
        if self._redact:
            # Same _REDACT_KEYS as in training script. Kept inline so this
            # module is self-contained.
            redact_keys = {
                "player_name", "player", "teammates", "team", "opponent_team",
                "round_number", "round",
                "time_remaining", "time", "round_time", "tick",
                "score", "current_player",
            }

            def _strip(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _strip(v) for k, v in obj.items() if k not in redact_keys}
                if isinstance(obj, list):
                    return [_strip(v) for v in obj]
                return obj
            data = _strip(gs)
        return json.dumps(data, sort_keys=True, ensure_ascii=False)

    def __call__(self, game_state: dict[str, Any]) -> np.ndarray:
        text = self._state_text(game_state)
        emb = self._encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return emb.astype(np.float32)

    def __repr__(self) -> str:
        return f"LearnedStateEmbedder({self._encoder_path})"
