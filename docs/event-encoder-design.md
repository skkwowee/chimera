# Event Encoder Design (Level 2)

This document specifies chimera's Level 2 perception layer — the "Situate"
step in the See → Situate → Think hierarchy. Architectural parent:
`docs/three-level-architecture.md`. Evaluation gates: `docs/methodology.md`
axes 1, 2, 5.

This document supersedes an earlier draft that specified a fixed
sliding-window transformer over per-tick features. That design was rejected
because fixed-tick windows smuggle the same arbitrary-quantization mistake
(see `claude-progress.txt` 2026-04-23 F08v4 incident, "the unit of work was
incorrect") back in at a coarser granularity. The current design treats
events as discrete tokens — the same way LLMs treat words — and lets a
transformer learn the temporal relational structure without prescribing
boundaries.

## 1. Purpose and constraints

The encoder turns a round's sequence of *events* into a sequence of *event
embeddings*, each ~256-dim, that capture the tactical situation surrounding
each event. It is trained once, self-supervised, then frozen. Downstream
consumers (RECALL, BT-head, clustering, GRPO conditioning) read its frozen
outputs.

Four hard constraints, in priority order:

- **C1. Self-supervised only.** No training objective consumes `round_won`.
  This is non-negotiable; the F2 collapse measured in commit 1b387b4's
  positive-rule dose-response is the empirical justification.
  `graf2021dissecting` is the formal justification.
- **C2. Labels derivable from demos.** Every training signal must come from
  awpy-parseable structure of the demo file. No human labeling.
- **C3. Right-sized for the data.** ~80 rounds × ~5–15 events per round × 4
  demos ≈ 2K–5K event tokens. Encoder must be small enough to train without
  overfitting at this scale (~1–3M params target).
- **C4. Frozen at inference.** Downstream training never updates encoder
  weights. The encoder is a tokenizer-equivalent: trained once, then a
  fixed transform.

## 2. Event tokenization

The fundamental unit is the **event token**, not the tick.

### 2.1 Why event-tokens, not tick-windows

awpy parses the demo into discrete event streams (`kills.parquet`,
`bomb.parquet`, `damages.parquet`, `weapon_fires.parquet`, etc.). Each event
already has a tick number, a type, and rich metadata. The game engine
*already* did the boundary detection for us — every prediction-error spike
(`docs/three-level-architecture.md` section 2.2, Event Segmentation Theory
anchor) is logged.

A fixed window of 128 ticks around each event would:

- Slice through engagement chains (a kill at tick T might trigger a trade
  at T+30; a 64-tick post-window cuts the trade in half).
- Treat sub-second flashes and 30-second retakes as equal-magnitude events.
- Re-introduce the determinism problem at coarser scale.

Treating events as tokens avoids all three. The "window" of context is
*learned by attention* over the event sequence rather than imposed by
preprocessing.

### 2.2 What is a token

One token = one awpy event. The token's feature vector encodes:

| Component | Source | Dim |
|---|---|---|
| Event type (one-hot) | `{kill_T, kill_CT, plant, defuse, util_throw, weapon_fire, damage, freeze_end, round_end, contact}` | 10 |
| Event time | Round-relative seconds (continuous) + sinusoidal encoding | 8 |
| Actor | Player ID hash, side, role hash | ~16 |
| Position | (x, y, z) of the principal actor, map-normalized | 3 |
| Target / victim | Player ID hash (if applicable), distance to actor | ~8 |
| Weapon / utility | One-hot over the ~40 game weapons + util types | ~40 |
| Post-event delta | Alive counts post-event (T, CT), bomb status, score | 5 |
| Round context | Round number, score (T, CT), economy phase | 5 |
| Map | One-hot over the 4 maps | 4 |
| **Per-tick context** (optional) | Mean-pooled Level-1 `game_state` numerics over ±0.5s around the event | ~16 |

Approximate per-token feature dim: **~115**. The exact schema lives in
`feature_schema_v1.json` produced by `scripts/build_event_tokens.py` (see
§5) and is the canonical contract every downstream consumer reads.

### 2.3 Sequence definition

A training sample is **one round of event tokens** in chronological order.
Each round is a sequence of length 5–25 (median ~10). No window selection,
no boundary detection — the round itself is the sequence and `round_end` is
the closing token.

For models that need a longer context (e.g., half-level momentum patterns),
half-sequences (12 rounds concatenated) are an open option, but the first
pass operates round-by-round.

### 2.4 The "what about tick-level detail" objection

Some downstream uses genuinely need sub-event-level state (precise spatial
geometry for retake angles, exact crosshair placement). Two answers:

- The per-token feature vector already includes spatial fields (position,
  distance, map). For most tactical decisions this is enough.
- For the rare case where finer detail is needed, attach a small fixed
  budget of "context ticks" to each event token: the Level-1 `game_state`
  at {event_tick − 32, event_tick − 16, event_tick, event_tick + 16,
  event_tick + 32} (≈ ±0.5s at 64Hz). These are mean-pooled into the
  per-token feature row (see §2.2 last row); they do *not* become their
  own tokens.

This is the same trade-off LLMs make with subword tokenization: tokens are
the unit of reasoning, but each token aggregates sub-token information.

## 3. Architecture

A small transformer over event token sequences. The architecture mirrors a
miniature LLM, trained on event sequences instead of word sequences.

```
Input:  (B, L, ~115)         batch × round_length × per-token features
        └── LayerNorm + Linear → (B, L, 256)
        └── + sinusoidal position encoding on event_time (continuous)
        └── + sinusoidal position encoding on event_index (discrete)
        └── Transformer encoder: 4 blocks, d_model=256, n_heads=4,
                                 d_ff=1024, pre-norm, GELU
Output: (B, L, 256)          per-token event embeddings
        + (B, 256)           [CLS] token → round-level embedding
```

Key choices:

- **Encoder, not decoder.** BERT-style, not GPT-style. Each token sees the
  whole round (bidirectional attention) for objectives that benefit from
  full context. The next-event-prediction objective applies a causal mask
  on top of the same encoder weights.
- **Two positional encodings.** Event index (1st event, 2nd event, ...)
  and event time (seconds-since-round-start). Index encoding gives
  ordering; time encoding gives the actual temporal gaps. Two events at
  the same index but different time deltas should be representable.
- **~2M parameters.** 4 blocks × (256² × 4 + 256 × 1024 × 2) ≈ 2M. Sized
  to match the ~3K-token training corpus.
- **Output is per-token.** Each event gets its own 256-dim embedding —
  that's the unit downstream consumers retrieve over. The [CLS] embedding
  is the round-level summary used only for the round-level probe.

## 4. Training objectives

Five self-supervised objectives, all derivable from demo data, none
touching `round_won`. Weighted sum:

| Objective | Description | Loss | Weight | F-collapse risk |
|---|---|---|---|---|
| **Next-event-type prediction** | From positions 0..i (causal mask), predict the type of event i+1. | CE over 10 classes | 1.0 | Low — labels are event types, not outcomes |
| **Time-to-next-event regression** | Predict seconds-to-next-event from token i's embedding. | MSE | 0.5 | Low |
| **Masked event reconstruction** | BERT-style: mask 15% of event tokens' feature vectors, reconstruct from the rest using bidirectional attention. | MSE over masked features | 1.0 | Low |
| **Forward state prediction** | From token i's embedding, predict the structured `game_state` numeric fields at event_time + 2s. | MSE | 0.5 | Low |
| **Per-event side-relative deltas** | Predict signed deltas in (alive_T, alive_CT, bomb_status) caused by event i+1. Forces the model to encode "what kind of consequence is coming." | MSE + CE | 0.2 | Medium — alive-delta correlates with round outcome at the trajectory level; per-event signal is local. Mitigated by side-invariant framing (predict side-relative, not T-relative). |

The five objectives are heterogeneous on purpose. If one collapses or
saturates, the others still produce gradient.

Conspicuously absent: any objective consuming `round_won`. The
probe-outcome metric in §6 measures correlation post-hoc but is **not** a
training signal.

## 5. Data pipeline

Three new scripts. Target paths and responsibilities:

### `scripts/build_event_tokens.py` (TO BUILD)

- Input: `data/processed/demos/*.parquet` (already exists, ~50MB across 4 demos)
- Process: per demo, per round, walk awpy event tables in chronological
  order. For each event, emit a token with the §2.2 schema. Mean-pool the
  ±0.5s context window into the per-token features. Tag with
  (demo_stem, round_num, token_index, event_tick).
- Output: `data/processed/event_tokens/{train,val}.pt`. Two files because
  train/val split is **demo-level** (3 demos train, 1 demo val) — avoids
  cross-round leakage at the sequence level.
- Also outputs `data/processed/event_tokens/feature_schema_v1.json` — the
  canonical schema definition. The encoder, downstream consumers, and the
  evaluation scripts all consult this file.

### `scripts/train_event_encoder.py` (TO BUILD)

- Input: `data/processed/event_tokens/train.pt`, `val.pt`
- Multi-task loss head with the five §4 objectives, weighted sum
- AdamW, lr 3e-4 with linear warmup over 5% of steps then cosine decay
- Mixed precision (bf16 on H100), batch size 16 sequences, ~50 epochs
- Total step count target: ~5K steps; trains in <2hr on a single H100
- Output: `outputs/event_encoder/<run_id>/{encoder.safetensors, train_log.jsonl, val_metrics.json, feature_schema_v1.json}`
- Resumable from checkpoint

### `scripts/build_event_index.py` (TO BUILD)

- Loads the trained encoder + all event tokens
- Emits per-token embeddings into a FAISS index
- Output: `outputs/event_encoder/<run_id>/event_index.faiss` + a parallel
  metadata JSONL with (demo_stem, round_num, token_index, event_type)
- This index replaces the 19-dim `tactical_embedding` FAISS index that
  `src/training/recall.py:RECALLIndex` currently builds

## 6. Evaluation per training run

Six metrics, gating downstream use. Metrics 1–4 are objective accuracy on
a demo-disjoint validation set. Metrics 5–6 are the methodology gates the
encoder must pass before being plugged into Level-3 code paths.

| # | Metric | Target | Floor | Source |
|---|---|---|---|---|
| 1 | Next-event-type val accuracy | ≥ 0.55 | 0.10 (chance over 10 classes) | Training loop |
| 2 | Time-to-next-event val MAE | ≤ 4s | — | Training loop |
| 3 | Masked-recon val MSE | ≥ 30% lower than mean-baseline | mean-baseline | Training loop |
| 4 | Forward state val R² | ≥ 0.40 | 0.0 | Training loop |
| 5 | **σ_s on event embeddings** | **median ∈ [0.15, 0.45]** with same-round mask | — | `scripts/recall_variance_diagnostic.py` (existing) |
| 6 | **Probe-outcome accuracy** (round_won predicted from [CLS]) | ≥ 0.65 val accuracy | 0.50 | `scripts/probe_event_encoder.py` (to build); diagnostic only — see §7 |

**Metrics 5 and 6 are the load-bearing gates.** If σ_s falls outside the
Goldilocks band or probe-outcome is at chance, the encoder is rejected
regardless of how well it scored on the SSL objectives.

The probe-outcome metric is delicate: it must be **diagnostic, not
training-supervised** (per C1). The probe is a tiny MLP trained *post-hoc*
on the frozen encoder; the encoder itself never sees `round_won`.

## 7. What downstream code consumes this

The frozen encoder feeds into:

- **`src/training/recall.py:RECALLIndex.__init__(state_embedder=...)`** —
  the existing plug-in point. Replace the default 19-dim
  `tactical_embedding` with the trained event encoder. Per-event retrieval
  replaces per-tick retrieval. The same-round-mask infrastructure
  (commit 1b387b4) stays in place; it now masks by (demo, round) at
  event-token granularity.
- **`src/training/grpo_trainer.py`** — the GRPO policy receives the
  per-event embedding for the current decision moment as additional
  context. Concretely: when sampling a training instance, look up the
  event embedding at the prompt's anchor tick and concatenate it into the
  model's input representation. Open question on the fusion mechanism
  (concat vs cross-attention; recommend concat first, §9.5).
- **`scripts/eval_scorer.py`** — the `recall_mask` candidate now reads
  "RECALL using the trained event encoder + same-round mask." The
  candidate's σ_s + probe gates inherit from §6 metrics 5–6.
- **`scripts/label_app.py`** — cluster all event tokens via the encoder,
  pick balanced pairs (one per cluster) so BT-head labelers see diverse
  rather than redundant pairs.

The probe used for metric 6 is its own small script —
`scripts/probe_event_encoder.py` (to build). Loads the frozen encoder,
freezes it, attaches a 2-layer MLP head, trains the head on a
demo-disjoint split, reports val accuracy on `round_won`,
`next_action_category`, `forward_state`.

## 8. What this doc does NOT cover

- **Level 1 SFT.** Shipped (F04). Outputs are the input to this layer;
  no redesign needed.
- **Level 3 GRPO loop.** Existing. Only the wiring change in §7 is in
  scope.
- **The reward family choice (judge / RECALL / BT-head).** That is
  `docs/reward-candidates.md` territory and is independent of Level 2 —
  except that any RECALL-family scorer downstream now consumes this
  encoder, and the encoder's gates must pass first
  (`docs/methodology.md` decision protocol, level-dependency rule).
- **Top-down minimap rendering.** Flagged as an enhancement for a v2
  encoder, not v1. The v1 spatial signal is the per-token position field.

## 9. Open design questions

Recommended defaults in parentheses. Each requires a call before
implementation; deferring them by picking the default is fine for the
first pass.

1. **Sequence granularity.** Round-level vs half-level sequences.
   *(Default: round-level. Half-level only if round-level can't capture
   momentum/economy effects, which can be tested once round-level is
   trained.)*
2. **Per-tick context budget.** How many tick samples to mean-pool into
   the per-token features. *(Default: 5 samples at ±0.5s.)*
3. **Loss weighting strategy.** Fixed equal weights vs gradient-magnitude
   balancing vs auto-balancing (GradNorm). *(Default: fixed at the §4
   weights; revisit if any objective saturates.)*
4. **Causal vs bidirectional attention.** Bidirectional for masked-recon,
   forward state, side-relative-deltas; causal for next-event-type. Two
   options: (a) one model with task-conditional attention masks, (b) two
   separate output heads sharing the encoder but applying different masks
   downstream. *(Default: option (a), task-conditional masks — simpler.)*
5. **Fusion into GRPO.** Concatenate event embedding to model input vs
   cross-attention from model to event embedding. *(Default: concat. The
   embedding is small (~256d) and concat is a 1-line wiring change.)*
6. **Train/val split.** Demo-level (3 train, 1 val) vs round-level.
   *(Default: demo-level. Round-level risks F1-style leakage.)*
7. **Map encoding.** Per-map one-hot vs per-map learned embedding.
   *(Default: one-hot for the 4-map first pass. Learnable embedding once
   we have more maps.)*
8. **Encoder reuse for non-event "decision points".** What if GRPO wants
   advice at a moment between awpy events (e.g., mid-rotation)?
   *(Default: for v1, only sample GRPO decision moments at event ticks.
   Generalize later if needed.)*

## 10. Acceptance checklist

Before this encoder is considered shippable for Level 3 downstream use:

- [ ] `scripts/build_event_tokens.py` exists; produces `train.pt` / `val.pt`
- [ ] `scripts/train_event_encoder.py` exists; runs end-to-end
- [ ] Trained checkpoint at `outputs/event_encoder/<run_id>/`
- [ ] §6 metrics 1–4 (SSL objective accuracy) meet targets
- [ ] §6 metric 5 (σ_s in Goldilocks band) passes via
  `scripts/recall_variance_diagnostic.py` with the new encoder as
  `state_embedder`
- [ ] §6 metric 6 (probe-outcome ≥ 0.65) passes via
  `scripts/probe_event_encoder.py`
- [ ] `scripts/build_event_index.py` exists; produces a FAISS index
- [ ] `src/training/recall.py` wired to consume the new encoder (zero-line
  change — `RECALLIndex(state_embedder=...)` already accepts any callable)
- [ ] Smoke run: `scripts/eval_scorer.py` with `--scorers recall_mask`
  produces non-degenerate scores. (If the pseudo-gold path gets replaced
  by BT-pair labeling per `docs/alignment-delta.md`, substitute the
  BT-pair eval as the acceptance test.)

After this checklist passes, Level 2 is shippable. Level 3 scorer-comparison
work resumes with `recall_mask` consuming a real encoder rather than the
19-dim baseline.
