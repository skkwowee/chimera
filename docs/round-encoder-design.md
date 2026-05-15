# Round Encoder Design (Level 2)

This document specifies chimera's Level 2 perception layer — the "Situate"
step in the See → Situate → Think hierarchy. Architectural parent:
`docs/three-level-architecture.md`. Evaluation gates: `docs/methodology.md`
axes 1, 2, 5.

## Design history (read this once, then skip)

This is the fourth revision of the Level 2 design and supersedes the prior
three. The filename change `event-encoder-design.md` → `round-encoder-design.md`
reflects the v3 architectural reframe; v4 keeps the round-as-sequence
framing but switches from a bidirectional encoder to a **causal decoder**.

| Rev | Design | Why rejected / superseded |
|---|---|---|
| v1 | Fixed-tick windows around each event (e.g., ±128 ticks) | Smuggled arbitrary quantization back at coarser scale |
| v2 | Event-token transformer (awpy events as discrete tokens) | awpy event times are *outcome*-timed, not *cause*-timed — the kill tick is the answer key, not the question |
| v3 | Round-as-sequence-of-downsampled-ticks, **bidirectional** transformer, awpy events as query positions | Bidirectional attention lets future-state (esp. round outcome) leak backward into every per-tick embedding — exactly the F2 collapse path |
| **v4 (current)** | **Round-as-sequence-of-downsampled-ticks, *causal decoder*, four predict-forward objectives** | Causal attention structurally prevents future-leakage; matches the OpenAI Five precedent (causal LSTM); aligns with decision-support use case where embedding-at-T should know the past, not the future |

The v4 design — implemented in `scripts/train_round_encoder.py` and
`scripts/build_tick_sequences.py` — uses causal self-attention so that
`h_T` is a function of ticks `0..T` only. F2-safe by construction. The
four objectives are all predict-forward (next-tick MSE, multi-horizon
MSE, time-to-next-event, next-event-type CE) — none consume `round_won`.

§3 (Architecture) and §4 (Training objectives) below were written for v3
and contain references to bidirectional attention + masked-reconstruction
objectives. **The shipped code follows v4** (see the script docstrings for
the canonical current spec). A full doc rewrite to v4 is pending; for now,
treat v3 sections as historical and the scripts as ground truth.

## 1. Purpose and constraints

The encoder turns a round's sequence of game-state observations into a
contextualized per-tick embedding stream. At any tick T in the round, the
embedding represents "what is tactically happening at moment T, in the
context of everything else in this round." Downstream consumers query the
stream at the tick they care about — typically an awpy event tick, but they
can query at any tick.

It is trained once, self-supervised, then frozen. Downstream consumers
(RECALL, BT-head, clustering, GRPO conditioning) read its frozen outputs.

Four hard constraints, in priority order:

- **C1. Self-supervised only.** No training objective consumes `round_won`.
  This is non-negotiable; the F2 collapse measured in commit 1b387b4's
  positive-rule dose-response is the empirical justification.
  `graf2021dissecting` is the formal justification.
- **C2. Labels derivable from demos.** Every training signal must come from
  awpy-parseable structure of the demo file. No human labeling.
- **C3. No fixed temporal windows.** The round is the unit of context. No
  preprocessing imposes "events live in this many ticks before/after." The
  encoder attends over the whole round and figures out its own temporal
  scope per query.
- **C4. Frozen at inference.** Downstream training never updates encoder
  weights. The encoder is a tokenizer-equivalent: trained once, then a
  fixed transform.

## 2. Input: round-as-sequence-of-downsampled-ticks

### 2.1 Why downsampled, not raw 64 Hz

CS2 servers tick at 64 Hz. A 2-minute round is ~7,500 ticks. Acting at full
tickrate is wasteful for strategic representation — adjacent ticks are
near-duplicates and strategic decisions don't change at sub-second
granularity. OpenAI Five made the same call for Dota: 30 Hz game → 7.5 Hz
acted-on rate (every 4th frame).

We downsample to **8 Hz** (every 8th tick). This gives:

- ~960 tokens for a 2-minute round
- ~125 ms between tokens — finer than human reaction time (200-300 ms)
- ~6× compute saving vs raw 64 Hz with no strategic information loss

Token sequence per round is therefore short enough that a small transformer
can apply full bidirectional attention without efficient-attention tricks.

### 2.2 Per-tick feature vector (~750 dim)

At each downsampled tick, the input feature row encodes:

| Component | Source | Per-tick dim |
|---|---|---|
| **Per-player block × 10 players** | | ~70 each = 700 |
| ├─ Position | (x, y, z), map-normalized | 3 |
| ├─ View angle | (yaw, pitch), sinusoidal | 4 |
| ├─ Velocity | (dx, dy, dz) finite-difference | 3 |
| ├─ HP / armor | Normalized | 2 |
| ├─ Money | Log-normalized | 1 |
| ├─ Weapon primary | One-hot over ~40 game weapons | 40 |
| ├─ Weapon secondary | One-hot pistol/shotgun set | 10 |
| ├─ Utility loadout | Multi-hot over 6 util types | 6 |
| └─ Status bits | isAlive, isScoped, isCrouched | 3 |
| **Global state block** | | ~50 |
| ├─ Bomb status | One-hot {none, carried, planted_A, planted_B, defused} | 5 |
| ├─ Bomb position | (x, y), map-normalized; defined if planted | 2 |
| ├─ Bomb timer | Seconds remaining post-plant | 1 |
| ├─ Round timer | Seconds remaining | 1 |
| ├─ Score | (T, CT) | 2 |
| ├─ Round number | Discrete | 1 |
| ├─ Map | One-hot over 4 maps | 4 |
| ├─ Phase | Freeze / live / post-plant / end | 4 |
| └─ Round-relative time | Continuous + sinusoidal encoding | 16 |
| **Active utility on map** | Smoke / molly / flash positions + timers, max-pooled over instances (variable cardinality) | ~14 |

Approximate per-tick feature dim: **~750**. The exact schema lives in
`feature_schema_v1.json` produced by `scripts/build_tick_sequences.py`.

### 2.3 Player identity

Per-player features don't include identity (no player ID embeddings). Each
player slot is a positional slot: T1, T2, T3, T4, T5, CT1, CT2, CT3, CT4,
CT5. This is intentional: the encoder should learn *role* patterns, not
*player* patterns. If we later want player-specific behavior, we can add
a per-player learned embedding as an opt-in feature.

### 2.4 The awpy events question

awpy event ticks are not used as token boundaries. They are used in two
narrower ways:

- **Query positions.** Downstream code that wants "the embedding at the
  moment of the X kill" maps the kill's tick to the nearest downsampled
  position and reads the encoder's output there.
- **Time-to-next-event regression label.** A training objective: at each
  tick, predict seconds-to-next-awpy-event. The supervision is *time
  until event*, not "what is an event." The encoder is never told the
  exact tick an event happens; it's only told an event happened *within a
  given seconds horizon*.

This sidesteps the cause-vs-outcome ambiguity in awpy timestamps. The
encoder's attention pattern around the awpy tick reveals what the encoder
considers "the event."

## 3. Architecture

Bidirectional transformer over per-tick features. The architecture mirrors
a small BERT trained on tick sequences instead of token sequences.

```
Input:  (B, L, ~750)         B rounds × L≈1000 downsampled ticks × per-tick features
        └── LayerNorm + Linear → (B, L, 1024)
        └── + sinusoidal position encoding (on tick index)
        └── + sinusoidal time encoding (on round-relative seconds)
        └── Transformer encoder: 4 blocks, d_model=1024, n_heads=8,
                                 d_ff=4096, pre-norm, GELU, dropout=0.15
Output: (B, L, 1024)         per-tick contextualized embeddings
```

Sizing rationale (vs the prior 256-dim spec): CS2 has more latent state
than fits in 256 dim — per-player util loadouts (~16 dim each × 10), last-
seen positions, smoke/molly timers, round economy context, momentum. 1024
dim gives room. Per-tick input is already ~750 dim, so the hidden dim
should be comparable.

Parameter count:

- Per block: ~12.6M (attention 4M + FFN 8.4M)
- 4 blocks + input projection + LayerNorms: ~50M total

This is on the edge of what our data scale supports — see §5 regularization.

## 4. Training objectives

Five self-supervised objectives. Every tick contributes a gradient (no
masking-based sparsity); each loss is computed at every position. None
touch `round_won`.

| Objective | Description | Loss | Weight | Why it's safe (no F2 collapse) |
|---|---|---|---|---|
| **Forward tick prediction** | From tick T's embedding, predict the structured per-tick features at T+1 (next downsampled tick). | MSE | 1.0 | Predicts *physics*, not outcome |
| **Forward state regression — multi-horizon** | From tick T's embedding, predict features at T+8, T+32 (≈1s, 4s ahead). Multi-horizon prevents short-horizon shortcut. | MSE | 0.8 | Same |
| **Time-to-next-event regression** | At each tick, predict seconds-to-next-awpy-event (any type). | Smooth L1 | 0.5 | Continuous regression, no class label |
| **Next-event-type prediction** | At each tick, classify the type of the next awpy event ∈ {kill_T, kill_CT, plant, defuse, util_throw, weapon_fire, freeze_end, round_end}. | CE | 0.5 | Labels are event types, not outcomes |
| **Tick reconstruction** | At each tick, reconstruct the per-tick input features from the embedding (autoencoder head). | MSE | 0.3 | Forces information preservation |

No 15%-masking-style objective. Every position contributes every loss at
every step. Total per-token loss = weighted sum.

Conspicuously absent:
- No `round_won` objective at any level. Probe-outcome (§6 metric 6) is
  evaluated post-hoc on the frozen encoder; it is **not** in the training
  loop.
- No contrastive objective. The forward-prediction objectives already pull
  temporally-close ticks together (predicting T+1 from T means the
  embedding of T must carry T+1-relevant info).

## 5. Training procedure

### 5.1 Optimizer

```
AdamW(
  lr           = 3e-4
  betas        = (0.9, 0.95)        # transformer-tuned, less momentum than default
  eps          = 1e-8
  weight_decay = 0.01
)
```

### 5.2 Schedule

- **Warmup:** linear over the first 5% of total steps (LR 0 → 3e-4)
- **Decay:** cosine to 10% of peak LR over remaining 95% of steps
- **Total steps:** ~8,000 — see §5.6 for the math

### 5.3 Batching

- **Sequence:** one round = one sequence, length ~1000 tokens
- **Per-device batch size:** 8 rounds (8,000 tokens per device)
- **Gradient accumulation:** 4 (effective batch: 32 rounds = 32K tokens)
- **Precision:** bf16 mixed precision
- **Hardware:** 1 × H100 (80GB) or 1 × H200 (144GB)

At bf16 + batch 8 + d_model 1024, peak memory is ~30GB. No gradient
checkpointing needed.

### 5.4 Regularization (load-bearing for this model size)

Without this stack the 50M-param model will memorize the small dataset:

- **Dropout:** 0.15 throughout the transformer (attention + FFN)
- **Weight decay:** 0.01 (in AdamW)
- **Per-tick input dropout:** randomly zero individual per-player slots
  with probability 0.10 — forces the encoder to not over-rely on any
  single player's observation
- **Tick-subsample augmentation:** each epoch, vary the downsampling phase
  offset (start at tick 0, 2, 4, 6) — every demo becomes effectively 4-8
  variants
- **Label smoothing:** 0.1 on next-event-type CE
- **Gradient clipping:** norm 1.0

### 5.5 Train/val split

**Demo-level** split, not round-level: 3 demos for train, 1 demo held out
for val. Round-level splits risk F1-style leakage (sibling rounds within
the same demo share player styles, map setups, momentum).

If the demo set grows past 4 (see §11 prerequisite), keep at least 20% of
demos held out, never less than 1 full demo.

### 5.6 Step / epoch budget

With 4 demos × 80 rounds = ~300 train rounds and effective batch 32:

- ~10 batches per epoch
- 100 epochs target
- ≈ 1000 batches
- Total optimizer steps after grad-accum: ~250

That's too few for a 50M-param model to converge. Two options:

- **Option A (default):** scale demos to 40 first (§11), giving ~3000 train
  rounds, ~100 batches/epoch, ~80 epochs, ~8000 steps. **This is the
  recommended path.**
- **Option B (smoke-test only):** stay at 4 demos, run 200 epochs with
  heavier regularization, accept that val-loss-plateau happens early. Use
  this only to validate the loss curves converge at all.

### 5.7 Validation & checkpointing

- **Eval frequency:** every 200 optimizer steps
- **Eval metrics on val demo:** all five objective losses + the §6 gates
- **Checkpoint frequency:** every 500 steps
- **Keep:** the best-on-val-loss + the latest
- **Stored alongside weights:** `feature_schema_v1.json`, `train_config.yaml`,
  git commit hash, full hyperparameter set
- **Early stop:** if val MSE has not improved for 1000 steps, stop

### 5.8 Wall time estimate

- 8000 steps × ~3 s/step (bf16, batch 32 × ~1000 tokens) = ~7 hours on a
  single H100
- ~10 hours on H200 (slower memory but more headroom)
- Round to **8-12 hours** end to end per training run

### 5.9 Reproducibility

- Seed everything via `torch.manual_seed(42)` + numpy + python random
- Save: random seed, git commit hash, full config, package versions, GPU
  arch — into the checkpoint dir
- Do not require deterministic mode (slower; SSL is robust to nondeterminism)

## 6. Evaluation per training run

Six metrics. Metrics 1–4 are objective accuracy on the demo-disjoint
validation set. Metrics 5–6 are the load-bearing methodology gates the
encoder must pass before being plugged into Level-3 code paths.

| # | Metric | Target | Floor | Source |
|---|---|---|---|---|
| 1 | Forward-tick val MSE | ≥ 30% lower than mean-baseline | mean-baseline | Training loop |
| 2 | Forward-state val R² at T+8 horizon | ≥ 0.40 | 0.0 | Training loop |
| 3 | Time-to-next-event val MAE | ≤ 4 s | — | Training loop |
| 4 | Next-event-type val accuracy | ≥ 0.50 | 0.125 (chance over 8 classes) | Training loop |
| 5 | **σ_s on event-tick embeddings** | **median ∈ [0.15, 0.45]** with same-round mask | — | `scripts/recall_variance_diagnostic.py` (existing) |
| 6 | **Probe-outcome accuracy** (round_won predicted from end-of-round embedding) | ≥ 0.65 val accuracy | 0.50 | `scripts/probe_round_encoder.py` (to build); diagnostic only |

If σ_s falls outside the Goldilocks band or probe-outcome is at chance, the
encoder is rejected regardless of SSL objective scores.

**Probe-outcome must be diagnostic, not training-supervised.** The probe is
a tiny MLP trained *post-hoc* on the frozen encoder.

## 7. Data pipeline

Four new scripts. Target paths and responsibilities:

### `scripts/build_tick_sequences.py` (TO BUILD)

- Input: `data/processed/demos/*.parquet`
- Process: per demo, per round, walk all ticks. Downsample to 8 Hz. Build
  the per-tick feature vector per §2.2. Emit per-round tensor + awpy event
  index for the round.
- Output: `data/processed/tick_sequences/{train,val}.pt` (split is
  demo-level per §5.5). Plus
  `data/processed/tick_sequences/feature_schema_v1.json` (canonical schema).

### `scripts/train_round_encoder.py` (TO BUILD)

- Input: `data/processed/tick_sequences/{train,val}.pt`
- Multi-task SSL loss per §4, schedule per §5
- Output: `outputs/round_encoder/<run_id>/{encoder.safetensors, train_log.jsonl, val_metrics.json, feature_schema_v1.json, train_config.yaml, git_commit.txt}`
- Resumable from checkpoint

### `scripts/build_event_index.py` (TO BUILD)

- Loads the trained encoder, applies it round-by-round, emits per-event-tick
  embeddings into a FAISS index
- Output: `outputs/round_encoder/<run_id>/event_index.faiss` + metadata JSONL
- Replaces the 19-dim `tactical_embedding` FAISS index that
  `src/training/recall.py:RECALLIndex` currently builds

### `scripts/probe_round_encoder.py` (TO BUILD)

- Trains tiny MLP probes on top of the frozen encoder for the §6 diagnostic
  gates (probe-outcome, probe-action-next, probe-forward-state)
- Reports val accuracy / R² on demo-disjoint split

## 8. What downstream code consumes this

- **`src/training/recall.py:RECALLIndex.__init__(state_embedder=...)`** —
  existing plug-in point. Replace the 19-dim `tactical_embedding` with
  `lambda gs: round_encoder.query_at(round_id, tick)`. The encoder is a
  callable; same-round-mask infrastructure (commit 1b387b4) stays in place.
- **`src/training/grpo_trainer.py`** — the GRPO policy receives the event
  embedding for the decision moment as additional input. Open question on
  the fusion mechanism (concat vs cross-attention; recommend concat first).
- **`scripts/eval_scorer.py`** — `recall_mask` candidate now reads
  "RECALL using the trained round encoder + same-round mask." Inherits §6
  gates 5–6.
- **`scripts/label_app.py`** — cluster all event-tick embeddings via the
  encoder; pick diverse rather than redundant pairs for BT labelers.

## 9. What this doc does NOT cover

- **Level 1 SFT.** Shipped (F04). Outputs are the input to this layer; no
  redesign needed.
- **Level 3 GRPO loop.** Existing. Only the wiring change in §8 is in scope.
- **The reward family choice (judge / RECALL / BT-head).** Lives in
  `docs/reward-candidates.md`. Independent of Level 2 except that the
  RECALL-family scorer downstream consumes this encoder.
- **Top-down minimap rendering.** Flagged as a v2 enhancement, not v1.
  The v1 spatial signal is the per-player position fields.

## 10. Open design questions

Recommended defaults in parentheses. Picking the default is fine for the
first pass.

1. **Downsample rate.** 8 Hz vs 4 Hz vs adaptive. *(Default: 8 Hz fixed.
   Adaptive — denser around active events, sparser during freeze time —
   is an enhancement.)*
2. **Player-identity features.** Position-only slots vs learned per-player
   embeddings. *(Default: position-only. Learned embeddings if generalizing
   beyond Furia/Vitality demos.)*
3. **Map encoding.** Per-map one-hot vs learned embedding. *(Default:
   one-hot for the 4-map first pass.)*
4. **Smoke/molly handling.** Max-pool over variable-cardinality vs fixed
   ordering (e.g., by smoke age). *(Default: max-pool — OpenAI Five
   precedent for variable-cardinality.)*
5. **Fusion into GRPO.** Concatenate event embedding to model input vs
   cross-attention from model to event embedding. *(Default: concat.)*
6. **Loss weighting.** Fixed equal-weights vs gradient-magnitude balancing
   (GradNorm). *(Default: fixed §4 weights; revisit if any objective
   saturates.)*
7. **Sequence length variability.** Pad to fixed 1024 vs pack-and-attend.
   *(Default: pad to 1024 with attention mask. Pack-and-attend only if
   batch-size pressure shows up.)*

## 11. Prerequisite: scale demos from 4 to 40

Training a 50M-param encoder on 4 demos = 80 rounds is a near-certain
overfit even with the §5.4 regularization stack. The acceptance bar for
shipping Level 2 includes:

- ≥ 40 demos parsed via awpy into `data/processed/demos/`
- ≥ 32 demos in train, ≥ 8 in val (still demo-disjoint)
- ≈ 3,200 rounds in train, ≈ 800 in val
- Demo source: HLTV + FaceIT pro VOD corpus (publicly available, awpy
  parses them)

Engineering work: the existing demo-pull/parse pipeline at
`scripts/data.py` already handles parquet output from .dem files. The
extension is mostly "automate the URL list" — ~1-2 days. Not blocked on
labeling, GPU time, or anything else.

**Strict gate:** the encoder is not trained on the 4-demo set as the final
artifact. The 4-demo set is a smoke-test config to validate the training
loop runs end-to-end; the actual encoder ships from the 40-demo training.

## 12. Acceptance checklist

Before this encoder is considered shippable for Level 3 downstream use:

- [ ] §11 prerequisite — 40 demos parsed; demo-disjoint split in place
- [ ] `scripts/build_tick_sequences.py` exists; produces `train.pt` / `val.pt`
- [ ] `scripts/train_round_encoder.py` exists; runs end-to-end on the 4-demo
  smoke-test config without errors
- [ ] Full training run on 40-demo set: completes in ~8-12 hr on H100/H200
- [ ] Trained checkpoint at `outputs/round_encoder/<run_id>/`
- [ ] §6 metrics 1–4 (SSL objective accuracy) meet targets
- [ ] §6 metric 5 (σ_s in Goldilocks band) passes via
  `scripts/recall_variance_diagnostic.py` with the new encoder
- [ ] §6 metric 6 (probe-outcome ≥ 0.65) passes via
  `scripts/probe_round_encoder.py`
- [ ] `scripts/build_event_index.py` exists; produces a FAISS index
- [ ] `src/training/recall.py` wired to consume the new encoder
- [ ] Smoke run: `scripts/eval_scorer.py --scorers recall_mask` produces
  non-degenerate scores on the chosen benchmark (pseudo-gold or BT-pair
  per `docs/alignment-delta.md`)

After this checklist passes, Level 2 is shippable. Level 3 scorer-comparison
work resumes with `recall_mask` consuming the real encoder rather than the
19-dim baseline.
