# Chimera

**A next-state-prediction world model for Counter-Strike 2.**

Chimera learns Counter-Strike 2 by predicting the future. A causal spatiotemporal transformer is trained over sequences of engine-accurate game-state frames — state → state, no text — exactly like language-model pretraining, but over game states instead of tokens. The hypothesis is that a model forced to predict where ten players, the bomb, and the economy will be a fraction of a second to a few seconds from now must internalize the game's tactics, geometry, and causality. Value functions, event detection, and (in a later phase) natural-language reasoning are then built as heads and bridges on top of the learned latent.

> **Status (2026-06):** the world model is the project's forward direction and is **just starting** — the 597-d state tensors are built and the cleanup is done, but no world model has been trained yet. The prior VLM line of work ("See, Then Think") is **superseded and parked**; its real findings are preserved below because they are what motivated the pivot. Nothing in the world-model sections describes completed training or measured results.

## Why the pivot — what the VLM line taught us

The original project was a vision-language pipeline ("See, Then Think": L1 See → L2 Situate → L3 Think) with a round-encoder trained by change-point segmentation and caster-commentary grounding for the language layer. We parked it after a series of consistent, load-bearing negative results:

- **Outcome supervision is information-starved.** The round encoder **saturates at ~16 demos** (negative slope beyond that). Round outcome is ~1 bit/round — far too little signal to learn a rich state representation from.
- **Change-point losses find statistics, not semantics.** They locate *statistical* boundaries in the feature stream, not the *semantic* "events" we cared about — the wrong objective.
- **Claude-generated captions are circular.** They only paraphrase the structured features fed to the captioner; a discriminative check showed only **+0.008** over the structured-feature ceiling. Distilling them teaches nothing new.
- **Per-event commentary grounding hits a ceiling.** Global VOD↔demo alignment locks cleanly at **4.6σ** (kills ↔ caster-name-mention cross-correlation), but per-event anchoring tops out at **~25%** — bounded by auto-caption ASR name-recall. Parked to phase 2.

The common thread: every one of these used a sparse, downstream, or circular signal. Next-frame prediction is the opposite — dense, self-supervised, and grounded in the actual dynamics of the game. That is the bet.

## The world model (forward direction)

**State as a "document."** Each frame is the existing `feature_schema_v2` vector: **597-d** (10 players × 56 per-player features + 37 global features), sampled at **8 Hz (125 ms/frame)**. A round is one document: the model does **not** attend across round resets — economy, score, and round number are carried as features instead.

**Objective: next-state prediction.** Train the transformer to predict a future frame from the past, then **sweep the horizon** (125 ms → 2 s). At k = +1 frame the task is MLMove-style positional inertia; the strategic signal lives at the longer horizons. The prediction head is **distributional** (discretized / GMM) rather than a point regressor, so the model expresses multimodal futures instead of mode-averaging into blur.

**Judge by transfer, not by loss.** Prediction MSE is not the metric. The world model is evaluated by **probe transfer** — freeze the latent, train lightweight value/event probes on it, and measure those. A lower prediction loss that doesn't move the probes is not progress.

**Engineer perception, learn tactics.** Feature engineering borrows MLMove's recipe (per-player tokens, derived visibility/geometry) but engineers **only L1/See perception primitives**. Tactics (L2) are left for the model to learn. We do **not** hand-engineer tactical labels.

**What falls out of the world model:**
- **Events** emerge from **prediction surprise** (this subsumes the parked change-point work — surprise is the semantic boundary the change-point losses failed to find).
- **Value / policy** are MuZero-style heads on the world-model latent.
- **Reasoning** is verbalizing model rollouts.

## Language layer (phase 2)

Bridge a **frozen LLM** (Qwen 3.6 / 3.7, 35B-A3B MoE) into the world-model latents:

1. Start **Flamingo-style** — a resampler turns the state latent into read tokens, injected via gated cross-attention. The LLM is LoRA-tuned because the state latent is far out-of-distribution for a text model.
2. Graduate to a **Mixture-of-Transformers (MoT)** fusion.

Training stages: templated grounding → contrastive commentary → **GRPO reasoning** rewarded by checking the model's *verbalized* predictions against the *actual* futures in the demo.

**Discipline — ablate the latent.** If removing the world-model latent doesn't hurt the language head, the language head is being circular (paraphrasing inputs, as the captions were). The ablation is the test that keeps us honest.

## Roadmap (world model)

Each step reads defined inputs, writes defined outputs, and validates itself. The harness state lives in `feature-list.json` (`passes` boolean per feature). World-model features below are all `passes=false` — none are built.

- [x] **State tensors built.** 81 demos parsed to per-tick parquet via `scripts/parse_demos.py`; `scripts/build_tick_sequences.py` assembles 597-d `feature_schema_v2` tensors at `data/processed/tick_sequences/{train,val}.pt` (69 train / 12 val demos; 1,471 + 262 rounds). Round-scoped, 8 Hz.
- [ ] **Train the world model.** Causal spatiotemporal transformer, next-state prediction, round-scoped attention. Distributional head (discretize/GMM).
- [ ] **Horizon sweep.** Train/evaluate across prediction horizons 125 ms → 2 s; characterize the inertia→strategy transition.
- [ ] **Probe transfer eval.** Freeze the latent; train value/event probes on it; this — not prediction loss — is the model's score.
- [ ] **Events from surprise.** Derive event boundaries from prediction surprise; compare against ground-truth kill/plant/defuse events.
- [ ] **Value/policy heads.** MuZero-style heads on the latent.
- [ ] **Phase 2 — language bridge.** Flamingo-style resampler + gated cross-attention into a LoRA'd Qwen MoE; ablate the latent.

## Prior / superseded method — "See, Then Think" VLM (parked)

> The two-phase VLM method below is **parked**, not deleted — it documents the approach the project pivoted away from, and the design rationale lives on in `decisions.md` (D013–D015, D024). The associated scripts have been moved to `scripts/_archive/`.

### Phase 1: Visual Grounding (SFT on Screenshots)

Supervised fine-tuning on screenshot–game state pairs. Ground truth comes from demo data synchronized to VOD frames (engine-accurate health, armor, weapons, player counts, etc. — not model-labeled). The model learns to read the HUD correctly. LoRA on both vision and language layers.

### Phase 2: Strategic Reasoning (GRPO on Demo-Derived Rewards)

GRPO with 2 reward signals, a multiplicative format gate, and KL regularization. Vision layers are frozen — only language layers train. The model already knows what's on screen from Phase 1; Phase 2 teaches it what to do about it.

**Reward signals:**

| Signal | Weight | Role |
|--------|--------|------|
| Format gate | multiplicative | Invalid JSON → zero total reward |
| R_percept | α ≈ 0.20 | Prevents SFT regression (field accuracy) |
| R_strategy | 1 − α ≈ 0.80 | Strategic reasoning via RECALL (kNN advantage estimation) or simplified outcome baseline |
| KL penalty | λ=0.02 | Prevents mode collapse vs SFT reference |

R_strategy is computed via **RECALL** (Retrieval-based Advantage Estimation): game states are encoded as ~25-dim tactical embeddings, indexed in FAISS, and k-nearest neighbors are retrieved to compute A(s,a) = Q̂(s,a) − V̂(s). This provides a counterfactual advantage signal without requiring explicit behavioral feature labels. As a simpler ablation baseline, a **simplified outcome signal** uses Ω = a·w + (1−a)·(1−w), where *a* is pro-agreement and *w* is round outcome. R_decision and R_outcome from D013 are retained as ablation baselines only, not the default reward architecture.

**Outcome asymmetry — learning beyond the expert.** The simplified outcome baseline uses an asymmetric signal: agreeing with a winning play is strongly rewarded (1.0), but agreeing with a losing play is penalized (0.2). Deviating from a losing play gets moderate reward (0.5). This means the model can learn from the pro's mistakes — when the pro rushed and lost, GRPO discovers that completions suggesting something else get higher relative reward. Over thousands of examples, this teaches counterfactual reasoning: inferring better strategies from observed failures. A symmetric design (penalize all deviation equally) collapses to pure imitation and can never exceed the expert. RECALL subsumes this by directly estimating advantage from retrieved outcomes, but the asymmetry insight motivates the approach.

**Credit assignment with φ.** In 5v5, round outcome is a team signal. If the spectated player died at second 5 and the team won at second 90, the outcome says nothing about that player's decision. Player contribution φ ∈ [0, 1] weights the outcome signal by damage dealt, survival time, and objective interaction (plant/defuse). High-φ samples get strong outcome signal; low-φ samples get attenuated. This applies to the simplified outcome baseline and is equivalent to advantage-weighted regression with sample quality weighting (Peng et al., 2019).

**Why GRPO over PPO.** GRPO generates G=16 completions per prompt and normalizes advantages within the group: `Â_i = (r_i - mean) / std`. It only needs correct *relative* ordering, not calibrated absolute rewards. PPO would need a value function estimating expected reward of a CS2 screenshot — but expected reward depends on skill level, teammates, and opponent tendencies, so the value estimate would be terrible. GRPO sidesteps this entirely.

**Effective gradient structure.** After early training, R_percept has near-zero within-group variance (same screenshot → same HUD reading for all 16 completions → zero gradient). The actual training signal is ~100% from R_strategy — "which strategic recommendations correlate with winning." Perception acts as a floor, not a gradient source. Each signal contributes gradient only during the phase where it's informative, then gracefully drops out.

**Handling mechanical skill noise.** Round outcome depends on decision quality, aim, teammates, opponents, and randomness. The model only sees the HUD and outputs strategy — aim is a latent variable that corrupts the training signal but never appears at inference. Three mechanisms handle this: (1) φ attenuates outcome signal when the player's contribution was low (good decision + bad aim → low damage → damped signal), (2) R_strategy (via RECALL) measures strategic similarity to states with known outcomes, which is orthogonal to mechanical execution, and (3) population averaging over multiple pros washes out mechanically-dependent plays because aim quality is uncorrelated with HUD state. The primary vulnerability — bad decision bailed out by good aim — produces high φ and amplified bad signal, but is outvoted across the training set by the population playing the same state conventionally. See `decisions.md` D015 for the full analysis.

See `decisions.md` D013–D015, D024 for the full mathematical formulation, design rationale, and mechanical skill confound analysis.

## Experiment

| Model | Description |
|-------|-------------|
| A | Qwen3.5-35B-A3B zero-shot (no training) |
| B | Qwen3.5-35B-A3B + SFT only (visual grounding, no GRPO) |
| C | Qwen3.5-35B-A3B + SFT + GRPO (visual grounding + strategic reasoning) |
| D | Qwen3.5-35B-A3B + GRPO only (no SFT, tests whether SFT phase is necessary) |

If C > B, GRPO improves reasoning beyond what SFT alone provides. If C > D, SFT visual grounding is a necessary foundation for effective GRPO.

### Pipeline (superseded VLM pipeline)

These steps describe the parked VLM pipeline. Steps 1–5b reached the marked state before the pivot; they are retained for provenance. Each step is isolated: it reads from defined inputs, writes to defined outputs, and validates its own results.

- [x] **Step 1 — Data schema & manifest.** Unified data manifest (`src/data/manifest.py`, JSONL append-only) for tracking screenshot provenance, source, timestamps, and transcript context. Collection scripts write to `data/manifest.jsonl`, `ScreenshotDataset` loads/filters by manifest fields, training utils accept manifest filtering.
- [x] **Step 2 — Demo data pipeline.** Parse pro demos with awpy into full-tick Parquet + metadata JSONs ([cs2-tools](https://github.com/skkwowee/cs2-tools)). Interactive demo viewer ([cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)) with radar canvas, vision cones (wall-clipped via raycasting), kill/damage lines, shot tracers, timeline scrubbing, and split upper/lower rendering for multi-level maps. 4 demos parsed (Furia vs Vitality, maps: Mirage/Inferno/Nuke/Overpass, 83 rounds, 563 kills).
- [x] **Step 3 — Screenshot capture.** Capture screenshots from the CS2 client via SendKeys automation during demo playback. Produces (screenshot, exact_game_state) pairs with engine-accurate ground truth from Parquet tick data. 4 maps complete: Mirage (548), Inferno (1476), Nuke (791), Overpass (1538). Total: 4,353 screenshots, 5,309 labels.
- [x] **Step 4 — Phase 1: Visual grounding (SFT).** SFT on Qwen3.5-35B-A3B with screenshot–game state pairs. LoRA r=4 α=8 on vision + language layers, 1 epoch (304 steps), 1h40m on H200. Re-trained from checkpoint-150 to checkpoint-304. Result: 84.8% token accuracy (vs 50% Opus 4.6, 50% base Qwen). Perfect on health/armor/money/round_phase. LoRA adapters saved in `checkpoints/` (250, 300, 304) and on HuggingFace.
- [ ] **Step 5 — GRPO dataset from demos.** Convert demo snapshots into decision training format. Each sample: game state → pro behavioral features (movement, utility, engagement timing from tick data) + round_won + player_contribution (φ). Active fight frames (enemies on screen) filtered to SFT-only; planning frames enter GRPO.
- [x] **Step 5a — Data sparsity diagnostic.** Measure state bucket coverage across the dataset before GRPO training (D023). Hierarchical bucketing over 14 state dimensions with contrastive pair availability analysis. Results: L0 96.2% coverage (23/30 buckets), L1 27.0% (89/2167). L0 marginal, L1 needs 100+ demos.
- [x] **Step 5b — Small-scale GRPO smoke test.** End-to-end GRPO pipeline test on a small slice (B-site post-plants). Manual GRPO loop bypassing TRL bug (image_grid_thw index OOB). 20 steps, rewards differentiate successfully. Verified gradient flow and loss decrease.
- [ ] **Step 5c — RECALL implementation.** Retrieval-based advantage estimation via kNN over tactical embeddings (`src/training/recall.py`). Computes A(s,a) = Q̂(s,a) − V̂(s) for R_strategy.
- [ ] **Step 6 — Phase 2: Strategic reasoning (GRPO).** Train Models B, C, D. 2 reward signals + RECALL + multiplicative format gate + KL regularization (see D024). Compare SFT-only vs SFT+GRPO vs GRPO-only.
- [ ] **Step 7 — Evaluation + analysis.** Per-field accuracy, consistency scores, reasoning quality across all models. Write up findings.

### Resuming work

Start a session by reading this pipeline checklist. Each step lists its inputs, outputs, and validation criteria. Validate the last completed step before starting the next.

## Output Format

All models produce structured JSON with three sections:

```json
{
  "game_state": {
    "map_name": "de_dust2",
    "round_phase": "playing",
    "player_side": "CT",
    "player_health": 100,
    "player_armor": 100,
    "player_money": 4750,
    "team_money_total": 20000,
    "weapon_primary": "M4A1-S",
    "weapon_secondary": "USP-S",
    "utility": ["smoke", "flashbang", "HE grenade"],
    "alive_teammates": 4,
    "alive_enemies": 5,
    "bomb_status": "carried",
    "site": "A",
    "visible_enemies": 0
  },
  "analysis": {
    "situation_summary": "Early round, full team alive, holding A site",
    "economy_assessment": "full-buy",
    "round_importance": "medium",
    "immediate_threats": ["Potential A execute"],
    "opportunities": ["Strong utility for retake"]
  },
  "advice": {
    "primary_action": "Hold crossfire position with teammate",
    "reasoning": "2v5 retake is difficult, better to get picks early",
    "fallback": "Fall back to site if smoked off",
    "callout": "Two holding A, need info mid"
  }
}
```

## Data

| | |
|---|---|
| Raw demos | **85** local `.dem` files (~37 GB) in `data/demos/` |
| Parsed | **81** parsed to per-tick parquet (`scripts/parse_demos.py`, awpy) |
| World-model tensors | `data/processed/tick_sequences/{train,val}.pt` — 597-d `feature_schema_v2`, 8 Hz, round-scoped (69 train + 12 val demos; 1,471 + 262 rounds) |

Demos are ingested by a separate **HLTV → `.dem` → HuggingFace** zero-local-storage pipeline ([chimera-demo-pipeline](https://github.com/skkwowee/chimera-demo-pipeline)). The world-model data path is `parse_demos.py` (→ parquet) then `build_tick_sequences.py` (→ `.pt` tensors).

## Standalone Repos

Components that live in their own repositories:

- **[chimera-demo-pipeline](https://github.com/skkwowee/chimera-demo-pipeline)** — HLTV scrape → demo download → tick-sequence build on HF, zero local storage. Hosts the parked commentary-grounding work.
- **[cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)** — Interactive Next.js viewer for CS2 demo replays; kept and repurposed for visualizing world-model rollouts.
- **[cs2-tools](https://github.com/skkwowee/cs2-tools)** — Python utilities for demo parsing, viewer data export, and screenshot capture (`pip install cs2-tools[parse]`)

## Project Structure

```
chimera/
├── decisions.md                # Design decision log (D001–D024)
├── feature-list.json           # Harness feature inventory (passes boolean per feature)
├── claude-progress.txt         # Session-handoff doc — read first every session
├── paper/                      # NeurIPS 2026 submission
├── checkpoints/                # LoRA adapter checkpoints (prior VLM SFT)
├── config/config.yaml          # Configuration settings
├── data/
│   ├── demos/                  # Raw .dem files (85 local, ~37 GB)
│   ├── processed/
│   │   └── tick_sequences/     # WORLD-MODEL DATA
│   │       ├── train.pt        # 597-d feature_schema_v2 tensors (69 demos)
│   │       ├── val.pt          # (12 demos)
│   │       └── feature_schema_v1.json  # schema doc (version: feature_schema_v2)
│   ├── manifest.jsonl          # Data provenance (VLM screenshots)
│   ├── raw/ labeled/           # Screenshots + demo-derived labels (parked VLM)
│   └── predictions/            # VLM predictions (parked)
├── src/
│   ├── data/                   # Data loading + manifest utilities
│   ├── inference/              # VLM inference (parked)
│   ├── training/               # SFT + GRPO training modules (parked VLM)
│   └── prompts.py              # Shared prompts (parked VLM)
└── scripts/
    ├── parse_demos.py          # WORLD-MODEL: .dem → per-tick parquet (awpy)
    ├── build_tick_sequences.py # WORLD-MODEL: parquet → 597-d tensors (.pt)
    ├── data.py                 # HF Hub data management (pull/push/clean)
    ├── train_sft.py            # Parked VLM: Phase 1 SFT
    ├── train_grpo.py           # Parked VLM: Phase 2 GRPO
    └── _archive/               # 18 superseded scripts: round-encoder,
                                #   change-point, probes, caption generation
```

> **Cleanup (2026-06):** 18 superseded scripts (round-encoder, change-point, probes, captions) were moved to `scripts/_archive/`. The commentary-grounding work is parked in the separate demo-pipeline repo. `cs2-demo-viewer` is kept — repurposed for visualizing world-model rollouts.

## Setup

```bash
git clone https://github.com/skkwowee/chimera.git
cd chimera
uv sync

cp .env.example .env
# Edit .env with your HF_TOKEN
```

## Usage

### Data Management

Data lives on HuggingFace Hub. The repo stays lean — pull what you need, train, clean up.

```bash
# Check what's available locally and remotely
python scripts/data.py status

# Pull dataset from Hub into data/raw/ and data/labeled/
python scripts/data.py pull
python scripts/data.py pull --subset 100  # quick iteration

# Push local data to Hub (assembles from data/captures/ first)
python scripts/data.py push --captures

# Upload trained model
python scripts/data.py push --model outputs/sft/final_model/merged_16bit

# Clean up local data copies
python scripts/data.py clean        # removes raw/, labeled/, .hf_cache/
python scripts/data.py clean --all  # also removes processed/, predictions/, outputs/
```

### Run Inference

```bash
python scripts/run_inference.py --single screenshot.png
python scripts/run_inference.py --input data/raw --output data/predictions
python scripts/run_inference.py --labeled-only --input data/raw --output data/predictions
```

### Evaluate

```bash
python scripts/evaluate.py --predictions data/predictions --labels data/labeled
```

### Train

```bash
# Phase 1: SFT (run first)
python scripts/train_sft.py --screenshots data/raw --labels data/labeled
python scripts/train_sft.py --dry-run  # check VRAM

# Phase 1: Resume SFT from checkpoint
python scripts/train_sft.py --resume checkpoints/sft-r4-checkpoint-304

# Evaluate SFT model
python scripts/compare_models.py --lora-adapter checkpoints/sft-r4-checkpoint-304

# Phase 2: GRPO (uses SFT output, --manual bypasses TRL bug)
python scripts/train_grpo.py --manual \
    --screenshots data/raw --labels data/labeled \
    --reward-mode simplified --kl-coef 0.02
python scripts/train_grpo.py --manual --reward-mode recall  # uses RECALL advantage estimation
python scripts/train_grpo.py --dry-run  # check VRAM
```

### Review

```bash
python scripts/generate_review.py --images data/raw --labels data/labeled --embed
python scripts/generate_review.py --compare --embed
```

## Hardware Requirements

| Task | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (Qwen3.5-35B-A3B bf16) | ~70 GB | H200 SXM (141GB) |
| SFT Training (bf16 + LoRA) | ~80 GB | H200 SXM (141GB) |
| GRPO Training (bf16 + LoRA) | ~90 GB | H200 SXM (141GB) |

## Artifact Locations

Training artifacts are split between this repo and HuggingFace. Use `python scripts/data.py` to sync.

| Artifact | Location | How to access |
|----------|----------|---------------|
| Screenshots (4,353 PNGs) | [HF: skkwowee/chimera-cs2](https://huggingface.co/datasets/skkwowee/chimera-cs2) | `python scripts/data.py pull` |
| Labels (5,309 JSONs) | [HF: skkwowee/chimera-cs2](https://huggingface.co/datasets/skkwowee/chimera-cs2) | `python scripts/data.py pull` |
| SFT checkpoints (150, 250, 300, 304) | [HF: skkwowee/chimera-cs2/checkpoints/](https://huggingface.co/datasets/skkwowee/chimera-cs2/tree/main/checkpoints) | `python scripts/data.py pull` |
| Demo Parquets (4 matches) | [HF: skkwowee/chimera-cs2/demos/](https://huggingface.co/datasets/skkwowee/chimera-cs2/tree/main/demos) | `python scripts/data.py pull --all` |
| GRPO checkpoint-25 | Pod only (`outputs/grpo/recall_run1/`) | **Not yet pushed to HF** |
| GRPO smoke test data | Git tracked | `data/training/grpo/` |
| Prediction comparisons | Git tracked | `data/predictions/` |
| Training logs | Git tracked | `data/training/` |
| Checkpoint configs (JSON) | Git tracked | `checkpoints/*/adapter_config.json` etc. |
| Checkpoint weights (safetensors) | HF only (untracked from git) | `python scripts/data.py pull` |

> **Note:** `.safetensors` and `tokenizer.json` are excluded from git via `.gitignore` — the canonical copies live on HuggingFace. Config/metadata JSONs remain in git for quick reference.

## Why This Matters Beyond Games

The general principle — *learn dynamics from cheap self-supervised next-state prediction, then read the learned latent for value, events, and language* — is the world-model recipe applied to a structured multi-agent domain. The same shape applies to robotics (predict sensor/state futures), autonomous driving (predict scene evolution), and any setting where dense self-supervision beats sparse outcome labels. CS2 is the controlled environment to prove it: engine-accurate state, clear outcomes, and a rich tactical layer to recover.

## License

MIT
