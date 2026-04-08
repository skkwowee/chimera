# Chimera

**See, Then Think: Two-Phase VLM Training for Game Understanding**

Vision-language models struggle with domain-specific visual grounding in competitive gaming, where rapid scene understanding is critical for strategic decision-making. Chimera is a two-phase training paradigm: first, supervised fine-tuning on screenshot–game state pairs teaches the model to read the HUD accurately (visual grounding); then, Group Relative Policy Optimization with demo-derived rewards teaches strategic reasoning scored against pro play and round outcomes. We evaluate on Counter-Strike 2, demonstrating improved performance over single-phase approaches.

## Motivation

Frontier VLMs fail at basic CS2 visual grounding (e.g., weapon identification, player counts), and they lack the strategic knowledge to advise on play. Both problems have cheap data sources: demo files provide engine-accurate game state for grounding, and pro match outcomes provide a reward signal for reasoning. The key insight: perception and reasoning are separable training objectives — teach them in sequence rather than asking RL to learn both at once.

**Hypothesis:** A VLM that first learns accurate visual grounding via SFT on screenshot–demo pairs, then learns strategic reasoning via GRPO scored against pro decisions and round outcomes, produces better tactical advice than either phase alone — and the two-phase approach is more data-efficient than end-to-end training.

## Method

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

### Pipeline

Each step is isolated: it reads from defined inputs, writes to defined outputs, and validates its own results. Intermediate artifacts are cleaned up.

- [x] **Step 1 — Data schema & manifest.** Unified data manifest (`src/data/manifest.py`, JSONL append-only) for tracking screenshot provenance, source, timestamps, and transcript context. Collection scripts write to `data/manifest.jsonl`, `ScreenshotDataset` loads/filters by manifest fields, training utils accept manifest filtering.
- [x] **Step 2 — Demo data pipeline.** Parse pro demos with awpy into full-tick Parquet + metadata JSONs ([cs2-tools](https://github.com/skkwowee/cs2-tools)). Interactive demo viewer ([cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)) with radar canvas, vision cones (wall-clipped via raycasting), kill/damage lines, shot tracers, timeline scrubbing, and split upper/lower rendering for multi-level maps. 4 demos parsed (Furia vs Vitality, maps: Mirage/Inferno/Nuke/Overpass, 83 rounds, 563 kills).
- [x] **Step 3 — Screenshot capture.** Capture screenshots from the CS2 client via SendKeys automation during demo playback. Produces (screenshot, exact_game_state) pairs with engine-accurate ground truth from Parquet tick data. 4 maps complete: Mirage (548), Inferno (1476), Nuke (791), Overpass (1538). Total: 4,353 screenshots, 5,309 labels.
- [x] **Step 4 — Phase 1: Visual grounding (SFT).** SFT on Qwen3.5-35B-A3B with screenshot–game state pairs. LoRA r=4 α=8 on vision + language layers, 1 epoch (304 steps), 1h40m on 2×H100. Result: 67.3% per-field accuracy (vs 50% Opus 4.6, 50% base Qwen). Perfect on health/armor/money/round_phase.
- [ ] **Step 5 — GRPO dataset from demos.** Convert demo snapshots into decision training format. Each sample: game state → pro behavioral features (movement, utility, engagement timing from tick data) + round_won + player_contribution (φ). Active fight frames (enemies on screen) filtered to SFT-only; planning frames enter GRPO.
- [x] **Step 5a — Data sparsity diagnostic.** Measure state bucket coverage across the dataset before GRPO training (D023). Hierarchical bucketing over 14 state dimensions with contrastive pair availability analysis. Results: L0 96.2% coverage (23/30 buckets), L1 27.0% (89/2167). L0 marginal, L1 needs 100+ demos.
- [ ] **Step 5b — Small-scale GRPO smoke test.** End-to-end GRPO pipeline test on a small slice (B-site post-plants). Verify training loop, reward differentiation, gradient flow, loss decrease.
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

## Standalone Repos

Two components live in their own repositories and are managed as dependencies:

- **[cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)** — Interactive Next.js viewer for CS2 demo replays
- **[cs2-tools](https://github.com/skkwowee/cs2-tools)** — Python utilities for demo parsing, viewer data export, and screenshot capture (`pip install cs2-tools[parse]`)

## Project Structure

```
chimera/
├── decisions.md                # Design decision log (D001–D024)
├── paper/                      # NeurIPS 2026 submission
│   ├── main.tex                # Main paper
│   ├── references.bib          # Bibliography
│   └── figures/                # Paper figures
├── config/config.yaml          # Configuration settings
├── data/
│   ├── manifest.jsonl          # Data provenance tracking
│   ├── raw/                    # Raw screenshots
│   ├── labeled/                # Demo-derived ground truth
│   ├── processed/              # Processed data
│   └── predictions/            # VLM predictions
├── src/
│   ├── data/                   # Data loading + manifest utilities
│   ├── inference/              # VLM inference (Qwen3.5-35B-A3B)
│   ├── training/               # SFT + GRPO training modules
│   └── prompts.py              # Shared prompts for all models
└── scripts/
    ├── run_inference.py        # Run VLM inference
    ├── evaluate.py             # Evaluate predictions
    ├── train_sft.py            # Phase 1: SFT fine-tuning
    ├── train_grpo.py           # Phase 2: GRPO fine-tuning
    ├── data.py                 # HF Hub data management (pull/push/clean)
    └── generate_review.py      # Generate HTML viewer for review
```

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

# Phase 2: GRPO (uses SFT output)
python scripts/train_grpo.py \
    --model-name outputs/sft/final_model/merged_16bit \
    --screenshots data/raw --labels data/labeled \
    --reward-mode simplified --kl-coef 0.02
python scripts/train_grpo.py --reward-mode recall  # uses RECALL advantage estimation
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

## Why This Matters Beyond Games

The general principle — *learn reasoning from cheap structured data, then ground in vision with minimal labels* — applies to robotics (sim logs → real-world), medical imaging (patient records → radiology), and autonomous driving (telemetry → camera perception). Games are the controlled environment to prove the paradigm.

## License

MIT
