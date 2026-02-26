# Chimera — Design Decisions Log

Technical decisions, alternatives considered, and rationale. For the paper writeup and technical deepdive.

---

## D001: Two-phase training — SFT then GRPO (2026-02-10)

**Decision:** Train in two explicit phases: SFT for visual grounding, then GRPO for strategic reasoning. Not end-to-end.

**Why:** Perception and reasoning are separable objectives with different data sources and loss functions. SFT has a clear supervised target (demo game state). Reasoning has no single correct answer — it needs RL with a reward signal. Asking RL to learn both perception and reasoning simultaneously is harder and less data-efficient.

**Alternatives considered:**
- End-to-end GRPO from base model (Model D in the ablation). Hypothesis: this underperforms because the model can't reason about game state it can't perceive.
- Single-phase SFT with reasoning targets from Claude labels. Rejected: Claude's strategic advice isn't ground truth — it's another model's opinion.
- Three-phase (SFT vision → SFT strategy on text → GRPO). The text strategy phase was cut — demo-derived rewards subsume it.

**Status:** Core thesis. Validated by experiment design (Models A/B/C/D).

---

## D002: Demo-derived ground truth over model labeling (2026-02-15)

**Decision:** SFT ground truth comes from demo files parsed with awpy, not from Claude (or any model) labeling screenshots.

**Why:** Demo data is engine truth. Health, armor, weapons, player positions, bomb status — all exact. Model labeling introduces errors (misreading HUD numbers, hallucinating weapons) that would propagate into the SFT training signal. The whole point of SFT is to teach accurate perception — you can't do that with noisy labels.

**Alternatives considered:**
- Claude API labeling (the original approach in early pipeline). Rejected: expensive, noisy, and redundant once demo sync exists.
- Human annotation. Rejected: doesn't scale, still error-prone for exact numbers.
- Self-training / pseudo-labels. Rejected: circular — model labels its own training data.

**Implication:** Requires screenshot-demo synchronization (F03). More engineering work upfront, but the label quality is strictly better.

**Status:** Decided. Demo parsing complete (4 demos, 83 rounds). Screenshot capture pipeline built (F03).

---

## D003: Tournament VOD screenshots vs demo playback screenshots (2026-02-20)

**Decision:** Capture screenshots by playing back .dem files in CS2's built-in demo viewer, not by extracting frames from tournament broadcast VODs.

**Why:** Tournament broadcasts have custom overlays (ESL/BLAST HUD, caster cameras, replays, picture-in-picture) that differ from the standard CS2 HUD. A model trained on broadcast frames wouldn't generalize to what a real player sees. Demo playback uses the standard CS2 HUD — the same one any player would see in-game.

**Alternatives considered:**
- VOD frame extraction + overlay-aware labeling. Rejected: fragile (overlays change per tournament), doesn't match the deployment scenario (advising a real player).
- Mix of both. Considered but adds complexity. Demo-only is cleaner for a first paper.

**Approach:** CS2 netcon (TCP console via `-netconport`) driven from WSL Python. Jump to specific ticks, control spectator POV, capture JPEGs. Automated via `scripts/capture_screenshots.py`.

**Trade-off:** Demo playback requires CS2 running on Windows with the demo files. Can't be done headlessly or in CI. Manual setup step for the researcher. Acceptable for research scale.

**Status:** Decided. Pipeline built (`src/netcon.py`, `scripts/plan_captures.py`, `scripts/capture_screenshots.py`).

---

## D004: Outcome reward asymmetry (2026-02-15)

**Decision:** The outcome reward uses an asymmetric signal matrix:

| | Pro wins | Pro loses |
|---|---------|-----------|
| Model agrees | 1.0 | 0.2 |
| Model deviates | 0.4 | 0.6 |

Formula: `won ? 0.4 + 0.6 * alignment : 0.6 - 0.4 * alignment`

**Why:** Pro play is a noisy oracle. Pros make mistakes, especially in losing rounds. The asymmetry encodes:
- Agreeing with a winning play is the strongest positive signal.
- Agreeing with a losing play is mildly penalized — endorsing what didn't work.
- Deviating from a winning play is mild — the model's alternative might also work.
- Deviating from a losing play is mildly rewarded — the model may have seen something better.

**Symmetric alternative (rejected):** Equal penalty for deviation regardless of outcome. This collapses to imitation learning — the model can never learn beyond the expert because deviation is always penalized.

**Why this matters for the paper:** If the model learns to deviate from pro play on losing rounds and those deviations are strategically sensible, that's evidence the model is reasoning, not just imitating. This is the strongest possible result.

**Status:** Implemented. Weight = 0.30 (highest single signal).

---

## D005: Seven decomposed reward signals (2026-02-15)

**Decision:** GRPO uses 7 separate reward functions, not one collapsed scalar.

**Signals:**

| # | Signal | Category | Weight | Purpose |
|---|--------|----------|--------|---------|
| 1 | Format gate | Vision | 0.05 | Valid JSON with required sections |
| 2 | Hard field accuracy | Vision | 0.15 | HUD-readable fields (health, weapons, alive counts) |
| 3 | Soft field accuracy | Vision | 0.05 | Harder fields (money, map, bomb status) |
| 4 | Decision alignment | Reasoning | 0.15 | Jaccard similarity to pro action categories |
| 5 | Outcome reward | Reasoning | 0.30 | Alignment weighted by round result |
| 6 | Consistency | Reasoning | 0.20 | Perception-reasoning coherence (6 heuristic checks) |
| 7 | Reasoning quality | Reasoning | 0.10 | Structural quality of analysis/advice |

**Why decompose:** Different signals have different variance profiles in GRPO. Format gate saturates early (all completions produce valid JSON). Field accuracy has low within-group variance (same screenshot = same HUD). Decision alignment and consistency have high variance (different completions suggest different actions). Keeping them separate lets GRPO compute meaningful advantages per signal.

**Weight rationale:** Vision = 25%, reasoning = 75%. Outcome gets the highest weight because it's the unique signal this pipeline provides — pro decisions paired with round results. Consistency gets second-highest because coherent reasoning chains are the end goal.

**Status:** Implemented. Weights are provisional — need empirical validation in F06. See D006 for planned revisions.

---

## D006: Reward design — open issues and planned revisions (2026-02-26)

After detailed review, several issues with the current reward design were identified. These should be addressed before or during F06 (GRPO training).

### Issue 1: Format gate should be multiplicative, not additive

**Problem:** At 0.05 additive weight, invalid JSON still collects 0.95 of reward from other signals. The gate doesn't actually gate anything.

**Planned fix:** Make format gate a multiplicative mask: `total_reward *= format_gate`. Invalid JSON gets zero total reward. One-line change in `rewards.py`.

### Issue 2: Decision alignment is partially redundant with outcome reward

**Problem:** Outcome reward already contains alignment (`0.4 + 0.6 * alignment`). A separate pure-alignment signal at 0.15 double-counts "agree with pro," overweighting imitation relative to novel strategy discovery.

**Planned fix:** Drop decision alignment to 0.05 or remove entirely. Boost outcome to 0.35. This biases the model toward "do what wins" over "do what the pro did" — better for the paper narrative.

**Open question:** If we want to show the model can *exceed* pro play (deviate on losing rounds), pure alignment is counterproductive. But if training is unstable without it, the anchor to pro play might be necessary for stability. Empirical question.

### Issue 3: Consistency reward is gameable and overweighted

**Problem:** The 6 hand-coded coherence rules (low health -> caution words, bomb planted -> defuse words) are keyword-matching proxies for reasoning. The model will learn to sprinkle trigger words without genuine reasoning. At 0.20 weight, this is the second-strongest signal for something so fragile.

**Planned fix (short term):** Drop weight from 0.20 to 0.10. Move the freed weight to outcome (0.35) or a new signal.

**Planned fix (longer term):** Consider replacing with an LLM judge (frozen small model) that scores whether the analysis logically follows from game state. More expensive per sample but much harder to Goodhart.

**Alternative considered:** Anneal consistency weight over training — high early (teaches basic coherence fast), decay to near-zero later (stops constraining creative reasoning). Adds schedule complexity; save for V2.

### Issue 4: No KL penalty / reference model constraint

**Problem:** Nothing prevents the reasoning distribution from diverging arbitrarily from the SFT checkpoint. Vision rewards anchor perception, but the model's action recommendations could mode-collapse onto safe generic advice ("hold position," "wait for info") that scores well across diverse game states.

**Planned fix:** Add `kl_coef` to GRPOConfig (TRL supports this). Start at 0.01-0.05. Track KL divergence during training — if it spikes, rewards are too aggressive.

### Issue 5: Action taxonomy is coarse

**Problem:** 6 categories (aggressive, hold, rotate, fall_back, utility, engage) miss tactical nuance. "Push A with flash" vs "push A dry" are very different calls but both map to "aggressive." Jaccard similarity over coarse categories gives high alignment for strategically different actions.

**Planned fix (V2):** Add second tier using Parquet data — utility usage, position deltas, engagement timing could distinguish "execute with utility" from "dry entry" from "info peek." Not a blocker for first experiment; outcome reward implicitly captures whether the type of push mattered.

### Issue 6: Within-group variance determines effective weights

**Observation:** In GRPO, advantages are `(reward - mean) / std` within the generation group. A signal with near-zero within-group variance contributes near-zero gradient regardless of its weight. Predicted variance by signal:

- Format gate: ~0 after early training (all completions valid JSON)
- Hard/soft field accuracy: ~0 (same screenshot, same HUD reading)
- Decision alignment: moderate (different completions suggest different actions)
- Outcome: moderate (varies via alignment component; outcome itself is fixed per example)
- Consistency: moderate-high (different actions trigger different coherence scores)
- Reasoning quality: low (structural quality is easy to learn)

**Implication:** The effective weight distribution is probably not the nominal one. The actual gradient is ~95% from reasoning signals (decision, outcome, consistency) after early training. The vision signals act as a floor/baseline, not as a gradient source.

**Action:** Log per-signal reward variance during training. If a signal's within-group std drops below a threshold, it's dead weight. Consider removing or replacing.

### Revised provisional weights (for first training run)

| Signal | Current | Proposed | Reason |
|--------|---------|----------|--------|
| Format gate | 0.05 (additive) | multiplicative mask | Actually enforce the gate |
| Hard field accuracy | 0.15 | 0.15 | Keep as perception floor |
| Soft field accuracy | 0.05 | 0.05 | Keep as perception floor |
| Decision alignment | 0.15 | 0.05 | Reduce imitation bias |
| Outcome reward | 0.30 | 0.40 | Primary gradient signal |
| Consistency | 0.20 | 0.10 | Reduce gameability risk |
| Reasoning quality | 0.10 | 0.10 | Keep for structural quality |
| KL penalty | (none) | kl_coef=0.02 | Prevent mode collapse |

These sum to 0.85 (with format as multiplicative). Renormalize the remaining 6 to sum to 1.0, or let TRL handle per-signal weighting.

---

## D007: Frozen vision layers during GRPO (2026-02-15)

**Decision:** During GRPO, freeze the vision encoder and LoRA on vision layers. Only train language layers.

**Why:** SFT teaches the model to perceive the HUD. GRPO teaches it to reason about what it sees. If vision layers are trainable during RL, the reward signal could corrupt perception — the model might learn to "see" game states that score well on reasoning rewards rather than accurately reading the HUD. Analogous to freezing the vision encoder in LLaVA-style training.

**Additional benefit:** vLLM (used for GRPO generation) requires static vision weights for efficient batched inference. Trainable vision layers would require reloading the model each generation step.

**Status:** Decided. Implemented in config (`finetune_vision_layers: false` for GRPO).

---

## D008: Data scale and generalization limits (2026-02-26)

**Current data:** 4 demos from one match (Furia vs Vitality, IEM Krakow 2026). 83 rounds, ~3500-5000 screenshots. 10 unique players, 4 maps.

**Concern:** The model might overfit to Vitality/Furia's playstyle rather than learning general CS2 strategy. The action distribution in these demos reflects these specific teams' tendencies — Vitality's structured CT setups, Furia's aggressive T-side. Other teams play differently.

**Mitigation for paper:** This is a method paper, not a deployment paper. The contribution is proving the two-phase approach works (C > B and C > D), not building a production advisor. Small-scale data is explicitly acknowledged as a limitation.

**Scaling path:** Parse more demos from different teams/events/maps. The parsing pipeline (`scripts/parse_demos.py`) and capture pipeline are already generalized. Adding data is mechanical, not architectural.

**GRPO-specific concern:** With K=4 generations and ~3000 training samples, that's 12K generations per epoch. K=4 is small for advantage estimation (DeepSeek used K=64 for math). If training is unstable, increase K to 8-16 at the cost of more inference compute.

**Status:** Known limitation. Not a blocker for first experiment.

---

## D009: Capture plan sampling strategy (2026-02-20)

**Decision:** Sample screenshots every ~3 seconds during active rounds (freeze_end to round_end), plus event ticks (first kill, bomb plant, post-plant). For each tick, capture 1 T-side + 1 CT-side POV.

**Why 3 seconds:** Balances coverage vs redundancy. Game state changes meaningfully every 2-5 seconds. Faster sampling adds near-duplicate frames. Slower sampling misses important moments.

**Why 1T + 1CT:** Both sides face different decision problems at the same tick. A CT holding a site sees different information than a T executing. Training on both perspectives doubles the data and teaches the model to reason from both sides.

**POV selection:** T-side prefers the bomb carrier (they face the most consequential decisions). CT-side takes first alive player (no strong prior on who matters more).

**Event ticks:** First kill and bomb plant are inflection points where the optimal strategy changes sharply. Over-representing these moments means the model sees more diverse game states (pre/post contact, pre/post plant).

**Status:** Implemented in `scripts/plan_captures.py`.

---

## D010: Unified extraction harness for two repos (2026-02-26)

**Decision:** Merge `viewer-harness/` into a new `harness/` directory that covers both the cs2-demo-viewer extraction and a new cs2-tools extraction. Add cs2-tools scope (5 features: T01–T05) alongside the existing viewer features (F01–F05).

**Why:** The chimera project has two independent pieces worth extracting as standalone repos: the Next.js demo viewer (`site/`) and the Python demo processing tools (`src/netcon.py`, `scripts/parse_demos.py`, `scripts/export_viewer_data.py`, `scripts/plan_captures.py`, `scripts/capture_screenshots.py`). Having a single harness directory with unified health checks and progress tracking keeps extraction work organized. The root-level `feature-list.json` and `init.sh` remain untouched — they track the training pipeline, which is a separate concern.

**Scope of cs2-tools:**
- `netcon.py` — TCP console driver for CS2 demo playback
- `parse_demos.py` — Parse .dem files via awpy into Parquet + metadata
- `export_viewer_data.py` — Export parsed data to viewer JSON format
- `plan_captures.py` — Generate capture plans (tick + POV sampling)
- `capture_screenshots.py` — Drive CS2 demo playback to capture JPEGs

**Key issue:** `parse_demos.py` has a hard import of `src.data.manifest` which is chimera-internal. T01 makes this conditional so the tool works standalone.

**Alternatives considered:**
- Keep viewer-harness separate, add a tools-harness. Rejected: two harness directories for the same kind of work (extraction) is redundant.
- Put extraction features into the root feature-list.json. Rejected: that file tracks the training pipeline (F01–F07), which is a different concern with different lifecycle.

**Status:** Decided. Harness restructured.

---

## D011: Revised reward architecture — 3 signals + multiplicative gate (2026-02-26)

**Decision:** Replace the 7 additive reward signals with a cleaner 3-signal architecture plus a multiplicative format gate. Drop consistency and reasoning quality entirely. Introduce behavioral feature vectors for decision alignment, continuous outcome modulation with player contribution weighting, and KL regularization.

**Why:** The original 7-signal design (D005/D006) had several problems identified through detailed analysis:

1. **Format gate was toothless.** At 0.05 additive weight, invalid JSON still collected 0.95 of reward. The gate didn't gate.
2. **Decision alignment was redundant with outcome.** Outcome already contains alignment (`0.4 + 0.6*a`). A separate pure-alignment signal at 0.15 double-counted "agree with pro," overweighting imitation.
3. **Consistency was gameable.** The 6 keyword-matching heuristics (low health → "caution words") taught pattern matching, not reasoning. At 0.20 weight, the model would learn to sprinkle trigger words. This is exactly the "rigid reasoning" that VLAA-Thinker (2025) found SFT produces.
4. **No KL constraint.** Nothing prevented mode collapse onto generic "safe" advice.
5. **Action taxonomy was too coarse.** 6 categories missed tactical nuance — "push with flash" vs "push dry" both mapped to "aggressive."
6. **Too many objectives for the data scale.** 7 signals with ~3000 samples and G=16 asks the model to satisfy 7 different objectives simultaneously. Cleaner gradient with fewer signals.

### Revised reward function

Total reward with multiplicative format gate:

```
r(y, o_t) = 𝟙[valid_json(y)] · (α·R_percept + β·R_decision + γ·R_outcome)
```

where α = 0.20, β = 0.30, γ = 0.50.

Invalid JSON → zero total reward, regardless of other signal quality.

### Component 1: Perceptual accuracy R_percept (α = 0.20)

Merged hard + soft field accuracy. Prevents SFT regression.

```
R_percept(y, s_t) = (1/|F|) Σ_{f∈F} match(y_f, s_{t,f})
```

F is the union of hard fields (health, armor, weapons, player counts) and soft fields (money, map, bomb status). `match(·)` is exact for categoricals, ±10% tolerance for numerics, Jaccard for lists.

### Component 2: Decision alignment R_decision (β = 0.30)

Compares model's advised action against the pro's behavioral signature extracted from the next Δ ≈ 10–15 seconds of tick data (~900–1350 ticks at 90Hz).

**Behavioral feature vector** from tick data:

```
b_t^{pro} = (d_move, d_obj, u_type, e_timing, δ_engage)
```

| Feature | Domain | Description |
|---------|--------|-------------|
| d_move | {-1, 0, 1} | Movement relative to enemies (retreat, hold, advance) |
| d_obj | {-1, 0, 1} | Movement relative to bomb/site (away, neutral, toward) |
| u_type | {0,1}^k | Binary vector of utility types used (smoke, flash, HE, molly) |
| e_timing | {0, 1} | Whether player initiated engagement (fired first) |
| δ_engage | [0, 1] | Normalized time to first damage (0=immediate, 1=none) |

Model text is mapped to the same feature space via keyword extraction (Approach A):

```
R_decision(y, τ^{pro}) = (1/|b|) Σ_j 𝟙[b_{t,j}^{model} = b_{t,j}^{pro}]
```

Exact match for binary/categorical features, ±0.2 tolerance for δ_engage, Jaccard for u_type.

Falls back to Jaccard over coarse action categories when behavioral features are unavailable (pre-F05).

### Component 3: Outcome-modulated decision reward R_outcome (γ = 0.50)

The core learning signal. Modulates decision alignment by round outcome, weighted by the spectated player's causal contribution to the result:

```
R_outcome = R_decision · Ω(W, φ, a)
```

where a = R_decision, W ∈ {0,1} is round outcome, and:

```
Ω(W, φ, a) = W·φ·(0.5 + 0.5·a) + (1−W)·φ·(0.5 − 0.3·a)
```

**Player contribution** φ ∈ [0, 1]:

```
φ = 0.4·(damage_dealt / max_damage_in_round)
  + 0.3·(survival_time / round_duration)
  + 0.3·𝟙[objective_action]
```

where objective_action ∈ {plant, defuse, last_alive}.

**Signal matrix at φ = 1:**

| | Pro wins | Pro loses |
|---|---------|-----------|
| Model agrees (a=1) | 1.0 | 0.2 |
| Model deviates (a=0) | 0.5 | 0.5 |

**Why φ matters (credit assignment):** If the spectated player died in 5 seconds without dealing damage, φ ≈ 0 and the outcome signal is nearly zeroed out — the round result tells us nothing about advice quality at that moment. Without φ, noisy outcome labels on low-contribution snapshots inject high-variance gradients that destabilize training.

**Why the asymmetry matters (strategy discovery):** Deviating from a losing play gets moderate reward (0.5), while endorsing a losing play gets low reward (0.2). Over many GRPO groups, this pushes the policy away from losing strategies even when they matched the pro's choice. This is how the model can potentially exceed pro play — the strongest result for the paper.

### KL regularization

```
L_total = L_GRPO + λ_KL · KL(π_θ || π_ref)
```

λ_KL = 0.02 (via TRL's `kl_coef` parameter). Reference model is the SFT checkpoint. Prevents the reasoning distribution from diverging arbitrarily — the model shouldn't mode-collapse onto narrow "safe" advice that scores well across all game states.

### GRPO dynamics

For each prompt x, GRPO samples G=16 completions and normalizes advantages within the group:

```
Â_i = (r_i − mean({r_j})) / (std({r_j}) + ε)
```

Key property: GRPO only needs correct relative ordering within the group, not calibrated absolute rewards. This means:

- **Low within-group variance → no gradient.** Format gate and perceptual accuracy saturate early (same screenshot = same HUD reading for all completions). After early training, ~95% of the gradient comes from R_decision and R_outcome. This is by design — perception is a floor, not a training signal.
- **φ reduces variance on noisy samples.** Outcome labels where the spectated player had minimal impact are automatically downweighted, preventing high-variance gradients from destabilizing training.
- **3 signals give a cleaner optimization landscape.** With 7 signals, the model had to balance conflicting objectives. With 3, the gradient direction is clearer.

### Weight summary

| Signal | Type | Weight | Purpose |
|--------|------|--------|---------|
| Format gate | Multiplicative mask | — | Invalid JSON → zero reward |
| R_percept | Additive | 0.20 | Prevent SFT perception regression |
| R_decision | Additive | 0.30 | Learn behavioral alignment with pro |
| R_outcome | Additive | 0.50 | Core: outcome-weighted strategy learning |
| KL penalty | Regularizer | λ=0.02 | Prevent mode collapse |

### What was dropped and why

| Old signal | Old weight | Why dropped |
|------------|------------|-------------|
| Soft field accuracy | 0.05 | Merged into R_percept |
| Consistency | 0.20 | Gameable keyword matching, caps out fast, teaches rigid patterns. Let GRPO discover coherent reasoning through outcome signal instead. |
| Reasoning quality | 0.10 | Structural quality saturates early (all completions learn valid JSON structure fast). Dead gradient weight. |
| Decision alignment (separate) | 0.15 | Redundant with outcome, which already contains alignment. Merged into single R_decision + R_outcome architecture. |

### Relationship to prior work

- **PeBR-R1** (2025): Also separates perception and reasoning into RL phases, but uses CLIP scores for perception. Our R_percept uses engine-accurate demo data — a stronger supervision signal.
- **Game-RL** (Fudan, 2025): GRPO on games with verifiable rewards. Their games have provable answers; CS2 has noisy outcomes. Our Ω function handles this noise via φ-weighting.
- **VLAA-Thinker** (2025): Found SFT locks models into "rigid reasoning modes." Our architecture explicitly addresses this: SFT only teaches perception, GRPO teaches reasoning from scratch.
- **Praxis-VLM** (2025): Found GRPO models internalize deeper decision-making than SFT baselines. Supports our hypothesis that C > B.

### Ground truth schema (for F05 implementation)

```json
{
    "game_state": { ... },
    "pro_action": {
        "categories": ["aggressive", "utility"],
        "description": "Pushed A site with flash",
        "behavior": {
            "movement_direction": 1,
            "objective_direction": 1,
            "utility_used": ["flash"],
            "initiated_engagement": true,
            "engagement_delay": 0.15
        }
    },
    "round_won": true,
    "player_contribution": {
        "damage_dealt": 120,
        "max_round_damage": 300,
        "survival_time": 45.0,
        "round_duration": 90.0,
        "objective_action": false
    }
}
```

**Status:** Implemented. Supersedes D005 and D006 reward designs. Files changed: `src/training/rewards.py`, `src/training/grpo_trainer.py`, `src/training/__init__.py`, `scripts/train_grpo.py`.
