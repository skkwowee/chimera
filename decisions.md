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

## D011: Conditional manifest import in parse_demos.py (2026-02-26)

**Decision:** Wrap `from src.data.manifest import append_to_manifest` in a `try/except ImportError`, setting it to `None` when unavailable. Guard the call site with `if append_to_manifest is not None`.

**Why:** `parse_demos.py` is being extracted into cs2-tools where `src.data.manifest` won't exist. The manifest is only used in the legacy `--snapshots` write path — the default parquet mode never calls it. Making it conditional lets the script work identically in chimera (manifest available) and standalone (manifest absent, snapshots just skip manifest logging).

**Alternatives considered:**
- Remove the manifest call entirely. Rejected: chimera's snapshot workflow still uses it.
- Pass a flag like `--no-manifest`. Rejected: more complex, and the try/except is invisible to callers who have the module.

**Status:** Done. Harness confirms conditional import detected.

---

## D012: Git subtree split for cs2-demo-viewer extraction (2026-02-26)

**Decision:** Use `git subtree split --prefix=site -b demo-viewer` to extract `site/` into `~/cs2-demo-viewer/` as a standalone repo with `main` branch.

**Why:** Preserves the 9-commit history of the viewer code. The new repo contains only the Next.js app — no chimera training code, no Python scripts. Users who want to visualize CS2 demo data don't need to clone the entire research repo.

**Method:** Subtree split creates a branch with rewritten history where `site/` is the root. We `git pull` that branch into a fresh `~/cs2-demo-viewer/` repo and rename the branch to `main`. `site/` remains in chimera (subtree split is non-destructive).

**Alternatives considered:**
- Copy files without history. Rejected: losing git history makes it harder to understand why code is structured the way it is.
- `git filter-branch`. Rejected: deprecated in favor of subtree split for this exact use case.

**Status:** Done. `~/cs2-demo-viewer/` exists with 9 commits on `main`.

---

## D013: Revised reward architecture — 3 signals + multiplicative gate (2026-02-26)

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

---

## D014: Why the two-phase reward architecture works — design rationale (2026-02-26)

This decision records the reasoning behind D013's design, tracing each component to the information structure of the problem and the failure modes it addresses.

### Core insight: decompose along the supervised/RL boundary

Perception has exact ground truth from engine data (health is 73 or it isn't). Strategy has no single correct answer — only noisy outcome signal from round results across 10 players over 90 seconds. These are fundamentally different learning problems:

- **Exact supervision → SFT.** Phase 1 solves perception. Cheap, fast convergence, no ambiguity.
- **Outcome-conditioned signal → RL.** Phase 2 solves reasoning. The model already knows what's on screen, so the RL search space is limited to "what to do about it" rather than "what am I looking at AND what to do about it."

Skipping Phase 1 (Model D) forces RL to solve both simultaneously through one noisy channel. The search space is too large. This is the same decomposition as AlphaGo (supervised value network → self-play policy improvement) and InstructGPT (SFT → RLHF).

### Why each reward signal exists

**R_percept (0.20) is a constraint, not an objective.** It prevents catastrophic forgetting of Phase 1. In GRPO, same screenshot → same HUD reading → same score for all G=16 completions → zero within-group variance → zero gradient. After early training, R_percept contributes no gradient. It's a regularizer expressed as a reward signal, using the same GRPO machinery without needing a separate loss term.

**R_decision (0.30) is the imitation anchor.** Pure RL from sparse outcomes would be too unstable with ~3000 samples. Behavioral feature alignment gives dense signal about what kind of action to recommend. But it's not SFT — the model isn't forced to match the pro. GRPO's group-relative normalization means the model only needs some completions that match and some that don't; the relative advantage does the rest.

**R_outcome (0.50) is the strategy discovery signal.** The asymmetric Ω function creates an optimization landscape where imitating winning strategies is strongly rewarded, imitating losing strategies is penalized, and deviation gets moderate reward regardless. This is offline policy improvement: the pro's trajectory is the behavior policy, the model's output is the target policy, and Ω reweights the imitation signal by returns.

### Why the outcome asymmetry enables learning beyond the expert

In a symmetric design (penalize all deviation equally), the model converges to pure imitation — it can never exceed the expert because disagreeing is always punished. The asymmetry changes this:

1. Model sees screenshot where pro rushed A and lost.
2. 16 completions generated: some say "rush A" (matching), some say "hold" or "rotate."
3. "Rush A" completions get low reward (agree + lose = 0.2 × φ).
4. "Hold" completions get moderate reward (deviate + lose = 0.5 × φ).
5. GRPO computes positive advantage for deviating completions.

Over thousands of such examples, the model learns counterfactual reasoning: "when the game state looks like X and the pro rushed and lost, something else would have been better." This is learning from negative examples — inferring better strategies from observed failures. Not imitation, but reasoning about consequences.

### Why φ solves credit assignment

In a 5v5 game, round outcome is a team signal. If the spectated player died at second 5 and the team won at second 90, the outcome says nothing about what that player should have done at second 3. Without φ, these samples inject random gradients — high variance for zero information.

φ acts as attention over the training data. High-φ samples (player dealt damage, survived, planted/defused) get strong outcome signal. Low-φ samples get attenuated signal. Mathematically equivalent to advantage-weighted regression with sample quality weighting — a standard technique in offline RL (Peng et al., 2019).

### Why GRPO specifically

GRPO's group-relative normalization sidesteps the need for absolute reward calibration. PPO would need a value function estimating expected reward of a CS2 screenshot — but expected reward depends on skill level, teammates, opponent tendencies. The value estimate would be terrible.

GRPO only asks: "which of these 16 completions is relatively better?" Even noisy absolute rewards can produce informative relative orderings. φ ensures that when the absolute signal is truly uninformative, the relative ordering is damped rather than random.

### The effective gradient structure

The nominal weight split is R_percept 20% / R_decision 30% / R_outcome 50%. But GRPO advantages are `(r - mean) / std` within each group. Low within-group variance → no gradient. After early training:

- R_percept variance ≈ 0 (perception converges) → ~0% of gradient
- R_decision + R_outcome variance is high (different advice → different alignment) → ~100% of gradient

The actual training signal is almost entirely "which strategic recommendations correlate with winning." The perception floor just keeps the model honest about what it sees. This is by design — each signal contributes gradient only during the training phase where it's informative, then gracefully drops out.

### Intellectual lineage

| Component | Draws from |
|-----------|-----------|
| Two-phase SFT → RL | InstructGPT pipeline; AlphaGo supervised → self-play |
| Outcome-weighted imitation | Advantage-weighted regression (Peng 2019), Decision Transformer (Chen 2021) |
| φ credit assignment | Per-trajectory importance weighting in offline RL |
| Asymmetric deviation handling | Game theory: in imperfect info games, expert strategy is one equilibrium among many |
| GRPO for non-verifiable rewards | Extension of DeepSeek-Math/R1 (verifiable) to noisy outcome domains |
| 7→3 signal pruning | Analyzing gradient flow and cutting dead weight — GRPO-specific insight that low-variance signals contribute no gradient regardless of nominal weight |

**Status:** Rationale documented. Informs D013 implementation and paper Method section.

---

## D015: Mechanical skill as a latent confound — how the architecture handles aim noise (2026-02-26)

Round outcome is a function of at least five variables: decision quality, mechanical skill (aim), teammate quality, opponent quality, and randomness. The model learns decision quality. Everything else is noise. This decision documents how the architecture handles — and fails to handle — mechanical skill as the primary confound.

### The model's relationship to aim

The model reads a HUD screenshot and outputs a strategic recommendation in text. It never controls a mouse. Aim is completely outside its input and output space — it is a **latent variable that corrupts the training signal** but does not appear at inference time. The model is a coach, not a player.

### How each component handles aim noise

**φ dampens aim-correlated outcome noise.**

Consider two problematic cases:

*Good decision + bad aim* (player holds the correct angle, whiffs the shot, dies at second 5, team loses at second 90):
- φ is low: low damage dealt, early death, no objective interaction.
- R_outcome is multiplied by small φ → attenuated.
- The model is not penalized much for the correct decision. The noisy negative outcome is dampened. Working as intended.

*Bad decision + good aim* (player dry-peeks into a stack, hits three headshots, team wins):
- φ is high: lots of damage, kills, survival.
- R_outcome is multiplied by large φ → amplified.
- The model is rewarded for imitating a bad decision that was mechanically bailed out.

This second case is the primary vulnerability. φ cannot distinguish "player contributed because of good decisions" from "player contributed because of cracked aim." Both produce high damage and kills.

**R_decision provides aim-independent signal.**

Behavioral feature alignment — aggression level, positioning type, utility usage, rotation timing — can be assessed regardless of whether shots landed. R_decision says "the pro held a passive angle here" or "the pro used a flashbang before peeking." These are decision-level features, not aim-level features.

Even when R_outcome is corrupted by mechanical noise, R_decision provides a clean 30% signal about what kind of strategy was employed. This anchors the model to strategic patterns while R_outcome provides the noisy-but-informative "did it work?" correction.

**Population averaging is the statistical defense.**

Training data includes multiple pros across multiple matches. For any given game state:
- Most pros hold the angle conventionally → works ~65% of the time
- Occasionally a pro dry-peeks aggressively → works ~40% in general (but ~90% when a mechanically exceptional player does it)

Across the training set, the expected outcome conditioned on the decision averages over the distribution of mechanical executions:

```
E[outcome | decision, game_state] averages over aim_skill ~ P(aim | pro_population)
```

Mechanically-dependent outliers (dry-peeking works because of exceptional aim) are outvoted by the population who play the same state conventionally with better expected value. The law of large numbers works because aim quality is uncorrelated with the game state visible in the HUD.

### The "donk problem" — unconventional play that works

Players like donk make plays that appear to defy conventional wisdom. These decompose into two categories:

**1. Superior game reading expressed as aggression.** donk peeks because he has read the opponent's utility usage, timing patterns, and information state — he knows the opponent is not ready. This IS visible in the HUD (killfeed timing, minimap positions, round time, economy). The model should learn this. It is not "defying logic" — it is deeper logic.

**2. Purely mechanical plays.** Dry-peeking an AWP because you can headshot faster than the AWP can scope. This works only with exceptional reaction time and is genuinely bad coaching advice for most players.

How the architecture handles each:
- Category 1: R_decision captures the aggressive posture, R_outcome rewards it because the game state supports it. When other pros face the same state and play passively, R_decision anchors differently. The model learns "aggression is viable here" — which is correct.
- Category 2: Gets rewarded in that player's demos, but is outvoted by the broader population playing the same state conventionally. With only 4 demos, if one features a mechanically exceptional player, their influence could be outsized.

### What the model actually learns

The model learns **decision-level expected value averaged over the mechanical skill distribution in the training set** (pro-level play). This is the correct target for a coaching model — it advises what a pro-level player should do, given pro-level mechanics. It is not calibrated to other skill levels.

The three-layer defense:
1. **φ attenuation**: When mechanics dominate the outcome (player dies instantly or gets an improbable multi-kill), φ weights the outcome signal by actual contribution rather than treating it as pure decision feedback.
2. **R_decision aim-independence**: 30% of the reward signal measures strategic features that are orthogonal to mechanical execution.
3. **Population averaging**: Over many similar game states across multiple players, aim noise is uncorrelated with HUD state and averages out.

### Acknowledged limitations

**Small sample size.** With ~3000 samples from 4 demos, the statistical power for population averaging is limited. Mechanically-exceptional plays from one demo can have outsized influence. This is mitigated by R_decision's dense signal but cannot be fully resolved without more data.

**φ conflates aim and decisions.** φ measures damage dealt, survival, and objective interaction — all of which correlate with both good decisions and good aim. A principled separation would require observing aim quality directly (crosshair placement, reaction time), which is not available from HUD screenshots.

**No explicit aim latent variable.** We do not model aim as a latent variable because: (1) it is unobservable from HUD data, making the model unidentifiable; (2) the output space does not include aim, so separating aim from decisions in the reward would not change what the model generates; (3) φ already serves as an implicit proxy.

### Data curation as the primary mitigation

The strongest defense against mechanical confounds is **more data from more diverse players**. Stratifying demos by playstyle (passive/aggressive/hybrid) ensures population averaging works even with moderate N. This is a data curation decision, not an architectural change, but it likely matters more than any reward function modification.

### Implications for the paper

This analysis belongs in the Limitations section and informs the framing of R_outcome in the Method section. The key claim: the architecture handles mechanical noise through three complementary mechanisms (φ attenuation, aim-independent R_decision, population averaging), but small dataset size remains the primary bottleneck for robust noise averaging. The model learns expected value at the training population's skill level, not universal strategic truth.

**Status:** Analysis documented. Informs paper Method (Section 3.4) and Limitations. Connects to D008 (data scale) and D013 (reward architecture).

---

## D016: Temporal credit assignment — from bandit framing to causal proximity (2026-02-26)

The current GRPO architecture treats each screenshot as an independent contextual bandit: one observation, one response, one reward. The round outcome W and player contribution φ are round-level aggregates applied identically to every decision point within a round. This decision documents the credit assignment problem and the planned solution.

### The problem: round outcome is temporally diffuse

A round lasts 30-90 seconds. Screenshots are captured every ~3 seconds, producing 10-30 decision points per round. All receive the same W ∈ {0,1} and the same φ. But the decision that determined the round outcome was typically made 20-40 seconds before the round ended — the execute timing, the rotation call, the utility usage that opened a site. Late-round events (clutch plays, defuse attempts) are resolutions of earlier decisions, not the decisions themselves.

Concrete failure case: a buy-phase screenshot (t=3s) where the pro correctly purchases AK + armor gets the same outcome penalty as a screenshot at t=25s where the pro made the bad rotation that actually lost the round. The model is penalized for correct buy advice because of a decision made 22 seconds later.

### Why forward windows don't work

An initial proposal was per-timestep φ computed from a forward window (damage/survival/objective in the next Δ≈12 seconds after each decision point). This is wrong for the same reason: the window captures what happened *after* the decision point, but the outcome relevance of a decision may not manifest for 30-60 seconds. The flash thrown at t=15 that enabled the site take at t=20 that led to the plant at t=25 — the forward window from t=15 might not even reach the plant.

More fundamentally: kill events are outcomes, not decisions. Weighting by proximity to kills conflates results with the choices that caused them.

### Detecting actual decision moments in tick data

A "decision" is a behavioral state transition — the moment a player commits to a new course of action. These are detectable from the Parquet tick data (90Hz position, velocity, view angles, active weapon):

| Signal | State transition | Decision type |
|--------|-----------------|---------------|
| Velocity: 0 → moving | Started moving after holding | Commit to rotate/push/reposition |
| Velocity: moving → 0 | Stopped after moving | Set up at new position |
| View angle: >30° snap in 0.5s | Rapid aim redirect | Received info, shifted attention |
| Active weapon → grenade | Switched to utility | About to commit a consumable |
| Grenade → rifle | Utility thrown | Executed utility play, ready to fight |
| Walking ↔ running | Shift key change | Chose speed vs stealth |

The causal chain for a kill at tick T is:

```
T-800: weapon switch to flash     ← DECISION (commit utility)
T-650: flash thrown                ← EXECUTION
T-500: weapon switch to rifle     ← DECISION (re-equip)
T-450: velocity 0→moving          ← DECISION (commit to push)
T-200: enemy flashed              ← CONSEQUENCE
T:     kill                        ← OUTCOME
```

The decision that matters is at T-800 (committing the flash) and T-450 (committing to push), not T (the kill).

### Proposed solution: causal proximity to decision-linked outcomes

1. For each round, detect behavioral state transitions from tick data (decision moments)
2. For each decisive outcome event (first blood, entry frag, bomb plant, round-ending kill), trace backward to the nearest decision moment that preceded it
3. For each screenshot/decision point, compute proximity to these decision-linked moments
4. Use proximity as a multiplier on the outcome signal

```
causal_weight(t) = max over (decision_moment d linked to outcome e):
    exp(-|t - d.tick| / τ)     where τ ≈ 5-8 seconds

φ_causal(t) = causal_weight(t) / Σ_t' causal_weight(t')
```

This replaces the current round-level φ with a per-timestep weight that tracks actual causal relevance: screenshots near the decisions that caused the outcome get strong signal, screenshots during buy phase or post-resolution get weak signal.

### Economy as an additional confound

Round type (eco/force/full buy/pistol) dominates expected outcome independent of decision quality. Eco rounds have ~10-15% win rate regardless of strategy. The outcome signal on eco rounds is near-zero mutual information with decision quality.

Planned addition: economy informativeness ε derived from team equipment value ratios:

```
ε = sigmoid(k · (team_equipment / enemy_equipment - 0.5))
```

Full buy mirrors: ε ≈ 1.0 (outcome reflects decisions). Eco vs full buy: ε ≈ 0.2 (outcome reflects economy). Folds into Ω multiplicatively alongside φ_causal.

### The combined outcome signal

```
R_outcome(t) = R_decision(t) · ε · φ_causal(t) · asymmetry(W, a)
```

Three layered filters, each removing a different source of noise:

| Filter | Question it answers | Noise it removes |
|--------|-------------------|-----------------|
| ε (economy) | Was outcome economically predetermined? | Economic confound |
| φ_causal(t) | Was this decision causally relevant? | Temporal diffusion |
| asymmetry(W, a) | Did the strategy work or fail? | Imitation bias |

### Mathematical interpretation: heuristics as variance reduction

Each component is formally a variance reduction technique on the policy gradient estimate:

- **φ / φ_causal**: Control variate — multiplies noisy gradients by ~0 when signal is pure noise
- **ε**: Prior on signal-to-noise ratio — downweights samples where I(decision; outcome) ≈ 0
- **R_decision**: Reward shaping — dense signal that guides policy search without changing optimal convergence point
- **Causal proximity**: Importance weight on the temporal dimension — corrects for non-uniform relevance of decision points within a round

Each heuristic trades a small, understood bias for a large reduction in gradient variance. This tradeoff is favorable in low-data regimes and becomes less necessary as data scales.

### Scaling behavior: heuristics vs data

Every heuristic becomes unnecessary with sufficient data — the law of large numbers does the work instead:

| Heuristic | Replaced by | Approximate data needed |
|-----------|------------|------------------------|
| φ (contribution) | Population averaging over diverse players per state | ~100x current |
| ε (economy) | Enough samples per economy level per state | ~5x per round type |
| R_decision (alignment) | Pure outcome signal with enough samples per state | ~1000x current |
| Causal proximity | Enough rounds that irrelevant timestamps' noise averages out | ~50x current |

With thousands of demos (planned training scale), several heuristics may become unnecessary. The ablation curve — which heuristics can be removed at which data scales — is itself an interesting empirical result for the paper.

### Paper framing

The reward architecture should be presented as domain-informed variance reduction for RL in a regime where standard solutions (learned reward model, simulator, verifiable answers) are unavailable:

1. No learned reward model (no pairwise preference data)
2. No simulator (can't run counterfactuals — no CS2 engine that takes text advice as input)
3. No verifiable answers (no ground truth for "correct strategy")
4. Only: offline expert demos with noisy terminal reward

The heuristic decomposition IS the method contribution. It makes RL viable where naive outcome-based training would fail. The intellectual lineage is offline RL (advantage-weighted regression, Decision Transformer) adapted to a text-output, non-interactive setting.

**Status:** Analysis documented. Requires implementation in F05 (ground truth generation). Connects to D013 (reward architecture), D014 (design rationale), D015 (mechanical skill confound).

---

## D017: Data scale revision — thousands of demos (2026-02-26)

**Decision:** Training will use thousands of demos, not just the initial 4 from Furia vs Vitality. This significantly changes the data regime and the relative importance of reward heuristics.

**Why this matters:**
- D008 identified data scale as the primary limitation. With thousands of demos from diverse teams/events/maps, the population averaging defense (D015) becomes robust rather than theoretical.
- Several reward heuristics (D016) exist specifically to compensate for low-N variance. At thousands of demos, some become unnecessary and their bias becomes the dominant concern.
- The ablation study gains statistical power: can meaningfully test which heuristics are needed at which data scales.

**Implications:**
- The parsing pipeline (`cs2-parse-demos`, `cs2-tools`) and capture pipeline (`cs2-capture`) are already generalized. Scaling is mechanical, not architectural.
- Per-state sample counts will be high enough that rare game states (eco clutches, 1v5s) have meaningful representation.
- The "donk problem" (D015) is mitigated by population diversity — unconventional playstyles are outvoted by the broader population.
- Economy informativeness ε (D016) may still be valuable even at scale, since eco rounds are fundamentally low-information regardless of sample count.

**Status:** Decided. Updates D008 (data scale) assessment. Training data collection to scale with pipeline tooling.

---

## D018: Context-aware observation model — multi-image + round context (2026-02-27)

**Decision:** The model receives multi-image input (current + up to 2 prior screenshots) plus a structured round context string c_t, rather than a single screenshot with a generic prompt.

**Observation model:** `o_t = (I_{t-k}, ..., I_t, c_t)`

### Why this matters

The previous implementation passed a single screenshot with a static prompt: "Analyze this CS2 screenshot." The model had to infer the entire round state from one HUD frame — who died, what utility was used, what the economy is, what happened 20 seconds ago. This is both unrealistic (a real player/coach has full round awareness) and wasteful (the information exists in the tick data but was being discarded).

With context, the task shifts from "infer everything from one frame" to "given full round context + current visual, advise on strategy." This is a much better-specified problem and matches the information a real coach would have.

### What c_t contains

Generated from Parquet tick data by `scripts/generate_sft_labels.py`:

- **Round header:** Round number, score, map name, round time elapsed
- **Economy (at round start):** Both teams' buy type and total equipment value
- **Chronological events (up to tick t):**
  - Deaths — detected from health transitions (>0 → 0) in consecutive ticks
  - Utility usage — detected from grenade items disappearing from inventory
  - Bomb events — from the bomb event JSON (plant, defuse, drop)
- **Current state:** Alive counts, bomb status, POV player loadout, teammate states

All derived from engine data. No inference, no model labeling.

### Multi-image input

Up to 2 prior screenshots from the same round and POV player are included before the current screenshot. This provides visual continuity — the model can observe:
- A smoke that was present 6s ago and has now faded
- An enemy that was visible and has moved
- Teammates who were alive and are now dead
- Position changes of the POV player

Qwen3.5-27B supports multi-image natively. At typical resolution, 3 images add ~1500 image tokens — manageable with 4-bit quantization on 24GB VRAM.

### Impact on reward terms

**R_percept:** The model has no excuse for getting alive counts wrong when the context says who died. Perceptual accuracy should be higher with context, making R_percept more of a floor/constraint than before.

**R_decision:** Becomes much more meaningful. Without context, advice is reactive to a single frame. With context, the model can reason about round progression — "two smokes have been used A, they're probably executing B" — and the behavioral alignment score measures whether that reasoning leads to the right call.

**R_outcome:** Credit assignment partially improves. The model at t=25s knows that a trade happened at mid and CTs have no AWP. If it advises "execute A now" and the round is won, the reward is attributable to informed advice. But temporal credit assignment (D016) is still needed to weight decision points by causal relevance within the round.

### Files changed

- `src/prompts.py` — System prompt updated for multi-image + context. Added `build_user_prompt(context)` function.
- `scripts/generate_sft_labels.py` — Added `generate_round_context()`, `detect_round_events()`, `find_prior_screenshots()`, economy classification. Labels now include `context` and `prior_screenshots` fields.
- `src/training/data_utils.py` — `GRPODataItem` now has `prior_image_paths` and `context`. `_build_prompt_content()` assembles multi-image + context prompts. `prepare_conversation_format()` supports prior images.
- `src/training/rewards.py` — Docstring updated to reflect observation model.

### Backward compatibility

Labels without `context` or `prior_screenshots` fields still work — the data loading falls back to the legacy single-image, generic-prompt behavior. Existing labels on Hub don't need regeneration for SFT (though they should be regenerated to include context before GRPO training).

**Status:** Implemented. Labels need regeneration with updated `generate_sft_labels.py` to include context. Connects to D013 (reward architecture), D016 (credit assignment).

---

## D019: Base model — Qwen3.5-27B dense over Qwen3.5-35B-A3B MoE (2026-02-27)

**Decision:** Use Qwen3.5-27B (dense, 27B params) as the base model, not Qwen3.5-35B-A3B (MoE, 35B total / 3B active).

### Why

The MoE model cannot be trained on our hardware (RTX 4090, 24GB VRAM). Two independent blockers:

1. **BnB 4-bit quantization incompatibility.** Qwen3.5-35B-A3B packs all 256 experts into single 3D tensors (e.g. `mlp.experts.down_proj` → shape `[256, 512, 2048]`). BnB only quantizes `nn.Linear` modules — packed expert tensors aren't `nn.Linear`, so BnB treats them as bf16. The device map calculator then estimates ~60GB and pushes layers to CPU, triggering `validate_environment()` rejection. The final quantized model *would* fit (~20GB), but loading never completes.

2. **No fused kernel support for Qwen3.5.** The `qwen3-moe-fused` kernel (which fuses experts into single tensors to fix the peak memory spike) only supports Qwen3, not Qwen3.5. Contributing Qwen3.5 support is possible but a time sink orthogonal to the research.

3. **Unsloth confirms.** Their Qwen3.5 docs explicitly state: "MoE QLoRA 4-bit is not recommended due to BitsAndBytes limitations." bf16 LoRA requires 74GB VRAM.

The older Qwen3-30B-A3B stored experts individually (`experts.0.down_proj`, `experts.1.down_proj`, ...) as separate `nn.Linear` modules, so BnB worked. Qwen3.5 changed the internal format.

### Alternatives considered

- **Qwen3.5-35B-A3B (MoE) with fused kernel.** Blocked: fused kernel doesn't support Qwen3.5 yet. Would need Triton 3.6+, PyTorch 2.10+, and an `OutputRecorder` stub. Investigated thoroughly, dependency chain too fragile.
- **Qwen3-30B-A3B (MoE, older generation).** Tooling works, similar architecture. Rejected: older model family, weaker vision understanding. The base model's zero-shot ability to parse CS2 screenshots is the foundation for everything downstream.
- **Pre-quantized BnB checkpoint.** Chicken-and-egg: creating one requires an 80GB+ GPU. None published for Qwen3.5-35B-A3B.
- **GGUF format.** Inference-only — no autograd, no training. Incompatible with QLoRA/GRPO.
- **Qwen2.5-VL-8B (original base model).** Would work trivially on hardware. Rejected: much less capable for vision and reasoning.

### Why dense is fine for Chimera

- **LoRA expressiveness is sufficient.** We train ~0.1-1% of parameters via QLoRA adapters. The LoRA rank (r=16-64) determines how many "directions" each layer's correction can adjust — r=64 modifies ~1.5% of directions per layer, but corrections compound across 50+ layers. Empirically, r=32-64 handles complex behavioral changes like strategic reasoning patterns.
- **Simpler RL dynamics.** Dense models have clean gradient flow — no router instability, no expert collapse, no reward signal dilution through discrete top-k routing. MoE RL works (DeepSeek-R1 proved it) but is less forgiving. One fewer failure mode for GRPO.
- **Training speed is acceptable.** Dense 27B is ~3-5x slower per step than 3B-active MoE (all 27B params in forward/backward vs ~3B). But iteration speed is gated by reward function design and data quality, not wall-clock time per step. We'll run fewer experiments more carefully.
- **Data requirements unchanged.** Data needs are driven by task complexity, not architecture. QLoRA trains the same tiny adapter fraction either way.

### Technical details: MoE vs dense

**MoE (Mixture of Experts):** Router network selects top-k experts per token from a pool. Only activated experts compute. Advantage: knowledge capacity of 35B with compute cost of ~3B. Disadvantage: all 35B params must be in VRAM (only compute is saved, not memory), training needs load balancing losses to prevent expert collapse, RL gradient flow through discrete routing is complex.

**Unfused vs fused experts:** Standard (unfused) MoE stores each expert's weights as separate `nn.Linear` modules — 256 individual tensors that get separate matmuls. Fused MoE packs all experts into single stacked tensors (e.g. `[256, 512, 2048]`) and uses one batched matmul with expert indexing. Fusion is a memory layout optimization (fewer kernel launches, better GPU utilization) that doesn't affect model quality. Qwen3.5 ships pre-fused at the architecture level — which ironically breaks BnB because the packed tensors aren't `nn.Linear`.

**Dense:** All parameters active for every token. Simpler, more predictable, but more compute per token. For QLoRA fine-tuning, the base model weights are frozen regardless — the question is only how fast forward/backward passes are through the frozen weights.

### Hardware constraints

- RTX 4090: 24GB VRAM
- System: 32GB RAM (WSL2, `.wslconfig` set to `memory=32GB`)
- Qwen3.5-27B in 4-bit: ~15GB VRAM post-load. ~8GB headroom for QLoRA adapters, optimizer states, gradients, activations.
- Loading spike: BnB deserializes bf16 weights through CPU RAM before quantizing to 4-bit. 32GB system RAM is sufficient (was failing with WSL's default 16GB limit).

**Status:** Decided. All config/scripts updated. Supersedes the Qwen3.5-35B-A3B commit (4891cef).
