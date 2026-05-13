# Alignment Delta — Adopting the Three-Level Perception/Reasoning Hierarchy

Status: planning document. NOTHING below has been applied. Inventory of what
changes if chimera reframes from "two-phase training (SFT → GRPO)" to a
three-level hierarchy (See → Situate → Think). Companion docs (drafted in
parallel and assumed to land in the same commit as this one):
`docs/three-level-architecture.md` (architecture overview) and
`docs/event-encoder-design.md` (Level-2 encoder spec).

## 1. Summary

Chimera is currently described as a two-phase pipeline: SFT for HUD
perception, then GRPO for strategy. Under the new framing it is a
three-level hierarchy: **Level 1 — See** (visual perception via SFT,
shipped as F04); **Level 2 — Situate** (an event encoder mapping a window
of game states to a tactical event embedding — NEW, not yet built); and
**Level 3 — Think** (GRPO strategic reasoning that consumes Level 1 + 2
outputs). The reframing's diagnostic claim: every F-axis failure from
`docs/methodology.md` (F1 sibling-tick leakage, F2 outcome-correlated
supervision collapse, F3 action-vec noise) is a symptom of conflating
Level 2 (representation) and Level 3 (scoring) inside RECALL. The
migration isolates them. Unchanged: the SFT recipe and 67.3% number, the
GRPO manual-loop trainer, the format gate, KL regularization, pod
infra, the data pipeline.

## 2. README.md changes

Wrong or stale under the new framing:

- **Abstract (lines 3–5) and Hypothesis (line 11):** "two-phase training
  paradigm" is the framing being retired. Replace with "three-level
  perception/reasoning hierarchy." Title `See, Then Think` survives as
  catchphrase; subtitle becomes See / Situate / Think.
- **`## Method` (lines 13–44):** currently `Phase 1`/`Phase 2`. Becomes
  Level 1 / Level 2 / Level 3 subsections. Phase 2 (lines 19–44) bundles
  scorer machinery (R_percept, R_strategy, RECALL, asymmetric outcome,
  φ, mechanical-skill noise) into one block; under the new framing,
  asymmetric outcome / φ / mechanical-skill content is Level-3 scorer
  machinery, not Level-2 representation. Split accordingly.
- **Reward table (lines 25–32) and line 29 ("R_strategy ... via RECALL"):**
  presents RECALL as THE strategy reward. Single biggest framing error.
  RECALL is now one of three Level-3 scorer families (kNN-over-Level-2,
  LM-judge, BT-head) with no privileged status.
- **Line 32 ("R_strategy is computed via RECALL..."):** rewrite as
  "R_strategy is a Level-3 scorer; current candidates are RECALL (kNN
  over Level-2 embeddings), Claude judge, and a BT head."
- **Pipeline diagram:** README has no graphical diagram. The paper's
  placeholder `\fbox` (`paper/main.tex` line 209–214) needs redrawing
  as three levels. README's flat Step 1–7 list (lines 61–70) needs
  regrouping under L1/L2/L3 headers.
- **Step 5c (line 68):** rename "RECALL implementation" → "Level-3 RECALL
  scorer (depends on Level-2 encoder)."
- **Step 6 (line 69):** "Level-3 GRPO training (gated on Level-2 readiness)."
- **Quickstart `### Train` (lines 205–222):** new canonical sequence:
  1. `python scripts/train_sft.py …` (Level 1; unchanged)
  2. `python scripts/train_event_encoder.py …` (Level 2; NEW)
  3. `python scripts/eval_event_encoder.py …` (Level-2 gates; NEW)
  4. `python scripts/train_grpo.py --reward-mode recall|judge|bt --level2-encoder PATH …`
- **`Project Structure` (lines 124–151):** add
  `src/training/event_encoder.py`, `scripts/train_event_encoder.py`,
  `scripts/eval_event_encoder.py`, `docs/three-level-architecture.md`,
  `docs/event-encoder-design.md`.

Proposed new section list (do NOT write prose):

1. `# Chimera — See, Situate, Think: A Three-Level Hierarchy for VLM Game Understanding`
2. `## Motivation`
3. `## Architecture: three levels`
   - `### Level 1 — See (SFT visual perception)`
   - `### Level 2 — Situate (event encoder)`
   - `### Level 3 — Think (GRPO strategic reasoning)`
4. `## Reward design` (Level-3-only; RECALL / judge / BT as siblings)
5. `## Pipeline status` (regrouped F-list by level)
6. `## Quickstart` (four-step sequence)
7. `## Project structure`
8. `## Setup` (unchanged)
9. `## Hardware requirements` (unchanged)
10. `## Artifact locations` (+ Level-2 artifacts)
11. `## Why this matters beyond games` (unchanged)
12. `## License`

## 3. feature-list.json changes

Add a top-level `level` field on every entry: `1`, `2`, `3`, or
`"cross-cutting"`.

Existing entry changes:

- **F01 (Data manifest)** — `level: "cross-cutting"`. KEEP.
- **F02 (Demo pipeline & viewer)** — `level: "cross-cutting"`. KEEP.
- **F03 (Screenshot–demo sync)** — `level: 1`. KEEP.
- **F04 (SFT — visual grounding)** — `level: 1`. KEEP. Rename to
  "Level 1 — SFT visual grounding."
- **F05 (GRPO dataset from demos)** — `level: "cross-cutting"`. KEEP +
  rename "Shared demo decision dataset"; note it feeds Level-2
  encoder training AND Level-3 GRPO. Still `passes: false`.
- **F05a (Sparsity diagnostic)** — RECLASSIFY `level: 2`. Bucket
  population is a Level-2 representational property.
- **F05b (GRPO smoke test)** — `level: 3`. KEEP.
- **F05c (RECALL)** — **SPLIT into two:**
  - **F05c-L2 (kNN state representation):** `level: 2`. The 19-dim
    `tactical_embedding` is now the Level-2 fallback baseline.
  - **F05c-L3 (kNN-over-Level-2 scorer):** `level: 3`. The
    `A = Q̂ − V̂` formula, same-round mask, action matching, reward
    wiring. Independent of Level-2 encoder choice.
- **F06 (GRPO training)** — `level: 3`. KEEP. Depends on F05c-L3 AND
  on the chosen Level-2 encoder passing its gates.
- **F07 (Evaluation & analysis)** — `level: "cross-cutting"`. KEEP.
  Must evaluate at Level 2 (probes, σ_s) AND Level 3 (passer-spread,
  Goodhart).

New entries to ADD:

- **F08 — Level-2 event-encoder design spec** — `level: 2`. The
  `docs/event-encoder-design.md` document. `passes: true` once committed.
- **F09 — Level-2 event-encoder training** — `level: 2`.
  `src/training/event_encoder.py` + `scripts/train_event_encoder.py`,
  consuming F05 dataset. `passes: false`.
- **F10 — Level-2 encoder gates** — `level: 2`. σ_s ∈ [0.15, 0.45],
  probe_outcome ≥ 0.65, probe_action ≥ 0.45, qualitative clustering.
  Blocks F11 and any Level-3 run that consumes this encoder.
  `passes: false`.
- **F11 — Level-2 → Level-3 integration** — `level: 3`. Plug trained
  encoder into `LearnedStateEmbedder`, route via
  `RECALLIndex(state_embedder=…)`, plumb `train_grpo.py --level2-encoder`.
  `passes: false`.

Suggested JSON shape:
```json
{ "id": "F04", "level": 1, "name": "Level 1 — SFT visual grounding",
  "description": "...", "passes": true }
```

## 4. methodology.md changes

The 5 axes all survive. Re-label each with its level. Current axis
anchors in `docs/methodology.md`:

- **Axis 1: σ_s** (lines 22–71) → `Level 2`. Property of the state
  encoder; does NOT apply to judge (not a vector encoder) or BT head
  (text-scoring).
- **Axis 2: Probe accuracy** (lines 73–119) → `Level 2`. Probes
  operate on encoder output φ(s). Same caveat.
- **Axis 3: Pseudo-gold AUC** (lines 121–169) → `Level 3`. Doc on
  lines 137–139 already lists "RECALL+mask, judge, BT-head" as
  scorers. Confirms Level 3.
- **Axis 4: In-training passer-spread** (lines 171–212) → `Level 3`.
  The scorer in action during a real training loop.
- **Axis 5: Encoder disagreement** (lines 214–244) → `Level 2`.
  Encoder × encoder comparison; line 219 explicit.

Decision-protocol re-sequencing (currently lines 308–346, Step 0 →
Step 4 by cost). Re-order so LEVEL is the primary key, cost the
secondary key within a level. Pre-existing Step 0 (pseudo-gold AUC)
was first by cost; under the new framing it's a Level-3 test and
cannot be the FIRST gate for a RECALL-family scorer because RECALL
needs a Level-2 encoder picked first. The new sequence makes this
dependency explicit (see §8).

Add at top of `docs/methodology.md`: "Gates are evaluated bottom-up.
Level-2 gates qualify an encoder; Level-3 gates qualify a scorer; a
Level-3 scorer that depends on a Level-2 encoder INHERITS both
sets." Do NOT delete existing 5-axis text — additive labeling only.

## 5. reward-candidates.md changes

Current doc (`docs/reward-candidates.md`) presents RECALL+mask, Judge,
and BT-head as apples-to-apples. Under the new framing they are NOT
comparable that way:

- **RECALL+mask:** Level-3 kNN scorer that DEPENDS ON a Level-2
  encoder choice. 19-dim `tactical_embedding` is the de facto Level-2
  input; MiniLM-on-JSON is an alternative. The Level-2 swap is the
  dominant lever; scorer machinery is invariant.
- **Judge:** Level-3 LM scorer that BYPASSES Level 2 — no state vector
  input. The LM does implicit Level-2 work in its context window.
- **BT-head:** Level-3 learned scorer that MAY OR MAY NOT consume
  Level-2 embeddings. Currently uses `learned_v3_alive` (a Level-2
  encoder) + MiniLM response encoder. Should be re-trained on the
  trained event encoder once F11 lands.

Proposed restructure of `docs/reward-candidates.md`:

- **§ Where we are** (lines 14–28): unchanged.
- **§ The three candidates (table)** (lines 30–37): KEEP table, ADD a
  new column `Level-2 dependency` per row
  (`tactical_19d / MiniLM-JSON / none / learned_v3_alive / event_encoder`).
- **§ (a) RECALL with same-round mask + encoder choice** (lines 39–115):
  rewrite as "RECALL family: Level-3 scorer parameterized by Level-2
  encoder choice." Line 47's "encoder choice is now a separate,
  unresolved knob" becomes the whole framing of the section.
- **§ (b) Claude judge** (lines 116–179): rewrite as "Level-3 scorer
  with no Level-2 input." Center the Goodhart concern.
- **§ (c) BT-head** (lines 181–246): rewrite as "Level-3 scorer with
  configurable Level-2 input." Flag that head signal quality is
  upper-bounded by Level-2 encoder quality.
- **§ Decision protocol** (lines 251–278): reference the new L2 → L3
  ordering from §8 below.
- **§ Combination strategies / § Goodhart concerns** (lines 283–340):
  unchanged.
- **§ Open infrastructure gaps** (lines 345–366): ADD Level-2 entries —
  `src/training/event_encoder.py`, `scripts/train_event_encoder.py`,
  `scripts/eval_event_encoder.py`, `data/eval/event_encoder_probe.jsonl`.

## 6. paper/main.tex changes

Section-by-section, with paper anchors:

- **Title** (line 12): subtitle changes "Two-Phase VLM Training" →
  "A Three-Level Hierarchy for VLM Game Understanding."
- **Abstract** (lines 24–26): "two-phase training paradigm" → "three-level
  perception/reasoning hierarchy." The RECALL failure-mode content
  (52.6% saturation, σ_s 0.000 → 0.331) survives as a contribution
  but the FRAMING shifts to "this failure motivated isolating L2
  from L3."
- **§ Introduction** (lines 28–37): line 34 — "perception and reasoning
  benefit from separate training phases" → "perception, situational
  abstraction, and reasoning benefit from separate training levels."
  Contribution list (line 37) gets a new bullet: "(5) a Level-2 event
  encoder mapping a window of states to a tactical event embedding
  (TBD pending training)."
- **§ Method** (lines 54–58): TODO at line 57 — "two-stage training
  pipeline" → "three-level architecture."
- **§ Phase 1: Visual Grounding via SFT** (lines 67–74): rename to
  `§ Level 1: Visual Grounding (See)`. Body unchanged.
- **§ Phase 2: Strategic Reasoning via GRPO** (lines 76–88): SPLIT into:
  - `§ Level 2: Tactical Situation Encoding (Situate)` — NEW. TBD
    placeholder. State motivation (decouple representation from
    scoring), target gates (σ_s, probe accuracy, clustering), and
    that empirical content is deferred.
  - `§ Level 3: Strategic Reasoning (Think)` — existing Phase 2 text
    with "Phase 2" → "Level 3."
- **§ Reward Design** (lines 90–109): TODO block must clarify
  R_strategy is a Level-3 family (RECALL / judge / BT-head), each
  with its own L2 dependency.
- **§ RECALL** (lines 111–122): rename "Level-3 kNN scorer." Make
  explicit that τ (state embedding) is the Level-2 input and the
  choice of τ is separate from RECALL mechanics.
- **§ Diagnosis** (lines 123–196): reframe — "Level-2 representation
  collapse causes Level-3 scorer collapse." F1 (line 193) and F2
  (line 196) are Level-2 failures propagating into Level 3.
- **§ Fix** (lines 198–207): Layer 0 (line 202) is a Level-3 patch;
  Layer 1 (line 205) is a Level-2 training attempt. Make explicit.
- **Architecture figure** (lines 209–214): redraw the `[Placeholder]`
  as three levels, not two phases.
- **§ Experiments / Results / Analysis / Conclusion:** global
  rename "Phase 1 / Phase 2" → "Level 1 / Level 3." Add a Level-2
  placeholder in `§ Results` and `tab:results` (lines 309–323).

## 7. Scripts and code changes

Inventory with level assignment.

**Level 1 (SFT):**
- `scripts/train_sft.py`, `scripts/run_sft_pod.sh`,
  `scripts/build_sft_dataset.py`, `scripts/generate_sft_labels.py`,
  `src/training/sft_trainer.py` — KEEP all as-is.

**Level 2 (NEW; target paths from `docs/event-encoder-design.md`):**
- `src/training/event_encoder.py` — NEW. Model class.
- `scripts/train_event_encoder.py` — NEW. Training entrypoint.
- `scripts/eval_event_encoder.py` — NEW. Runs L2 gates (σ_s, probes,
  clustering).
- `scripts/probe_features.py` — NEW. Tiny MLP probes for axis 2.
  Already listed in `docs/methodology.md` as a coverage gap (line 103).
- `data/training/event_encoder/` — NEW. Pre-built window dataset.

**Level 3 (GRPO + scorers):**
- `scripts/train_grpo.py` — KEEP. ADD `--level2-encoder PATH` flag
  that routes through `LearnedStateEmbedder` instead of the default
  `tactical_embedding`.
- `scripts/run_grpo_pod.sh`, `scripts/run_grpo_smoke.sh`,
  `scripts/run_grpo_with_auto_stop.sh`, `scripts/auto_stop_pod.sh`,
  `scripts/wait_and_stop_pod.sh`, `scripts/extract_grpo_samples.py` —
  KEEP all unchanged.
- `src/training/grpo_trainer.py` — KEEP. Will need to assume a Level-2
  encoder is available when `--level2-encoder` is set; current code
  silently falls back to `tactical_embedding`.
- `src/training/recall.py` — KEEP. **FLAG:** `tactical_embedding` at
  line 105 becomes the "Level-2 fallback baseline." Update its
  docstring. `RECALLIndex.__init__` at line 235 already accepts a
  custom `state_embedder` — THAT is the Level-2 plug-in point; no
  code change, just doc clarification.
- `src/training/learned_state_embedder.py` — KEEP. This file IS the
  Level-2 adapter. Once F09 ships, point its `encoder_path` at the
  trained encoder. NOTE: current contract is "JSON → text →
  SentenceTransformer encode." The event encoder takes a window of
  states, so this file's API may need to generalize (real change,
  not just a weights swap).
- `src/training/judge_reward.py` — KEEP. Pure Level-3, no L2 dep.
- `src/training/bt_reward.py` — KEEP. Level-3, depends on whichever
  state encoder is loaded.
- `src/training/rewards.py` — KEEP. Mixed (perceptual accuracy = L1;
  format gate = cross-cutting).
- `src/training/data_utils.py` — KEEP. Cross-cutting.

**Cross-cutting / diagnostics:**
- `scripts/recall_diagnostic.py` — L2/L3 boundary.
- `scripts/recall_variance_diagnostic.py` — L2 (Axis 1: σ_s).
- `scripts/validate_recall_masking.py` — L2 + L3 (validates F1 fix).
- `scripts/passer_spread_audit.py` — L3 (Axis 4).
- `scripts/eval_scorer.py` — L3 (Axis 3, pseudo-gold AUC).
- `scripts/build_pseudo_gold.py`, `scripts/draft_pseudo_gold_abc.py`,
  `scripts/label_d_advices.py`, `scripts/label_d_app.py` — L3 eval-set
  construction.
- `scripts/label_app.py`, `scripts/build_label_candidates.py`,
  `scripts/train_bt_head.py` — L3 BT-head pipeline.
- `scripts/data.py`, `scripts/pod_setup_grpo.sh`, `init.sh` —
  cross-cutting.

Concrete annotations to apply (no edits, just flag):
- `src/training/recall.py:105` (`def tactical_embedding`): add note
  "Level-2 fallback baseline."
- `src/training/recall.py:235` (`RECALLIndex.__init__`): note "Level-2
  plug-in point."
- `src/training/learned_state_embedder.py` top docstring: note
  "designated Level-2 → Level-3 bridge."

## 8. Methodology decision protocol changes

Proposed new gate sequence (replaces `docs/methodology.md` lines 308–346
ordering):

**Level 1: shipped, no gate needed.** F04's 67.3% per-field accuracy
is sufficient on its own.

**Level 2 gates** (must all pass BEFORE a Level-3 run that consumes
this encoder; cost ~$0 + ~10 min CPU each):

- **L2-G1: σ_s ∈ [0.15, 0.45]** on a 200-query sample with same-round
  mask. Run via `scripts/recall_variance_diagnostic.py`. Below 0.15:
  F1/F2 collapse. Above 0.45: encoder is random.
- **L2-G2: probe_outcome val accuracy ≥ 0.65** on (demo, round)-disjoint
  split. Run via `scripts/probe_features.py` (not yet built).
- **L2-G3: clustering qualitatively meaningful.** 2-D projection
  (UMAP/t-SNE/PCA) colored by pro-action category vs. round_won
  should show category structure, not outcome collapse. Subjective
  but required.

**Level 3 gates** (must all pass BEFORE a 100-step GRPO run; Level-2
must have already passed for any Level-3 scorer that depends on
that encoder):

- **L3-G1: pseudo-gold pairwise AUC ≥ 0.70** on the 30×4 hand-authored
  set. Run via `scripts/eval_scorer.py`. (Or revised version using
  BT-pair preferences once labels exist.)
- **L3-G2: in-training passer-spread, k=2 median ≥ 0.025 AND
  zero-spread fraction at k=2 < 15%**, audited by step 20. Run
  `scripts/passer_spread_audit.py` on `useful_jumps.jsonl`.
- **L3-G3: cross-rubric Goodhart guard ≥ 0.05.** Judge-family
  scorers only. Cross-rubric score must exceed base+SFT by ≥ 0.05.

Level-dependency rule (the new rule):

> A Level-3 scorer that consumes a Level-2 encoder INHERITS that
> encoder's Level-2 gates. RECALL+τ_A and RECALL+τ_B are different
> Level-3 candidates with respect to gating; each must pass L2-G1
> through L2-G3 for ITS τ before L3 gates are meaningful. Judge
> inherits no L2 gates (no L2 dependency). BT-head inherits whichever
> L2 encoder it was trained on.

## 9. What stays the same

Level-agnostic; no update needed:

- Pod setup / torch+CUDA pinning (`scripts/pod_setup_grpo.sh`,
  `init.sh`, the causal-conv1d / Mamba match gotcha).
- The manual GRPO trainer loop in `src/training/grpo_trainer.py`
  (sees only rewards + completions; agnostic to where rewards come
  from).
- `data/training/grpo/smoke_test.jsonl` and the smoke-test pattern.
- Auto-stop pod scripts (`auto_stop_pod.sh`, `wait_and_stop_pod.sh`).
- Format gate (multiplicative, JSON-validity), α ≈ 0.20 / 0.80
  weighting, KL coef λ = 0.02. Level-3 hyperparameters but reframing
  doesn't touch them.
- HF Hub data sync (`scripts/data.py`).
- `decisions.md` — historical, append-only.
- F01 (manifest), F02 (demo pipeline), F03 (screenshot–demo sync).
- `claude-progress.txt` — pure history; only forward-going entries
  use the new vocabulary.

## 10. Migration order

a. **Commit the three new docs (THIS commit).** No code changes.
   Files: `docs/three-level-architecture.md` (overview),
   `docs/event-encoder-design.md` (Level-2 spec),
   `docs/alignment-delta.md` (this inventory). After this commit
   the project has a written reference later commits can cite.

b. **Update `feature-list.json` (next session).** Apply §3: add
   `level` field to F01–F07, split F05c, add F08/F09/F10/F11.
   Tracker-only; no code changes. Verify with
   `python -c "import json; json.load(open('feature-list.json'))"`.

c. **Update `README.md`.** Apply §2: new section headers, new
   Quickstart, pipeline list regrouped by level. Defer any prose
   claiming Level 2 is shipped (F10 must clear first).

d. **Update `docs/methodology.md` and `docs/reward-candidates.md`.**
   Apply §4 and §5. Add the level-dependency rule (§8) to
   methodology.md. Additive only — do NOT delete the existing 5-axis
   content.

e. **Build the event encoder (real work, multi-day).** Implements
   F09 (training) and the L2 gates F10. Files:
   `src/training/event_encoder.py`, `scripts/train_event_encoder.py`,
   `scripts/eval_event_encoder.py`, `scripts/probe_features.py`.
   Output: trained encoder checkpoint + JSON gate report.

f. **Integrate trained encoder into `recall.py` + judge_eval +
   grpo_trainer.** Implements F11. Plug-in point already exists
   (`RECALLIndex(state_embedder=...)`); work is plumbing
   `--level2-encoder` through `train_grpo.py` and generalizing
   `LearnedStateEmbedder` from "JSON → SentenceTransformer" to
   "window-of-states → event encoder." BT-head re-training on the
   new encoder is gated on F11.

g. **Update `paper/main.tex`.** Apply §6: rename Phase 1 / Phase 2
   to Level 1 / Level 3, add Level 2 section with empirical results
   from step (e), redraw the architecture figure as three levels.
   Last because it needs (e)'s real numbers.

Steps (a)–(d) are reversible and cheap. Step (e) is the real
investment. Steps (f) and (g) are mechanical once (e) lands.
