# Measurement Methodology — Evaluating GRPO Strategy Rewards & State Abstractions

## Why this document exists

In F08v4 + F08v4_resumed (100 GRPO steps, ~6 hours of pod time, ~$25 spent) we
trained the model on a strategy reward whose median within-group spread among
format-passing completions was **0.000**. RECALL was producing identical scores
for genuinely different strategic advice in the same state, ~50% of the time.
The model could not learn strategy because there was no strategy gradient — it
only learned format discipline. See `claude-progress.txt` 2026-04-23 entry for
the full diagnosis.

The lesson: **never invest pod time in a reward function or state abstraction
that has not passed an offline scientific test of its discriminative power.**
This document names the tests. It is the canonical reference for how chimera
evaluates feature engineering quality before any training run.

The five axes below are not a shopping list — they are a gating sequence.
Cheaper tests run first; failure on an early test disqualifies the candidate
without spending later test budget.

## The five measurement axes

### 1. σ_s — within-neighborhood outcome variance

**Plain English.** For each state in our dataset, ask "if I find the k most
similar states under this representation, do their round outcomes vary, or do
they all share the same outcome?" If they all share the same outcome, the
representation has collapsed onto the outcome axis (or is trivially leaking
sibling ticks of the same round) — neighbors carry no counterfactual signal
because there is no counterfactual to retrieve.

**Operational.** For each query state s_q, retrieve top-k neighbors under the
candidate state encoder. Compute the std of `round_won` across those k
neighbors. Aggregate over a held-out sample of 200+ queries; report the
median, p25, p75, and the fraction of queries with σ_s < 0.05. With binary
outcomes at base rate p ≈ 0.6, the maximum achievable std is √(p(1−p)) ≈ 0.49.

**What it catches.** F1 sibling-trajectory leakage (neighbors are the same
round; round_won is shared by construction; σ_s saturates at 0). F2
outcome-correlated supervision collapse (a learned encoder trained with
outcome-correlated positives maps everything onto a 1-D outcome axis;
neighbors share outcome trivially; σ_s saturates at 0). See `paper/main.tex`
F1/F2 diagnosis sections and `graf2021dissecting` (SupCon collapse) for the
formal grounding.

**What it cannot catch.** Action-side noise — σ_s scores the state encoder
alone. RECALL's full failure required both a leaky state and a noisy action
encoder; σ_s only diagnoses the state half. A state encoder with σ_s ≈ 0.35
can still produce a useless reward if the action encoder destroys the signal
downstream.

**Implementation.** `scripts/recall_variance_diagnostic.py` (per-query σ_s
across embeddings × same-round-mask conditions). `scripts/recall_variance_sweep.py`
(σ_s across k ∈ {8, 16, 32, 64, 128} × cosine threshold ∈ {0.05, 0.10, 0.20}).
`scripts/validate_recall_masking.py` (CPU-only pre-flight that exits non-zero
if same-round mask fails to lift σ_s above a threshold).

**Pass thresholds (chimera, 4 demos, ~2K samples, p ≈ 0.6).**

| σ_s median | Interpretation                            |
|-----------:|-------------------------------------------|
| < 0.10     | F1 or F2 collapse — do not run            |
| 0.10–0.15  | Marginal — sweep k/threshold first        |
| **0.15–0.45** | **Goldilocks — proceed**               |
| > 0.45     | Random / encoder useless — neighbors are arbitrary |

The 1b387b4 same-round mask lifted production median σ_s from 0.000 → 0.331 —
into the Goldilocks band — without changing the encoder. That number was the
first quantitative evidence that the F1 fix was a real fix.

### 2. Probe accuracy battery

**Plain English.** If a representation is decision-relevant, a *small* model
trained on top of it should be able to predict the things a player needs to
predict — outcome, what the pro will do next, what the next state will look
like. Low probe accuracy means the information is missing from the
representation, not just hard to extract.

**Operational.** For each candidate state encoder φ(s), train three tiny
probes (1–2 layer MLP, ~10K params each) on a (demo, round)-disjoint train/val
split:

  - `probe_outcome`: predict round_won from φ(s). Binary classification.
  - `probe_action`: predict the pro's action category from φ(s). 5–10-class.
  - `probe_next`: predict φ(s_{t+1}) from φ(s_t) (forward dynamics). Regression.

Report val accuracy / regression-R² per probe. Held-out split must be by
(demo, round) to avoid sibling-tick leakage — the same failure mode that broke
RECALL will silently inflate within-round probe accuracy.

**What it catches.** A representation that has discarded decision-relevant
information. If `probe_outcome` is at chance, the representation doesn't
encode anything predictive of winning; if `probe_action` is at chance, it
doesn't encode what the pro found relevant.

**What it cannot catch.** Invariance to nuisance variables. A probe that
succeeds with high accuracy could still be using a nuisance feature (e.g.
exact tick number) as a proxy. A counterfactual probe — train on (s, s')
pairs that differ only in a nuisance dimension — is the next layer of rigor;
not yet built for chimera.

**Implementation.** Not built. Should live at `scripts/probe_features.py`.
Inputs: candidate encoder identifier (env var or path), training data
JSONL. Outputs: per-probe accuracy + drop-one-dim ablation table.

**Pass thresholds.**

| Probe                | Pass            | Floor (chance)        |
|----------------------|-----------------|-----------------------|
| `probe_outcome`      | val acc ≥ 0.65  | 0.50 (binary, balanced) |
| `probe_action`       | val acc ≥ 0.45  | ~0.20 (5-class)       |
| `probe_next` (R²)    | ≥ 0.50          | 0.0                   |

Any candidate failing two or more probe thresholds is disqualified before
σ_s is computed — probes are cheaper and more discriminating at the encoder
level.

### 3. Pseudo-gold AUC

**Plain English.** Hand-construct cases where you already know the right
ranking, then check whether the candidate scorer reproduces that ranking.
This is the only test that directly measures what we actually care about
(does the scorer rank advice correctly) on data we built ourselves.

**Operational.** Pick 30 diverse states stratified across (map × side ×
round_phase). For each state, hand-author four advices labeled by
construction:

  - **A_correct**: parrots `pro_action` with concrete reasoning. Should rank highest.
  - **B_anti_pro**: opposite of `pro_action`. Should rank lowest (when pro won).
  - **C_generic**: vague platitudes ("play smart", "communicate"). Should rank medium-low.
  - **D_plausible_wrong**: confident, polished, uses pro CS2 vocabulary, but
    tactically wrong for *this* state. Should rank low. The diagnostic case.

Score each candidate scorer (RECALL+mask, judge, BT-head) on all 120
(state, advice) pairs. Compute pairwise AUC against the construction labels:
fraction of (A, X) pairs where the scorer correctly ranks A above X, summed
over X ∈ {B, C, D}.

**What it catches.** Any scorer that cannot distinguish constructed-correct
from constructed-wrong advice on cases we built. The most discriminating
single number we have for picking among RECALL+mask, judge, and BT-head
without spending a pod hour. Specifically catches the failure mode where
"polish" beats "tactics" — a scorer that ranks D above B has learned to
reward confident phrasing rather than strategic content.

**What it cannot catch.** Generalization to advice the human author did not
think to construct. Adversarial advice generated by the policy itself
(once-trained) will probe the scorer's blind spots in ways the static set
will not. Cross-rubric (axis 5) and held-out training-sample audit (axis 4)
mitigate this.

**Implementation.** Stub generator: `scripts/build_pseudo_gold.py` (built
this session — produces the 30 stub records for the human author).
Eval harness: `scripts/eval_scorer.py` (not yet built — should consume the
authored file and run all candidate scorers on it).

**Pass thresholds.**

| Pairwise AUC vs construction labels | Verdict          |
|------------------------------------:|------------------|
| < 0.60                              | At/near chance — disqualified |
| 0.60–0.70                           | Marginal — sweep before pod   |
| **≥ 0.70**                          | **Proceed to pod gating**    |
| ≥ 0.85                              | Strong — preferred candidate  |

Below 0.70, no GRPO run. The cost of authoring the 30×4 set is ~3 hours of
human time, one-time, amortized across every future scorer.

### 4. Within-group passer-spread (in-training)

**Plain English.** During an actual training run, look at the G=4 completions
generated for the same state. Among the ones that passed the format gate,
how different are their reward scores? If the scorer gives near-identical
rewards to genuinely different format-passing advices in the same state,
the policy has nothing to learn — GRPO's `advantage = (r − mean)/std` zeroes
out.

**Operational.** Read `useful_jumps.jsonl` produced by
`src/training/grpo_trainer.py`. For each step, identify the format-passers
(reward > ε proxy, since the format gate is multiplicative), compute std
among their rewards, group by passer count k ∈ {1..G}. Report median, p25,
p75, and zero-spread fraction per k. The k=2 row is the diagnostic case —
it is the most common bucket and where RECALL specifically failed.

**What it catches.** Any reward function that is degenerate among same-state
completions. This is the only test that uses the *actual* policy outputs the
scorer will be ranking during training — the scorer is being asked to do its
real job, with realistic input distribution.

**What it cannot catch.** Same-quality completions that genuinely deserve
the same score (the test reads degenerate-quality the same way). In practice
this is rare — identical scores from sampling-diverse completions almost
always indicate scorer collapse, not legitimate ties.

**Implementation.** `scripts/passer_spread_audit.py` (built this session).
Reads any `useful_jumps.jsonl`. Flags PASS / WEAK / DEAD against the F08v4
RECALL baseline.

**Pass thresholds (k=2 passers, the diagnostic case).**

| k=2 median spread | Zero-spread fraction | Verdict                  |
|------------------:|---------------------:|--------------------------|
| ≥ 0.025           | < 15%                | ALIVE — keep running     |
| 0.015–0.025       | 15–35%               | WEAK — extend audit window |
| **< 0.015**       | **≥ 35%**            | **DEAD — kill the run, RECALL floor** |

This audit must run by step 20 of any new GRPO run. Killing a run at step
20 if the diagnostic fails costs ~30 pod minutes; missing it costs the
F08v4 outcome (100 wasted steps).

### 5. Encoder disagreement

**Plain English.** Take two reasonable but independent state encoders. For
random pairs of states, do they agree on which pairs are similar? If two
defensible encoders disagree wildly on similarity rankings, at least one is
missing the signal — and you can localize *which* pairs they disagree on,
which often points at the missing dimension.

**Operational.** Sample 1000 (s_i, s_j) random state pairs. Compute cosine
similarity under encoder φ_A and φ_B. Spearman-correlate the two similarity
rankings across the 1000 pairs. Report the coefficient and the top-10 pairs
where the two encoders most disagree (highest |rank_A − rank_B|).

**What it catches.** Hidden representational disagreement. If the
hand-engineered 19-dim encoder agrees with untrained MiniLM at ρ = 0.7+, both
are capturing roughly the same structure (and either can be used). If they
agree at ρ < 0.4, at least one is wrong, and the disagreement set is the
diagnostic input.

**What it cannot catch.** Agreement-on-the-wrong-thing. Two encoders trained
with the same outcome-correlated positives will both collapse onto outcome
and agree perfectly with each other (and both fail σ_s). Disagreement is
informative in one direction; agreement is not.

**Implementation.** Not built. Should live at `scripts/encoder_disagreement.py`.
Inputs: two encoder identifiers, sample size. Output: Spearman ρ and a
disagreement-pair JSONL for human inspection.

**Pass thresholds.** Less binary than the others — disagreement is a
diagnostic, not a gate. Use it when σ_s passes for two candidates and you
need to pick one.

## Formal literature anchors

References already in `paper/references.bib` are cited by key. Others are
marked **[verify]** — the citation is widely-known but the bib entry is not
yet in the repo and should be added before paper submission.

- **Bisimulation metrics** — Ferns, Panangaden, Precup 2004 **[verify]**.
  Defines a quasi-metric on MDP states such that the distance between two
  states upper-bounds the difference in their optimal value functions. Anchors
  axis 1 (σ_s): a state encoder whose induced distance approximates
  bisimulation distance will retrieve neighbors with similar value, so σ_s in
  the Goldilocks band reflects useful state structure rather than coincidence.

- **MDP homomorphism** — Ravindran & Barto 2004 **[verify]**. Formalizes
  state abstraction as a value-preserving map from a fine MDP to a coarse
  one. Anchors the philosophical framing in this doc: "tactical equivalence"
  is a homomorphism property, not a geometric one.

- **DeepMDP** — Gelada et al 2019 **[verify]**. Shows that an encoder
  trained to make ‖φ(s) − φ(s')‖ approximate |V(s) − V(s')| produces good
  behavioral representations. Justifies axis 2's `probe_outcome`: if the
  encoder is value-aligned, a small probe should recover round_won.

- **Successor representations** — Dayan 1993 (original); Barreto et al 2017
  (successor features) **[verify]**. Encodes a state by its expected future
  visitation distribution. Conceptually the closest "right answer" for
  tactical equivalence in CS2 — two states are equivalent if they lead to
  similar futures — but requires rollout-style data we cannot produce
  without a simulator (chimera has no executable game).

- **Information bottleneck** — Tishby & Zaslavsky 2015 **[verify]**.
  Optimal representation Z minimizes I(Z; X | Y) (compresses input) while
  maximizing I(Z; Y) (preserves task signal). Axis 2's probe battery is the
  practical estimator for I(Z; Y); the dimensions-dropped ablation table
  measures the X-side compression.

- **SupCon collapse** — `graf2021dissecting`. Supervised contrastive losses
  with class-correlated positives collapse the embedding onto the class
  axis. Directly anchors chimera's F2 finding (commit 1b387b4): triplet
  loss with outcome-correlated positives collapses the state encoder onto
  the outcome axis.

- **DIAMOND** — `alonso2024diamond`. Closest precedent for ML on CS:GO
  pro-demo data. Solves a different problem (pixel-level diffusion world
  model), but the data constraint (no simulator for counterfactuals) is
  the same as ours and validates the choice to use historical-data-only
  diagnostics rather than rollout-based ones.

- **Counterfactual data augmentation** — `pitis2020counterfactual`. The
  general framework for distinguishing causal from correlational signal in
  RL representations without a simulator. Future axis-6 candidate.

- **Bisimulation in deep RL** — `castro2020scalable`, `zhang2021learning`,
  `agarwal2021contrastive`. Modern operationalizations of bisimulation
  metrics for deep representations; cited as the methodological lineage
  for axis 1.

- **Zero-variance GRPO** — `zerovariance2025`. The exact failure mode
  chimera hit at F08v4: when the reward function gives identical scores
  to all G generations in a group, GRPO's advantage normalization zeros
  out. Anchors axis 4 (passer-spread) as the live diagnostic for this
  failure.

## The decision protocol

Pre-flight gating for any GRPO run. Order is OFFLINE FIRST (cheapest,
highest-signal). A candidate must pass every gate to advance to the next.

**Step 0 — Pseudo-gold AUC (offline, ~30 min, $0.50 if judge).**
Score the candidate against the 30×4 hand-authored set. Required:
**pairwise AUC ≥ 0.70**. Below this, the scorer cannot distinguish
constructed-correct from constructed-wrong on cases we built ourselves;
do not run.

**Step 1 — σ_s sanity (RECALL family only, offline, ~5 min, $0).**
Run `scripts/recall_variance_diagnostic.py` on the candidate state encoder
+ same-round-mask config. Required: **σ_s median ∈ [0.15, 0.45]** on a
200-query sample. Below 0.15: F1/F2 collapse. Above 0.45: encoder is
random.

**Step 2 — Probe accuracy (RECALL family only, offline, ~10 min, $0).**
Run `scripts/probe_features.py` (not yet built). Required:
**probe_outcome val accuracy ≥ 0.65** on (demo, round)-disjoint split.

**Step 3 — In-training passer-spread (any candidate, ~30 pod min, ~$0.30
if judge).** Launch a 20-step smoke run. By step 20, run
`scripts/passer_spread_audit.py` on `useful_jumps.jsonl`. Required:
**k=2 median spread ≥ 0.025** AND **zero-spread fraction at k=2 < 15%**.
Below this: kill the run; the reward is at or below RECALL's noise floor.

**Step 4 — Goodhart cross-rubric (post-training, judge runs only,
~15 min, ~$0.75).** Score `f08vN/final_model` outputs with a *different*
judge rubric than the training rubric (different scoring criteria, same
model). Required: **cross-rubric mean strategy score must exceed
base+SFT cross-rubric score by ≥ 0.05**. If the gap is smaller, the
model has learned the training-judge's idiosyncratic preferences rather
than improving tactically; revert.

The protocol exists to make pod time and human-labeling time conditional
on offline evidence. Steps 0–2 cost ~$1 and 30 minutes; steps 3–4 cost
~$1 and ~45 pod minutes. The full gate sequence costs <2% of a single
F08v4-style training run and prevents 100% of an F08v4-style waste.

## Why each technique is specifically vulnerable

**RECALL + same-round mask + good encoder.** Axis 1 (σ_s) is the live
diagnostic — already lifted 0.000 → 0.331 offline by the F1 fix; never
tested in actual training. Axis 3 (pseudo-gold AUC) is the unmade test
that would expose the F3 action-side noise (5-dim keyword extraction)
even after F1/F2 are fixed.

**Claude judge.** Axes 1–2 do not apply (the judge is not a vector
encoder). Axes 3 (pseudo-gold AUC) and 5 in the protocol (cross-rubric
Goodhart guard) are the two real tests. Goodhart risk is the central
concern: the judge is another LM's prior; the policy can learn to please
its phrasing without being tactically better. Cross-rubric eval is
mandatory before any 100-step run is declared a win.

**BT-head.** Axis 2 (probe accuracy) extends naturally — train probes
from (state, advice) pairs to predict the human label, report held-out
accuracy. Axis 3 (pseudo-gold AUC) catches generalization failure beyond
label-app idiosyncrasies. The label-volume risk is independent: below
~300 labeled pairs, the BT head's own variance dominates the signal it
is supposed to provide; the doc-level minimum is 500.

## Coverage gap to fix next

Scripts that should exist but don't:

- **`scripts/eval_scorer.py`** — unified harness: feed a scorer fn, runs
  σ_s + within-group spread + pseudo-gold AUC, writes one JSON per
  scorer for direct comparison. Single file enables every future
  candidate to be evaluated identically.

- **`scripts/probe_features.py`** — tiny MLP probes for (round_won,
  pro_action, next-tick) from candidate state encoders, with
  drop-one-dim ablation. Implements axis 2 end-to-end.

- **`scripts/encoder_disagreement.py`** — Spearman correlation of
  similarity rankings across two encoders, plus disagreement-pair dump.
  Implements axis 5.

- **`scripts/build_holdout_eval.py`** — carve a (demo, round)-disjoint
  100-sample evaluation set from `data/training/grpo/smoke_test.jsonl`
  for cross-rubric and post-training audits. The current 50-sample eval
  reuses training states.

- **`scripts/judge_eval_crossrubric.py`** — Goodhart guard. Score a
  trained model's outputs with a rubric different from the training
  rubric; report Δ vs base+SFT.

- **`data/eval/pseudo_gold_advices.jsonl`** — the actual hand-authored
  data. The stub generator (`scripts/build_pseudo_gold.py`) exists; the
  authoring is human time.

After this list closes, every reward candidate gets the same 5-axis
scoreboard, and no GRPO run starts without a pre-flight scoring
artifact in the run's output directory.
