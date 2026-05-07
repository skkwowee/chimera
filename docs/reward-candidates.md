# GRPO Strategy-Reward Candidates

Side-by-side comparison of the three reward signals chimera has in flight for
the GRPO strategic-reasoning phase, plus the gating protocol that decides
which one runs in the next pod session. This doc is the canonical comparison
surface; the experimental gating thresholds themselves live in
`docs/methodology.md` and are referenced (not redefined) here.

This doc does NOT recommend a winner. The candidates are gated on experiments
not yet run; the orchestrator's job is to run them, not to pre-judge them.

---

## Where we are

After F08v4_resumed (100 effective GRPO steps, full schema, RECALL reward,
SFT-merged base) the within-group spread among format-passing completions
collapsed to a median of 0.000 in the most common case (2/4 passing) and
~0.0154 otherwise (claude-progress.txt 2026-04-23). The model learned format
discipline (format_passes 1.86 → 2.75 over the run) but RECALL was not a
strategic-content signal at our data scale and state representation. Three
candidate replacements are now in flight at very different maturity levels:
RECALL with the same-round mask + a better encoder choice (offline-validated
2026-04-30, never actually trained on); a single-shot Claude-judge reward
(smoke-validated 2026-04-23, never run for >5 steps); and a Bradley-Terry
preference head trained on human-labeled advice pairs (infrastructure built,
zero labels collected).

---

## The three candidates (side-by-side)

| Candidate | Family | Cost / 100-step run | Status | Strongest evidence | Biggest risk |
|---|---|---|---|---|---|
| RECALL + same-round mask + encoder choice | Non-parametric kNN over historical pro-play outcomes | $0 (CPU index, no API) | Offline validated; never trained on | Median σ\_s 0.000 → 0.331 on production 19-dim tactical embedding once F1 leakage is masked (commit 1b387b4, 2026-04-30) | σ\_s lift may not survive in-training; encoder choice unsettled (untrained MiniLM > every supervised variant per F2 ablation) |
| Claude judge | Group-aware single API call ranking G completions on a CS2 rubric | ~$1–2 / 100 steps with claude-opus-4-7 | Smoke-validated (5 steps, 0 failures) | Passer-spread median 0.035 vs RECALL 0.0154; zero-spread among passers 5% vs ~50% (2026-04-23) | Goodhart against judge phrasing; another model's prior is not coach truth |
| BT-head | Local MLP trained on human pairwise preferences over advice | $0 at inference (after one-time labeling effort) | Infra built, zero labels collected | Pattern is standard RLHF reward modeling | < ~300 labels: head fits label-app idiosyncrasies, not strategic content |

### (a) RECALL with same-round mask + encoder choice

**How it works.** Per `src/training/recall.py`: encode each historical demo
sample's game_state to a vector, index in FAISS, retrieve top-K=32 neighbors
of the query state, and compute advantage A = Q̂(s, a) − V̂(s) where Q̂ is
the mean round_won of action-matched neighbors and V̂ is the mean over all
K. The 1b387b4 (2026-04-30) change adds a per-sample (demo\_stem, round\_num)
source key plumbed through the FAISS query, so neighbors from the SAME round
are dropped before kNN — this fixes failure mode F1 (same-trajectory
outcome leakage). Encoder choice is now a separate, unresolved knob: the
shipped 19-dim hand-engineered `tactical_embedding`, an off-the-shelf MiniLM
on the game_state JSON, or a learned encoder; the `--positive-rule` ablation
on `scripts/train_state_embedding.py` makes the F2 (outcome-correlated
supervision) dose-response runnable. Reward signature stays unchanged so
the trainer wiring needs no edits.

**Observed evidence.**
- F08v4_resumed end-of-run eval: mean\_strategy −0.031, weighted\_total
  −0.0003 (claude-progress.txt 2026-04-23). Format gate 0.98, perceptual
  accuracy 0.123 — schema and perception both held; the strategy term moved
  in the wrong direction.
- Within-group reward spread on the same run: among completions that passed
  the format gate, the median spread was 0.000 in 2/4-passing steps (the
  modal case), 0.0101 in 3/4-passing, 0.0154 in 4/4-passing
  (claude-progress.txt 2026-04-23). RECALL was not differentiating
  format-valid advices.
- Per-query neighbor outcome std (σ\_s) on the production 19-dim embedding:
  0.000 for 52% of queries before the same-round mask. After 1b387b4: median
  σ\_s lifts to 0.331 on the production code path, validated end-to-end via
  `scripts/validate_recall_masking.py` (commit 1b387b4 message,
  2026-04-30).
- F2 dose-response (commit 1b387b4): saturation rate scales monotonically
  with outcome-correlation strength of the triplet-loss positive label;
  **untrained MiniLM beats every supervised variant.**

**Failure modes that would disqualify it.**
- Median σ\_s on the held-out 30×4 set drops back below ~0.1 once the
  state representation is paired with the action-extractor — would mean the
  signal we're seeing offline is being filtered out by the keyword-counter
  action matcher (`_extract_action_from_text` in `recall.py`).
- 25-step in-training passer-spread median fails the gate from
  `methodology.md` — RECALL repeating the F08v4 collapse despite a non-zero
  offline σ\_s.
- AUC on the pseudo-gold 30×4 set < 0.55 (no better than chance ranking
  of constructed-correct vs constructed-wrong advices).

**Next test.** Pseudo-gold AUC on the 30 hand-authored states × 4 advices
once `pseudo_gold_advices.jsonl` ships (in flight this session). Costs $0
and finishes in seconds; it is the cheapest discriminator across all three
candidates and it discriminates encoder choice for RECALL specifically.
Second cheapest: a 25-step training run with RECALL + mask + the
AUC-winning encoder, watched via the in-training passer-spread audit.

**Implementation status.**
- `src/training/recall.py` — same-round mask plumbed through `query()` and
  `recall_advantage()` via `query_source_key`. Idempotent / backward
  compatible (None → previous behavior).
- `scripts/train_grpo.py` — passes source key through to the index at build
  time; reward path forwards it on every call.
- `scripts/recall_variance_diagnostic.py` + `scripts/recall_variance_sweep.py`
  — CPU-only diagnostics over k ∈ {8,16,32,64,128}, thresholds {0.05, 0.10,
  0.20}, same-round vs whole-demo holdout.
- `scripts/validate_recall_masking.py` — end-to-end offline check.
- `scripts/train_state_embedding.py --positive-rule` — F2 ablation flag.
- Open: which encoder to ship by default. The 19-dim hand-engineered one is
  the production default; MiniLM and learned variants exist but are not
  wired as the trainer's default.

**Open questions.**
- Does the offline σ\_s lift translate to in-training passer-spread above
  the F08v4 baseline? Untested.
- Is the action-extraction step (5-dim keyword-count vector) still the
  bottleneck even after F1+F2 are addressed, or does fixing the state side
  unlock the existing matcher?
- Does the encoder-choice ranking hold under the 1b387b4 same-round mask?
  The F2 ablation predates the mask; numbers may shift.

### (b) Claude judge

**How it works.** Per `src/training/judge_reward.py`: each GRPO step makes a
single Anthropic API call (`claude-opus-4-7` by default, overridable via
`CHIMERA_JUDGE_MODEL`) that sees the ground-truth game\_state, the pro's
actual action, the round\_won outcome, and ALL G sibling completions at
once. The judge returns a JSON list of G floats in [0, 1] under a fixed
rubric anchoring on action-match-with-pro and outcome. The trainer's
per-completion reward loop calls `judge_reward` G times; an `lru_cache`
keyed on (state\_json, gt\_json, completions\_tuple, model) collapses those
to a single API call. On any failure (network, JSON parse, malformed score
list) the function returns neutral 0.5 for every completion — that step
contributes no gradient signal but training does not crash.

**Observed evidence.**
- 5-step smoke run, 20 useful\_jumps samples, 0 errors, 0 judge API failures
  (claude-progress.txt 2026-04-23).
- Passer-spread median: judge 0.035 vs RECALL f08v4 0.0154 (~2.3×).
  Max passer-spread: judge 0.175 vs RECALL ~0.10. Zero-spread among
  passers: judge 1/20 (5%) vs RECALL ~50% on 2/4-passing steps
  (claude-progress.txt 2026-04-23).
- Final 50-sample eval after the 5-step smoke: mean\_strategy 0.4800 vs
  RECALL f08v4 −0.031; weighted\_total 0.4037 (claude-progress.txt
  2026-04-23). Five steps is too few to attribute to learning, but the
  numbers confirm scores land non-degenerately in the middle of [0, 1] out
  of the gate, where RECALL clustered at 0 with random sign.

**Failure modes that would disqualify it.**
- Judge API failure rate > 5% across a 100-step run — neutral-0.5 fallback
  becomes the modal step and gradient density collapses.
- Cross-rubric AUC (judge scored under rubric A vs rubric B on the same
  pseudo-gold set) < 0.7 — would mean the judge's ranking is dominated by
  rubric phrasing, not advice content. (Tooling for this not yet built; see
  Open Infrastructure Gaps.)
- Eval mean\_strategy climbs but a hand-authored "polished but tactically
  wrong" holdout-set score also climbs — direct Goodhart evidence.
- Per-step latency adds > 30s/step relative to no-API baseline (currently
  ~75s/step from claude-progress.txt 2026-04-19) — would push 100 steps
  past pod-budget threshold.

**Next test.** Pseudo-gold AUC on the same 30×4 set used for RECALL.
Same cost, same scale, head-to-head comparable. If AUC > RECALL+mask, the
judge advances to a 25-step in-training trial under the passer-spread gate.

**Implementation status.**
- `src/training/judge_reward.py` exists, group-cached via
  `functools.lru_cache(maxsize=128)`, soft-fail on every error path.
- `train_grpo.py` `--reward-mode judge` is wired (claude-progress.txt
  2026-04-23); the trainer passes `siblings` so the cache works correctly.
- `requirements.txt` already includes `anthropic` (was there for
  `baseline_eval.py`).
- Open: no cross-rubric eval harness; no rubric-randomization knob; only one
  fixed rubric (the `_RUBRIC` constant in `judge_reward.py`).

**Open questions.**
- Does the 0.035 passer-spread hold over 100 steps once the model adapts to
  the rubric, or does it collapse like RECALL did?
- How much of the +0.48 mean\_strategy is "judge gives reasonable scores
  by default" vs "the SFT-merged model already produces decent advice"? A
  baseline run scoring random advice with the judge would disambiguate.
- Cost stability: does the lru\_cache hit rate on real training stay where
  the smoke test put it (1 call/step), or does cache-key drift (sibling
  text varies subtly per generation) push us toward G calls/step?

### (c) BT-head

**How it works.** Per `scripts/train_bt_head.py` and `src/training/bt_reward.py`:
a small MLP over [state\_emb || response\_emb] from two FROZEN encoders —
state encoder is `outputs/embedding/learned_v3_alive` (CS2 state → 384-d),
response encoder is `all-MiniLM-L6-v2` (advice text → 384-d). Loss is the
standard Bradley-Terry pairwise objective: −log σ(r\_chosen − r\_rejected)
over labeled (chosen, rejected) advice pairs. Pairs come from a labeling
loop: `scripts/build_label_candidates.py` mines high-informativeness pairs
from existing GRPO audit logs (`outputs/grpo/f09/useful_jumps.jsonl`),
`scripts/label_app.py` (Streamlit) presents them shuffled to an expert CS2
player, and the resulting `preferences.jsonl` feeds `train_bt_head.py`. At
GRPO time, `bt_reward(response, ground_truth)` does one local forward pass
per completion — no API, no group context needed.

**Observed evidence.**
- `src/training/bt_reward.py` interface mirrors `judge_reward.py` exactly
  (drop-in replacement in the trainer's reward loop) and fail-soft to
  neutral 0.5 if `CHIMERA_BT_HEAD_PATH` is unset, the directory is missing,
  the checkpoint won't load, or the forward pass raises.
- `scripts/build_label_candidates.py` filters: drops pairs where both
  completions are format-fails, drops near-duplicate pairs (|reward\_diff|
  < 0.02 AND text similarity > 0.95 via MiniLM or trigram-Jaccard),
  caps at 500 pairs sorted by an informativeness score that prefers
  high-text-difference + low-judge-confidence. This is the only
  evidence-grade signal so far — it says we know which pairs to label, not
  that the head will score well.
- Zero `preferences.jsonl` collected. No empirical evidence on this
  candidate's strategic-content signal because no labels exist.

**Failure modes that would disqualify it.**
- Held-out validation accuracy on labeled pairs < ~65% — head is not
  separating chosen from rejected reliably; with 500 pairs and ~50/50
  labels, anything below ~65% is at noise floor.
- < ~300 labeled pairs collected: BT loss has too few constraints; the head
  fits label-app idiosyncrasies (pair ordering bias, labeler-specific
  habits) rather than tactical content. Even if labels are eventually
  obtained, < 300 means we can't trust differentiation between (b) and (c).
- AUC on the pseudo-gold 30×4 set is ≤ random (< 0.5) once enough labels
  exist — would mean the BT head learned something other than tactical
  preference (length bias, lexical preference, etc.).

**Next test.** Two stages:
1. Cold-start evaluation: train on whatever labels exist (≥ 300 pairs
   minimum) and score the pseudo-gold 30×4 set. AUC vs RECALL+mask, judge.
2. If AUC competitive, 25-step in-training trial with passer-spread audit.

**Implementation status.**
- `scripts/build_label_candidates.py` — done, deterministic ordering by
  informativeness, dedupes via embedding or trigram fallback.
- `scripts/label_app.py` — Streamlit UI, A/B position randomized,
  resumable via `preferences.jsonl` scan, ui\_a\_was\_originally key
  preserved so the trainer can recover canonical order.
- `scripts/train_bt_head.py` — done, BT loss, val-acc reporting,
  redaction-keyset matched to `learned_v3_alive` training.
- `src/training/bt_reward.py` — done, drop-in replacement, fail-soft.
- Open: no labels, hence no trained head, hence the head never gets loaded.
  Pipeline waits on the labeler.

**Open questions.**
- Is `learned_v3_alive` the right state encoder, or does the F2 outcome-
  correlation issue from RECALL apply here too? Same encoder family.
- How many labels before BT-head AUC stabilizes? Standard RLHF guidance is
  500–1500 minimum for a usable signal but we have no chimera-specific
  curve.
- Does "expert CS2 player" availability constrain timeline more than
  pod-cost or dev-time? Single bottleneck on a human.

---

## The decision protocol

Single canonical procedure for picking the next pod-bet. The thresholds
(numerical gate values) live in `docs/methodology.md`; this section
describes the SEQUENCE.

1. **Pseudo-gold AUC (offline, ~$0.50, no pod).** Build the 30 hand-authored
   states × 4 advices set (`scripts/build_pseudo_gold.py`, in flight this
   session). Score every candidate against it. The candidate with the
   highest AUC moves to step 2. This is the head-to-head test; the rubric
   for "constructed-correct" advice is hand-authored, so this validates
   "is this scorer better than chance" not "is it absolutely correct."
2. **σ\_s gate (RECALL only).** If candidate (a) is the AUC winner, it must
   first clear the per-query σ\_s threshold from `methodology.md` on the
   production code path. Already cleared offline (median 0.331 post-mask);
   the gate exists to catch regressions in the encoder pick.
3. **25-step in-training passer-spread gate.** Whichever candidate is up,
   run a 25-step GRPO trial and audit `useful_jumps.jsonl` for among-passer
   spread. If the median passer-spread is below the `methodology.md`
   threshold, the candidate fails the gate and we go back to step 1 with
   the next candidate.
4. **Goodhart cross-rubric guard (judge only).** Judge candidate (b) must
   additionally clear a cross-rubric eval — score the pseudo-gold set under
   rubric A and rubric B; if AUC drops by more than the `methodology.md`
   tolerance, the judge is over-fitting rubric phrasing and is disqualified
   for the F08v6 slot. Tooling for this is not yet built (see Open
   Infrastructure Gaps).

A candidate that clears all gates that apply to it is the next pod-bet.

---

## Combination strategies

**Reward ensembling.** Weighted sum of judge + BT-head, or judge + RECALL+
mask. The motivation is variance reduction: each scorer's idiosyncratic
errors are a non-trivial fraction of its raw output, but those errors are
likely uncorrelated across families (kNN-over-state, LLM-judge,
preference-MLP). Concretely, if BT-head AUC plateaus around 0.7 and judge
AUC sits at 0.75, an equal-weight sum should clear ~0.78 if the error
signs are independent. Wiring is small: the trainer's reward loop already
sums weighted reward functions; this is just adding a second strategy entry
to `REWARD_MODES` in `train_grpo.py`. The risk is that both signals share
the same Goodhart hole (e.g., both prefer confident phrasing) and the
ensemble amplifies it.

**Curriculum.** Run the judge for the first ~50 steps to bootstrap schema-
fluency and get the model emitting tactically-coherent advice, then switch
to BT-head once ~300+ labels exist. Or judge first, then ensemble with
BT-head once it's trained. Curriculum buys two things: (1) cheap signal
during the early-noise phase when most generations are format-broken or
generic, and (2) reduced API spend (the second half of training is
free). Risk: discontinuity at the switchover step. The model may have
gradient-fit features that please the judge specifically; switching the
reward signal mid-run can destabilize it.

**Ablation.** F08v6 picks one of (a, b, c) and labels the run as a
baseline; future paper writeup needs at minimum one head-to-head to claim
"this scorer is better than that one." Ablations are not gated by the
decision protocol — they are deliberately uneconomic single-shot runs to
fill the comparison table. F08v6 should be whichever candidate clears the
gating protocol; the OTHER two should be queued for back-to-back ablation
runs in subsequent pod sessions, not weighed against the gating protocol.

---

## Goodhart-specific concerns

The judge IS another model's prior. The policy can learn to please the
judge's lexical preferences (CS2 vocabulary density, confident phrasing,
the exact rubric anchor terms in `_RUBRIC`) without being tactically
better. RECALL has a Goodhart surface too — keyword-count action matching
in `_extract_action_from_text` rewards specific phrases like "push site"
over equivalents like "execute together" — but it is bounded by historical
outcomes. BT-head Goodhart is bounded by labeler-specific preferences,
which is also a flavor of "another model's prior" if the labeler is one
human.

Mitigations:
- **Cross-rubric eval.** Score the same pseudo-gold set under two
  semantically-equivalent rubrics. If AUC differs significantly, the
  scorer is fitting rubric not content. NOT BUILT.
- **Hand-authored polished-but-wrong holdout set.** A small set of advices
  written in the judge's preferred lexical style but with tactically wrong
  recommendations. If the policy's mean\_strategy on this set climbs over
  training, that's direct Goodhart evidence. NOT BUILT.
- **Randomized rubric per-step.** Sample one of N rubric variants per
  GRPO step. Increases variance (probably bad for convergence) but flattens
  rubric-Goodhart. NOT BUILT.

None of these mitigations exist as code yet. They are specified here so the
gap is visible.

---

## Open infrastructure gaps

Files that should exist but do not (or are in flight this session):

- `scripts/eval_scorer.py` — head-to-head AUC harness across all candidate
  scorers on a fixed eval set. Used in Decision Protocol step 1.
- `scripts/probe_features.py` — sanity-check what features each scorer is
  actually keying on (lexical, length, confidence words, JSON structure).
- `scripts/judge_eval_crossrubric.py` — cross-rubric AUC for the
  Goodhart guard in Decision Protocol step 4.
- `scripts/build_holdout_eval.py` — generates the polished-but-wrong
  holdout set referenced in the Goodhart section.
- `scripts/build_pseudo_gold.py` — IN FLIGHT this session. Generates the
  30×4 pseudo-gold set used by the AUC harness.
- `scripts/passer_spread_audit.py` — IN FLIGHT this session. Replays a
  `useful_jumps.jsonl` through the passer-spread gate from Decision
  Protocol step 3 to score in-training runs offline.
- `data/eval/pseudo_gold_advices.jsonl` — the actual hand-authored data
  the AUC harness consumes. Construction is intuition-labeled (A correct,
  B wrong, C generic, D plausible-but-wrong); only differential
  comparisons are valid.
- `outputs/labels/preferences.jsonl` — empty. Bottleneck for candidate (c).
