# The Three-Level Perception / Reasoning Hierarchy

> **PIVOT (2026-06): the world model is now how L1→L2 understanding is
> learned.** The three-level decomposition (See / Situate / Think) still
> holds as the organizing principle, but the *mechanism* for learning the
> L1→L2 understanding has changed. Originally L2 was a round encoder
> trained on self-supervised SSL heads and gated on outcome probes; that
> encoder **saturated at ~16 demos** (`docs/learning-curve-finding.md`)
> because outcome supervision (~1 bit/round) is info-starved. The new
> foundation is a **next-state-prediction world model**
> (`docs/world-model-design.md`): a causal spatiotemporal transformer
> over game-state frames (state → state, no text) whose dense
> distributional objective learns the *dynamics* of the game. Under this
> framing:
> - **L1 (See)** = the engineered **perception primitives** that make up
>   each frame (position, view angle, derived visibility). Still
>   hand-built — but only perception, never tactics.
> - **L2 (Situate)** = the **latent of the world model**. Tactics are
>   *learned* as the structure the model needs to predict the future, not
>   trained against tactical labels. Events fall out of **prediction
>   surprise**; value/policy are MuZero-style heads on the latent.
> - **L3 (Think)** = a **downstream language bridge** into that latent
>   (phase 2): a frozen Qwen 3.6/3.7 MoE wired in Flamingo-style, with
>   reasoning trained as *verbalized rollouts scored against the actual
>   future*.
>
> The sections below describe the prior three-level pipeline (VLM "See,
> Then Think" with a round encoder at L2 and GRPO advice at L3). Read them
> as the architectural lineage; `docs/world-model-design.md` is the
> current central design doc. The per-level *evaluation discipline*
> (probe transfer, σ_s, ablate-the-signal) carries over unchanged and is
> still load-bearing.

Organizing principle for chimera. Treats "look at the game", "understand what
just happened", and "say what to do" as three different learning problems with
three different ground truths, three different evaluation harnesses, and three
different training objectives. Every reward-signal failure in F08v4 and v5 is
re-read here as a symptom of having merged levels 2 and 3 into a single GRPO
loop.

## 1. Why this document exists

F08v4 (`outputs/grpo/f08v4/`) and F08v4_resumed spent ~100 effective GRPO steps
and ~$25 of pod time pushing the policy against a collapsed reward. The
post-mortem (`claude-progress.txt`, 2026-04-23) showed the standard deviation
of RECALL scores among format-passing siblings was a *median* of 0.000 in the
modal case (2/4 passers). `methodology.md` formalizes the five offline gates
that would have caught that before the pod spend. `reward-candidates.md` lays
out the three reward families currently in flight (RECALL+mask, Claude judge,
BT-head).

What neither doc yet says out loud: the deeper problem was not that *this
particular* reward signal was bad. It was that the GRPO loop was being asked
to do two unrelated things in one parameter update — turn noisy continuous
tick streams into clean event-level representations *and* generate tactical
advice on top of those representations — using a single gradient whose only
supervision was a final scalar at the end of a long pipeline. The
state-equivalence problem RECALL was trying to solve is a perception problem,
not a reasoning problem; collapsing it into the reasoning loss starved both.

The paper title (`paper/main.tex` line 12) is *See, Then Think*. The framing
is incomplete. Between "See" (read the HUD from one screenshot) and "Think"
(emit strategic advice) there is a missing middle step: aggregate a window of
sees into an event-level summary. Call it Situate. The expanded framing is
**See, Situate, Then Think.**

The rest of this document specifies the three levels, the contracts between
them, and the failure modes each level now isolates. Encoder choice,
window definition, and the actual training objective for the new middle layer
are deferred to `docs/round-encoder-design.md` (in flight in parallel).

## 2. The three levels

| Level | Name     | Input                                                    | Output                              | Status                                          |
|------:|----------|----------------------------------------------------------|-------------------------------------|-------------------------------------------------|
| 1     | See      | Single CS2 screenshot (`PIL.Image`, ~1920x1080)         | `game_state` JSON (typed)           | Shipped — F04 SFT, ~67% field accuracy           |
| 2     | Situate  | Sequence of Level-1 outputs across a tick window around an event | Event embedding, ~256-dim vector | Missing — the new piece this doc justifies      |
| 3     | Think    | Level-1 output (current) + Level-2 output (situation) jointly | Natural-language tactical advice  | Target of GRPO; currently trained against fused signal |

### 2.1 Level 1 — See (visual perception)

**Inputs.** One CS2 screenshot at HUD resolution. Realistically captured at
the same tickrate as the demo (128 Hz) but only sampled at event-relevant
ticks; full-stream perception is out of scope.

**Outputs.** A typed `game_state` JSON. The shipped schema covers HP, armor,
money, weapon, round_phase, map, side, score, plus a 19-dim
`tactical_embedding` (see `reward-candidates.md` section (a) for the
hand-engineered version). Each field is independently checkable against demo
parquet ground truth.

**Ground truth source.** *Free.* Demo parquets parsed via awpy provide
per-tick ground truth for every screenshot taken during recorded playback
(`data/demos/` parquets, F02). The screenshot–parquet pairing is fixed and
deterministic (F03; SendKeys-driven capture pinned to specific ticks).
Cross-map coverage: Mirage / Inferno / Nuke / Overpass, 4 demos, 4353
screenshots, 5309 field-labels. No human labeling.

**Training objective.** Supervised fine-tuning. LoRA r=4, alpha=8, on vision
and language layers, 1 epoch over 3322 paired (screenshot, game_state JSON)
samples. Loss is plain next-token cross-entropy on the target JSON string. See
F04 in `feature-list.json` for full hyperparameters.

**Evaluation metric.** Per-field accuracy on a held-out (demo, round)-disjoint
split. Numeric fields use tolerance buckets (HP within 5, money within 100,
score exact). Pass threshold: per-field accuracy >= 0.60 on the held-out
split; current shipped checkpoint: 67.3%. Field-level breakdown reveals which
HUD elements remain unreliable (currently still-life elements like money are
near-perfect; spatial elements like equipment around a teammate are weaker).
Floor (random / base Qwen3.5): ~50%.

**Frozen vs trainable during downstream training.** **Frozen.** Once F04
shipped, the LoRA was merged (`outputs/sft/final_model/merged_16bit/`, 66 GB)
and no Level-3 gradient flows back through the vision stack. Any later
improvement to Level 1 is a separate SFT pass with its own per-field
accuracy regression test.

**Where this layer lives in the codebase.**
- Training: `scripts/train_sft.py`, `src/training/sft_trainer.py`.
- Data: `src/data/manifest.py`, `src/data/loader.py`, `scripts/build_sft_dataset.py`.
- Eval: `scripts/generate_sft_labels.py` (label generation), and the
  perceptual-accuracy reward in `src/training/rewards.py` (used at Level 3 as
  the multiplicative format / fidelity gate, but the underlying metric is the
  Level-1 eval).
- Merged model: `outputs/sft/final_model/merged_16bit/`.

### 2.2 Level 2 — Situate (event / temporal perception)

**Inputs.** A tick window around a designated event. Each tick in the window
contributes a Level-1 `game_state` JSON (so Level 2 consumes Level-1 outputs,
not raw screenshots; it is a temporal aggregator over a structured stream).
Window size, anchor offset, and resolution policy are open and deferred to
`docs/round-encoder-design.md`. For the purposes of this doc the window is
"a contiguous sub-sequence of Level-1 outputs around the event tick."

**Outputs.** An event embedding, target dimensionality ~256. The point of
Level 2 is that this vector is what Level 3 should retrieve neighbors over,
key its conditioning on, and key its evaluation on — not the single-tick
19-dim `tactical_embedding`. The single-tick vector is the input feature; the
event embedding is the output of Situate over a window of those features.

**Ground truth source.** *Weakly supervised, free.* Two parallel sources:

1. *Forward dynamics.* Given the Level-1 sequence on the first half of a
   window, predict the Level-1 sequence on the second half. Forward-dynamics
   prediction has no labeling cost — the labels are the demo itself.
   Motivated by DeepMDP-style behavioral encoders (see
   `methodology.md`'s anchor section: `castro2020scalable`, `zhang2021learning`,
   `agarwal2021contrastive`, plus DeepMDP / `gelada2019deepmdp` [verify] —
   not yet in `paper/references.bib`).

2. *Event metadata.* Awpy already supplies per-event annotation: kill events,
   bomb plant / defuse, round_end, side_won, post-plant bucket, etc. These
   are demo-derived, exact, and per-event — they are stronger labels than
   round-level `round_won` because they are localized to the window. Each
   window receives a small typed event-metadata tuple from awpy.

`docs/round-encoder-design.md` will pick between (1), (2), a combination, or
something contrastive over windows. This doc does not prescribe.

**Training objective.** Self-supervised or weakly supervised, never
outcome-correlated alone. The F2 collapse documented in `methodology.md`
section 1 (axis sigma_s) was caused by using `round_won` as the *only*
positive-pair label — a class-correlated positives setup that `graf2021dissecting`
shows collapses representations onto the class axis (and which chimera
empirically confirmed in commit 1b387b4's F2 dose-response). Level 2's
objective must therefore not be "predict round_won from event embedding"
directly. The objective space is one of: forward dynamics on Level-1 stream,
event-metadata prediction, masked-tick reconstruction, contrastive over
nearby vs distant windows, or a combination — chosen and motivated in
`docs/round-encoder-design.md`.

**Evaluation metrics.** Three independent probes (`methodology.md` axis 2,
"Probe accuracy battery"; the existing pass thresholds carry over verbatim
once we re-target the probes from single-tick state to event embedding):

- `probe_outcome`: predict `round_won` from event embedding. Pass: val acc
  >= 0.65 on (demo, round)-disjoint split. Floor 0.50.
- `probe_action`: predict the pro's next-action category from event embedding.
  Pass: val acc >= 0.45 on 5-class. Floor 0.20.
- `probe_next`: forward dynamics R^2 on the next K Level-1 outputs. Pass: R^2
  >= 0.50.
- `sigma_s` (`methodology.md` axis 1): median per-query neighbor outcome std
  in the Goldilocks band [0.15, 0.45]. Same gate as for Level 3's retrieval
  state; reusing the existing diagnostic infrastructure
  (`scripts/recall_variance_diagnostic.py`).
- Disqualification: any candidate that fails two or more of the above is not
  promoted to Level-3 input.

**Frozen vs trainable during downstream training.** **Frozen** during the
Level-3 GRPO loop, *after* Level-2 has independently passed its probe
battery. Training Level 2 jointly with Level 3 is the exact mistake F08v4
made; the lesson is to gate promotion of a Level-2 encoder on the Level-2
metrics alone.

**Where this layer lives in the codebase.** **To be written.** Target paths:

- `src/perception/event_encoder.py` (new) — `EventEncoder` class consuming a
  list of Level-1 outputs, returning a fixed-dim vector. Frozen after
  training.
- `scripts/train_event_encoder.py` (new) — training entry point.
- `scripts/probe_event_encoder.py` (new) — Level-2 probe battery
  (`methodology.md` axis 2 retargeted at event embeddings). Subsumes the
  not-yet-built `scripts/probe_features.py` listed in `methodology.md`'s
  "Coverage gap to fix next."
- `outputs/event_encoder/<run_id>/` (new) — checkpoint directory pattern
  matching `outputs/sft/` and `outputs/grpo/`.

### 2.3 Level 3 — Think (strategic reasoning)

**Inputs.** Two things jointly: the Level-1 output at the decision tick
(the *current* game state) AND the Level-2 output for the window leading up
to it (the *situation*). Crucially: Level 3 should never see raw
single-tick `tactical_embedding` features in place of the event embedding.
This is the structural fix.

**Outputs.** Natural-language tactical advice. Output schema matches the
existing SFT-merged base: a JSON object with reasoning + concrete actionable
advice plus structured fields (target_site, weapon_priority, timing,
coordination). Schema details are in `src/training/grpo_trainer.py` and the
SFT label generator.

**Ground truth source.** *Indirect and structurally hard.* This is the
problem RECALL was trying to solve via retrieval and the judge route bypassed
via another LM's prior. Three families of supervision are now disentangled
from one another, since they each target a different downstream object:

- *Outcome.* `round_won` and `player_contribution` (F05 schema). These
  remain — but they now condition on the event embedding rather than on a
  single-tick state, so the retrieval at Level 3 is over events, not ticks.
- *Pro-action match.* The pro's actual next action (extracted from
  awpy-parsed demos) is a per-event label. This was always there; it now has
  a clean home as a Level-3 conditioning signal.
- *Judge / preference.* The Claude-judge and BT-head rewards from
  `reward-candidates.md` remain as drop-in scoring modules at this level.

**Training objective.** GRPO with a multiplicative format gate
(`src/training/grpo_trainer.py`), perceptual-accuracy reward (Level-1 fidelity
check), and one strategy reward selected per
`reward-candidates.md` decision protocol. The strategy reward must clear the
gates in `methodology.md` (sections "The five measurement axes" 3 and 4,
"The decision protocol" steps 0 and 3) before pod spend, and now
*additionally* must consume Level-2 output, not single-tick features. This is
the structural change: the reward function's "state" argument is the event
embedding, not the 19-dim `tactical_embedding`.

**Evaluation metrics.** Inherited from `methodology.md` and re-anchored at
Level 3:

- Pseudo-gold AUC on the 30x4 hand-authored set (`methodology.md` axis 3) >=
  0.70 — unchanged.
- Within-group passer-spread on `useful_jumps.jsonl` (axis 4): k=2 median
  spread >= 0.025 and zero-spread fraction at k=2 < 15% — unchanged.
- Cross-rubric Goodhart guard (`methodology.md` decision-protocol step 4):
  cross-rubric mean_strategy improvement >= 0.05 over base+SFT — unchanged.
- Mean strategy score on the 50-sample held-out eval (existing harness in
  `scripts/run_grpo_smoke.sh`) reported but not gated.

Notice: nothing on the Level-3 gate list now needs to *also* validate that
the state representation is decision-relevant. That work is done at Level 2.
Level-3 gates run *only* on Level-3 metrics. This is the per-level evaluation
discipline of section 5.

**Frozen vs trainable during downstream training.** **Trainable.** This is
the only level whose weights move during GRPO. Level 1's LoRA is merged and
frozen; Level 2's encoder is frozen post-promotion. Level 3 has LoRA r=4,
alpha=8 on the language adapter (current configuration in
`scripts/train_grpo.py`).

**Where this layer lives in the codebase.**
- Training: `scripts/train_grpo.py`, `src/training/grpo_trainer.py`.
- Reward modules: `src/training/rewards.py` (perceptual accuracy),
  `src/training/recall.py` (kNN, RECALL family),
  `src/training/judge_reward.py` (Claude judge),
  `src/training/bt_reward.py` (Bradley-Terry head).
- All three reward modules need a small interface change to consume the
  event-embedding argument (currently they consume the single-tick state
  JSON). That migration is part of integrating Level 2; see section 6.

## 3. The contracts between layers

The hierarchy is only useful if the boundaries are sharp. Each contract is a
function-signature-in-plain-English plus what must NOT cross.

**Level 1 -> Level 2.** Level 2 consumes a *sequence of Level-1 outputs*
across a tick window. It does not consume the raw screenshots, the VLM's
hidden states, or any non-Level-1 derived feature. Plain-English signature:
`situate(level1_outputs: list[GameStateJSON], event_tick: int) -> EventEmbedding`,
where `EventEmbedding` is a fixed-dimensional vector (~256-d, exact dim
deferred to `docs/round-encoder-design.md`). If a Level-2 candidate wants
visual information not currently in the Level-1 schema, the right fix is to
extend the Level-1 schema and re-SFT — not to pipe images around Level 1.

**Level 2 -> Level 3.** Level 3 consumes both the current Level-1 output and
the Level-2 event embedding. Plain-English signature:
`think(current_state: GameStateJSON, event_embedding: EventEmbedding) -> AdviceJSON`.
The reward functions at Level 3 must consume the same `(current_state,
event_embedding)` pair when computing their score. The kNN index in
`src/training/recall.py` must be rebuilt over event embeddings; the judge
prompt in `src/training/judge_reward.py` may consume Level-1 state directly
(it is a text prompt, not a vector retrieval) but it should *also* receive
event-summary context so its rubric anchors on the same situation Level 2
encodes. The BT-head in `src/training/bt_reward.py` swaps the state encoder
from `learned_v3_alive` to the promoted event encoder.

**What MUST NOT cross.** Level-3 gradients into Level 2; outcome labels
(`round_won`) as Level-2 positives without auxiliary objectives; single-tick
state used in place of the event embedding in any Level-3 reward function.

## 4. Why this fixes the failure modes

| Prior failure mode                                  | Where it was wrongly placed                           | Where it is now isolated / fixed                                            |
|-----------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------|
| F1 sibling-tick leakage (`methodology.md` axis 1)   | Conflated into RECALL's retrieval; "fixed" by ad-hoc mask in `recall.py` | Level 2's positives are defined over events / windows, not ticks. Same-round bleed has no natural surface to leak through because retrieval is event-level. |
| F2 outcome-correlated supervision collapse (`graf2021dissecting`, commit 1b387b4) | Conflated into RECALL's encoder training (state encoder used `round_won` as the SupCon positive label) | Level 2's objective is forward dynamics + event metadata, **not** `round_won` directly. Outcome can be ONE probe among several but never the only positive-pair signal. See section 2.2 "Training objective." |
| F3 action-vector noise (`_extract_action_from_text` 5-dim keyword count) | Conflated into Level-3 reward as an action-side encoder | Level 3's reward function operates on advice text and event embeddings, not on a keyword-count proxy. The judge reward already does this; RECALL needs the same migration. |
| Tick-determinism (single-tick state is dominated by surface features: map one-hot, side, HP) | The "state" the GRPO loop saw was a single tick, so sibling generations within a step were all conditioned on near-identical features and the scorer collapsed | Level 3's "state" is now (current Level-1 output, Level-2 event embedding). The event embedding aggregates a window so two genuinely different situations no longer look identical to the scorer. |
| Sample-author labeling difficulty (BT-head needs 300+ pairs; nobody knows what "good" is at the single-tick level) | Asking a labeler to compare advice for a single tick out of context | Level 3 labelers compare advice for a *situated event*. The event embedding fixes the context the labeler is operating in, so pair comparisons become tractable and reusable across ticks within the same situation. |

Each row used to live in the same "fix the reward" bucket. Each one now has a
distinct, addressable home and a distinct evaluation harness.

## 5. Per-level evaluation methodology

Each level is evaluated independently. Concretely:

- **Level 1** has *one* metric: per-field accuracy on the (demo,
  round)-disjoint held-out split, with field-level breakdown. Pass threshold
  applies before the level's checkpoint is merged. No Level-2 or Level-3
  outcome is permitted to vote on Level-1 promotion.

- **Level 2** has its own probe battery (section 2.2, retargeted from
  `methodology.md` axis 2) plus `sigma_s` (axis 1) on event embeddings. A
  Level-2 candidate is promoted on these metrics alone. Pseudo-gold AUC and
  passer-spread are *Level-3* metrics; they do not participate in Level-2
  promotion. If a Level-2 candidate passes its battery but Level-3 results
  underperform downstream, the right next step is to debug the Level-3
  reward / objective on a fixed Level-2 encoder, NOT to retrain Level 2
  against Level-3 outcomes (which is precisely the failure F08v4 made).

- **Level 3** runs the full `methodology.md` decision protocol (steps 0–4):
  pseudo-gold AUC, sigma_s on Level-3 retrieval state (= event embedding),
  probe accuracy on Level-3 retrieval state, 25-step in-training
  passer-spread audit, cross-rubric Goodhart guard. None of these audit
  Level-1 visual accuracy or Level-2 event-embedding quality directly;
  upstream failures show up as Level-3 instability and trigger an *upstream*
  investigation, not a Level-3 retrain.

This eliminates conflated comparisons. "F08v4's strategy reward had
median 0.000 within-group spread" is no longer a question about RECALL
vs the judge vs BT-head in the abstract. It is now three separable questions:

1. Was the Level-1 perception accurate at the decision ticks? (~67% per-field
   right now — adequate, not a bottleneck.)
2. Did the Level-2 event embedding pass its probe battery on this slice of
   data? (Currently undefined — Level 2 does not yet exist.)
3. Given (1) and (2), did the Level-3 reward function differentiate among
   sibling completions in a window where the state genuinely varies?

The F08v4 result is now correctly read as: question 2 was implicitly answered
"no" because there was no Level 2, and question 3 was therefore unanswerable.

## 6. What this changes about the project

Artifacts that need updating to make the three-level hierarchy the project's
organizing principle (alignment-delta details — exactly which sections of
each file change — live in a separate doc; this list is just categories):

- **`docs/methodology.md`.** Add a "Per-level evaluation" preamble pointing
  out that the five axes live at Level 2 (axis 1 sigma_s, axis 2 probes, axis
  5 encoder disagreement) and at Level 3 (axis 3 pseudo-gold AUC, axis 4
  in-training passer-spread). Mark which axis gates which level.

- **`docs/reward-candidates.md`.** Add a paragraph at the top noting that all
  three candidates (RECALL, judge, BT-head) are Level-3 reward functions and
  that they consume event embeddings, not single-tick state, going forward.
  Update the RECALL "state encoder" discussion to point at the event encoder
  rather than the 19-dim `tactical_embedding`.

- **`feature-list.json`.** The F-axis numbering predates this hierarchy and
  the methodology doc already supersedes the F-numbering as project framing.
  New entries should be tagged by level (L1/L2/L3) and the Level-2 build-out
  should be enumerated as a series of features (event encoder training,
  probe battery, promotion gate) rather than absorbed into F06 ("GRPO
  training — strategic reasoning"). F06's strategy-reward dependency
  splits into a Level-2 promotion and a Level-3 reward selection.

- **`README.md`.** The project tagline currently echoes the paper's
  "See, Then Think." Update to "See, Situate, Then Think" or equivalent, and
  add a one-line description of the hierarchy with a link to this doc.

- **`paper/main.tex`.** Title, abstract, and method section all assume a
  two-phase pipeline. The three-level framing is the stronger contribution
  (see section 7). The paper update is its own task; this doc just flags
  that the change is needed.

- **`docs/round-encoder-design.md` (new, in parallel).** Specifies what this
  doc deliberately defers: encoder architecture, training objective, window
  definition, target dimensionality, probe battery wiring.

## 7. Research framing

The current paper sells "GRPO on a VLM for game understanding." That framing
flattens the contribution into a method-application combo and gives reviewers
a single hook ("does GRPO help here") on which to anchor accept/reject. The
three-level reframing is stronger and more generalizable:
*continuous-game strategic reasoning decomposes into visual perception
(per-tick HUD parsing), event-level temporal aggregation (situating a window
of perceptions as an event), and natural-language reasoning (advice
generation), each independently trainable against its own ground-truth
objective, with explicit gating between levels.* The empirical contribution
becomes (a) the per-level evaluation harness, (b) the demonstration that
prior collapsed-reward failure modes (`zerovariance2025`, our F08v4
diagnosis, `graf2021dissecting`-style F2 collapse) are symptoms of
cross-level conflation rather than scorer-family inadequacy, and (c) the
specific event-encoder design that resolves the missing middle level for
CS2 demo data. The application to CS2 becomes one instantiation of a recipe
that also applies to other continuous, partially-observed, demo-rich domains
(StarCraft, Dota, real-world dashcam driving — all share the perception /
situation / decision decomposition). Pre-existing precedents we cite remain
the same — `alonso2024diamond` for ML on CS:GO demo data,
`pitis2020counterfactual` for non-simulator counterfactual reasoning,
`castro2020scalable` / `zhang2021learning` / `agarwal2021contrastive` for
bisimulation-style behavioral encoders — but they now anchor specific
*levels* of the hierarchy rather than a single fused pipeline. `gelada2019deepmdp`
[verify] (DeepMDP) is the natural anchor for Level 2's forward-dynamics
objective and should be added to `paper/references.bib`. The
"missing middle" framing is, to our knowledge, new for natural-language
strategic advice scored against continuous-game demos; the closest prior
art (chess/Go RL, RLHF code-tasks, RLHF chat) all sidestep the situation
level by virtue of discrete or text-only state.
