# World Model — Architecture Reference

This is the architecture reference for the Chimera world model: the objective,
state schema, token layout, scoping rules, and the probe-first evaluation
philosophy. Read it when you need to know how the model is shaped and why.
What to *train* — step size, schema arms, seeds, budgets, gates — is locked and
pre-registered in `docs/retrain-recipe.md` (+ `retrain-recipe-knobs4-7.md`),
which supersedes this doc wherever the two overlap; measured results live in
`claude-progress.txt`.

Built and measured 2026-06: a 19.2M per-player-token transformer trained on
next-state prediction with a distributional xy head and a value head. Those
checkpoints are baselines only; the canonical model comes from the recipe's
retrain.

## 1. Why a world model

The prior direction was a VLM "See, Then Think" pipeline with a Level-2 round
encoder trained on self-supervised forward-prediction objectives
(`docs/round-encoder-design.md`). Two measured results killed it as the main
lever. First, the round encoder saturates at ~16 demos: beyond ~328 training
rounds, probe accuracy stopped improving and the top-end slope went slightly
negative; architectural scaling (1024-d + salience) did not help
(`docs/learning-curve-finding.md`). Second, outcome supervision is
information-starved: `round_won` is ~1 bit per round, so an encoder gated on
`probe_outcome` extracts that bit almost immediately and then has nothing left
to learn. Change-point objectives found statistical boundaries, not semantic
ones, and Claude-caption supervision was circular — it paraphrased the same
structured features the encoder already had (discriminative-check gap +0.008).

The diagnosis: a sparse target was starving the representation. The fix changes
the *target*, not the data scale or the encoder size. Next-state prediction
supplies a dense, high-bandwidth signal — hundreds of bits per frame, thousands
of frames per round. It is the same bet language modeling makes: pretrain on a
huge self-supervised signal, then read everything downstream off the learned
latent. Here the "tokens" are game-state frames.

This reframes the three levels (`docs/three-level-architecture.md`): the world
model is *how* L1→L2 understanding is learned (the dynamics of the game), and
language (L3) becomes a downstream bridge into the learned latent rather than a
co-trained reasoning loss.

## 2. Objective

Train a causal spatiotemporal transformer to predict the next game-state frame
from the history of frames within a round:

```
p(state_{t+k} | state_{≤t})        state → state, no text in the loop
```

This is a self-supervised dynamics model. The labels are the demos themselves —
no human annotation, no caster commentary. The dynamics objective carries no
outcome label: a value head predicting P(CT win) is co-trained for downstream
use, but it is detached (stop-grad), so the trunk's gradients are identical to
training with no value head at all (recipe, Knob 5).

Key discipline: the model is judged by probe transfer, not by prediction loss.
A model can drive next-frame error down by predicting inertia — everything
keeps moving the way it was — without learning any tactics. The question is
whether the *latent* the model builds to make those predictions is
decision-relevant, which is exactly what the probe battery in
`docs/methodology.md` measures. Prediction loss is a training-health metric;
probe transfer is the acceptance gate (§8).

## 3. State schema

`feature_schema_v2` is a fixed 597-d vector per frame, and is
canonical-by-default (recipe, Knob 2):

- **10 players × 56-d = 560.** Per-player blocks in positional slots
  (T1..T5, CT1..CT5) with no player-identity embedding — the model should learn
  *role*, not *player*; identity is an opt-in feature later. Each block carries
  position, view angle, HP/armor, money, weapon, utility loadout, and status
  bits. There are no velocity dims (recipe, Knob 3: velocity is added only if
  the facing-shortcut check demonstrably fails on the canonical model).
- **37-d global.** Bomb status/position/timer, round timer, score, round
  number, map, phase, and round-relative time encodings.

`feature_schema_v3` (687-d) adds 9 derived perception dims per player —
visibility computed per §7 — and is the ablation arm of the recipe's
perception deconfound (Knob 2). The canonical schema is emitted by the
tensor-build step and travels with every checkpoint; corpus facts and defects
live in `docs/datasheet.md`.

**Cadence: 8 Hz** — every 8th tick of the 64 Hz server. That is ~125 ms between
frames, finer than human reaction time, and ~960 frames for a 2-minute round —
short enough for full attention over a round without efficient-attention
tricks.

## 4. Token layout and backbone

The backbone is a causal transformer with RoPE, ~19.2M parameters,
d_model = 512. Each frame becomes 11 tokens — one per player plus one global —
a per-player token layout borrowed from MLMove. The model's output is a
contextualized token grid `[B, L, 11, 512]`; this grid is the interface for
everything downstream (probes, value head, the bridge).

The backbone design is reused from the round encoder
(`docs/round-encoder-design.md` §3 — superseded as a *direction*, kept as the
reusable architecture). What changed is the objective (dense distributional
next-state prediction instead of five SSL heads), the scoping discipline
(round-as-document, below), and the downstream story (heads plus the bridge,
instead of a frozen embedding feeding RECALL).

## 5. Round scoping: the round is the document

Attention is round-scoped. A round is the unit of context, like a document in
LM pretraining. There is no cross-reset attention — a frame in round 7 never
attends to round 6; resets (buy phase, side switch, score change) are hard
boundaries. Information that *would* carry across rounds — economy, score,
round number — enters as features inside the frame (the 37-d global block),
not as attention reach. The model therefore cannot leak future-round outcomes
backward and cannot use cross-round trajectory bleed as a shortcut (the F1
leakage failure that haunted the round encoder has no surface here).

Causal attention within the round means `h_t` is a function of frames `0..t`
only — F2-safe by construction, the same property the v4 round encoder relied
on.

## 6. Prediction head and horizons

The future is multimodal — a player at a chokepoint might push, hold, or
retreat. A point-estimate regression head (MSE) averages those modes and
produces blurred, physically-implausible "average" states. The non-negotiable
architectural principle is that the head represents a *distribution* over next
states, for the same reason image/world-model work (discretized tokens,
DIAMOND's diffusion head) avoids plain MSE.

How that principle is instantiated is locked in the recipe (Knob 1):
rollout-native — a single short-step model (k = 4 frames = 500 ms) with a
distributional xy head (97-class classify-then-refine), reaching all horizons
by sampled autoregressive rollout and evaluated on coverage (minADE-K). This
supersedes the design-time horizon sweep (direct +1/+4/+8/+16 heads) and the
discretize-vs-GMM open point that earlier versions of this doc carried.

## 7. Feature-engineering boundary: perception, not tactics

The single most important design rule, carried over from the three-level
framing: engineer only perception primitives (L1 / "See"); the model learns
tactics (L2). No hand-engineered tactical labels enter the model.

Derived visibility — who can see whom, from positions, view angles, and map
geometry — is a *perception* primitive: it is what a player would perceive, so
computing it for the model is fair game (implementation and its known
limitations are locked in recipe Knob 2). By contrast, "this is an execute,"
"this is a retake," "this is a default" are tactical abstractions; if they are
real, the model should discover them as structure in its latent. Injecting
them as features would repeat the circularity that sank Claude-caption
supervision.

The line: if a primitive is something the player's *senses* deliver (position,
what's visible, what's audible-in-principle), engineer it. If it is an
*interpretation* of the situation, let the model learn it.

## 8. Evaluation: probe transfer, not prediction loss

Acceptance is the probe / σ_s discipline of `docs/methodology.md` (the
canonical statement of the gates), retargeted onto the world-model latent:

- **Probe battery** (methodology axis 2). Freeze the world model; train tiny
  MLP probes on its latent for `round_won`, `pro_action_next`, and
  forward-state. Pass thresholds carry over: `probe_outcome` ≥ 0.65,
  `probe_action` ≥ 0.45, `probe_next` R² ≥ 0.50. The decisive question versus
  the round encoder: does next-state pretraining push probe accuracy *above*
  the saturation ceiling (`docs/learning-curve-finding.md`) at fixed data,
  especially on the multi-class / forward targets that `round_won` could not
  move? The keystone (C1) pass/fail criteria for this comparison are
  pre-registered in the recipe (Knob 7).
- **σ_s** (methodology axis 1). Neighbor-outcome variance on world-model
  latents, same Goldilocks band [0.15, 0.45], same
  `scripts/recall_variance_diagnostic.py` infrastructure.
- **Encoder disagreement** (methodology axis 5) as a tiebreaker.

Prediction loss (next-frame NLL / per-field accuracy) is reported as a
training-health metric, not an acceptance gate. A model that wins on prediction
loss but fails probe transfer has learned physics, not understanding.

## 9. What derives from the world model

The world model is the substrate; everything else is a head or a read-out on
its latent.

- **Events fall out of prediction surprise.** An event is where the model is
  surprised — where the realized next state has low likelihood under the
  prediction. This subsumes the change-point work: instead of fitting
  statistical boundaries (which found non-semantic splits), semantically
  meaningful moments (a kill, a flash, a fake) are exactly the high-surprise
  frames. No separate change-point objective.
- **Value / policy are heads on the latent (MuZero-style).** The value head
  predicts `round_won`; a policy head predicts the pro's next action. These
  are downstream heads — the value head detached per recipe Knob 5 — keeping
  outcome out of the representation-learning loop.
- **Reasoning is verbalizing rollouts.** With a learned dynamics model you can
  roll the latent forward and have the bridge describe what it predicts will
  happen — "they're going to fake A and rotate B." This is the L3 "Think"
  step, grounded in an actual forward model rather than a single-state advice
  generator.

## 10. Language bridge (phase 2)

Canonical design: `docs/bridge-design.md`. In brief: the frozen world-model
token grid feeds a trainable resampler into Qwen3.6-35B-A3B (QLoRA), the NLA
gate scores round-trip faithfulness (does the text render the latent or
hallucinate past it), and the phase-3 GRPO reward is grounded — the verbalized
prediction is scored against the demo's *actual* future, in contrast to the
prior judge/RECALL rewards (`docs/reward-candidates.md`), which had no
ground-truth future to check against.

One discipline from this doc still binds every language result: ablate the
latent. Each result must be re-run with the world-model latent zeroed or
shuffled, to prove the language head is using the latent and not just the
LLM's prior. This is the language-side analogue of the probe-transfer gate.

## 11. Data

Corpus composition, splits, exclusions, and defects are canonical in
`docs/datasheet.md`; the corpus's future is `docs/corpus-strategy.md`. The
design-relevant fact: per `docs/learning-curve-finding.md`, data scale is
*not* the bottleneck for the saturated recipe — the bet here is squarely on
the denser objective, not on more demos. More / more-diverse demos (different
era, skill bracket, MR12) remain a distribution-shift lever for later, but
they are not the reason to expect the world model to beat the saturation
ceiling.

## 12. Design questions: settled and open

Settled since this doc was first written — see the recipe: head form and
horizon strategy (Knob 1), visibility computation (Knob 2), velocity inputs
(Knob 3), and bridge staging (`docs/bridge-design.md`).

Still open:

1. **Latent read-out point** — which layer / pooling feeds the probes and the
   bridge (a tap-layer sweep is planned before bridge training; see the
   runbook in `claude-progress.txt`).
2. **Surprise calibration** — what likelihood threshold defines an "event,"
   and whether it recovers awpy events (kills, plants) as a sanity check. The
   adaptive event clock in `docs/bridge-design.md` depends on this.
