# World Model Design — Next-State Prediction over CS2 Game State

**Status: PLANNED / STARTING.** This is the new central design doc for
chimera. Nothing in the world-model line is trained yet. The data
substrate exists (see §9); the model, the horizon sweep, the probe
battery wiring, and the language bridge are forward work. No metrics in
this doc are measured — measured findings live in
`docs/learning-curve-finding.md` (the saturation result that motivates
this pivot) and in `docs/methodology.md` (the probe / σ_s gates we keep).

## 0. Why we pivoted

The prior direction was a VLM "See, Then Think" pipeline with a Level-2
round encoder trained on self-supervised forward-prediction objectives,
gated on probe transfer (`docs/round-encoder-design.md`). Two measured
results killed that recipe as the *main* lever:

1. **The round encoder saturates at ~16 demos.** Adding demos beyond
   ~328 training rounds produced no probe-accuracy improvement and a
   slightly negative top-end slope (`docs/learning-curve-finding.md`).
   Architectural scaling (1024-d + salience) did not help either.

2. **Outcome supervision is information-starved.** `round_won` is
   ~1 bit per round. A round encoder gated on `probe_outcome` is
   learning to predict a single noisy bit; it extracts essentially all
   of that bit's worth of structure almost immediately and then has
   nothing left to learn. Change-point / event objectives found
   *statistical* boundaries, not *semantic* ones, and Claude-caption
   supervision was circular (it paraphrased the same structured features
   the encoder already had; discriminative-check gap was +0.008).

The diagnosis: we were starving the representation with a sparse target.
The fix is to change the *target*, not the data scale or the encoder
size. **Next-state prediction supplies a dense, high-bandwidth signal:**
predicting the full 597-d game state one step ahead is hundreds of bits
per frame, thousands of frames per round. This is the same bet language
modeling makes — pretrain on next-token prediction over a huge,
self-supervised signal, then read off everything downstream from the
learned latent. Here the "tokens" are game-state frames.

This reframes the three levels (`docs/three-level-architecture.md`):
the world model is *how L1→L2 understanding is learned* (the dynamics
of the game), and language (L3) becomes a downstream bridge into the
learned latent rather than a co-trained reasoning loss.

## 1. Objective

Train a **causal spatiotemporal transformer** to predict the next
game-state frame from the history of frames within a round:

```
p(state_{t+k} | state_{≤t})        state → state, NO text in the loop
```

This is a self-supervised dynamics model. The labels are the demos
themselves — no human annotation, no outcome label in the training
objective, no caster commentary. It is LM-style pretraining over game
states.

Key discipline: **we judge the model by probe transfer, not by
prediction loss.** A model can drive next-frame MSE down by predicting
inertia (everything keeps moving the way it was) without learning any
tactics. The question is whether the *latent* the model builds to make
those predictions is decision-relevant — which is exactly what the
probe battery in `docs/methodology.md` measures. Prediction loss is a
training signal; probe transfer is the acceptance gate.

## 2. State schema (feature_schema_v2, 597-d/frame)

Each frame is a fixed **597-dimensional** vector:

- **10 players × 56-d = 560** — per-player block, positional slots
  (T1..T5, CT1..CT5), no player-identity embedding (learn *role*, not
  *player*; identity is an opt-in feature later). Per-player block
  carries the perception primitives: position, view angle, velocity,
  HP/armor, money, weapon, utility loadout, status bits, and derived
  visibility (see §6).
- **37-d global** — bomb status / position / timer, round timer, score,
  round number, map, phase, and round-relative time encodings.

This succeeds `feature_schema_v1` (~582–750-d in
`docs/round-encoder-design.md`); the canonical schema is emitted by the
tensor-build step (see §9) and travels with every checkpoint.

### Cadence

**8 Hz** (every 8th tick of the 64 Hz server). ~125 ms between frames,
finer than human reaction time; ~960 frames for a 2-minute round. Same
cadence the round encoder used — short enough for full attention over a
round without efficient-attention tricks.

## 3. Round-scoping: the round is the "document"

Attention is **round-scoped**. A round is the unit of context, like a
document in LM pretraining. There is **no cross-reset attention** — a
frame in round 7 never attends to round 6. Resets (buy phase, side
switch, score change) are hard boundaries.

The information that *would* carry across rounds — economy, score,
round number — enters as **features inside the frame** (the 37-d
global block), not as attention reach. This keeps the model honest:
it cannot leak future-round outcomes backward, and it cannot use
cross-round trajectory bleed as a shortcut (the F1 leakage failure
that haunted the round encoder has no surface here).

Causal attention within the round means `h_t` is a function of frames
`0..t` only — F2-safe by construction, same property the v4 round
encoder relied on.

## 4. Horizon sweep

We do not commit to a single prediction horizon. We **sweep k**:

| Horizon k | Wall time @ 8 Hz | What it should capture |
|---|---|---|
| +1 | 125 ms | Inertia / physics — near-trivial, mostly extrapolation |
| +4 | ~0.5 s | Short-term motion, peeks |
| +8 | ~1 s | Engagement-scale dynamics |
| +16 | ~2 s | Strategy-scale — rotations, executes, repositioning |

The hypothesis: **short horizons are dominated by inertia and reward a
model that learns physics; longer horizons force the model to encode
intent and tactics** because position alone no longer predicts where a
player will be in 2 seconds — you have to know what they are *trying to
do*. The horizon at which probe transfer peaks is itself a finding.

## 5. Distributional output head

The future is multimodal — a player at a chokepoint might push, hold,
or retreat. A plain regression (MSE) head averages those modes and
produces blurred, physically-implausible "average" states. To avoid
mode-averaging blur, the prediction head is **distributional**:

- **Discretize** continuous fields (position, velocity) into bins and
  predict a categorical over bins (LM-style), and/or
- **GMM** head — predict a mixture of Gaussians per continuous field.

The choice (discretize vs GMM, per-field) is an open design point; the
non-negotiable is that the head represents a *distribution* over next
states, not a point estimate. This is the same reason image/world-model
work (e.g. discretized tokens, DIAMOND's diffusion head) avoids plain
MSE.

## 6. Feature-engineering boundary: perception, not tactics

The single most important design rule, carried over from the
three-level framing:

**We engineer ONLY perception primitives (L1 / "See"). The model LEARNS
tactics (L2). No hand-engineered tactical labels enter the model.**

- Borrow from **MLMove** the per-player token layout and **derived
  visibility** (who can see whom, line-of-sight from positions + view
  angles + map geometry). Visibility is a *perception* primitive — it
  is what a player would perceive — so computing it for the model is
  fair game.
- Do **not** engineer "this is an execute," "this is a retake," "this
  is a default." Those are tactical abstractions; if they are real,
  the model should discover them as structure in its latent. Injecting
  them as features would be the same circularity that sank the
  Claude-caption supervision.

The line: if a primitive is something the player's *senses* deliver
(position, what's visible, what's audible-in-principle), engineer it.
If it is an *interpretation* of the situation, let the model learn it.

## 7. Evaluation: probe transfer, not prediction loss

Acceptance is the existing probe / σ_s discipline from
`docs/methodology.md`, retargeted from the round-encoder latent onto
the world-model latent:

- **Probe battery** (methodology axis 2). Freeze the world model; train
  tiny MLP probes on its latent for `round_won`, `pro_action_next`, and
  forward-state. Pass thresholds carry over (`probe_outcome` ≥ 0.65,
  `probe_action` ≥ 0.45, `probe_next` R² ≥ 0.50). The decisive
  question vs the round encoder: does next-state pretraining push
  probe accuracy *above* the saturation ceiling
  (`docs/learning-curve-finding.md`) at fixed data, especially on
  multi-class / forward targets that `round_won` could not move?
- **σ_s** (methodology axis 1). Neighbor-outcome variance on world-model
  latents, same Goldilocks band [0.15, 0.45], same
  `scripts/recall_variance_diagnostic.py` infrastructure.
- **Encoder disagreement** (methodology axis 5) as a tiebreaker.

Prediction loss (next-frame NLL / per-field accuracy) is reported as a
training-health metric, **not** an acceptance gate. A model that wins on
prediction loss but fails probe transfer has learned physics, not
understanding.

## 8. What derives from the world model

The world model is the substrate; everything else is a head or a
read-out on its latent.

- **Events fall out of prediction surprise.** An event is where the
  model is surprised — where the realized next state has low likelihood
  under the prediction. This *subsumes the change-point work*: instead
  of fitting statistical boundaries (which found non-semantic splits),
  semantically meaningful moments (a kill, a flash, a fake) are exactly
  the high-surprise frames. No separate change-point objective.
- **Value / policy = heads on the latent (MuZero-style).** A value head
  predicts `round_won`; a policy head predicts the pro's next action.
  These are downstream heads on a frozen (or lightly adapted) latent,
  not part of the dynamics objective — keeping outcome out of the
  representation-learning loop.
- **Reasoning = verbalizing rollouts.** With a learned dynamics model
  you can roll the latent forward and have the language bridge (§9)
  describe what it predicts will happen — "they're going to fake A and
  rotate B." This is the L3 "Think" step, now grounded in an actual
  forward model rather than a single-state advice generator.

## 9. Language bridge (phase 2)

Language is **deferred to phase 2** and bridges a *frozen* LLM into the
world-model latent. The world-model latent is far out-of-distribution
for a text LLM, so we start Flamingo-style and graduate:

1. **Flamingo-style bridge.** A resampler over world-model latents +
   gated cross-attention into a frozen **Qwen 3.6 / 3.7 (35B-A3B MoE)**,
   LoRA on the LLM. Gated cross-attn lets the frozen LLM ignore the
   latent until the bridge learns to inject useful signal — appropriate
   when the latent is far OOD.
2. **Graduate to Mixture-of-Transformers (MoT)** once the bridge is
   working — deeper integration than cross-attention bolted on.

Training stages for the bridge:

1. **Templated grounding** — map latent → templated descriptions of
   state (positions, economy) to teach the LLM to read the latent.
2. **Contrastive commentary** — align latent windows with real caster
   commentary (the parked grounding line from
   `project_commentary_grounding`; global align was strong at 4.6σ,
   per-event ASR-limited).
3. **GRPO reasoning** — reward = verbalized prediction vs actual future.
   The model says what will happen; we score it against what *did*
   happen in the demo. This is a grounded, non-circular reasoning
   reward (contrast the prior Claude-judge / RECALL rewards in
   `docs/reward-candidates.md`, which had no ground-truth future to
   check against).

**Discipline: ABLATE the latent.** Every language result must be run
with the world-model latent removed (zeroed / shuffled) to prove the
language head is using the latent and not just the LLM's prior. This is
the language-side analogue of the probe-transfer gate.

## 10. Data

- **85 demos, 81 parsed.**
- **597-d tensors built** (feature_schema_v2).

Per `docs/learning-curve-finding.md`, data scale is *not* the bottleneck
for the saturated recipe — so the bet here is squarely on the denser
objective, not on more demos. More / more-diverse demos (different era,
skill bracket, MR12) remain a distribution-shift lever for later, but
they are not the reason to expect the world model to beat the saturation
ceiling.

## 11. Architecture reuse

The world model reuses the round encoder's backbone: a **causal
transformer with RoPE** over per-frame feature vectors (see
`docs/round-encoder-design.md` §3, now marked superseded as a *design
direction* but kept as the reusable architecture). What changes is the
objective (dense distributional next-state prediction vs the prior
five SSL heads), the scoping discipline (explicit round-as-document,
no cross-reset attention), and the downstream story (heads + language
bridge vs a frozen embedding feeding RECALL).

## 12. Open design questions

1. **Head form** — discretize vs GMM, per-field, bin resolution.
2. **Horizon** — single best k vs multi-horizon joint training vs
   the sweep as the deliverable.
3. **Visibility computation** — full ray-cast vs approximate; cost at
   8 Hz × 10 players × all-rounds.
4. **Latent read-out point** — which layer / pooling feeds the probes
   and the language bridge.
5. **Surprise calibration** — what likelihood threshold defines an
   "event," and does it recover awpy events (kills, plants) as a
   sanity check.
6. **Bridge depth** — how long to stay Flamingo-style before MoT.
