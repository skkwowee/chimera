# Chimera — Decisions Ledger

The rationale ledger: what we tried, killed, and kept, with the measurement
behind each verdict. Read it before proposing a feature or re-opening an old
decision. Where it conflicts with `claude-progress.txt`, this file wins on
rationale; the progress file wins on current run state. Add an entry when an
idea is killed, refined, or a methodology trap is found — not for routine
progress.

Entry convention: **what** · **verdict** · **evidence** · **lesson**.
Numbers struck or re-scoped after audit live in the corrections log (§6); the
entries above it already state the corrected facts.

---

## 0. The current pipeline (context for every entry)

```
HLTV demos ─► tick state (687-d/frame) ─► world model ─► language bridge ─► GRPO
                                          (Phase 1,      (Phase 2,          (Phase 3,
                                           ~done)         not started)       infra exists)
```

The bet: next-state prediction is dense, self-supervised, and grounded, so we
learn the game first, then attach language (the bridge) and reasoning quality
(GRPO — Group Relative Policy Optimization). Every killed idea below either
violated that bet or failed its own test.

---

## 1. Killed (tried, disproven — stays dead)

### Visual SFT / captioning (the "See, Then Think" VLM line)
- **What:** supervised fine-tuning (SFT) of a vision-language model on tactical
  captions of game states, hoping understanding would emerge as a byproduct,
  then GRPO on top.
- **Verdict:** dead — redundant and circular.
- **Evidence:** captions were generated *from* the same structured features the
  model received as input, so the target carried no new information —
  discriminative-check gap of only +0.008 over the structured-feature ceiling.
- **Lesson:** captioning creates understanding only when the target forces
  compression of a *richer* signal (true for noisy pixels, false for clean
  symbolic state). The parser had already solved perception; the task was
  lateral (features → prose), not forward. Captioning was a probe that
  returned null — a footnote, not a pipeline stage, and absent from the
  pipeline diagram.

### Naive VQ codebook (Path B, single code per player)
- **What:** a VQ-VAE compressing each player's 56-d state to one codebook entry.
- **Verdict:** dead as tested — a narrowly scoped kill (see the distributional
  head in §2 for the Path-B idea done right).
- **Evidence:** position reconstruction RMSE of 423u against the ~50u bar
  (player radius 32u).
- **Lesson:** one discrete code cannot hold continuous 2D position. This killed
  single-code-per-player VQ, not learned codebooks in general (corrections
  log, entry 6).

### cv-residual target (predict the correction over constant velocity)
- **What:** train the head to predict the tactical delta on top of a momentum
  prior.
- **Verdict:** dead — kept as a documented negative result.
- **Evidence:** better single-step aggregate (−16% vs −54%), but the momentum
  prior extrapolates into turns and is unstable in closed loop: 8s rollout
  drift of 1235u vs 490u for the plain-residual model (constant-velocity
  baseline 669u).
- **Lesson:** single-step loss gains are not simulator gains; only closed-loop
  (rollout) evaluation catches this. Don't optimize a proxy the deployment
  doesn't use.

### Engine roaming to learn map boundaries
- **What:** drive bots around a CS2 server to teach the model walls.
- **Verdict:** killed for now; re-opens only with action-conditioning.
- **Evidence:** the `.tri` collision mesh *is* the boundary set — exact, on
  disk. Roaming would spend days rediscovering a file, and random-walk data
  pollutes the behavioral prior (the model should learn physics ∘ *pro*
  policy, not random).
- **Lesson:** engineer ground-truth environment data; don't relearn it.
  Roaming becomes the *right* data only for a future action-conditioned model
  (system identification).

### Duration counters as input features (planned, pre-empted)
- **What:** hand-feed "time stationary" and "time since repositioned" per
  player.
- **Verdict:** killed before building — probe the latent instead.
- **Evidence:** both are computable from the model's own 96-frame window, so
  they fail criterion 1 of the line in the sand (§4).
- **Lesson:** a failed latent probe is a publishable model-deficiency finding,
  not a feature request.

---

## 2. Refined / kept (survivors and why)

### World model = next-state prediction (the foundation)
A causal spatiotemporal transformer with per-player-token factoring (10 players
plus a global token), a block-causal mask, round-scoped sequences. It learns
`P(next|past) = engine ∘ pro-policy`. It is the right target because the future
is a *consequence*, not a *reformatting*: you cannot cheat by paraphrasing; you
must model dynamics.

### Distributional head (97-class classify-then-refine displacement)
Replaced the continuous regression head, which mode-averaged (stationary
jitter; turn means landing between modes).
- **Evidence:** argmax point-decode still loses to copy on decision frames —
  the *choice* isn't in the state, so that comparison is partly unwinnable —
  but minADE-16 sampling *covers* the modes (hard turn 42u vs copy's 87u;
  reversal 38u vs 61u), and medADE ≈ argmax, so the spread is healthy.
- **Lesson:** the win is not better point prediction; it is representing the
  *set* of plausible futures. Sixteen samples make one GRPO group — this is
  the concrete "generate the group" mechanism. It is also the cheap Path-B win
  the naive VQ missed: a hand-designed codebook over *displacements*, not
  positions, plus a refinement offset.

### Value head co-trained on the latent (P(CT win), weight 0.3)
- **Why:** as the *primary* signal, value starved (Era 1: ~1 bit/round). As a
  co-trained head on a dense foundation it works, and value-through-rollout
  becomes GRPO's group generator (the grounded reward does the scoring —
  bridge-design §5 and the GRPO clarification in §5 below).
- **Gate it passed:** value-through-rollout AUC stays roughly flat through 8s
  of *imagined* frames (≈0.86), so rollouts carry value signal and GRPO is
  viable.
- **Open caveats:** the flatness is partly window-confounded — at depth 8 only
  32 of 96 frames are real (corrections log, entry 3) — so cite the gate with
  that caveat until it is tested at depth greater than the window. The
  0.856/0.865 gate numbers also pooled excluded and off-training maps; treat
  them as historical until recomputed per-map on the clean corpus
  (corrections log, entry 4).

### v3 derived perception (LOS/FOV/exposure raycasts, input-only)
Input-only and loss-masked; deterministic functions of state × the `.tri` mesh.
- **Caveat:** its headline value-transfer win is *confounded* with the value
  co-train — both shipped in the v2→v3 jump. The once-cited linear-probe
  parity number was phantom and has been purged (corrections log, entry 1).
  The v2-vs-v3 deconfound runs as part of the canonical retrain
  (retrain-recipe Knob 2).

### Geometry-gated decode (GeoGate)
Inference-only constrained decoding: mask wall-infeasible displacement classes
via the `.tri` BVH — grammar-constrained decoding where the grammar is the
ground-truth map.
- **Evidence:** 464u vs 487u drift at 8s (~5%), and wall-clipping is impossible
  by construction. The learned model does ~95% of the work; the gate is a
  legality rail. Veto-rate logging was added per council requirement.

### Adaptive tactical clock (surprise-gated frame selection)
Rescued from frozen. Distributional-head NLL is the surprise signal;
integrate-and-fire emits frames where the model is surprised and skips where it
is not.
- **Evidence:** event-detection AUC went from 0.519 (summed — a broken metric)
  to 0.698 (per-alive mean — corrected), clearing the council's ~0.65 gate.
- **Lesson:** the metric, not the idea, was broken (§3).

### NLA reconstruction objective (text-only decoder → the faithfulness leg)
- **What:** close the latent→text bridge into a Natural-Language Autoencoder
  (NLA, per Anthropic: the bottleneck is *text*). The bridge is the encoder
  (latent → reasoning text); we add a text-only decoder that reconstructs the
  frozen world-model latent — headline target the pooled-512
  `z = h.mean(dim=2)`, the exact vector the value head reads — from Qwen's
  *generated* text. The decoder reads the re-tokenized output string only; a
  firewall guarantees zero tensor path from the latent or soft tokens,
  enforced by assertion and unit test. The metric is reconstruction fidelity
  (cosine / variance explained vs the true latent).
- **Verdict:** adopted — as a metric first (a separate, frozen-verbalizer,
  held-out decoder; λ=0 for the milestone). The auxiliary and RL legs (a small
  annealed λ regularizer; Track-2 REINFORCE) are gated behind milestone
  success. In GRPO it acts as a *constraint filter* — zero the advantage of
  completions below τ before group normalization — not a summed reward. The
  quality signal is the *grounded* reward: claims scored against the actual
  demo future (CRPS/Brier). Value-through-rollout is model-authored and
  circular, so it only generates the group (bridge-design §5; corrections
  log, entry 5).
- **Evidence (design-stage, gates defined):** it supplies the faithfulness leg
  that ablate-the-latent alone could not — ablation proves the latent is
  *used*; reconstruction proves the output text *faithfully renders* it rather
  than hallucinating past it. The two checks are orthogonal and both ship. The
  green light is two numbers: `latent-on > latent-off`, and
  `recon(real text) > recon(shuffled / ablated-bridge text)` above the
  latent-mean capacity floor, with value-head agreement and readability
  intact. A cheap local capacity kill-test (text-budget × target sweep on
  cached latents, no pod) can falsify it before any QLoRA spend.
- **Lesson:** it is a label-free training/eval signal — the reward is "does
  the text preserve enough to rebuild the latent," with no answer key — so it
  sidesteps the "where do correct descriptions come from / is the teacher LLM
  stable enough to read raw ticks" problem. (LLMs are bad at tick state; that
  is *why* the world model exists. The NLA loop needs no teacher.) Caveats it
  must carry: reconstruction-faithful is not good reasoning — it certifies
  information preservation only; GRPO owns reasoning quality, templated
  grounding owns format and vocabulary. Never report fidelity bare — pair it
  with value-head agreement and readability, or it can be gamed by
  steganographic gibberish. Reconstruct the *pre-augmentation* raw latent (not
  the appended value/rollout channels) and weight toward predictive channels,
  or it inherits the circularity it was built to detect.

---

## 3. Evaluation lessons (the methodology scars)

These are the traps that produced *wrong green numbers*. They cost the most and
are the easiest to repeat.

- **Selection effect in decision_eval (the big one).** The original bucket key
  *was* const-vel's own error, so within-bucket comparisons were mechanically
  biased — any constant-error predictor showed the same −915%/+39% pattern,
  and the "+39–57% learned tactics" claim was partly an artifact. Fix: bucket
  by truth-trajectory turn angle (no predictor's error in the key) and compare
  against the best of {copy, const-vel, smoothed-CV, damped-CV}. Under the
  honest eval, the tactics claim *died* — copy wins on turns.
- **Summed vs per-alive-mean surprise.** Summed NLL scales with alive count,
  so kills mechanically *depress* post-kill surprise, which falsely killed the
  event signal (AUC 0.519). The per-alive mean fixed it (0.698). Any per-frame
  aggregate must be invariant to the thing that changes during the event.
- **Read value from `best.pt`, not `best_ns.pt`.** Value AUC peaks early, then
  the value head overfits; the late next-state checkpoint has a saturated
  value head (the viewer once showed 2% CT at round start because of this).
- **Quarantine thin slices.** The dust2 per-map decision number came from one
  match (18 val rounds). Don't report per-map numbers until the slice is real
  (now 163).
- **Closed-loop beats single-step.** cv-residual looked good single-step and
  diverged in rollout. The deployment metric — rollout drift,
  value-through-rollout — is the one that counts.
- **Systematic beats anecdote.** "Spike at 43s = a kill," cherry-picked from
  one round, was overturned by the 387-kill systematic test. Run the
  population version before believing the example.

---

## 4. The line in the sand (apply to every future bias proposal)

> Engineer only what the model architecturally cannot see; never what it
> merely fails to compute.

A candidate input feature ships only if all four hold:

1. It is a deterministic function of *(current state × environment data
   outside the model's inputs)*. The `.tri` mesh qualifies; the model's own
   96-frame window does not.
2. It is input-only and loss-masked — never a training target.
3. It arrives with a shortcut audit (a facing-bias-style conflict test plus an
   input-corruption eval) *and* a measured closed-loop delta (rollout/value,
   not single-step loss).
4. It is re-derived live from predicted state during rollout — no frozen
   observations.

If the quantity is computable from the model's own context, probe the latent
instead. A failed probe is a publishable finding, not a feature request.

Applied so far: wall-rays pass (1) but are blocked on (3) until the GeoGate
veto rate is measured. Duration counters fail (1) — probe. Δz classes are an
output parameterization — judge by coverage eval. Ground-snap is a decode-time
environment query, allowed inference-only.

One suspected violation: **the facing shortcut**. Yaw is a direct input (dims
3–6) while velocity has no input dims, so the asymmetry is real — but its
magnitude is unverified: the "+27.2pp" headline was never reproduced (a cached
run shows +3pp), and the yaw-shuffle causal test has never been run
(corrections log, entry 2). Controlling docs: datasheet §7 and retrain-recipe
Knob 3 — velocity (v4) is deferred until the yaw-shuffle test on the new
canonical model shows the shortcut survives.

---

## 5. Conceptual clarifications (so we stop re-deriving them)

- **SFT-as-understanding (dead) vs SFT-as-translation (kept).** Era-1 SFT
  asked language to *create* understanding from captioning — circular. Bridge
  SFT only *translates* a latent that already understands (Phase 1 baked it).
  Same word, opposite job. Don't conflate them.
- **The bridge is the encoder half of an NLA; the text-only decoder closes the
  faithfulness loop** (§2).
- **CS2-from-demos is state-modeling, not perception.** The parser hands over
  complete privileged state for free. Perception is solved; understanding is
  learned by next-state prediction; language is the bridge; reasoning is GRPO.
- **Vision is parked, not deleted.** Its only legitimate role is *tick-less
  inference* — live VOD → estimated tick state → the same world model. A
  swappable front-end codec, not a core stage; well-posed via paired VOD↔demo.
- **GRPO supervises its own reasoning.** The group is sampled world-model
  rollouts (not retrieval/RECALL). The reward is grounded: verbalized claims
  scored against the realized demo future (CRPS/Brier). Value-through-rollout
  generates the group; it does not score it — scoring with it would be
  circular. No human reasoning labels are needed for RL; only the *bridge
  bootstrap* SFT needs some grounding.
- **The model is "world + pro players," not pure physics.** It learns
  dynamics ∘ pro-policy — a feature for value and reasoning, a caveat for any
  future action-conditioned (counterfactual) variant.

---

## 6. Corrections log

Numbers and claims struck or re-scoped after audit. The entries above already
state the corrected facts; this log exists for auditability.

1. **2026-07-18 — phantom linear-probe parity (v3 perception, §2).** The
   oft-cited "0.803 vs 0.807" parity was never computed on any checkpoint
   (methodology-review F2). Purged.
2. **2026-07-18 — facing-shortcut magnitude (§4).** An earlier entry claimed
   "+19.3pp, causal, confirmed" — a number with no provenance
   (adversarial-review T2). Struck. Verified state: the +27.2pp headline was
   never reproduced, a cached run shows +3pp, and the yaw-shuffle test has
   never been run.
3. **Step-vs-frame miscount in the value gate (§2).** An earlier caveat said
   88 of 96 frames are real at rollout depth 8; the correct count is 32 of 96
   (methodology-review F4).
4. **2026-07-18 — value-gate pooling (§2).** The 0.856/0.865
   value-through-rollout gate numbers pooled excluded and off-training maps
   (adversarial-review E1). Historical until recomputed per-map on the clean
   corpus.
5. **2026-07-18 — CHANGE A (GRPO reward, §2/§5).** The quality signal changed
   from value-through-rollout (model-authored, circular) to the grounded
   reward — claims scored against the actual demo future (CRPS/Brier) —
   with value-through-rollout demoted to group generator (bridge-design §5).
6. **Naive-VQ kill scope (§1).** The kill conclusion was first written broader
   than the test: the 423u result kills single-code-per-player VQ only, not
   learned codebooks in general.
