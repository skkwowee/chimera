# Chimera — Decisions Ledger (what we tried, killed, and kept)

The single source of truth for *why the project is shaped the way it is*. Every
entry is a decision backed by a measurement, not a hunch. When something here
conflicts with `claude-progress.txt`, this file wins for *rationale*; the
progress file wins for *current run state*. Add to this file when an idea is
killed, refined, or a methodology trap is found — not for routine progress.

Convention per entry: **what** · **verdict** · **evidence** · **lesson**.

---

## 0. The current pipeline (so every entry has context)

```
HLTV demos ─► tick state (687-d/frame) ─► WORLD MODEL ─► LANGUAGE BRIDGE ─► GRPO
                                          (Phase 1,        (Phase 2,         (Phase 3,
                                           ~done)           not started)      infra exists)
```
Bet: next-state prediction is dense/self-supervised/grounded, so we **learn the
game first**, then attach language (bridge) and reasoning quality (GRPO). Every
killed idea below is a thing that violated that bet or failed its own test.

---

## 1. KILLED (tried, disproven — stays dead)

### Visual SFT / captioning ("See, Then Think" VLM line)
- **What:** SFT a VLM on tactical captions of game states, hoping understanding
  would emerge as a byproduct, then GRPO it into reasoning.
- **Verdict:** DEAD. Redundant and circular.
- **Evidence:** captions were generated *from* the structured features that were
  also the input, so the target carried no new information — discriminative-check
  gap only **+0.008** over the structured-feature ceiling.
- **Lesson:** captioning creates understanding only when the target forces
  compression of a *richer* signal (true for noisy pixels, false for clean
  symbolic state). Perception was already solved by the parser; the task was
  lateral (features→prose), not forward. **Captioning was a probe that returned
  null — a footnote, not a pipeline stage.** It is NOT in the pipeline diagram.

### Naive VQ codebook (Path B, single code per player)
- **What:** VQ-VAE compressing each player's 56-d state to one codebook entry.
- **Verdict:** DEAD (as tested). Narrowly scoped — see refined entry §2.
- **Evidence:** position recon RMSE **423u** vs the ~50u bar (player radius 32u).
- **Lesson:** one discrete code can't hold continuous 2D position. This killed
  *single-code-per-player VQ*, NOT learned codebooks in general (the kill-test
  was scoped narrower than the conclusion was first written).

### cv-residual target (predict the correction over const-velocity)
- **What:** train the head to predict the tactical delta on top of a momentum prior.
- **Verdict:** DEAD — kept as a documented NEGATIVE RESULT.
- **Evidence:** better single-step aggregate (−16% vs −54%) but the momentum
  prior extrapolates into turns and is **unstable in closed loop**: 8s rollout
  drift **1235u vs 490u** for the plain-residual model (CV baseline 669u).
- **Lesson:** single-step loss gains ≠ simulator gains. Only closed-loop
  (rollout) eval catches this. Don't optimize a proxy the deployment doesn't use.

### Engine roaming to learn map boundaries
- **What:** drive bots around a CS2 server to teach the model walls.
- **Verdict:** KILLED for now (re-opens only with action-conditioning).
- **Evidence:** the `.tri` collision mesh *is* the boundary set, exact, on disk.
  Roaming would spend days rediscovering a file; and random-walk data pollutes
  the behavioral prior (the model learns physics ∘ *pro* policy, not random).
- **Lesson:** engineer ground-truth env data; don't relearn it. Roaming becomes
  the *right* data only for a future action-conditioned model (system-ID).

### Duration counters as input features (planned, pre-empted)
- **What:** hand-feed "time-stationary", "time-since-repositioned" per player.
- **Verdict:** KILLED before building — probe the latent instead.
- **Evidence:** computable from the model's own 96-frame window.
- **Lesson:** see the Line in the Sand (§4). A failed latent probe is a
  publishable model-deficiency finding, not a feature request.

---

## 2. REFINED / KEPT (survivors and why)

### World model = next-state prediction (the foundation)
- Causal spatiotemporal transformer, per-player-token factoring (10 + global),
  block-causal mask, round-scoped. Learns `P(next|past) = engine ∘ pro-policy`.
- **Why it's the right target:** the future is a *consequence*, not a
  *reformatting* — you can't cheat by paraphrasing; you must model dynamics.

### Distributional displacement head (97-class classify-then-refine)
- Replaced the continuous regression head, which mode-averaged (stationary
  jitter; turn means landing between modes).
- **Evidence:** argmax point-decode still loses to copy on decision frames (the
  *choice* isn't in the state — partly unwinnable), BUT minADE-16 sampling
  COVERS the modes (hard turn 42u vs copy 87u; reversal 38u vs 61u). medADE ≈
  argmax → healthy spread.
- **Lesson:** the win isn't better point prediction; it's representing the *set*
  of plausible futures. **16 samples = a GRPO group** — this is the concrete
  "generate the group" mechanism. (Also the cheap Path-B win the naive VQ missed:
  hand-designed codebook over *displacements*, not positions, + refine offset.)

### Value head co-trained on the latent (P(CT win), weight 0.3)
- **Why:** as the *primary* signal value starved (Era-1, ~1 bit/round); as a
  *co-trained head* on a dense foundation it works and becomes GRPO's reward.
- **Gate it passed:** value-through-rollout AUC stays ~flat through 8s of
  *imagined* frames (≈0.86) → rollouts carry value signal → GRPO viable.
- **Open caveat:** flatness is partly window-confounded — at depth 8 only 32/96
  frames are real (the earlier "88/96" was a step-vs-frame miscount; see
  methodology-review F4) — cite the gate with that caveat until tested at
  depth > window. NOTE (2026-07-18): the 0.856/0.865 gate numbers also pooled
  excluded/off-training maps (adversarial-review E1) — treat as historical until
  recomputed per-map on the clean corpus.

### v3 derived perception (LOS/FOV/exposure raycasts, input-only)
- Input-only (loss-masked); deterministic functions of state × the `.tri` mesh.
- **Refined caveat:** its headline value-transfer win is *confounded* with the
  value co-train (both shipped in the v2→v3 jump). The oft-cited "linear-probe
  parity 0.803 vs 0.807" was PHANTOM — never computed on any checkpoint
  (methodology-review F2); purged 2026-07-18. The v2-vs-v3 deconfound runs as
  part of the canonical retrain (retrain-recipe Knob 2).

### Geometry-gated decode (GeoGate)
- Inference-only constrained decoding: mask wall-infeasible displacement classes
  via the `.tri` BVH. Grammar-constrained decoding, grammar = ground-truth map.
- **Evidence:** 464u vs 487u @8s drift (~5%) AND wall-clipping impossible by
  construction. The learned model does ~95% of the work; the gate is a legality
  rail. (Veto-rate logging added per council requirement.)

### Adaptive "tactical clock" (surprise-gated frame selection)
- **Status:** RESCUED from frozen. Dist-head NLL = surprise; integrate-and-fire
  emits frames where the model is surprised, skips where it's not.
- **Evidence:** event-detection AUC 0.519 (summed, broken) → **0.698**
  (per-alive-mean, corrected) — cleared the council's ~0.65 gate.
- **Lesson:** see §3 — the metric, not the idea, was broken.

### NLA reconstruction objective (text-only decoder → faithfulness leg)
- **What:** close the latent→text bridge into a Natural-Language Autoencoder
  (Anthropic NLA: the bottleneck is *text*). The bridge is the encoder
  (latent→reasoning text); ADD a **text-only decoder** that reconstructs the
  frozen world-model latent — headline target the pooled-512 `z = h.mean(dim=2)`,
  the exact vector `value_head` reads — from Qwen's *generated* text. Decoder reads
  the re-tokenized output STRING only (firewall: zero tensor path from the latent /
  soft tokens, enforced by assertion + unit test). Reconstruction fidelity
  (cosine/variance-explained vs the true latent) is the metric.
- **Verdict:** ADOPTED — **as a metric first** (separate, frozen-verbalizer,
  held-out decoder; `λ=0` for the milestone). Aux/RL legs (small annealed λ
  regularizer; Track-2 REINFORCE) are gated behind milestone success. In GRPO it is
  a **constraint-filter** (zero the advantage of completions below τ before
  group-norm), NOT a summed reward — value-through-rollout stays the sole quality
  signal.
- **Evidence (design-stage, gates defined):** it supplies the faithfulness leg
  **ablate-the-latent alone could not** — ablate proves the latent is USED, recon
  proves the OUTPUT TEXT *faithfully renders* it vs hallucinating past it; the two
  are orthogonal and both ship. Green light is now TWO numbers: `latent-on >
  latent-off` AND `recon(real text) > recon(shuffled / ablated-bridge text)` above
  the latent-mean capacity floor, with value-head-agreement and readability intact.
  A cheap LOCAL capacity kill-test (text-budget × target sweep on cached latents,
  no pod) can falsify it before any QLoRA spend.
- **Lesson:** it is a **label-free** training/eval signal — reward = "does the text
  preserve enough to rebuild the latent," **no answer key** — so it sidesteps the
  "where do correct descriptions come from / is the teacher LLM stable enough to
  read raw ticks" problem (LLMs are bad at tick-state — that is WHY the world model
  exists; the NLA loop needs no teacher). Caveats it must carry: recon-faithful ≠
  good reasoning (it certifies info-preservation only — GRPO owns reasoning quality,
  templated grounding owns format/vocab); never report fidelity bare (pair with
  value-head-agreement + readability or it can be gamed by steganographic gibberish);
  reconstruct the **pre-augmentation** raw latent (not the appended value/rollout
  channels) and weight toward predictive channels, or it inherits the circularity it
  was built to detect.

---

## 3. EVALUATION LESSONS (the methodology scars)

These are the traps that produced *wrong green numbers*. They cost the most and
are the easiest to repeat.

- **Selection effect in decision_eval (the big one):** the original bucket key
  WAS const-vel's own error, so within-bucket comparisons were mechanically
  biased — any constant-error predictor showed the same −915%/+39% pattern. The
  "+39–57% learned tactics" claim was partly an artifact. **Fix:** bucket by
  truth-trajectory turn-angle (no predictor's error in the key); compare vs the
  BEST of {copy, const-vel, smoothed-CV, damped-CV}. Under the honest eval, the
  tactics claim *died* (copy wins on turns).
- **Summed vs per-alive-mean surprise:** summed NLL scales with alive count, so
  kills mechanically *depress* post-kill surprise → falsely killed the event
  signal (AUC 0.519). Per-alive-mean fixed it (0.698). **Lesson:** any per-frame
  aggregate must be invariant to the thing that changes during the event.
- **Read value from best.pt, not best_ns.pt:** value AUC peaks early then the
  head overfits; the late next-state checkpoint has a saturated value head
  (viewer once showed 2% CT at round start because of this).
- **Quarantine thin slices:** the dust2 per-map decision number came from ONE
  match (18 val rounds). Don't report per-map until the slice is real (now 163).
- **Closed-loop > single-step:** cv-residual looked good single-step, diverged in
  rollout. The deployment metric (rollout drift / value-through-rollout) is the
  one that counts.
- **Adversarial / systematic > anecdote:** "spike at 43s = a kill" (cherry-picked
  from one round) was overturned by the 387-kill systematic test. Always run the
  population version before believing the example.

---

## 4. THE LINE IN THE SAND (apply to every future bias proposal)

> **Engineer only what the model architecturally cannot see; never what it
> merely fails to compute.**

A candidate input feature ships iff ALL four hold:
1. Deterministic function of *(current state × environment data outside the
   model's inputs)* — the `.tri` mesh qualifies; the model's own 96-frame window
   does NOT.
2. Input-only and loss-masked (never a training target).
3. Arrives with a shortcut audit (facing-bias-style conflict test + input-
   corruption eval) AND a measured *closed-loop* delta (rollout/value, not
   single-step loss).
4. Re-derived live from predicted state during rollout — no frozen observations.

If the quantity is computable from the model's own context: **probe the latent
instead.** A failed probe is a publishable finding, not a feature request.

Applied: wall-rays → passes (1), blocked on (3) until GeoGate veto-rate measured.
Duration counters → fail (1), probe. Δz classes → output parameterization, judge
by coverage eval. Ground-snap → decode-time env query, allowed inference-only.

Known SUSPECTED violation: **the facing shortcut.** Yaw is a direct input
(dims 3-6), velocity has NO input dims — the asymmetry is real. But the
magnitude is UNVERIFIED: the "+27.2pp" headline was never reproduced (cached
run shows +3pp) and the yaw-shuffle causal test has NEVER been run — an earlier
version of this entry claimed "+19.3pp, causal, confirmed," a number with no
provenance; struck 2026-07-18 (adversarial-review T2). Controlling docs:
datasheet §7 + retrain-recipe Knob 3 — velocity (v4) is DEFERRED until the
yaw-shuffle test on the NEW canonical model shows the shortcut survives.

---

## 5. CONCEPTUAL CLARIFICATIONS (so we stop re-deriving them)

- **SFT-as-understanding (dead) vs SFT-as-translation (kept).** Era-1 SFT asked
  language to *create* understanding from captioning → circular. Bridge SFT only
  *translates* a latent that already understands (Phase 1 baked it). Same word,
  opposite job. Don't conflate them.
- **The bridge is the encoder half of a Natural Language Autoencoder; adding the
  decoder half (text → reconstruct the latent) closes the faithfulness loop.** (§2
  NLA entry.)
- **CS2-from-demos is state-modeling, not perception.** The parser hands us
  complete privileged state for free. Perception = solved; understanding =
  learned by next-state; language = bridge; reasoning = GRPO.
- **Vision is parked, not deleted.** Its only legitimate role is *tick-less
  inference* (live VOD → estimate tick-state → feed the same world model). A
  swappable front-end codec, not a core stage. Well-posed via paired VOD↔demo.
- **GRPO supervises its own reasoning.** Group = sampled world-model rollouts
  (not retrieval/RECALL); reward = value-through-rollout. No human reasoning
  labels needed for RL — only the *bridge bootstrap* SFT needs some grounding.
- **The model is "world + pro-players," not pure physics.** It learns
  dynamics ∘ pro-policy. That's a feature for value/reasoning, a caveat for any
  future action-conditioned (counterfactual) variant.
