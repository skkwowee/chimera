# Canonical Next-State Model — Retrain Recipe (R0)

*The frozen spec for the ONE model we defend. Each knob is LOCKED with its reason.
Existing wm_3map* checkpoints are baselines only, never final. Written 2026-06-29.*

Corpus: clean (datasheet §5 exclusions) — 3,876 train / 705 val rounds. 8 Hz, round-scoped.

---

## Knob 1 — Horizon / prediction mechanism  ✅ LOCKED

**Decision: rollout-native. Train a single short-step distributional model; reach all
horizons by autoregressive sampling rollout. NOT direct t+k regression.**

- **Step size: k = 4 frames = 500 ms.** Coarser step accepted in exchange for less
  compounding drift per rolled step.
- **Head: distributional (97-class classify-then-refine) on player xy** — so each
  rollout step is *sampled*, giving a spread of trajectories, not a blurred mean.
- **Training: rollout-aware (scheduled sampling)** — feed the model its own predictions
  for short rollouts during training, so it learns to consume its own outputs and the
  train/test mismatch that kills naive 1-step rollout is mitigated.
- **Eval sweep {1,2,5,7,10}s = rollout depths {2,4,10,14,20} steps.**
  Metric: **coverage (minADE-K over K sampled rollouts)**, not point error — past ~4 s
  the honest claim is "the true future is in the sampled set," not "we hit it."
- **This rollout group is also what Phase-3 GRPO consumes** ("generate the group").

**Why rollout, not direct multi-horizon heads (rationale RESTATED 2026-07-18,
first-principles-plan CHANGE G):** the model-free `res/d` analysis (val_v3,
normalized coords: const-velocity residual/displacement = 0.68 @1s, 0.87 @2s,
1.22 @5s, 1.38 @7s, 1.57 @10s) establishes that motion is momentum-trivial <2s
and decision-dominated >4s — motivating distributional, tactical-horizon
prediction. It does NOT establish that direct long-k heads fail: an earlier
version claimed "a direct long-k head can only learn the average (mode
collapse)" — that is TRUE for point regression but FALSE for multimodal anchor
heads (MultiPath/MTR predict 6–8s directly and own the minADE-K metric family).
The load-bearing argument for rollout is CONSUMPTION: (a) GRPO's reward needs
value on *intermediate* imagined frames — round value is path-dependent (deaths
are absorbing, ordering matters); (b) the bridge verbalizes *lines*, not
endpoints; (c) 10 coupled agents stay jointly coherent only under stepwise
re-simulation. Control (cheap, on the trained trunk): pilot a direct
winner-take-all anchor head at t+{40,80} and report it next to rollout coverage.

---

## Knob 2 — Feature schema (v2 597-d vs v3 687-d)  ✅ LOCKED

**Decision: train BOTH v2 and v3 under the identical recipe; v2 is canonical-by-default,
v3 is the ablation arm.**

- The v2-vs-v3 run is the long-overdue **perception deconfound** (every prior checkpoint
  was 687-d — no v2 baseline ever existed; the "v3 hurt value" claim was confounded with
  the data jump).
- **Validity clause:** the two runs must be bit-identical except the schema — same seed,
  same clean corpus, same rollout schedule, same probes — or the ablation is inadmissible.
- v2 canonical because the thesis ("prediction teaches tactics") is cleanest on raw state;
  perception is an *enhancement hypothesis*, not a dependency. If v3 wins on probe
  transfer, it earns canonical status with evidence.
- Perception dims legitimacy: passes Line-in-the-Sand (collision mesh is architecturally
  unavailable to the model; visibility is underivable from state).

**Perception grounding kept on awpy** (VisibilityChecker, .tri physics mesh, eye+64,
53° half-FOV, distance-gated 3500u; 91.2% kill-agreement). Alternatives considered and
rejected as not-guaranteed-better: VRF+own raycaster (same paradigm, new bug surface),
engine TraceRay (better ruler for the same narrow single-ray question; no smokes/doors),
`spotted` flag as truth (radar bookkeeping w/ hysteresis). **No accessible oracle for
"what a player perceives" exists — so triangulate proxies instead of swapping them:**
- V1a (cheap, CPU): awpy LOS vs demo `spotted` flag → precision/recall per map, for the
  datasheet. Two independent imperfect rulers agreeing = real evidence.
- The v2-vs-v3 ablation carries the final weight: if noisy LOS moves probe transfer it
  earned its place despite imperfection; if not, its accuracy is moot.
Known optimism: static mesh only (smokes/flashes/mollies not modeled), single eye-ray
(no shoulder-peek partials), constant eye height, yaw-only cone.

## Knob 3 — Velocity inputs (facing-shortcut fix / v4)  ✅ LOCKED (deferred)

**Decision: NO velocity dims in the canonical retrain.** The facing-shortcut concern was
measured at +3pp (the +27.2pp headline was never reproduced), was largely a short-horizon
artifact, and the 500 ms rollout + coverage objective (Knob 1) drains most of its power.
Run `facing_bias_check.py --corrupt-yaw` on the NEW canonical model; add velocity (v4)
only if the shortcut demonstrably survives. Evidence before rebuilds.
## Knobs 4–7 — ✅ LOCKED 2026-07-18 → see `docs/retrain-recipe-knobs4-7.md`

Judge-panel decisions (3 designers × rigor/simplicity/frugal priors + judge per
question, harmonized). Headlines:
- **Knob 4 (maps/OOD):** train on 5 maps (ancient/dust2/inferno/mirage/nuke =
  3,573/641 clean rounds); **de_overpass held out entirely** (367 rounds = OOD set).
  Coords stay global /3000. OOD decode: map one-hot ZEROED (primary), plus an
  ID-maps-zeroed-one-hot control that turns the protocol from assumption into
  measurement.
- **Knob 5 (heads):** trunk trained by next-state ONLY; value head kept but
  **detached (stop-grad)**, end-phase-masked, BCE 1.0 — closes the F2 circularity
  structurally (trunk is gradient-identical to value_weight=0; unit-test enforced).
  Dist edges REFIT for k=4 on the 5-map clean train split (rule pre-registered in
  fit_dist_edges.py). Alive-masked displacement loss (subsumes the 2026-07-05
  sub-item). `best_ns.pt` is canonical; `best.pt` retired.
- **Knob 6 (budget/SS/seeds):** scheduled sampling = **one-step sample-and-swap**
  (measured 1.27× teacher-forcing cost); fixed 25k steps, NO early stop; 3 paired
  model seeds × both schemas + one SS-off control (7 pod runs); 5 probe seeds;
  checkpoint selection on val next-state loss only (probes never touch selection).
- **Knob 7 (keystone L2):** retrain the round encoder on the clean 5-map corpus
  (v6 config verbatim); probe six frozen representations with a linear-only gating
  probe; C1 gates pre-registered (Δ ≥ +0.02, paired CI excludes 0, ≥4 of 5 maps,
  plus a scaling clause) with committed failure branches — C1-REP failure
  falsifies C1, no salvage wording. Historical 0.759 retired to history.
- **Compute:** ~12–20 local 4090-h + ~28–44 pod-h ≈ $45–85.

Amendment to Knob 1: the scheduled-sampling algorithm is now PINNED (one-step
sample-and-swap with p-ramp, per Knob 6) — resolving adversarial-review R1's
"unpinned" finding. Committing these docs constitutes the pre-registration.
