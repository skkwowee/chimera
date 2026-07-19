# Canonical Next-State Model — Retrain Recipe (R0)

This is the pre-registered training contract for the one world model we defend.
Read it before any retrain, ablation, or eval run; no knob unlocks without
evidence hitting a written switch trigger. Existing wm_3map* checkpoints are
baselines only, never final.

Corpus: clean (datasheet §5 exclusions) — 3,876 train / 705 val rounds. 8 Hz,
round-scoped. Committing this doc and `retrain-recipe-knobs4-7.md` constitutes
the pre-registration.

---

## Knob 1 — Horizon / prediction mechanism — LOCKED

**Decision: rollout-native.** Train a single short-step distributional model and
reach all horizons by autoregressive sampling rollout — not direct t+k
regression.

- **Step size: k = 4 frames = 500 ms.** A coarser step is accepted in exchange
  for less compounding drift per rolled step.
- **Head: distributional head (97-class classify-then-refine) on player xy.**
  Each rollout step is *sampled*, giving a spread of trajectories rather than a
  blurred mean.
- **Training: rollout-aware scheduled sampling**, pinned as one-step
  sample-and-swap with p-ramp (see Knob 6). The model consumes its own
  predictions during short training rollouts, mitigating the train/test
  mismatch that kills naive 1-step rollout.
- **Eval sweep: {1,2,5,7,10} s = rollout depths {2,4,10,14,20} steps.** Metric:
  coverage (minADE-K over K sampled rollouts), not point error — past ~4 s the
  honest claim is "the true future is in the sampled set," not "we hit it."
- This rollout group is also what Phase-3 GRPO consumes ("generate the group").

**Why rollout, not direct multi-horizon heads** (first-principles-plan CHANGE G):
the model-free `res/d` analysis (val_v3, normalized coords: const-velocity
residual/displacement = 0.68 @1s, 0.87 @2s, 1.22 @5s, 1.38 @7s, 1.57 @10s)
establishes that motion is momentum-trivial below 2 s and decision-dominated
past 4 s — motivating distributional, tactical-horizon prediction. It does
*not* establish that direct long-k heads fail: "a direct long-k head can only
learn the average (mode collapse)" is true for point regression but false for
multimodal anchor heads — MultiPath/MTR predict 6–8 s directly and own the
minADE-K metric family. The load-bearing argument for rollout is consumption:
(a) GRPO's reward needs value on *intermediate* imagined frames — round value
is path-dependent (deaths are absorbing, ordering matters); (b) the bridge
verbalizes *lines*, not endpoints; (c) 10 coupled agents stay jointly coherent
only under stepwise re-simulation. Control (cheap, on the trained trunk): pilot
a direct winner-take-all anchor head at t+{40,80} and report it next to rollout
coverage.

---

## Knob 2 — Feature schema (v2 597-d vs v3 687-d) — LOCKED

**Decision: train both v2 and v3 under the identical recipe; v2 is
canonical-by-default, v3 is the ablation arm.**

The v2-vs-v3 run is the long-overdue perception deconfound: every prior
checkpoint was 687-d, no v2 baseline ever existed, and the "v3 hurt value"
claim was confounded with the data jump. **Validity clause:** the two runs must
be bit-identical except the schema — same seed, same clean corpus, same rollout
schedule, same probes — or the ablation is inadmissible.

v2 is canonical because the thesis ("prediction teaches tactics") is cleanest
on raw state; perception is an *enhancement hypothesis*, not a dependency. If
v3 wins on probe transfer, it earns canonical status with evidence. The
perception dims are legitimate: they pass Line-in-the-Sand (the collision mesh
is architecturally unavailable to the model; visibility is underivable from
state).

Perception grounding stays on awpy (VisibilityChecker, .tri physics mesh,
eye+64, 53° half-FOV, distance-gated 3500u; 91.2% kill-agreement). Alternatives
were considered and rejected as not-guaranteed-better: VRF plus our own
raycaster (same paradigm, new bug surface), engine TraceRay (a better ruler for
the same narrow single-ray question; no smokes/doors), the demo `spotted` flag
as truth (radar bookkeeping with hysteresis). No accessible oracle for "what a
player perceives" exists, so we triangulate proxies instead of swapping them:

- **V1a** (cheap, CPU): awpy LOS vs demo `spotted` flag → precision/recall per
  map, for the datasheet. Two independent imperfect rulers agreeing is real
  evidence.
- The v2-vs-v3 ablation carries the final weight: if noisy LOS moves probe
  transfer, it earned its place despite imperfection; if not, its accuracy is
  moot.

Known optimism: static mesh only (smokes/flashes/mollies not modeled), single
eye-ray (no shoulder-peek partials), constant eye height, yaw-only cone.

---

## Knob 3 — Velocity inputs (facing-shortcut fix / v4) — LOCKED (deferred)

**Decision: no velocity dims in the canonical retrain.** The facing-shortcut
concern was measured at +3pp (the +27.2pp headline was never reproduced), was
largely a short-horizon artifact, and the 500 ms rollout plus coverage
objective (Knob 1) drains most of its power. Run
`facing_bias_check.py --corrupt-yaw` on the new canonical model; add velocity
(v4) only if the shortcut demonstrably survives. Evidence before rebuilds.

---

## Knobs 4–7 — LOCKED → `docs/retrain-recipe-knobs4-7.md`

Decided by judge panel (3 designers with rigor/simplicity/frugal priors plus a
judge per question, harmonized). Headlines; the linked doc is canonical:

- **Knob 4 (maps/OOD):** train on 5 maps (ancient/dust2/inferno/mirage/nuke =
  3,573/641 clean rounds); de_overpass held out entirely (367 rounds = OOD
  set). Coords stay global /3000. OOD decode: map one-hot zeroed (primary),
  plus an ID-maps-zeroed-one-hot control that turns the protocol from
  assumption into measurement.
- **Knob 5 (heads):** the trunk is trained by next-state only; the value head
  is kept but detached (stop-grad), end-phase-masked, BCE 1.0 — closing the F2
  circularity structurally (the trunk is gradient-identical to value_weight=0;
  unit-test enforced). Distributional-head edges are refit for k=4 on the
  5-map clean train split (rule pre-registered in `fit_dist_edges.py`).
  Alive-masked displacement loss. `best_ns.pt` is canonical; `best.pt` is
  retired.
- **Knob 6 (budget/SS/seeds):** scheduled sampling = one-step sample-and-swap
  (measured 1.27× teacher-forcing cost); fixed 25k steps, no early stop; 3
  paired model seeds × both schemas plus one SS-off control (7 pod runs); 5
  probe seeds; checkpoint selection on val next-state loss only (probes never
  touch selection).
- **Knob 7 (keystone L2):** retrain the round encoder on the clean 5-map
  corpus (v6 config verbatim); probe six frozen representations with a
  linear-only gating probe; the keystone (C1) gates are pre-registered
  (Δ ≥ +0.02, paired CI excludes 0, ≥4 of 5 maps, plus a scaling clause) with
  committed failure branches — C1-REP failure falsifies C1, no salvage
  wording. The historical 0.759 is retired to history.
- **Compute:** ~12–20 local 4090-h + ~28–44 pod-h ≈ $45–85.

## Checkpoint naming (fixed before any run exists)

All R1-era checkpoints land in the HF model repo `skkwowee/chimera-wm` under
`runs/`: the six canonical arms are `r1-v2-s{0,1,2}` and `r1-v3-s{0,1,2}`, the
scheduled-sampling control is `r1-v2-ssoff-s0`, the retrained round encoder is
`r1-re-v6`, and the matched-capacity supervised ceiling is `r1-sup-ceiling`.
Each run directory contains `best_ns.pt`, `last.pt`, `train.log`, and
`run_meta.json` (argv, resolved args, seed, git sha, config hash, corpus blob
sha256s from the manifest). Named here, before any artifact exists, so nothing
gets renamed mid-analysis.
