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

**Why not direct multi-horizon heads:** model-free `res/d` analysis (val_v3, normalized
coords) shows const-velocity residual/displacement = 0.68 (1s), 0.87 (2s), 1.22 (5s),
1.38 (7s), 1.57 (10s). Above 1.0 momentum is worse than "don't move" → a direct long-k
head can only learn the average (mode collapse). Direct targets are only sound in the
{1,2}s band; rollout+sampling degrades gracefully (spreads) instead of collapsing.
The `res/d` curve is itself a headline figure: motion is momentum-trivial <2s and
decision-dominated >4s, motivating a distributional, tactical-horizon objective.

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
## Knob 4 — Maps (3-map dense vs all-map + OOD holdout)  ⬜ OPEN
## Knob 5 — Heads (dist + value weight)  ⬜ OPEN (value not horizon-indexed: one head, current frame)
Locked sub-item from the 2026-07-05 schema review: **alive-mask the displacement loss**
(exclude dead players' xy from next-state/dist targets — they're frozen bodies; ~13% of
player-frames are wasted capacity otherwise). One-line change, must be in the canonical run.
## Knob 6 — Budget / early-stop (existing stopped at 500–1500 steps — likely undertrained)  ⬜ OPEN
