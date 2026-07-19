# Chimera R1 — Locked Recipe: Knobs 4–6 + Keystone Protocol (Knob 7)

**Pre-registered 2026-07-18. Git-committing this document (plus the keystone subset manifests) constitutes the pre-registration. All four sections apply bit-identically to both schema arms (v2 canonical, v3 ablation) under the Knob-2 validity clause.**

## 1. Executive summary

1. **Knob 4 (maps/OOD):** Canonical train = 5 maps (ancient, dust2, inferno, mirage, nuke; 3,573/641 clean rounds); de_overpass held out entirely (367 rounds = OOD set); coordinates stay global /3000.
2. — *Key validity clause:* the mandatory ID-maps-with-zeroed-map-ID control converts the zeroed-one-hot OOD decode from assumption to measurement; if it degrades ID coverage beyond CI, OOD rows are declared confounded and the pre-registered map-dropout fallback fires.
3. **Knob 5 (heads):** Trunk trained by next-state only; value head kept but detached (stop-grad), end-phase-masked, BCE 1.0; dist edges refit for k=4; alive-masked displacement loss; `best.pt` deleted, `best_ns.pt` canonical.
4. — *Key validity clause:* `tests/test_no_value_leak.py` proves zero outcome gradient reaches the trunk, making the trunk gradient-identical to value_weight=0 — F2 (circular outcome supervision) is closed structurally, no control arm needed.
5. **Knob 6 (budget/SS/seeds):** One-step sample-and-swap scheduled sampling (measured 1.27× TF), fixed 25,000 steps with NO early stop, 3 paired model seeds × both schemas + one SS-off control (7 pod runs), 5 probe seeds, selection on val next-state loss only.
6. — *Key validity clause:* the config-identical p=0 control means SS must beat teacher forcing on depth-{10,20} coverage or it ships as a reported negative result; probes never touch checkpoint selection (C1 anti-circularity firewall).
7. **Knob 7 (keystone L2):** Retrain the round encoder on the clean 5-map corpus (forward-MSE heads only, v6 config verbatim); probe six frozen representations with a linear-only gating probe; C1 gates on G0 ∧ C1-REP (Δ ≥ +0.02, paired CI excludes 0, ≥4 of 5 maps) ∧ C1-SCALE (Δ ≥ +0.01 from N_sat to N_full).
8. — *Key validity clause:* committed failure branches with no salvage wording — C1-REP failure falsifies C1 outright; the RE's saturation behavior is reported, never gated, so C1 is not hostage to baseline replication.
9. **Cross-cutting:** every gating number is computed on the 5-map clean corpus (3,573/641) with per-map reporting; overpass appears only in the Knob-4 OOD protocol; the canonical downstream checkpoint is v2 seed-0 `best_ns.pt`, declared before any probe result is seen.
10. **Total new compute:** ~12–20 local 4090-hours + ~28–44 pod GPU-hours ≈ **$45–85** (or ~$30–50 if the keystone WM curve runs as local overnights); see §4.

## 2. Cross-section harmonization — contradictions found and resolved

The four sections were judged independently; the following conflicts were found and resolved. Each resolution is applied inline below (marked **[H\#]**):

- **[H1] Corpus counts.** Knob 6 asserted 3,876/705; Knob 4 locks 3,573/641 (5 maps). *Resolved:* two-stage guard — `clean_blob` output must equal 3,876/705 (6-map clean invariant), then the `--maps` filter output must equal 3,573/641. All training and gating numbers use 3,573/641.
- **[H2] Dist edges.** Knob 5's `[8, 26, 47, 58, 85, 117]/123u` were fit on the 6-map clean corpus (29.5M pairs incl. overpass). Knob 5's own sequencing clause fires under Knob 4: *the edge fit MUST be re-run on the 5-map clean train split before launch.* The rule is pre-registered in `fit_dist_edges.py`; the quoted values are provisional until refit.
- **[H3] `--value-weight`.** Knob 5 retires the flag (detached head); Knob 7 mandated a `--value-weight 0` arm. *Resolved:* under Knob 5 the canonical trunk is already gradient-identical to vw=0 (unit-test enforced), so **the keystone arm IS the canonical model** — no separate vw=0 arm, no flag. The "co-trained twin" becomes one explicitly labeled extra run (head attached, weight 0.3, no detach), secondary-only, per Knob 5's contingency clause.
- **[H4] `best.pt`.** Knob 5 deletes it; Knobs 6/7 referenced it as "monitoring." *Resolved:* deleted. Monitoring = the logged detached end-masked value AUC + the 2.5k-step snapshots. Checkpoints = `best_ns.pt`, `last.pt`, `step_*.pt`.
- **[H5] Early stopping.** Knob 6 forbids it; Knob 7 said "canonical budget and early-stop rule." *Resolved:* fixed 25k steps at every scaling-grid point, no early stop; small-N is handled by `best_ns.pt` selection landing earlier on its own, not by stopping.
- **[H6] Keystone eval set.** Knob 7 was drafted on 6-map val (705 rounds, "≥4 of 6 maps," 14 val matches). *Resolved per Knob 7's own re-derivation clause:* gating val = the 641 clean 5-map rounds; map clause = ≥4 of 5 maps; val-match count for the cluster bootstrap re-computed and frozen at manifest time. Overpass probing lives exclusively in the Knob-4 OOD protocol (which already mirrors C1's learning-curve structure).
- **[H7] Historical 0.759.** Knob 7 retires it; Knob 4's OOD baselines updated to use the retrained clean-corpus RE (Knob 7a), 0.759 cited as history only.
- **[H8] Run-count reuse.** Keystone's N_full anchor (3 seeds) is satisfied by the three canonical v2 runs from Knob 6 — identical recipe, corpus, budget. WM curve = **7 new runs**, not 10.
- **[H9] Epoch arithmetic.** 25k × 64 crops ≈ **~14 epochs** of the 3,573×32 crop inventory (was "13 epochs of 3,876×32").
- **[H10] SS/alive-mask coherence.** Knob 5's scheduled-sampling clause (targets always real, alive-mask from real frames) and Knob 6's SS spec agree verbatim — confirmed, no change.

## 3. Locked recipe sections

## Knob 4 — Maps + cross-map OOD  ✅ LOCKED

**Decision: canonical train = 5 maps — `--maps de_ancient,de_dust2,de_inferno,de_mirage,de_nuke`
(3,573 train / 641 val clean rounds); de_overpass held out ENTIRELY. All 367 overpass rounds
(303 train-split + 64 val-split — none seen in training) form the OOD eval set, evaluated on the
canonical model itself. Applies identically to both schema arms (v2 AND v3 — Knob-2 validity
clause: the map set is part of "bit-identical except the schema").**

- **Holdout choice (ex-ante, documented before the run):** overpass is (i) the cheapest holdout
  (303 rounds = 7.8% of clean train vs nuke's 14%) and (ii) a mid-complexity, mostly-planar
  layout — so the OOD row measures *layout novelty*, not the verticality confound. Nuke stays IN
  training deliberately: it is the Z-outlier, the model needs that physics regime, and a nuke-OOD
  failure would be uninterpretable ("new map" vs "new physics"). Not anubis (D1: no map identity),
  not train (D2: 16 rounds).
- **Coordinates: global /3000 xy, /500 z — UNCHANGED.** (1) The locked Knob-1 head's ring edges
  (`DIST_EDGES_U`, game units, train_world_model.py:146) are physically meaningful only under one
  global scale; per-map normalization redefines the 97-class bin geometry map-by-map, silently
  unlocking Knob 1 and invalidating the res/d analysis. (2) Egocentric/relative coords = schema
  rebuild without a passing gate (datasheet §7) and break the Knob-2 clause. (3) No coordinate
  scheme transfers absolute layout to an unseen map — that limitation is inherent to the OOD
  question, not to /3000. Conceded up front: absolute positions fingerprint the map; the claim is
  scoped below that attack (see claim scope). **Egocentric gate (future work):** pilot only if
  OOD coverage skill < 0 vs copy/const-vel at ALL horizons while the fine-tune anchor recovers —
  that pattern would implicate coordinate-frame memorization specifically.
- **OOD decode (pre-registered): map one-hot ZEROED is primary** (global dims
  [10·ppd + 0 : 10·ppd + 7] := 0; v2 560–566, v3 650–656; de_overpass = MAP_VOCAB index 5).
  Zeros contribute nothing through the linear input projection, whereas SETTING the overpass bit
  feeds an untrained (random-init) weight column = injected noise; zeroing is also the encoder's
  own unknown-map semantic (build_tick_sequences.py:575). **One-hot-SET is reported once as the
  appendix robustness row.** **Mandatory control: the 5 trained maps evaluated with map-ID
  zeroed** — this measures how much the model relies on the one-hot at all (absolute coords
  already fingerprint the map, so expected ≈ nothing). If the control degrades trained-map
  coverage beyond bootstrap CI, the OOD rows are declared confounded and the pre-registered
  fallback fires: ONE map-dropout (p=0.1) rerun per arm, reported as a recipe deviation. No
  map-dropout in the canonical run otherwise (no training-distribution change without a gate).
- **OOD eval protocol** (per-map reporting throughout; pooled numbers banned, datasheet §4):
  1. *Dynamics:* minADE-16 (locked Knob 1) at {1,2,5,7,10}s on all 367 overpass rounds vs the
     copy baseline and overpass model-free res/d, round-level 1,000-resample bootstrap CIs,
     printed beside the 5 ID per-map rows. Pre-registered expectation: ≈ID at 1–2 s, degraded
     at ≥5 s (routes are layout-bound).
  2. *Probe transfer (the C1-linked row):* frozen-latent outcome probe FIT on overpass
     train-split rounds, EVAL on the 64 val-split rounds (match-level split preserved); learning
     curve n ∈ {8,16,32,64,128,303}; baselines on identical data: raw-feature probe and the
     **retrained clean-corpus round encoder refit on the same overpass data (Knob 7a trainer;
     the historical 0.759 / saturates-~16-demos anchor is retired history, cited only as
     motivation) [H7]**; plus the zero-shot row (5-map-fit probe applied unchanged to overpass).
     5 probe seeds × bootstrap CIs. The load-bearing comparison is latent > raw ON overpass —
     this answers "your OOD AUC is just 5v3+bomb arithmetic."
  3. *Value-through-rollout AUC* (Knob-5 protocol: the head-only refit head on the frozen
     trunk), same 367 rounds, both decode conditions.
  4. *Fine-tune anchor:* branch the canonical checkpoint (`--init-from`), 1k steps on
     overpass-train only (~30–60 min, 4090); paper row reads
     "zero-shot / +303-round fine-tune / ID reference band."
- **One-time checks:** assert at load time that zero de_overpass rounds enter the training
  loader; measure whether overpass coords exceed the /3000 → [−1,1] envelope (one line, report).
  Disclose: 303 of 367 OOD rounds come from train-split *series* (same teams on sibling maps);
  the feature schema carries no player/team identity dims, and the 64-val-only agreement check is
  reported alongside.
- **Claim scope. CAN say:** the frozen latent of a 5-map next-state model transfers to a fully
  held-out map — dynamics above copy/model-free baselines at short horizons, and label-efficient
  outcome probing that beats raw-feature probes and the round-encoder at matched budgets —
  evidence of map-general tactical structure, not map memorization. **CANNOT say:** map-blind
  evaluation (absolute coords fingerprint maps — this is DATA-transfer, not identity-invariance);
  generalization over the map distribution (n=1 holdout); layout/route knowledge of unseen maps;
  anything below pro tier (datasheet §1). This scoping travels everywhere, abstract included.
- **Pre-registered contingencies (pinned configs, NOT run):** (i) v2 6-map reference run (adds
  overpass, identical recipe/seed) for the ID-ceiling denominator; (ii) v2 nuke-holdout stress
  run, pre-registered as the *boundary* of the claim, not its carrier. Either is producible in a
  rebuttal within days.
- **Downstream coherence:** Phases 2–3 inherit a model that has never seen overpass; bridge/GRPO
  data must exclude overpass or explicitly mark it OOD — propagate via `_corpus.py` / pipeline
  configs. **The Knob-7 round encoder and all keystone probes inherit the same 5-map corpus [H6].**

## Knob 5 — Heads ✅ LOCKED

**Decision: the trunk is trained by next-state prediction ONLY. The value head is kept
but DETACHED (stop-grad at the trunk); dist edges are refit for k=4 from the clean train
split; the displacement loss is alive-masked; pooling is unchanged; value-AUC checkpoint
selection is removed. Applies bit-identically to both v2 and v3 arms (Knob 2 clause).**

### (a) Value head — detached, no schedule, no third arm
- `value = self.value_head(h.detach().mean(dim=2))` — outcome BCE trains the 2-layer MLP
  head only; **zero outcome gradient reaches the trunk**, enforced by
  `tests/test_no_value_leak.py` (backprop v_loss alone; assert every non-head param grad
  is None). C1's "frozen next-state latent" becomes literally true and F2 is closed
  structurally: under stop-grad the trunk is gradient-identical to `value_weight=0`, so
  **no value_weight=0 control arm exists or is needed** — **and the keystone arm (Knob 7b)
  is therefore the canonical model itself, not a separate run [H3]**.
- `--value-weight` is retired; head BCE weight fixed at 1.0 (trunk-invariant by construction).
- **End-phase mask (closes adversarial O3):** value BCE and VALUE_AUC computed only where
  `x[..., 10*ppd + 10] < 0.5` (global block = map(7) + phase(4), 'end' = phase idx 3);
  eval AUC uses the last non-end frame per window, windows fully in end-phase dropped.
  New value numbers are restated as "previous numbers were partly end-phase leak," not a
  regression.
- **Reported value number = `value_probe.py` linear probe on the FROZEN trunk** (per-map,
  frac buckets 0.25/0.5/0.75) vs raw_last/raw_mean and the retrained round encoder
  **(Knob 7a; historical 0.759 retired [H7])**. The in-flight detached-head AUC is a
  monitoring curve ("value decodability vs pretraining steps") — never the paper number.
- **Checkpoints: `best_ns.pt` (best val next-state) + `last.pt` only. `best.pt`
  (value-AUC-selected) is DELETED** — selecting on outcome AUC is outcome supervision
  through model selection. The Phase-3/bridge value head is re-fit head-only on the
  frozen final trunk (minutes on the 4090).
- **Pre-registered contingency:** if the frozen-trunk probe lands below the raw baselines,
  report it honestly ("next-state alone does not yield value decodability") and rest C1 on
  the action/next-event probes; a co-trained variant (weight 0.3, no detach) may then be
  trained ONLY as a labeled Phase-3 pipeline component — never the canonical representation.
  **The Knob-7 "co-trained twin" secondary row is exactly this labeled variant [H3].**

### (b) Dist edges — refit for k=4
- Rule (pre-registered in `scripts/fit_dist_edges.py`): stationary floor 8u absolute;
  interior edges = 1/6..5/6 equal-occupancy quantiles of the alive-masked (alive at t AND
  t+4), moving (≥8u) |dxy| at k=4 over the CLEAN train split; open-ring representative =
  median of the top sextile.
- Provisional fit (train_v2m.pt, 6-map clean blob 3876/4043 rounds, 29,531,934 alive pairs,
  39.4% stationary, verified 2026-07-18): `DIST_EDGES_U = [8, 26, 47, 58, 85, 117]`;
  `OPEN_RING_MAG_U = 123` (replaces the k=8-era 700u in `dist_centers()`).
  **[H2] LAUNCH BLOCKER: per this section's own sequencing clause, the fit MUST be re-run
  on the Knob-4 5-map train split (3,573 rounds) before any canonical run; the refit values
  supersede the provisional ones everywhere, same pre-registered rule, no judgment calls.**
  97 classes unchanged; ring centers = geometric means of adjacent edges + open-ring rep.
- **One global edge set, identical for v2 and v3** (xy dims identical across schemas;
  per-arm edges would break bit-identity). Per-map mover quantiles go in the datasheet
  with the honest spread (provisional 6-map fit: edges 1–3 within ±3u across maps; the 4/6
  quantile spans 72 (nuke) to 95 (overpass) vs global 85 — table regenerated at refit).
- Edges + open-ring mag + fit provenance (script, corpus file, n pairs, date) are written
  into checkpoint meta; eval/decode read edges from meta, never a hardcoded constant.
- **Launch gate:** at first eval, log the 97-class occupancy histogram; assert every ring
  holds ≥5% of moving player-frames and every direction bin is nonzero (wiring check —
  kills R3's dead-class/teleport failure).

### (c) Pooling — unchanged
Value pooling stays `h.mean(dim=2)` over all 11 tokens. Fixed denominator (the surprise
scar was a varying-normalizer artifact; value must be COVARIANT with alive count — dead
tokens carry the man-advantage signal), and bridge-design pins `z = h.mean(dim=2)` as the
Phase-2 reconstruction target, so changing pooling silently re-specs Phase 2. Pooling
variants (alive-masked mean, global-token-only) are frozen-trunk PROBE experiments,
pre-declared in the probe protocol — never a training change.

### (d) Loss book (final, both arms)
`L = Huber(residual, non-xy reg dims, β=1) + 0.1·CE(97-class, alive-masked)
   + 1.0·Huber(refine offset, alive-masked) + 1.0·BCE(detached value, end-masked)`
- **Alive mask (locked sub-item, now implemented):** alive at t AND t+4, per-player dim 13
  — the SAME mask as the edge fit, so the trained class prior matches the fitted edges;
  `mask.any()` guard for empty batches; non-xy per-player dims stay in the reg loss
  unmasked (predicting zero residual for a corpse is free).
- **Decode companion (required):** `gen_residual()` zeroes the xy residual (no class
  sampling) for any player with alive < 0.5 in the input frame — their dist logits are
  untrained; sampling them teleports corpses into rollouts, coverage eval, and GRPO
  groups. Rollout evals score only players alive at anchor and target frame.
- **Scheduled-sampling clause (Knob 1, satisfied by Knob 6):** targets are always REAL
  frames and the alive mask reads ground-truth alive at both ends; only context frames may
  be model samples.
- Weights reg 1.0 / CE 0.1 / refine 1.0 inherited from the working dist runs — ungated
  retunes forbidden.
- **Removed:** value-AUC checkpoint selection (`best.pt`) and the `--value-weight` knob.
  Retained: residual head, global head, dist head, detached value head. `--cv-residual`
  stays excluded with the dist head; the categorical split-head TODO stays out.

## Knob 6 — Training budget / scheduled sampling / seeds  ✅ LOCKED

**Decision: one-step sample-and-swap scheduled sampling (parallel SS, Duckworth et al.
2019 / Bengio et al. 2015), fixed 25,000-step budget with NO early stop, selection on
val next-state loss only, 3 paired model seeds × both schemas + one SS-off control,
5 probe seeds. Measured, not estimated: TF step 498 ms / no-grad sampled forward 136 ms
on the 4090 at the canonical shape (2026-07-18) → SS = 1.27× teacher forcing.**

**Scheduled sampling (the Knob-1 unpinned mechanic — now pinned; the Knob-7 prerequisite
is satisfied).** Per training step, one extra no-grad forward samples the model's own t+k
frames through the SAME decode path eval rollout uses (`gen_residual(sample=True,
temperature=1.0)` — dist-class sample + refine offset, derived perception dims zeroed;
for dist checkpoints this IS `roll_step`, the gap correction there is cv_residual-only;
**dead-player xy zeroing per Knob 5(d) applies here too**). Each input frame is then
replaced by its model-generated version (prediction of frame i made from real history at
i−k) with probability p — per-(batch,position) Bernoulli, whole frame; first k positions
never swapped. Targets stay the REAL future computed against the corrupted input
(`true_res = y − x_ss`): an error-recovery objective — steer model-generated states
back to the data. Dist class/refine targets recomputed from the same corrected
residual (watch the ce component: steer-back targets can exceed the ring range; the
open last ring absorbs). Value loss unchanged (real outcome labels on possibly
generated frames = value-through-rollout training at depth 1; **detached + end-masked
per Knob 5**). Alive-mask (Knob 5) always from the REAL frames' alive bits, never
generated ones. Val eval stays clean teacher-forced (comparable across runs); rollout
behavior is measured offline.
**Ramp:** p = 0 for steps < 2,000 (sampler must stabilize), linear 0 → 0.5 over steps
2,000–15,000, hold. p_max = 0.5 keeps half the gradient on clean dynamics — the
cv-residual lesson (train-time trick improved 1-step loss, wrecked closed-loop).
- **Rejected, with measured numbers:** full 24-step in-loop AR unroll = 1 + 24×0.27 ≈
  7.5× TF with zero gradient through the chain absent REINFORCE/straight-through
  machinery (no gate); offline rollout-replay buffer (staleness policy = ungated
  machinery); multi-depth truncated rollout conditioning (best depth coverage but ~4×
  the code and subtle target-time bookkeeping = silent-bug surface).
- **SS earns its place or ships as a negative result:** one config-identical p=0
  control (v2, seed 0). If SS-on does not beat TF on depth-{10,20} coverage
  (minADE-K), the canonical recipe drops SS and reports it. If SS wins shallow but
  coverage still collapses ≥ depth 10, the pre-registered escalation is iterated
  2-pass corruption (+1 forward, 1.54× TF) — never the full unroll.

**Budget / config (identical for v2 and v3 — Knob-2 validity clause):** `--dist-head
--horizon 4 --window 96 --batch 64 --crops-per-round 32 --lr 3e-4
--maps de_ancient,de_dust2,de_inferno,de_mirage,de_nuke` **[H1]** (AdamW wd 0.01,
warmup 1,000, cosine → 0 at exactly 25,000 steps, grad-clip 1.0, AMP), alive-masked
xy losses. 25k ≈ **~14 epochs of the 3,573×32 crop inventory [H9]** — past every prior
stopping point. **No early stop / no plateau abort:** adaptive stopping gives arms
different effective LR schedules and breaks bit-identical comparability. Eval every 500
steps (50 clean val batches); snapshots every 2,500 (11/run, ~1 GB) feed the offline
probe-AUC-vs-compute curve. **Selection = best val next-state loss (`best_ns.pt`)
only.** Probe AUC and coverage are computed offline on snapshots and NEVER touch
selection (the C1 anti-circularity firewall). **`best.pt` no longer exists [H4]** —
value decodability is monitored via the logged detached end-masked AUC; prior runs
show value peaks at 500–1,500 steps while next-state peaks at the cap.
- **Pre-registered escalation:** pilot = v2 seed 0 (local). If best val_ns still
  improves > 1% relative over the final 20% of the budget, double the budget and
  re-pilot before launching the matrix; run the same check on a v3 seed-0 pilot (an
  undertrained v3 arm would confound the ablation). This rule is live: both prior
  dist runs' best_ns landed exactly at their caps (8,000/8,000 and 12,000/12,000).
- **Corpus guard (must-pass) [H1]:** load blobs through `scripts/_corpus.py::clean_blob`
  (datasheet §5) and hard-assert TWO stages: post-clean_blob counts = 3,876 train /
  705 val, then post-`--maps`-filter counts = 3,573 train / 641 val; refuse to run
  otherwise. The on-disk blobs are pre-exclusion (val_v2m holds 770 rounds). Also
  assert zero de_overpass rounds in the training loader (Knob 4).
- **Hardware:** pilot + all probes local (v2m fits: 9.3+1.8 GB blobs under the 15 GB
  WSL cap, num_workers 0; ~4.5 h/run measured with display contention, ~2× faster
  idle). **v3m (10.7+2.0 GB) does NOT fit local RAM — never run v3 locally.** All 7
  canonical runs (6 matrix + SS-off control) on ONE pod GPU class, never splitting
  seeds across machines: ≈ 12–20 GPU-h ≈ $25–45 (A100-class) or ~$15–20 (pod 4090,
  ~2 pod-days). Escalation doubles this.

**Seeds.** Model seeds {0,1,2} for BOTH v2 and v3, same list → the v2-vs-v3 ablation
is a per-seed PAIRED delta (Knob-2 clause operationalized). `--seed` drives init +
data order via named torch.Generators (data / SS-Bernoulli / decode-sampling) so
pairing is exact; claim "identical seed + config", never bitwise reproducibility
(AMP/cuDNN). Probes: 5 probe seeds per checkpoint (the linear probe is Adam-trained —
it has init/minibatch variance); headline cell = mean over probe seeds; error bar =
spread over the 3 model-seed means with all 3 per-seed points shown; no p-values at
n=3. Baselines carry bars too: raw-feature probe ×5 probe seeds; round-encoder
retrained ×3 seeds iff < 2 h/seed **(satisfied: ~20 min/run, Knob 7a [H7])**.
Bars on: headline probe AUCs (global + per-map), minADE-K per depth {2,4,10,14,20},
value-through-rollout AUC per depth; model-free res/d gets none (deterministic).
**C1 decision rule (pre-registered): the latent probe must beat BOTH baselines with
the same sign in all 3 model seeds;** the honest fallback if spread swamps the margin
is +2 seeds, not a softer criterion. Canonical downstream checkpoint = v2 seed-0
`best_ns.pt`, declared before any probe result is seen.

## Knob 7 — Keystone probe-transfer protocol (L2)  ✅ LOCKED (pre-registered)

**Decision: retrain the round encoder on the clean 5-map corpus (forward-MSE heads
only), probe the canonical detached-trunk model (gradient-identical to value_weight=0
by Knob 5 [H3]), and gate C1 on a linear outcome probe with pre-registered thresholds
over a 6-point nested match-subset scaling grid.**
Runs ONLY after R0 is fully pinned — Knobs 4/5/6 above satisfy the prerequisite,
**including the Knob-5 edge refit on the 5-map corpus [H2]**; this section + subset
manifests are committed BEFORE any treatment run (git hash = pre-registration).
**All counts below are re-derived for the Knob-4 5-map corpus per this section's own
re-derivation clause [H6].**

### C1, exact wording under test
"The frozen latent of a world model pretrained ONLY on next-state prediction
(zero outcome gradient into the trunk, unit-test enforced) is a better substrate for
round-outcome probes than raw features, window-mean raw features, and the retrained
round-encoder representation, under identical corpus, split, and probe budget — and its
probe transfer keeps improving with training data past the round-encoder's historical
saturation point."

### 7a — Round encoder: retrain (yes), minimally
- The historical 0.759 is **retired from all claims** (dirty 81-demo corpus, different
  split, 582-d schema); it survives only as motivation history.
- Port `scripts/_archive/train_round_encoder.py` → `scripts/train_round_encoder.py`.
  Changes ONLY: load `train_v2m.pt`/`val_v2m.pt`; apply `_corpus.clean_blob` (D1/D2);
  **apply the Knob-4 5-map filter [H6]**; add `--matches-manifest` filter keyed on
  `meta["match_id"]`. Config = v6 VERBATIM (`outputs/round_encoder/v6_81demos/config.json`:
  d_model 512, 4 layers, 8 heads, d_ff 2048, dropout 0.15, lr 3e-4, bs 8, **50 epochs**,
  seed 42; feature_dim is data-driven → 597). No re-tuning. Checkpoint = native `best.pt`
  (lowest val total loss; the Knob-5 best.pt deletion applies to the WORLD MODEL trainer
  only — the RE has no outcome head, so its native selection is admissible [H4]).
- **Event heads: dropped, disclosed.** The clean v2m blobs carry NO event labels
  (`merge_hf_tick_sequences.py` strips `event_*`, lines 111/152); the trainer skips
  absent event heads natively (archive lines 182–196), leaving the 3 forward-MSE
  horizons {1,8,32}. Bias direction favors the baseline: v6's own log shows event-CE
  catastrophically overfits (val 1.07 → 6.84 by epoch 49), and the learning-curve
  finding blames it for the negative top-end slope — removing it can only help the
  encoder's curve. Do NOT rebuild an event sidecar (silent-misalignment surface for
  a secondary target).
- **Validity clause (fixture test):** before any clean run, the ported trainer must
  reproduce v6's val forward-MSE curve on the OLD `train.pt`/`val.pt` (which still
  carry event labels) within noise. Port drift fails the fixture → no clean runs.
- Cost: verified 7.6 s/epoch at 1,471 rounds → ≤ ~20 min/run on the clean 5-map corpus.

### 7b — Keystone world-model arm: the canonical model itself [H3]
- Under Knob 5 the canonical trunk receives **zero outcome gradient by construction**
  (detached value head; `tests/test_no_value_leak.py` proves gradient-identity with
  value_weight=0). The keystone arm is therefore the canonical Knob-6 run — no separate
  arm, no `--value-weight` flag (retired). Checkpoint = **`best_ns.pt`** (value-AUC
  selection no longer exists), pre-registered.
- **Co-trained twin (secondary): one clearly labeled extra run** — identical seed/config
  but value head ATTACHED (no detach), weight 0.3 — probed on full corpus only. It
  measures co-training's contribution and doubles as Knob 5 evidence; it is Knob 5's
  pre-registered labeled variant and is never the canonical representation. Never gates.
- RE parity: its SSL never consumes `round_won` (trainer docstring) — neither
  representation sees outcome labels.

### 7c — Probe protocol (`scripts/keystone_probe.py`, extends `value_probe.py`)
Representations, all frozen, all fed the IDENTICAL (round, frame) sample manifest
(fracs {0.25, 0.5, 0.75} of round length; frame ≥ 96 so the WM window fits; label =
winner=='ct'):
1. `raw_last` — 597-d current frame; 2. `raw_mean` — mean of trailing 96-frame window;
3. `raw_mean_full` — mean from round start (context parity with the RE — favors the
   baseline, deliberately); 4. `rand_wm` — untrained WM latent, same arch/init seed
   (**G0 control**); 5. `re_h` — RE causal h_t, full round prefix (512-d);
6. `wm` — `model.latent(window)[:, -1, :]` from `best_ns.pt` (512-d) — THE TEST.
Disclose both asymmetries in the paper: RE sees full-round history vs the WM's 96
frames; raw has MORE dims (597 vs 512).
- **Probe family: LINEAR ONLY gates** — `value_probe.py::fit_probe` (standardize on
  probe-train stats; Adam, 300 epochs, lr 0.05). Weight decay swept {1e-4, 1e-3, 1e-2},
  selected per-representation on a fixed match-disjoint 15% train-internal probe-select
  split (never val), same grid for all reps — the dimensionality-fairness answer.
  Otherwise byte-identical probe code for every representation.
- SECONDARY (one labeled table, full data only, never gating): the historical L2-G2
  MLP (2-layer, 128 hidden) for continuity with the retired 0.759.
- 5 probe seeds {0..4}; headline = seed mean. **Primary CI = paired percentile
  bootstrap over VAL ROUNDS** (resample rounds, keep all fracs of a round together;
  1,000 resamples, 95%; all between-rep deltas computed within each resample).
  Match-level cluster bootstrap (**5-map val-match count, computed and frozen at
  manifest time [H6]**) reported as robustness.
- Reporting: pooled val AUC + per-frac (@0.25 highlighted) + per-map slices of the
  ONE pooled probe (no per-map fitting). **Val = the 641 clean 5-map rounds; overpass
  probing is exclusively the Knob-4 OOD protocol [H6].** Per-map mandatory at N ≥ 32
  and full; pooled-only below (small subsets may lack maps).

### 7d — Scaling grid
- Unit = MATCH (the split unit). Grid: N ∈ {4, 8, 16, 32, 64, N_full} train matches,
  where **N_full = train matches with ≥ 1 clean round on the 5 canonical maps** (from
  `split_manifest_v2.json`; exact count computed and frozen at manifest time **[H6]**).
  **Nested** subsets, seed 0, **map-stratified round-robin ordering** (largest map
  pools first) so small N covers all **5** maps. Manifests →
  `outputs/keystone/subsets/subset_N{n}.json`, committed before any run.
  Val fixed at the full **641** clean rounds at every point.
- At each point train BOTH on the identical subset: WM (canonical Knob-5/6 recipe,
  detached head, **fixed 25k-step Knob-6 budget, NO early stop — small-N lands its
  `best_ns.pt` earlier on its own [H5]**) and RE (v6 config, 50 epochs, `best.pt`).
  Model seeds: 1 per point + 2 extra at the gating anchors {N_sat, N_full} for both
  models; seed SD reported. **The N_full WM anchor reuses the three canonical v2 runs
  from Knob 6 (identical recipe/corpus/budget) — the WM curve adds 7 new runs, not
  10 [H8].** 5 probe seeds everywhere.
- **N_sat** = the grid point whose train-round count is closest to 328 rounds (the
  historical saturation point; expected N=8 ≈ 400 rounds — resolved and frozen at
  manifest time).
- **Stage 0 (before any curve compute):** run `value_probe.py` on the existing
  `wm_3map` / `wm_3map_dist_v3m` checkpoints (~minutes). Diagnostic + harness
  shakedown only — old checkpoints are dirty-corpus and undertrained; never cited.

### 7e — PRE-REGISTERED pass/fail for C1
- **G0 (sanity precondition):** `wm` beats `rand_wm`, paired 95% CI > 0. Fails → no
  representation claim of any kind.
- **C1-REP:** pooled linear val AUC, `wm`(N_full) − max(`raw_last`, `raw_mean`,
  `raw_mean_full`, `re_h`) ≥ **+0.02** AND paired-bootstrap 95% CI of the delta
  excludes 0; AND `wm` ≥ that best baseline on **≥ 4 of 5 maps [H6]** with NO map whose
  paired CI is entirely below 0.
- **C1-SCALE:** AUC_wm(N_full) − AUC_wm(N_sat) ≥ **+0.01** with paired 95% CI
  excluding 0. The RE curve's slope is REPORTED for the narrative but never gates —
  C1 is not hostage to whether the baseline's saturation replicates.
- **C1 passes iff G0 ∧ C1-REP ∧ C1-SCALE.** Secondary (reported, never gating):
  @0.25 early-bucket delta, MLP table, co-trained-twin row, E1/E2 probes, RE slope,
  and the Knob-4 overpass OOD probe rows.
- **Committed failure handling:** C1-REP fails → C1 is falsified; the paper reframes
  around the coverage/option-set result, no salvage wording. C1-SCALE fails alone →
  claim reduces to "better representation at matched data"; the scaling clause is
  dropped, not softened. RE scales on the clean corpus → the saturation narrative is
  rewritten as historical motivation only.

### 7f — Probe targets
- **Keystone (gates C1): round outcome P(CT win) ONLY.**
- In-scope secondary, full data only, same harness — labels derived from the feature
  tensors themselves (per-player `alive` bit; global `bomb_state_onehot`), so no
  labeling pipeline can misalign: **E1** any-death-within-5s (alive bits at t+40
  frames); **E2** bomb-plant-within-10s (pre-plant anchors only). Rebut "the latent
  only encodes economy/score" without gating.
- **Deferred:** action/movement/engage probes, 7-way next-event-type (needs an event
  sidecar the clean blobs can't support), time-to-event, anything needing new labels.

### 7g — Execution order
Stage 0 (harness shakedown) → RE fixture test → full-data six-way probe (settles
C1-REP) → grid (settles C1-SCALE). Costs consolidated in §5 below.

## 4. Consolidated implementation checklist for R1 (file-level, ordered)

**Phase A — code (all pre-launch; ~1 engineer-week total):**

1. **`scripts/fit_dist_edges.py`** — commit (currently untracked); no code changes. **[gates step 2]**
2. **Edge refit run** — re-run on the 5-map clean train split (Knob 4 map set) [H2]; record new `DIST_EDGES_U` / `OPEN_RING_MAG_U` + provenance. CPU, ~minutes.
3. **`scripts/train_world_model.py`** — single consolidated edit (~120 lines), all knobs land here:
   - line 146: refit `DIST_EDGES_U`; line 154: `OPEN_RING_MAG_U` module constant replacing 700.0, provenance comment (Knob 5b).
   - line 255: `value = self.value_head(h.detach().mean(dim=2)).squeeze(-1)` (Knob 5a).
   - lines 480–488: alive mask `(x[...,p*ppd+13]>0.5) & (y[...,p*ppd+13]>0.5)` on dist CE + refine Huber, `mask.any()` guard (Knob 5d); mask reads REAL tensors, never `x_ss` (Knob 6).
   - lines 491–493: end-phase mask on value BCE (`x[...,10*ppd+10] < 0.5`); remove `--value-weight` (line ~364, print at 414) (Knob 5a).
   - `evaluate()` lines 343–345: value AUC from last non-end frame per window (Knob 5a).
   - lines 459–519: delete `best_v`/`best.pt` branch; keep `best_ns.pt`, add `last.pt` + `step_*.pt` snapshots every 2,500 steps; meta gains `dist_edges_u`, `open_ring_mag_u`, edge-fit provenance (Knobs 5b, 6).
   - `gen_residual()` (~lines 263–297): zero xy residual for input-alive < 0.5 players (Knob 5d; line-140 comment already demands it).
   - SS block (~25 lines): `ss_p(step)` schedule (0 <2k, linear→0.5 at 15k, hold); no-grad `gen_residual(sample=True, T=1.0)`; whole-frame Bernoulli swap for positions ≥ k; losses on `true_res = y − x_ss`, `o = model.heads(x_ss)`; flags `--ss-pmax 0.5 --ss-ramp-start 2000 --ss-ramp-end 15000` (control = `--ss-pmax 0`) (Knob 6).
   - `--seed` + named torch.Generators (data / SS-Bernoulli seed+500 / decode seed+1000) (Knob 6).
   - `clean_blob` import + two-stage corpus assert (3,876/705 → 3,573/641 after `--maps`) + zero-overpass loader assert [H1] (Knobs 4, 6).
   - `--init-from <ckpt>` (~10 lines, fine-tune anchor); `--matches-manifest` subset filter (Knobs 4, 7).
   - defaults `--steps 25000 --warmup 1000`; canonical `--maps` list in run commands.
   - first-eval ring-occupancy histogram + assert (each ring ≥5% of movers, all direction bins nonzero) (Knob 5b).
4. **`tests/test_no_value_leak.py`** — new, ~15 lines, CPU: backprop v_loss alone, assert all non-value_head grads are None (Knob 5a). Run in CI before any launch.
5. **`scripts/_corpus.py`** — `zero_map_onehot(tensor, ppd)` (global dims [10·ppd : 10·ppd+7]; overpass = MAP_VOCAB idx 5) + unit test; OOD blob assembler concatenating overpass rounds from train_pt + val_pt via `clean_blob` (~25 lines) (Knob 4).
6. **`scripts/rollout_eval.py`, `scripts/dist_coverage_eval.py`, `scripts/value_probe.py`** — `--map-id {asis,zero,set:<map>}` input transform; wire `clean_blob` into rollout_eval.py (existing datasheet TODO); `value_probe.py`: `--probe-seed` + 5-seed loop (mean±sd), learning-curve mode (n ∈ {8,16,32,64,128,303}), round-level 1,000-resample bootstrap helper (Knobs 4, 6).
7. **Head-only value refit** — script/flag reusing value_probe.py machinery on frozen `best_ns.pt` (the Phase-3/bridge reader; rollout_eval/GRPO read this head, not an in-run head) (Knob 5a).
8. **`scripts/train_round_encoder.py`** — restore from `_archive/`, add `clean_blob` + 5-map filter + `--matches-manifest`, defaults → `train_v2m.pt`/`val_v2m.pt`; log the event-head skip loudly (Knob 7a).
9. **RE fixture test** — reproduce v6 val forward-MSE on old `train.pt`/`val.pt` within noise; port drift blocks all clean runs (Knob 7a).
10. **`scripts/keystone_probe.py`** — extend value_probe.py (keep `fit_probe` verbatim): reps `re_h`, `rand_wm`, `raw_mean_full`; wd sweep {1e-4,1e-3,1e-2} on match-disjoint 15% probe-select split; paired round-bootstrap + match-cluster CI; per-map/per-frac slices; E1/E2 tensor-derived labels; JSON → `outputs/keystone/` (Knob 7c/7f).
11. **`scripts/keystone_curve.py`** — adapt `_archive/learning_curve.py`: map-stratified nested manifest writer → `outputs/keystone/subsets/` (freeze N_full, N_sat, val-match count), grid driver for 7 new WM runs + 10 RE runs + probes → `curve.json` (Knob 7d).
12. **One-time checks** — overpass coordinate envelope vs [−1,1] (one line, report); commit this document + manifests (= pre-registration).

**Phase B — runs (ordered):**

1. Pilot v2 seed 0 local (escalation check: >1% val_ns improvement over final 20% → double budget) + v3 seed-0 pilot on pod (Knob 6).
2. 7-run canonical matrix on ONE pod GPU class: {0,1,2} × {v2m, v3m} + SS-off (v2, seed 0). Never run v3 locally (Knob 6).
3. Offline evals on snapshots: coverage per depth, probe-vs-compute curve, SS-vs-TF depth-{10,20} coverage gate (Knob 6).
4. Knob-4 OOD suite on canonical v2 seed-0 `best_ns.pt`: ID-zeroed control → OOD dynamics/probe/value rows (both decodes) → 1k-step fine-tune anchor.
5. Keystone Stage 0 shakedown (old checkpoints, minutes) → RE fixture → RE full-data retrain (3 seeds, ~20 min each, local) → full-data six-way probe (C1-REP) → scaling grid: 7 new WM runs + 10 RE runs + probes (C1-SCALE).
6. Co-trained twin run (secondary row) + head-only value refit for Phase 3.

## 5. Total cost (honest)

| Bucket | Compute | $ |
|---|---|---|
| Local 4090 (pilot ~5–9 h; probes/evals/OOD/anchor ~6–8 h; RE fixture+retrains+curve ~2–3 h) | **~13–20 4090-h** (overnights; display GPU — no daytime training) | $0 |
| Pod: 7-run canonical matrix (Knob 6) | 12–20 GPU-h | $25–45 (A100-class) / ~$15–20 (pod 4090) |
| Pod: keystone WM curve, 7 new runs [H8] | 14–21 GPU-h | $15–35 (or $0 as local v2 overnights, ~3–4 nights) |
| Pod: co-trained twin (1 run, secondary) | 2–3 GPU-h | ~$5 |
| **Total** | **~13–20 local h + ~28–44 pod GPU-h** | **≈ $45–85** (≈ $30–50 with curve run locally); pre-registered escalation (budget doubling) doubles the pod lines |