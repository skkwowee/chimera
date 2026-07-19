# Chimera Adversarial Review — Synthesis

**Scope:** full pipeline at `/home/soone/chimera`. 24 findings survived independent refutation attempts (4 critical, 15 major, 5 minor); 8 were refuted. Every confirmed finding below was verified in code and, where applicable, reproduced empirically on the actual corpus blobs.

---

## 1. Verdict

The pipeline does **not** hold up to scrutiny in its current state, and launching the R1 canonical retrain now would burn GPU on a run that is unreproducible by its own recipe and mis-specified on at least five axes. The core problems cluster into three failure classes. **(a) The corpus the recipe describes is not the corpus the code produces:** the trainer applies none of the datasheet §5 exclusions, defaults to a stale 1,471-round blob, and the feature tensors themselves carry four dead bomb-state dims, pause-contaminated round clocks, 17.5% forced-stationary freeze frames, and a v3 "dist_to_bomb" dim that is distance-to-origin on 84% of frames. **(b) The locked recipe is unimplementable as written:** Knob 1's scheduled-sampling training, the seed required by the Knob 2 validity clause, and the headline minADE-K coverage eval all have zero implementation, and the 97-class displacement head loses a third of its class book at the locked k=4. **(c) The measurement layer that would judge the run is compromised:** value AUC is inflated by post-round frames and computed on a different corpus than its baseline, the GRPO gate numbers pool excluded and off-distribution maps, the decision-eval turn buckets are contaminated by onset frames, and both bridge gates leak the gold answer into the "latent-off" arm. Before any GPU is spent: rebuild the corpus with the four data fixes, wire `clean_blob` and a `--seed` into the trainer, implement (or explicitly re-lock away from) scheduled sampling, rescale the ring edges for k=4, mask value loss/AUC to pre-end frames, and pin a multi-seed protocol for the v2-vs-v3 decision. The bridge and thesis-level fixes can trail the retrain launch but must land before the gates and the paper.

---

## 2. Findings by pipeline stage

### 2.1 Data / corpus construction

**D1 — bomb_state one-hot is entirely dead** — MAJOR, material — `scripts/build_tick_sequences.py:503`
The code builds the label `planted_bombsite_a/b` but the vocab expects `planted_a/planted_b`, and `carried` is never assigned — so all four bomb-state bits are zero on every frame corpus-wide (empirically: 87,588 post-plant frames sum to `[0,0,0,0]`; whole-corpus sums `[632830,0,0,0]`). The datasheet §4b "dimension-level audit — PASS" claim is falsified by this block, and any replication implementing the declared schema trains on a different feature space. *Fix:* normalize the site string to the vocab, drop or document `carried`, re-bake, amend datasheet §4b/§5.

**D2 — round_time_s baked with pauses/halftime included** — MAJOR, material — `scripts/build_tick_sequences.py:509`
Round time is anchored at the previous round's `official_end`, so halftime/timeout pauses (p90 64s, max 510s; ~28% of rounds exceed 30s) shift the normalized clock and all 8 sinusoid dims by an arbitrary offset — live play can begin at round_time_norm 1.7–2.7. Two tactically identical states get clock features differing by up to ~2.4 normalized units. Corrupts a training input and any round-time-binned analysis. *Fix:* anchor at freeze_end (minus fixed buy time), clamp at 0, rebuild.

**D3 — 17.5% of frames are freeze-phase, undisclosed, unmasked** — MAJOR, material — `scripts/train_world_model.py:87` (mechanism), corpus-wide
`RoundWindows` crops uniformly over full round tensors, and no eval filters freeze frames. One in six frames trains the stationary class on physically-immovable states and dilutes every pooled per-frame metric (displacement CE, minADE, model-vs-copy gap) — including the res/d const-velocity curve the recipe calls "itself a headline figure." Disclosed nowhere. *Fix:* crop/mask to freeze_end..end in train and eval; disclose phase composition and phase-stratified metrics in datasheet §5.

**D4 — v3 "dist_to_bomb" is distance-to-map-origin pre-plant** — MINOR severity, but material — `scripts/build_v3_features.py:146`
bomb_x/y are (0,0) until plant, and 84% of frames are pre-plant, so dim 7 of the 9 perception dims is a map-dependent origin-distance artifact for most of the corpus — directly contaminating the LOCKED v2-vs-v3 perception deconfound that "carries the final weight" for the keystone probe-transfer claim. *Fix:* gate on plant (sentinel 1.0 pre-plant, matching the min_aim_error convention) or use carrier position; rebuild v3 blobs.

**D5 — datasheet claims 128-tick source; demos are 64-tick** — MINOR, material — `docs/datasheet.md:16`
The stated provenance is wrong (and internally inconsistent — 128/8 = 16 Hz, not 8). Code and data confirm 64-tick. A replicator trusting the datasheet re-parses at half the intended resolution or concludes all second-denominated quantities are off by 2x. Same error repeated in `three-level-architecture.md:83`. *Fix:* doc correction plus audit of tick-derived numbers (event horizon = 4s; the "16 Hz ablation" TODO = downsample 4, not 8).

### 2.2 Objective / trainer

**O1 — canonical recipe's rollout-native training doesn't exist** — MAJOR (see also R1 below) — `scripts/train_world_model.py:477`
The only trainer is pure teacher-forced direct t+k regression with default k=8; grep finds no scheduled-sampling code anywhere. Every claim tied to the rollout-native recipe is backed by no runnable code. (Escalated to critical under recipe-gap, R1.)

**O2 — trainer never applies corpus exclusions; gate compares across corpora** — MAJOR, material — `scripts/train_world_model.py:394`
No `clean_blob` import; defaults point at stale `train_v3.pt/val_v3.pt` (1,471/262 rounds incl. de_anubis with all-zero map one-hots and de_train). The documented command cannot produce the documented 3,876/705 clean run, and the trainer's printed VALUE_AUC — which selects `best.pt` — is compared against a `value_probe.py` baseline computed on the cleaned corpus: two different eval sets deciding one gate. *Fix:* apply `clean_blob` immediately after `torch.load`, recompute the baseline on the identical cleaned val, re-run the Phase-1 gate.

**O3 — value training and VALUE_AUC include post-round frames** — MAJOR, material — `scripts/train_world_model.py:491`
5.6% of val frames are phase='end', where sign(alive_diff) identifies the winner in 240/241 rounds; the value head is supervised on every frame and ~6% of eval windows terminate in this trivially-labeled segment. The headline 0.849/0.841 AUC is partly a parlor trick, `best.pt` selection is biased toward exploiting it, and datasheet §4b's "no target/outcome leakage" is contradicted. Aggravating: `value_probe.py` deliberately avoids late-round frames — the team knows this trap and the protection doesn't reach the trainer. *Fix:* mask value loss and AUC to tick < end_tick, restate numbers, re-select checkpoints.

### 2.3 Retrain recipe (R0 spec vs code)

**R1 — Knob 1 "rollout-aware (scheduled sampling)" is unimplemented and unpinned** — CRITICAL, material — `scripts/train_world_model.py:476` / `docs/retrain-recipe.md:19-21`
A replication either silently gets teacher forcing (the train/test mismatch the recipe claims is mitigated is untouched) or must invent the algorithm — mixing schedule, replaced positions, stop-grad vs straight-through are all unpinned, and the ~L/k sequential-forward cost (~32 passes/step at window 128, k=4) is unbudgeted. Two faithful replications produce different models. The dynamics, coverage, and GRPO ladder rows all rest on a training regime that either never happened or is underspecified. *Fix:* pin the exact variant, implement behind a logged flag, re-budget wall-time — or amend the recipe to state the actual regime.

**R2 — Knob 2 validity clause is unimplementable: trainer has no seed** — CRITICAL, material — `scripts/train_world_model.py:349`
`grep seed` returns nothing: init, crops, shuffling, dropout, and multinomial decode all run on OS entropy. The recipe's own "same seed … or the ablation is inadmissible" clause makes the flagship v2-vs-v3 deconfound inadmissible as written. Sibling trainers (`train_sft.py`, `train_grpo.py`) seed properly — this is an omission unique to the world-model trainer. *Fix:* `--seed` seeding torch/numpy/random + DataLoader generator + worker_init_fn, recorded in run metadata; launch paired-seed arms.

**R3 — DIST_EDGES_U leaves 32 of 97 classes dead at the locked k=4** — MAJOR, material — `scripts/train_world_model.py:146`
Edges were tuned for 1s displacements; at 500ms, ring 4 gets 9 examples in 5.4M player-frames and ring 5 gets zero, while typical moving displacements collapse into 2 coarse bins. Temperature-1 sampling can draw a never-trained ring and emit a 348–700u/500ms teleport into the compounding rollout buffer — contaminating coverage eval, scheduled-sampling training, bridge latents, and GRPO groups. (`fit_dist_edges.py` exists but is untracked, never applied, and unconsumable by the trainer.) *Fix:* rescale edges (~halve, e.g. [4,12,28,60,124,256]) and the refine cap for the empirical 500ms distribution; report per-horizon class books.

**R4 — recipe's "clean corpus" is wired into value_probe only** — MAJOR, material — `scripts/train_world_model.py:394` (recipe-level view of O2)
Trainer and every rollout/coverage/dynamics eval load the dirty 4,043/770 corpus; the paper's corpus number is wrong for the model actually trained, and headline metrics juxtapose clean-705 and dirty-770 numbers as comparable. *Fix:* wire `clean_blob` into all blob loaders, retrain and rerun gates on 3,876/705.

**R5 — headline minADE-K coverage eval has no implementation** — MAJOR, material — `scripts/rollout_eval.py:71`
`rollout_eval` decodes argmax (deterministic point error — the metric the recipe itself calls dishonest past ~4s); `dist_coverage_eval` computes minADE-K but never rolls. No script can produce the locked depth-sweep {2,4,10,14,20} coverage numbers, and the same missing sampler is the Phase-3 GRPO group-generation path. *Fix:* extend rollout_eval to K sampled rollouts per anchor with frozen K/temperature/decoding, shared with the GRPO generator.

### 2.4 Evals

**E1 — rollout_eval (GRPO gate source) pools excluded and off-distribution maps** — MAJOR, material — `scripts/rollout_eval.py:219`
No `clean_blob`, no `--maps` option: the 0.856/0.865 value-through-rollout AUC — the Phase-3 go/no-go — pools 51% off-training-map rounds and 9% de_anubis rounds with all-zero map identity, directly violating the datasheet's "per-map, never pooled" mandate. Recomputation could flip the gate. *Fix:* wire `clean_blob`, filter to checkpoint training maps, report per-map, recompute before the F4 depth-sweep rerun.

**E2 — decision_eval turn buckets contaminated by movement-onset frames** — MAJOR, material — `scripts/decision_eval.py:173`
Zero-past-speed frames get theta = 90° by arithmetic and land deterministically in "hard turn" (37% of hard-turn, 22% of reversal frames have past speed <5 u/frame; ~24% of hard-turn is the exact cos=0 degenerate case), and the stationary bucket mechanically bounds copy's error <25u — contradicting the script's own printed no-bounded-baseline claim. Both the F1 verdict that killed the learned-tactics claim ("hard turn −52%") and the surviving coverage row (minADE-16 42u vs 87u) rest on these buckets. *Fix:* route low-speed/high-disp frames to a separate "onset" bucket; drop or caveat stationary-bucket skill; rerun.

**E3 — minADE-K "cover vs copy" has no fair sampled baseline** — MAJOR, material — `scripts/dist_coverage_eval.py:124`
Oracle min over 16 model samples vs a single deterministic point: any dispersed sampler wins by spread alone, so the coverage claim is unfalsifiable against pure variance. Flagged in methodology-review F1 as required Tier-1 fix #5 (2026-06-21); never added; the recipe canonizes the metric anyway. *Fix:* per-bucket train-fitted stochastic baseline (damped-CV + residual covariance), K draws, min-over-K scored identically; require model < baseline.

**E4 — dist_coverage_eval and event_boundary_check pool across maps** — MINOR, material — `scripts/dist_coverage_eval.py:130`
Mirage is ~55% of pooled rounds; a per-map inferno collapse would be invisible. Violates the datasheet mandate that decision_eval already complies with. *Fix:* add per-map breakdown sections mirroring decision_eval.

### 2.5 Bridge / NLA

**B1 — both gate evals leak the gold answer into the "prompt-only" slice** — CRITICAL, material — `scripts/eval_recon.py:45` (identical in `eval_ablate.py:69-72`)
The prompt length is computed as the batch-max count of -100 labels, which is padded_len − min(target_len) — so `input_ids[:, :T]` contains most or all of the target text (CPU repro: row slices decoded to near-complete gold answers including the value %). The latent-off arm is conditioned on the answer templated *from* the latent; r_abl converges to r_real, destroying the recon(on)−recon(off) honesty number and invalidating every gate #1/#2 figure in an unpredictable direction. *Fix:* per-row prompt length (first non−100 label index) with left-padded per-row slices.

**B2 — the value-agreement metric is not implemented** — MAJOR, material — `scripts/eval_ablate.py:56`
`value_agreement()` only regex-counts percents; `errs` is never filled and the true-% comparison is a TODO comment. The GROUNDED verdict gates solely on teacher-forced CE over the same templated distribution — clearable by template memorization while stated percents are uncorrelated with the value head: the exact Era-1 failure the gate exists to catch. *Fix:* compute |stated% − 100·sigmoid(value_logit)| aligned by order; make the verdict require the value-agreement delta.

**B3 — bridge SFT caches hardcode the h8/1s assumption; stale under k=4 with no guard** — MAJOR, material — `scripts/gen_bridge_sft.py:202`
REC2/REC4, movement thresholds (56/248u per step), and `horizon_s_per_step: 1` are hardcoded; `load_world_model` drops the checkpoint horizon so nothing can assert consistency. Post-retrain, captions are mislabeled 2x (or stale h8 latents pair with a k=4 model silently). Recipe never mentions regeneration. *Fix:* derive all three from `ck['args']`, add a checkpoint-hash guard on caches, add a mandatory regen step to the recipe.

**B4 — no EOS truncation / string round-trip before the recon decoder** — MINOR, material — `scripts/eval_recon.py:47`
Generation always emits max_new tokens; latent-conditioned post-EOS tokens (invisible in any decoded answer) feed the decoder, upward-biasing reported faithfulness vs the documented "decode to STRING → re-tokenize" firewall, and miscalibrating the GRPO recon-τ. *Fix:* truncate at first EOS or decode/re-tokenize before scoring; re-derive τ.

### 2.6 Thesis-level

**T1 — keystone C1 comparison is protocol-incommensurable** — CRITICAL, material — `docs/world-model-design.md:184`
The 0.759 ceiling is best-epoch MLP-probe *accuracy* on the old corpus's 262-round val with ≤1,471 train rounds; the planned world-model probe is a linear-probe *AUC* on the clean 705-round val with 3,876 train rounds (2.6x, violating the doc's own "at fixed data"). Metric family, probe capacity, model selection, val population, and train size each can move the number more than the ~0.03 margin at stake. No harmonization is scheduled anywhere. *Fix:* retrain the round encoder on the clean corpus and score both latents under one identical protocol (probe class, metric, val split, train count, selection rule), with a CI on the margin.

**T2 — fabricated causal confirmation in the canonical rationale doc** — MAJOR, material — `docs/decisions-ledger.md:215`
The ledger — the mandated first-read doc — asserts the facing shortcut is "+27.2pp; causal, confirmed by yaw-shuffle dropping it to +19.3pp." The +19.3pp figure has zero provenance (grep hits only that line; no artifact, no log), and two later canonical docs state the causal test was never run and +27.2pp never reproduced. The ledger also still carries two numbers the methodology review ordered purged (phantom 0.803/0.807 probe parity; 88/96 vs actual 32/96 depth-8 miscount). This neutralizes the datasheet §7 fence against the 9–13h v4 rebuild. *Fix:* strike +19.3pp, record the test as never-run, execute ladder item 9's purge, cross-reference datasheet §7 / Knob 3 as controlling.

**T3 — no multi-seed protocol anywhere; "same seed" clause is statistically vacuous** — MAJOR, material — `docs/retrain-recipe.md:46`
The canonical-schema decision will be made from one run per arm, and identical seeds cannot pair a 597-d and a 687-d model (different-shape input projections desynchronize RNG consumption — empirically verified: same-seed builds diverge in identically-shaped downstream layers). Precedent already exists: F2 treats a single-seed 0.008 AUC delta as a conclusion on a val set where the AUC SE alone is ~0.02. *Fix:* require n≥3 seed pairs per arm with a paired-difference CI; gate canonical promotion on the CI excluding zero.

---

## 3. Pre-R1 punch list

### Blocking — must land before the R1 retrain spends GPU

1. **Corpus rebuild (one pass, four fixes):** bomb_state vocab normalization (D1), round_time anchored at freeze_end (D2), freeze-frame crop/mask policy (D3), v3 dist_to_bomb plant-gating (D4). All change corpus bytes; doing them after R1 means retraining twice.
2. **Trainer: wire `clean_blob`** into blob loading and point defaults at the merged clean corpus (O2/R4).
3. **Trainer: add `--seed`** (torch/numpy/random + DataLoader generator + worker_init_fn), logged to run config (R2).
4. **Trainer: mask value loss + VALUE_AUC to pre-end frames** — this metric selects `best.pt`, so it must be fixed before training, not after (O3).
5. **Knob 1 decision:** implement pinned scheduled sampling with a wall-time budget, or formally re-lock the recipe to teacher forcing so the coverage claims match training (R1/O1).
6. **Rescale DIST_EDGES_U** and the refine cap for k=4/500ms (R3).
7. **Pin the multi-seed protocol** (n≥3 paired seeds, paired CI) into Knob 2 before any v2-vs-v3 arms launch (T3).
8. **Build the sampled minADE-K rollout harness** (shared with the GRPO group generator) with frozen K/temperature/decoding — the run's acceptance metric must exist before the run (R5).

### Blocking — before the bridge/GRPO gates and any gate-number citation

9. Fix the eval_recon/eval_ablate prompt-slice leak (per-row prompt lengths) (B1).
10. Implement value-agreement and gate the verdict on it (B2).
11. Parameterize gen_bridge_sft from checkpoint args + cache guard + mandatory regen step in the recipe (B3).
12. rollout_eval: clean_blob + map filter + per-map reporting; recompute the Phase-3 gate numbers (E1).
13. dist_coverage_eval: fair sampled best-of-K baseline column (E3).
14. decision_eval: onset bucket + stationary-bucket caveat; rerun hard-turn/reversal and coverage numbers (E2).
15. Harmonize the C1 keystone protocol (round-encoder retrain on clean corpus, identical probe/metric/val) before any "beats the ceiling" claim (T1).

### Non-blocking — documentation, before publication

16. Purge +19.3pp / restate yaw-shuffle status in decisions-ledger; execute the ordered number purges (T2).
17. Correct datasheet 128-tick → 64-tick and audit tick-derived doc numbers (D5).
18. Amend datasheet §4b/§5 for the bomb_state block and freeze-frame composition (D1, D3).
19. EOS truncation / string round-trip in eval_recon; re-derive GRPO τ (B4).
20. Per-map tables in dist_coverage_eval and event_boundary_check (E4).

---

## 4. Appendix — refuted findings (not re-litigated)

- **Unmasked dist-loss corpse frames bias the stationary prior:** already documented (datasheet §4b, recipe Knob 5 mandates the mask); the conditional under-movement mechanism is unconfirmed — dead slots are separable by the alive input bit.
- **Knob 5 alive-mask "absent" as a recipe violation:** the recipe is a prospective spec; the mask is an explicitly tracked pre-canonical-run to-do, and dead-frame class-0 targets are conditionally correct labels.
- **--corrupt-yaw leaves yaw-derived v3 dims intact, invalidating the causal test:** only ~5% of scored conflict frames have LOS exposure, so the leak bounds attenuation at a few pp — footnote-level impurity, not a false-acquittal mechanism.
- **NLA frame-level split inflates recon via same-round siblings:** empirically dead — 0.5% of same-round pairs share text, round-grouped vs frame split changes FVE by −0.006, and a 1-NN memorization probe scores negative FVE.
- **Recon gate contradicts bridge-design on target/beta/standardization:** the doc's dated §7 Step 0 result supersedes §2b and prescribes the grid target the code uses; the measured grid variance profile defeats the dominance mechanism (one-line standardization nit survives).
- **OOD holdout ill-posed via all-zero map one-hot:** de_overpass/de_train are in MAP_VOCAB, so the named mechanism doesn't apply; the all-zero path is exactly the D1 exclusion already gated on a vocab rebuild.
- **README self-contradiction / stale corpus table:** already flagged in methodology-review (fix scheduled, ladder item 9), and the mandated first-read protocol prevents the failure scenario.