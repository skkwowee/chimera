# World-Model Paper — Outline & Skeleton

This is the skeleton for the Chimera world-model paper: title candidates, an
abstract draft anchored only in established facts, the section plan, result-table
skeletons, and the limitations section. Read it when drafting paper text or when a
runbook gate lands a number that fills one of the bracketed slots. It restates
nothing that has a canonical home: training decisions live in
`docs/retrain-recipe.md` (+ `retrain-recipe-knobs4-7.md`), corpus facts in
`docs/datasheet.md`, rationale in `docs/first-principles-plan.md`, and the
phase-2/3 design in `docs/bridge-design.md`. The archived VLM-era draft
(`main.tex`) is covered by `PIVOT_NOTES.md`.

**Status: no headline result exists.** Every result slot below names the runbook
gate ([1]–[7] in `claude-progress.txt`) that fills it. Until runbook [7] lands,
this paper is a plan with pre-registered criteria, and must be labeled as such if
circulated.

---

## 1. Title candidates

1. *Predict, Then Talk: A Game-State World Model as a Foundation for Grounded
   Tactical Reasoning in Counter-Strike 2*
2. *Game-State Pretraining: Next-State Prediction as a Foundation for Value,
   Events, and Reasoning*
3. *World Models over Game State: Pre-Registered Probe-Transfer Evaluation for CS2*

## 2. The claim ladder (C1–C3)

The paper claims exactly three things, each behind a gate. A rung that fails ships
as its committed failure branch — no salvage wording.

**C1 — representation (the keystone).** Pre-registered exact wording
(`retrain-recipe-knobs4-7.md` §Knob 7):

> "The frozen latent of a world model pretrained ONLY on next-state prediction
> (zero outcome gradient into the trunk, unit-test enforced) is a better substrate
> for round-outcome probes than raw features, window-mean raw features, and the
> retrained round-encoder representation, under identical corpus, split, and probe
> budget — and its probe transfer keeps improving with training data past the
> round-encoder's historical saturation point."

Pass rule (pre-registered, §7e): **G0 ∧ C1-REP ∧ C1-SCALE** — G0: `wm` beats
`rand_wm`, paired 95% CI > 0; C1-REP: pooled linear val AUC delta over the best
baseline ≥ +0.02 with paired CI excluding 0, and ≥ 4 of 5 maps; C1-SCALE:
AUC_wm(N_full) − AUC_wm(N_sat) ≥ +0.01 with paired CI excluding 0. Committed
failure branches: C1-REP fails → C1 is falsified and the paper reframes around the
coverage/option-set result; C1-SCALE fails alone → the claim reduces to "better
representation at matched data" and the scaling clause is dropped, not softened.
Gate: runbook [6]–[7].

**C2 — faithful translation.** A small trainable bridge (featurizer + 32-query
Perceiver → soft tokens) lets a frozen Qwen3.6-35B-A3B verbalize the frozen
world-model latent, and the text both *uses* and *faithfully renders* it. Gates
(all four legs co-equal; `docs/bridge-design.md` §2b–§3, runbook [7] CHANGES D–E):
ablate-the-latent, NLA reconstruction fidelity above floor and controls
(with value-head agreement and readability), the exact-match fact audit against
tick ground truth, and the counterfactual flip-test. Preceded by the three
pre-QLoRA referees (grid-target capacity probe, SAE/probe audit, bridge-tap layer
sweep) before any 35B pod spend.

**C3 — grounded improvement.** GRPO whose reward scores verbalized predictions
against the *realized demo future* (CRPS/Brier over extracted claims) improves
forecast truthfulness of novel sampled text; value-through-rollout serves only as
the group generator, never the reward. Gates (runbook [7] CHANGE F,
`docs/bridge-design.md` §5): text→claim checker AUC ≥ ~0.75 on pseudo-gold,
reward-noise ICC ≥ ~0.2 under single-draw future noise, and one offline
ReST/rejection-sampling round before any on-policy GRPO.

## 3. Abstract draft

Established facts only; every result is a bracketed slot naming its gate.

> Competitive Counter-Strike 2 is a round-structured, zero-sum, partial-information
> game whose futures are genuinely multimodal at decision points. It has no fast
> headless simulator, so we treat the game state itself as the pretraining
> substrate: a 19M-parameter causal transformer over parsed professional demos
> (92 HLTV matches, 8 Hz, 11 tokens per frame, match-level split) trained only to
> predict the next state — 12 s of history to a 500 ms-ahead distribution over
> per-player displacement — with longer horizons reached by sampled autoregressive
> rollout. The design is forced by measured failures of the alternatives: outcome
> supervision carries ~1 bit per round and saturated a supervised round encoder at
> roughly 16 demos; caption supervision was circular (+0.008); and an L2 head
> mode-averages (regression jitter 28–32u against a 3u copy baseline), which
> mandates the distributional head. The value head is detached (zero outcome
> gradient into the trunk, unit-test enforced), and the representation is judged
> never by prediction loss but by frozen linear probes against pre-registered
> floors (raw features, an untrained latent) and ceilings (a retrained round
> encoder and a matched-capacity supervised model), plus a held-out map.
> [C1 keystone deltas — pending runbook [7] keystone probes.] [Coverage minADE-K
> vs the fair stochastic baseline — pending runbook [4]/[6].] [OOD overpass
> transfer — pending runbook [7], Knob 4.] In a second phase a small trainable
> bridge conditions a frozen Qwen3.6-35B-A3B on the frozen latent, gated by
> ablation, round-trip reconstruction (NLA), a fact audit, and counterfactual
> flip-tests [numbers — pending runbook [7] bridge gates]; in a third, GRPO
> rewards verbalized predictions against the realized demo future
> [numbers — pending runbook [7], CHANGE F].

Note: the historical round-encoder ceiling (0.759) is retired from all claims and
survives only as motivation history; the ceiling the paper gates against is
re-measured under the identical protocol (Knob 7a).

## 4. Section plan

1. **Introduction.** Game state as pretraining substrate; the causal chain
   compressed (`first-principles-plan.md` §1); C1–C3 stated with their gates.
   Prior findings that motivated the design — round-encoder saturation, caption
   circularity (+0.008), commentary alignment (4.6σ global but ~25% per-event,
   ASR-limited, parked), the mode-averaging verdict — presented as motivating
   measurements, not results of this paper.
2. **Related work.** World models (Dreamer, DIAMOND — `alonso2024diamond`, in
   the bib); MuZero-style heads-on-latent; distributional prediction
   (C51, MultiPath/MTR anchor-classify-refine, BeT); representation probing
   (Alain & Bengio, control tasks, SSL linear eval); frozen-LM bridges
   (Flamingo, BLIP-2, QLoRA); the NLA antecedent (citation still a placeholder —
   verify before submission); RLVR/GRPO; carried-over retrieval-RL and
   contrastive-collapse background.
3. **Corpus.** Summary only; `docs/datasheet.md` is canonical. 92 matches →
   78 train / 14 val (match-level, leak-audited); 5-map gating corpus with 641
   clean val rounds; de_overpass (367 rounds) fully held out; defect registry and
   patch lineage disclosed.
4. **World model.** 19M causal transformer, round-as-document attention;
   97-class classify-then-refine displacement head; scheduled sampling with the
   SS-off control (negative-result clause honored); detached value head.
   Canonical: recipe Knobs 1–7 (LOCKED, pre-registered). Include the cv-residual
   negative result (better single-step, 2.5× closed-loop blowup) as a
   design-motivating finding from the prior corpus.
5. **Evaluation protocol.** Frozen linear probes, six representations, identical
   sample manifest, pre-registered thresholds; the scaling grid; coverage
   (minADE-K per depth) with the fair stochastic baseline and the
   trajectory-coherence metric; OOD protocol (Knob 4).
6. **Results.** Skeleton in §5 below; maps to C1.
7. **Language bridge and faithfulness (C2).** `docs/bridge-design.md` is
   canonical; report all four gate legs plus the pre-QLoRA referee outcomes.
8. **Grounded GRPO (C3).** Reward design, feasibility gates, offline round first.
9. **Lessons from the prior approach.** The RECALL retrieval-advantage-collapse
   finding and the SFT-for-HUD perception result — cited from the standalone if
   published separately (see `PIVOT_NOTES.md`), summarized here otherwise.
10. **Limitations** (§6 below). 11. **Conclusion.**

## 5. Results-table skeletons

All cells empty until the named gate lands.

- **T1 — Keystone probes** (gate: runbook [6]–[7], Knob 7c/7e). Rows: `raw_last`,
  `raw_mean`, `raw_mean_full`, `rand_wm`, `re_h`, `wm`; secondary rows: co-trained
  twin, MLP continuity table, matched-capacity supervised ceiling (CHANGE C).
  Cols: pooled linear val AUC (95% paired bootstrap CI), per-map, @0.25 early
  bucket. Both disclosed asymmetries (RE full-round history vs 96-frame window;
  raw 597-d vs latent 512-d) footnoted.
- **T2 — Scaling curve** (gate: [6]–[7], Knob 7d). AUC_wm and AUC_re at
  N ∈ {4, 8, 16, 32, 64, N_full} matches; C1-SCALE anchor deltas; RE slope
  reported, never gating.
- **T3 — Coverage** (gate: [4]/[6], CHANGE B). minADE-K at depths
  {2, 4, 10, 14, 20} for the model, the damped-CV + fitted-residual-covariance
  baseline (K=16, scored identically), and the SS-off control; mode-switch-rate
  trajectory-coherence column.
- **T4 — OOD overpass** (gate: [7], Knob 4). Per-horizon coverage skill and the
  probe-transfer row, with the ID-zeroed control.
- **T5 — Faithfulness** (gate: [7], CHANGES D–E). Ablate delta; recon
  fraction-of-variance-explained over the latent-mean floor vs shuffled-text and
  ablated-bridge controls; value-head agreement; readability; fact-audit
  exact-match rate; counterfactual flip rate.
- **T6 — GRPO** (gate: [7], CHANGE F). Checker AUC on pseudo-gold; reward ICC;
  CRPS/Brier of verbalized forecasts before/after the offline round and after
  on-policy GRPO.

## 6. Limitations (from the weak-links register scope lines)

- **Skill tier.** All data is professional; nothing is claimed about lower tiers
  without an OOD test.
- **Generalization scope.** Global /3000 coordinates fingerprint maps and the
  holdout is n=1 (overpass), so the OOD claim is data-transfer plus
  label-efficiency, not layout invariance.
- **Reasoning := forecasting.** The corpus has no action-conditioning, so there
  are no verifiable counterfactuals; counterfactual advice is out of scope. The
  claim is faithful *translation* plus *forecast* quality.
- **Faithfulness ceiling.** Reconstruction certifies information *presence* in
  the text, never meaning or mechanism — which is why the ablation, fact-audit,
  and flip-test legs remain co-equal, and no "provably grounded" claim is made.
- **Selection channel.** Architecture and hyperparameters were tuned while value
  curves were visible in the prior era; the unit test closes the gradient path,
  not this channel. Disclosed; bounded empirically by the co-trained twin.
- **Cadence.** The 8 Hz rate is inherited and unablated against 100–250 ms
  peeker-advantage windows (standing datasheet TODO, post-R1).
- **Corpus size.** Defended by a saturation number measured under the abandoned
  sparse objective; the dense-objective scaling curve is itself a deliverable
  ([6]), not an assumption.
- **Schema gap.** Active utility (smokes/mollies) is invisible to the model
  despite being a perception primitive by the project's own rule (post-R1
  re-bake item).
- **Reward noise.** Each realized future is a single draw from a stochastic
  branch; the ICC gate measures, but does not remove, this noise.
- **Bounded channel (also a result).** The decision-relevant frame content is
  ~6–8 effective dims; the model's decision state is far simpler than the human
  tactical vocabulary, which bounds what any bridge text can faithfully contain.

## 7. Anti-fabrication rule

No number appears unhedged in any draft until its gate lands. C1's criteria and
failure branches are pre-registered by git commit; C2/C3 gates are written in the
runbook and bridge-design before implementation. Negative results (SS-off,
capacity-floor or firewall failures, checker/ICC failures) are reported, not
dropped.

## 8. References status (audited 2026-07-19)

Verified real: `shao2024deepseekmath`, `bai2023qwen`, `yang2024qwen2`, `hu2022lora`,
`vonwerra2023trl`, `liu2023visual`, `rafailov2024direct` (year corrected to 2023),
`achiam2023gpt4`, `alonso2024diamond`, `zhang2026memrl`, `msgrpo2025` (authors
corrected to the actual arXiv 2506.04746 author list), `zhang2025tompo`,
`chen2025vlaathinking`, `zerovariance2025` and `ngrpo2025` (both real; "Anonymous"
author fields must be filled before submission), `khosla2020supervised`,
`graf2021dissecting`, `schroff2015facenet`, `pritzel2017neural`,
`pitis2020counterfactual`, `castro2020scalable`, `gelada2019deepmdp`,
`zhang2021learning`, `agarwal2021contrastive`, `goyal2022retrieval`.

Unverified — flagged in the .bib, do not cite without independent verification:
`chen2024game`, `huang2022strategic`, `zhao2023counterstrike`. The Anthropic NLA
entry remains a placeholder scaffold (see `docs/bridge-design.md`); confirm exact
authors/venue/URL before submission.
