# PIVOT NOTES — Audit & Proposal

**Status:** Audit + proposal only. No `.tex`/`.bib` files were touched. No results
were fabricated. This document is a decision aid; the human decides the paper's fate.

**Context:** chimera is pivoting away from the VLM "See, Then Think" approach
(SFT for HUD grounding → GRPO with retrieval-based advantage) toward a
**next-state-prediction world model** as the foundation (causal spatiotemporal
transformer over 597-d game-state frames, trained state→state, judged by probe
transfer; language bridged in as phase 2). This note audits the current paper
against that pivot and proposes how to evolve.

---

## 1. AUDIT — section by section

Legend: **STANDS** = still valid/publishable as-is or with light edits ·
**SUPERSEDED** = the pivot removes its premise or it was always a TODO/placeholder ·
**SALVAGE** = real artifact worth keeping but needs re-homing/reframing.

### Title + Abstract (lines 12, 24–26)
- **SUPERSEDED (framing).** The title "See, Then Think: Two-Phase VLM Training"
  is the discarded thesis. The two-phase VLM-training framing (SFT→GRPO on a VLM)
  is exactly what the pivot abandons.
- **STANDS (one half of abstract).** The second half of the abstract — the RECALL
  retrieval-advantage-collapse characterization (σ_s 0.000 → 0.331 cross-round →
  0.390 untrained encoder; saturation 52.6%; ceiling 0.49 at p=0.60) — is a real,
  self-contained negative result and survives the pivot intact.

### §1 Introduction (`sec:intro`, lines 28–37)
- **SUPERSEDED:** the "perceive then reason on a VLM" hypothesis (the two-phase
  separation argument, lines 35) is the pivoted-away thesis.
- **STANDS / SALVAGE:** the framing motivation (CS2 as a partially-observable,
  HUD-mediated, demo-rich testbed) carries over to the world model. The
  frontier-VLM-HUD-failure observations (Opus 4.6 / base Qwen at 50%) belong to the
  SFT story, not the world model.
- Contribution list (line 37): (1) data pipeline — **STANDS** (reusable);
  (2) SFT recipe +17pp — **SALVAGE** (real but no longer the headline);
  (3)+(4) RECALL + collapse diagnostic — **STANDS** (the durable contribution).

### §2 Related Work (`sec:related`, lines 39–52)
- **STANDS:** Retrieval-based RL, bisimulation/behavioral embeddings, zero-variance
  GRPO, supervised-contrastive-collapse, counterfactual augmentation paragraphs
  (lines 42–50) — all support the RECALL finding and are unaffected by the pivot.
- **SALVAGE/SUPERSEDED:** "VLM training and game AI" paragraph (line 52) is tied to
  the SFT framing. For a world-model paper this section needs **new** related work:
  world models (Dreamer/DIAMOND — note `alonso2024diamond` is *already* in the .bib,
  unused), MuZero-style latent value/policy heads, distributional prediction,
  probing/linear-probe transfer, Flamingo / Mixture-of-Transformers for the language
  bridge. None of that is currently cited for the prediction-model direction.

### §3 Method (`sec:method`, lines 54–214)
- §3.1 Problem Formulation (`sec:problem`, 60–65): **SUPERSEDED** — all TODO; was to
  formalize screenshots→advice. World model needs a different formulation
  (state→state prediction over 597-d frames).
- §3.2 Phase 1 SFT (`sec:sft`, 67–74): **SALVAGE.** Concrete, real, reproducible
  (11-field schema, demo-synchronized GT, LoRA r=4). No longer the paper's core but
  a legitimate self-contained perception result. Note: the pivot's world model is
  text-free, so SFT-for-HUD is *orthogonal*, not foundational, to it.
- §3.3 Phase 2 GRPO (`sec:grpo`, 76–88): **MOSTLY SUPERSEDED.** Heavily TODO. The
  one durable artifact is the documented TRL multimodal GRPO bug + manual-loop
  workaround (lines 86–88) and the 20-step smoke test — methods notes, not results.
  Full GRPO training never ran. Pivot drops VLM-GRPO entirely (GRPO returns only in
  world-model phase 2 as a reasoning reward against actual futures — different setup).
- §3.4 Reward Design (`sec:reward`, 90–109): **SUPERSEDED** — entirely TODO/comments.
- §3.5 RECALL (`sec:recall`, 111–121): **STANDS.** The estimator definition
  (kNN V̂/Q̂, Â = Q̂−V̂, k_min gate) is the substrate for the whole finding.
- §3.6 Diagnosis (`sec:diagnostic`, 123–196): **STANDS — this is the crown jewel.**
  σ_s diagnostic, Tables 1–3 (variance, k-sweep, positive-rule ablation), F1
  same-trajectory leakage, F2 outcome-correlated positive labels. Fully realized,
  real numbers, honest.
- §3.7 Fix (`sec:fix`, 198–207): **STANDS.** Layer 0 retrieval-side mask (the win)
  + Layer 1 counterfactual contrastive (honestly reported null). Survives intact.
- Fig 1 architecture (209–214): **SUPERSEDED** (placeholder; depicts the pivoted-away
  two-phase VLM pipeline).

### §4 Experiments (`sec:experiments`, 216–324)
- §4.1 Setup (`sec:setup`, 221–230): **SALVAGE.** Data (4 demos, 4353 screenshots,
  5309 labels, 3322 SFT), H200, SFT hyperparams, overfitting note — all real.
- §4.2 Baselines (`sec:baselines`, 232–242): **SALVAGE** (Opus/Sonnet/Qwen zero-shot
  HUD baselines). GRPO rows are TODO/**SUPERSEDED**.
- §4.3 Results (`sec:results`, 244–306): **SPLIT.**
  - SFT perception (Table `tab:sft_results`, 67.3% overall, +17pp) — **SALVAGE**, real.
  - SFT training curve (Fig `tab:sft_curve`) — **SALVAGE**, real.
  - RECALL diagnostic paragraph (276) — **STANDS.**
  - Table `tab:results` (full pipeline) — **SUPERSEDED** (empty placeholder; GRPO
    never ran).
- §5 Analysis (`sec:analysis`, 326–343): **SPLIT.** "What SFT learns" / "overfitting
  boundary" — **SALVAGE** (real). "Why Layer 1 did not improve" + "Generalization
  caveats" — **STANDS** (core to RECALL honesty).
- §6 Conclusion (`sec:conclusion`, 345–355): explicitly frames *two contributions of
  differing scope* (SFT application + RECALL methodology). The RECALL half **STANDS**;
  the SFT half is **SALVAGE**; the two-phase-VLM thesis around them is **SUPERSEDED**.

### Net assessment
The paper already contains two loosely-coupled stories: (A) an SFT-for-HUD perception
result, and (B) the RECALL/retrieval-advantage-collapse methodological finding. The
pivot kills the *connective tissue* (two-phase VLM training as the thesis) and the
unfinished GRPO/reward machinery, but leaves (A) intact-but-demoted and (B) intact.
**Nothing real is invalidated by the pivot.** The pivot makes the *world model* the
new flagship — and that work has **no results yet**.

---

## 2. STRATEGIC FORK (for the human to decide)

### Option (a) — Split: standalone RECALL paper + separate new world-model paper
Carve out §3.5–3.7 + relevant Related Work + the RECALL parts of Experiments/Analysis
into a tight standalone methods paper. Start the world model as a brand-new paper.

- **Pros:**
  - RECALL is *already done and honest* — it is publishable now (workshop / short
    methods paper). Splitting lets it ship on its own timeline instead of being held
    hostage to world-model results that don't exist yet.
  - Clean narrative for each. RECALL's value is precisely that it's self-contained
    and generalizes beyond CS2; bolting it onto a world-model paper dilutes it.
  - The world-model paper can be written honestly as "new direction" without an
    awkward "...and also here's an unrelated retrieval finding" appendix.
  - De-risks the pivot: if the world model takes months, RECALL isn't stranded.
- **Cons:**
  - Two writing efforts. The SFT perception result needs a home (could fold into the
    RECALL paper as the "perception foundation / data pipeline" it was built on, or
    become a third short artifact / tech report).
  - Slightly thinner individual papers.

### Option (b) — Reframe THIS paper around the world model; RECALL becomes a lessons-learned subsection
Keep `main.tex` as the vessel, swap the thesis to the world model, demote RECALL +
SFT to "what we learned from the prior approach."

- **Pros:**
  - One document; continuity with existing draft scaffolding and bib.
  - "Here's why we pivoted" lessons-learned framing is honest and increasingly
    common.
- **Cons:**
  - **The world model has zero results.** A reframed paper would be ~90% plan +
    one real-but-now-tangential negative result. Reviewers read that as a position
    paper at best.
  - RECALL deserves to be a *result*, not a war story. Demoting it to a subsection
    undersells a finished, generalizable contribution and buries it under an
    unfinished flagship.
  - Couples RECALL's publishability to world-model progress (the exact risk to avoid).

### Option (c) — Hybrid (RECOMMENDED): ship RECALL standalone now; write the world model as a *plan/position* paper or tech report in parallel; keep SFT as a shared "data + perception" backbone
- Ship **(b-of-a)**: a standalone RECALL methods paper (the §3.5–3.7 + §2 retrieval/
  contrastive related work + RECALL experiments/analysis). Fold the SFT perception
  recipe + data pipeline in as the "system we built RECALL inside of," or spin it out
  as a short tech report — both are real and need no new experiments.
- Start the **world-model paper as a separate document now**, written as a
  *plan/design* paper (clearly labeled, results = TODO) so the thesis, architecture,
  and evaluation protocol get pinned down and reviewed early, but **publish it only
  once probe-transfer results exist** (see §4 below). Do not let it absorb RECALL.

### Recommendation
**Option (c).** Concretely:
1. **Do not delete or rewrite `main.tex` yet.** It holds the two finished artifacts
   (SFT perception + RECALL). Preserve it as the source for the RECALL standalone.
2. Treat RECALL as the publishable unit; it is the strongest finished thing chimera
   has and it stands fully independent of the pivot.
3. Begin the world-model paper as a new file (see §3) in *plan* mode, with every
   numeric claim marked TODO until experiments land.
4. Keep SFT/data-pipeline as a shared backbone asset; cite it from both.

**Why not (b):** the single largest risk in the pivot is conflating "we have a plan"
with "we have a result." Reframing the existing paper around an unbuilt world model
manufactures exactly that conflation and simultaneously demotes the one finished
contribution.

---

## 3. IF a world-model paper is written

### Proposed title
**"Predict, Then Talk: A Game-State World Model as a Foundation for Grounded
Reasoning in Counter-Strike 2"**
(Alternatives: "Game-State Pretraining: Next-Frame Prediction as a Foundation for
Value, Events, and Reasoning"; "World Models over Game State: Probe-Transfer
Evaluation for CS2".)

### DRAFT abstract — **DRAFT, PLAN ONLY. No results exist. All numbers are TODO.**

> *(DRAFT — promises the plan, not results. Every quantitative claim below is a
> placeholder to be filled only after the experiments in §4 of these notes are run.)*
>
> Prior approaches to game understanding bolt language onto perception and supervise
> reasoning with sparse outcome rewards — a regime we found to be info-starved and
> prone to circular supervision. We instead propose treating the **game state itself
> as the pretraining substrate**: a causal spatiotemporal transformer trained to
> predict the next game-state frame (597-d per frame) from past frames, with no text
> in the loop — analogous to language-model pretraining, but over CS2 game states.
> To avoid mode-averaging on multi-modal futures we use a **distributional prediction
> head**, and we sweep prediction horizons (125 ms → 2 s). Crucially, we evaluate the
> learned representation **not by prediction loss but by probe transfer**: linear/MLP
> probes for state value and discrete events read off the frozen latent. We
> hypothesize that (i) events emerge from prediction surprise, (ii) value and policy
> are lightweight heads on the latent (MuZero-style), and (iii) reasoning reduces to
> verbalizing latent rollouts. In a second phase we bridge a **frozen Qwen 3.6/3.7
> (35B-A3B MoE)** into the latent space (Flamingo-style cross-attention, extended
> toward a Mixture-of-Transformers), trained via templated grounding → contrastive
> caster-commentary alignment → GRPO reasoning rewarded against *actual observed
> futures*, with an **ablate-the-latent honesty check** that the language head's
> claims depend on the world-model state. *Results: TODO — probe-transfer AUROC/MAE,
> horizon curves, distributional vs. point-estimate ablation, language-bridge
> grounding and honesty-check numbers are all pending and will not be claimed until
> measured.*

### Proposed section outline
1. **Introduction** — game state as a pretraining substrate; why predict-then-talk;
   explicit "lessons that motivated the pivot" (round-encoder saturation at ~16 demos;
   change-point losses found statistical not semantic boundaries; Claude captions
   circular at +0.008; commentary alignment 4.6σ global but ~25% per-event, ASR-limited
   and parked). State these as *prior findings motivating the design*, not as new
   results of this paper.
2. **Related Work** — world models (Dreamer, DIAMOND [`alonso2024diamond`, already in
   bib], diffusion/transformer world models); MuZero-style latent value/policy;
   distributional prediction; representation probing / linear-probe transfer;
   Flamingo & Mixture-of-Transformers for the language bridge; (carry over) retrieval-
   RL and contrastive-collapse as background to the GRPO-against-futures reward.
3. **Game-State Representation** — the 597-d frame; tokenization/normalization;
   corpus construction from demos (reuse the existing data pipeline).
4. **World Model** — causal spatiotemporal transformer; distributional head; horizon
   sweep (125 ms → 2 s); training objective (state→state, text-free).
5. **Evaluation Protocol (probe transfer)** — value probe, event probes;
   prediction-surprise → event correspondence; *why probe transfer, not pred loss.*
6. **Heads on the Latent** — value/policy as MuZero-style heads; reasoning as rollout
   verbalization.
7. **Language Bridge (Phase 2)** — frozen Qwen bridged via Flamingo→MoT; training
   curriculum: templated grounding → contrastive caster-commentary → GRPO vs. actual
   futures; ablate-the-latent honesty check.
8. **Experiments** — (RESULTS TODO; see §4 of these notes for the gating list.)
9. **Lessons from the Prior Approach** — short: SFT-for-HUD perception result and the
   RECALL retrieval-advantage-collapse finding, *if not published separately*; cite
   the standalone otherwise.
10. **Limitations & Future Work.**
11. **Conclusion.**

---

## 4. RESULTS REQUIRED BEFORE ANY WORLD-MODEL CLAIM (anti-fabrication checklist)

None of the following exist yet. Each must be measured before the corresponding claim
may appear unhedged in any paper.

**World model core**
1. Trained world model exists; report corpus size (demos/frames) and that it trained
   to a stable objective. (Note prior round-encoder saturated at ~16 demos — must show
   the prediction objective does **not** saturate, or characterize where it does.)
2. **Horizon sweep** (125 ms → 2 s): prediction quality vs. horizon curve.
3. **Distributional vs. point head ablation**: evidence the distributional head
   actually avoids mode-averaging (e.g., calibration / multi-modal-future capture),
   not just lower loss.

**Probe transfer (the declared judge)**
4. **Value probe** transfer numbers (e.g., AUROC/MAE for round-win or state value)
   from frozen latent vs. baselines (raw-features probe; the saturated round-encoder;
   random-init latent control).
5. **Event probes**: detection metrics for discrete events from the latent.
6. **Events-from-surprise** claim: quantitative correspondence between prediction
   surprise spikes and real events (precision/recall), or it stays a hypothesis.
7. **Baselines/controls** for every probe: untrained-latent control, raw-597-d-input
   probe, and at least one prior representation (e.g., the round-encoder) — mirroring
   the honesty discipline already shown in RECALL (untrained-MiniLM control).

**Heads on latent**
8. Value/policy head performance (MuZero-style) with the same control baselines.

**Language bridge (Phase 2) — only after world model is validated**
9. Grounding quality after templated-grounding stage.
10. Contrastive caster-commentary alignment numbers (note prior: 4.6σ global but ~25%
    per-event, ASR-limited — must improve or be scoped honestly).
11. GRPO-vs-actual-futures reasoning reward results.
12. **Ablate-the-latent honesty check**: measured drop in language-head correctness
    when the latent is ablated/randomized — the core evidence that the model is
    *grounded* in the world model and not confabulating. Without this number, no
    "grounded reasoning" claim.

**General**
13. Any cross-setting/generalization claim (other maps, tournaments, non-CS2) requires
    held-out evaluation, not asserted transfer.

Until items 1–8 exist, the world-model paper is a **plan/position paper** and must be
labeled as such. Items 9–12 gate any language/grounding claims.

---

## Quick reference: disposition map

| Paper element | Disposition |
|---|---|
| Title / two-phase-VLM thesis | SUPERSEDED |
| RECALL estimator + diagnosis + fix (§3.5–3.7, Tables 1–3) | STANDS (publish standalone) |
| RECALL-related Related Work (retrieval-RL, bisimulation, GRPO zero-var, SupCon collapse, CoDA) | STANDS |
| SFT-for-HUD recipe + Table 1 results + training curve | SALVAGE (real; demote to backbone/tech report) |
| Data pipeline (4 demos, 4353 shots, 5309 labels) | SALVAGE (reusable for world model) |
| Zero-shot VLM baselines (Opus/Sonnet/Qwen) | SALVAGE (tied to SFT story) |
| Phase 2 GRPO machinery, Reward Design, full-pipeline table | SUPERSEDED (TODO/never ran) |
| TRL multimodal-bug + manual-loop note | SALVAGE (methods note) |
| Architecture figure (Fig 1) | SUPERSEDED (depicts old pipeline) |
| World model + language bridge | NEW (no results yet — see §4) |
