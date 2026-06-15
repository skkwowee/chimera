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
| NLA round-trip faithfulness objective (text-only decoder) | NEW (proposal — see §5; metric-first, no results yet) |

---

## 5. PROPOSAL — Bridge as a Natural-Language Autoencoder (NLA round-trip faithfulness)

**Status:** Proposal only. No `.tex`/`.bib` files were touched by this section either (see
§5.6 for the reason `main.tex` was deliberately left alone and the drop-in LaTeX the human
can place once the world-model paper exists). No results fabricated; every numeric claim
remains TODO and gated by §4 + the kill-tests below.

This extends the language-bridge plan (§3 item 7) with a single new piece: a **text-only
decoder** that reconstructs the frozen world-model latent from Qwen's *generated* reasoning,
turning the latent→text bridge into a **Natural-Language Autoencoder (NLA)** — an autoencoder
whose bottleneck is text. The bridge we already designed is the *encoder* half (latent→text);
this adds the *decoder* half (text→latent). The decoder is built and reported as a
**faithfulness metric first**; using it as a training signal is a hedged, gated follow-on.

### 5.1 The contribution claim (headline bullet for the world-model paper)

> **A round-trip faithfulness objective and metric for grounded reasoning.** We close the
> latent→text bridge into a Natural-Language Autoencoder by training a **text-only decoder**
> that reconstructs the frozen world-model latent from Qwen's generated reasoning.
> Reconstruction fidelity (cosine / variance-explained to the true latent, scored against
> shuffled-text, latent-mean, and ablated-bridge controls) is a **direct, label-free
> measure** of whether the language faithfully renders the model's understanding rather than
> hallucinating past it.

**Conservative phrasing (required, to match the project's honesty discipline):** high
reconstruction fidelity is *necessary evidence* of grounding; report it **alongside, not in
place of**, value/event agreement. **Do not** claim "provably grounded" — the result is
empirical and upper-bounded by decoder capacity.

### 5.2 Why this is worth adding — what it buys over the existing plan

The existing plan (§3 item 7, §4 item 12) gates grounding on **ablate-the-latent**: zero/shuffle
the soft tokens and show latent-on beats latent-off. That proves the bridge *uses* the latent.
It is **blind to embellishment**: a bridge can pass ablate (uses the latent) yet have Qwen add
fluent tactical detail that is *not in the latent* (the Qwen-embellishment risk already flagged
in the bridge design). Reconstruction answers the orthogonal question — *is the latent faithfully
rendered in the text, or hallucinated past?* — with a number. The two are **complementary; recon
subsumes nothing**:

| Eval | Question | Role |
|---|---|---|
| **Ablate-the-latent** | Does the text *use* the latent at all? (necessary) | **First gate** (the Era-1 circularity tripwire; unchanged). |
| **Recon-fidelity** | Is the latent *faithfully rendered*, or embellished past? (sufficiency) | **Second gate** (the missing faithfulness leg). |

Second, it is a **label-free** signal: training the decoder needs no answer key, sidestepping
the "where do correct descriptions come from / is a teacher LLM stable enough to read raw
ticks?" problem — the very problem the world model exists to solve, so the NLA loop must not
re-introduce a tick-reading teacher.

### 5.3 Method framing (for the Methods section of the world-model paper)

New subsection, slots into the Language Bridge section (§3 item 7): *"Round-trip grounding: a
Natural-Language Autoencoder over the latent."*

- **Target.** The decoder reconstructs the **pooled-512 latent** `z = h.mean(dim=2) ∈ R^512` —
  exactly the vector the trained world model's `value_head` consumes (verified in
  `scripts/train_world_model.py`: `_grid(x)` returns `h` of shape `[B,L,11,512]`, post-LayerNorm,
  and `value_head(h.mean(dim=2))`). It is provably value-bearing and low-enough-dim to be
  text-expressible. Targets are standardized per-dim (fixed train-set mean/std) before the loss.
  Fallback ladder if capacity fails (§5.5): (2) top-k PCA of the head-Jacobian subspace, then
  (3) head outputs directly (value logit, per-player top-1 displacement class, rollout
  value mean/spread). Full `[11,512]` grid is a stretch only after pooled-512 shows a clean gap
  (slots are identity-fixed by `slot_emb`, so per-token reconstruction needs no matching).
- **The text-only firewall (non-negotiable).** The decoder consumes **only the decoded answer
  string**, detached and re-tokenized in a fresh context (no soft-token prefix, no prompt, no
  KV-cache that ever saw the soft tokens), re-embedded by frozen Qwen, then a separate
  reverse-Perceiver (own learned queries, **weights not tied** to the forward resampler) →
  MLP → `ẑ`. There must be zero tensor path from the latent / soft tokens to the decoder's
  input. Enforced by `stop_gradient` + an assertion + a unit test: `recon_from_shuffled_text`
  and `recon_from_empty_text` must score at the latent-mean floor; if not, the firewall is
  broken and every fidelity number is invalid.
- **Loss.** `L_recon = (1 − cos(ẑ, z)) + β·MSE(ẑ, z)`, β ≈ 0.1; cosine is the headline (scale-
  invariant, matches how downstream heads read direction), MSE secondary on standardized targets.
- **Reported metric is ALWAYS a separate, frozen-verbalizer, held-out decoder** — trained on
  frozen bridge outputs with no gradient to Qwen/resampler/LoRA. A *jointly*-trained decoder
  co-adapts with the verbalizer into a private cipher (high fidelity, unreadable text — the
  documented NLA failure mode and this project's own circularity scar). Joint use is allowed
  only as a small, annealed (λ≈0.05–0.1), decoder-frozen *auxiliary regularizer* after ablate
  passes, never as the primary objective; the literal REINFORCE NLA loop is research-stretch,
  KL-anchored, gated behind milestone success.
- **In GRPO: a constraint, not a summed reward.** Do not add `λ·recon` to the GRPO reward
  (that lets the policy trade reasoning for info-density → stilted text, and double-counts
  grounding already in value-through-rollout). Instead **zero the advantage** of any group
  completion whose recon-fidelity < τ (≈25th percentile of the SFT-passing distribution) before
  group-normalizing — faithfulness as a *feasibility constraint*, value-through-rollout the sole
  quality signal. Eval-only fallback if the constraint starves the group.

### 5.4 Eval framing (for the Evaluation section of the world-model paper)

Promote ablate-the-latent + recon-fidelity into one **"Faithfulness"** subsection. Green light
requires **two numbers, not one** (replaces the single ablate gate in §4 item 12):

> `latent-on > latent-off` (grounding exists) **AND** `recon(real text) > recon(shuffled/ablated
> text)` above the capacity floor, with value-head-agreement and readability intact (grounding
> is faithful).

**Mandatory anti-gaming controls — report fidelity ONLY against these, never bare:**
(a) **shuffled-text** (decode a different example's text → must collapse to floor);
(b) **ablated-bridge** (text generated with soft tokens zeroed → must reconstruct worse; the
delta `recon(latent-on) − recon(latent-off)` is the single strongest honesty number — it shows
the text carries *latent-specific* information, not prompt patterns);
(c) **latent-mean baseline** (report fidelity as **fraction-of-variance-explained over this
floor**, not raw cosine — raw cosine on a low-rank post-LayerNorm manifold is deceptively high);
(d) **empty/scrambled-text invariant** (firewall audit, must hit floor).

**Faithfulness-gibberish guard (mandatory).** Recon-fidelity is never reported or gated *alone*
— it can be satisfied by latent-encoding steganographic gibberish. Always pair with: a
**value-head-agreement** number (feed `ẑ` through the frozen `value_head`, compare P(CT win) to
truth) **and** a **readability/perplexity guard** (base-Qwen perplexity or human check). A
passing faithfulness claim requires fidelity-above-controls **AND** value-agreement **AND**
readability.

### 5.5 First milestone & the cheap LOCAL kill-test (anti-fabrication, add to §4)

Front-load a **local capacity kill-test before any pod spend**:
- **Step 0 (local, 4090, no pod):** cache `(h, z=h.mean, value, 1-step rollout summary,
  templated text)` from `wm_3map_dist_v3m`; train *only the decoder* on **templated text →
  target** across a sweep: text-budget ∈ {32,64,128,256} tokens × target ∈ {pooled-512,
  +rollout, value+rollout+top-PCA}. Plot fidelity vs budget. **This can kill the NLA idea
  before QLoRA if text provably can't carry the signal** (no Qwen / no world-model forward at
  train time — latents cached).
- **Step 1 (pod):** QLoRA on ~20–50k templated-from-predictive-heads pairs (aux λ default 0).
- **Step 2:** run **both** gates (ablate + recon, with all §5.4 controls).

**Add to the §4 checklist:** item 12 (ablate-the-latent) is **necessary but not sufficient**;
a new **item 12b — recon-fidelity above the capacity floor and above shuffled/ablated controls,
with value-agreement + readability** — gates any *faithful*-grounding claim. New kill-criteria
(report as negative results, do not quietly drop): capacity-floor failure (≤256 tokens can't
beat the latent-mean floor even for value+rollout); firewall failure (shuffled/empty doesn't
collapse to floor); steganographic degenerate code (recon-as-reward drives perplexity down);
no-latent-specific-signal (`recon(real) ≈ recon(shuffled/ablated)`).

**Honest caveats (must appear in docs + paper):** recon-faithful ≠ good reasoning (NLA certifies
information preservation only; reasoning quality is GRPO's job); capacity is real (don't chase
full `11×512=5632`-dim reconstruction — measure the decision-relevant subspace); double-counting
guard (target the world-model latent `z=h.mean`, *not* the value/rollout channels appended to the
bridge input, else "faithfulness" only measures whether the text echoes the channels); static-
derivable confound (weight the target toward predictive/foresight channels per the Line-in-the-
Sand discipline, or recon inherits the circularity it was meant to detect).

### 5.6 Where it slots into the paper — and why `main.tex` was NOT touched

This objective belongs to the **world-model paper** (the new document recommended in §2 Option c
and outlined in §3), specifically:
1. **Methods** — new subsection inside the Language Bridge section (§3 item 7): the NLA round-trip,
   the text-only firewall, the recon-as-constraint design.
2. **Evaluation** — a unified **"Faithfulness"** subsection: *ablate proves the latent is USED;
   reconstruction proves the OUTPUT TEXT preserves it.*
3. **Related Work** — cite Anthropic **Natural Language Autoencoders** as the direct antecedent
   (`@misc{anthropic2025nla}` below; `alonso2024diamond` already in `references.bib`). Position
   novelty: NLA interpreted *LLM activations*; we apply the round-trip principle to a **harder
   bottleneck** — a frozen multi-agent spatiotemporal game-state latent — and use the decoder
   primarily as a **faithfulness metric for a reasoning system**. Differentiate from CoT-
   faithfulness (perturbs the trace) and probing (reads the latent directly): reconstruction is
   the only one that scores the **generated text's** information content end-to-end.
4. **Capacity-as-a-result figure:** the recon-cosine-vs-token-budget curve and the per-channel
   reconstruction profile (value/event recoverable from text vs. fine spatial jitter not) are
   themselves a contribution characterizing natural-language-as-a-channel; establish the
   in-principle ceiling with an oracle decoder trained on the templated text.

**Why `main.tex` was deliberately left untouched.** `main.tex` is firmly VLM-era and contains no
safe insertion point: the title ("See, Then Think: Two-Phase VLM Training"), abstract, intro, and
*every* Method subsection are about HUD-SFT + RECALL; its "Phase 2" is **VLM-GRPO**, not the
world-model language bridge (which does not appear anywhere in the document). There is no
approach/methods/future-work section discussing *this* bridge to extend. Adding an NLA subsection
there would orphan it next to a discarded thesis and risk implying VLM-era results. Per §2 (Option
c), `main.tex` is preserved as the source for the standalone RECALL paper; the NLA objective lives
here until the world-model paper exists. The drop-in LaTeX below is ready for the human to place
into that new document.

### 5.7 Drop-in LaTeX (for the world-model paper — NOT for `main.tex`)

```latex
% ===== Methods: inside the Language Bridge section =====
\subsection{Round-Trip Grounding: a Natural-Language Autoencoder over the Latent}
\label{sec:nla}
The language bridge maps the frozen world-model latent to reasoning text; viewed as the
\emph{encoder} half of a Natural-Language Autoencoder~\cite{anthropic2025nla}, it is
incomplete without a \emph{decoder} that reconstructs the latent from text. We add a
\textbf{text-only decoder} $R$ that, given \emph{only} the model's generated answer string
$y$ (detached, re-tokenized in a fresh context, re-embedded by the frozen language model),
predicts $\hat{z} = R(y)$, where the target $z = \bar{h} \in \mathbb{R}^{512}$ is the
slot-pooled world-model latent that the value head consumes. The decoder is a separate
reverse-Perceiver (its queries untied from the forward resampler) followed by an MLP, trained
to minimize $\mathcal{L}_{\text{recon}} = \bigl(1 - \cos(\hat{z}, z)\bigr) + \beta\,\lVert
\hat{z} - z\rVert_2^2$ on standardized targets. A firewall is enforced in code: there is no
tensor path from the latent or soft tokens to $R$'s input, audited by the invariant that
shuffled- and empty-text reconstruction collapses to the latent-mean floor. We report
reconstruction fidelity from a \emph{separate, frozen-verbalizer} decoder trained on held-out
bridge outputs; in GRPO it acts as a feasibility \emph{constraint} (zeroing the advantage of
completions below a fidelity threshold) rather than a summed reward, leaving value-through-
rollout as the sole quality signal.

% ===== Evaluation: Faithfulness subsection =====
\subsection{Faithfulness: Does the Text Render the Latent?}
\label{sec:faithfulness}
We separate two questions. \emph{Ablate-the-latent} (zero/shuffle the soft tokens) tests
whether the text \emph{uses} the latent. \emph{Reconstruction fidelity} tests whether the
generated text \emph{faithfully renders} it: we reconstruct $z$ from the model's own output
and report variance-explained over a latent-mean floor, against (a)~a shuffled-text control,
(b)~an ablated-bridge control --- the delta $\text{recon}(\text{latent-on}) -
\text{recon}(\text{latent-off})$ isolates latent-specific information --- and (c)~the latent-
mean baseline. To rule out steganographic codes, we additionally require value-head agreement
($\hat{z}$ pushed through the frozen value head must match $P(\text{win})$) and a readability/
perplexity guard. High fidelity is necessary evidence of grounding, reported \emph{alongside},
not in place of, value and event agreement; we do not claim provable grounding, as the metric
is upper-bounded by decoder capacity.
```

```bibtex
@misc{anthropic2025nla,
  title  = {Natural Language Autoencoders},
  author = {Anthropic},
  year   = {2025},
  note   = {Anthropic interpretability research; round-trip activation--text reconstruction
            trained on reconstruction fidelity. PLACEHOLDER — fill exact authors/venue/URL
            before submission.}
}
```

> **Citation caveat:** confirm the exact authors, title, year, and URL of the Anthropic
> Natural Language Autoencoders work before submission; the entry above is a placeholder
> scaffold, not a verified bibliographic record.
