# Phase 2 — Language Bridge: Architecture & SFT-Bootstrapping

**Goal.** Connect the *frozen* world-model latent to Qwen3.6-35B-A3B so the LLM
produces tactical reasoning **grounded in the learned game model** — not
paraphrased from features. This is the thesis-defining component and the longest
pole (35B on a pod, solo).

**Non-negotiable framing** (see `decisions-ledger.md` §5): the world model
*understands*; the bridge only *translates*. The understanding was baked by
Phase-1 next-state prediction. SFT here teaches a latent→language **mapping**,
not understanding. Reasoning *quality* comes later, from GRPO (Phase 3). So the
bridge SFT only has to be good enough to bootstrap — do not over-invest in it.

```
WORLD MODEL (frozen) ──latent──► RESAMPLER ──soft tokens──► QWEN3.6 (QLoRA) ──► reasoning text
   d_model=512                   (trainable)                  (base frozen,
   per-frame: 11 tokens                                        LoRA + new xattn
   (10 players + global)                                       trainable)

   z_hat <- RECONSTRUCTOR (reverse-Perceiver) <- re-embed TEXT ONLY <- generated text
   recon-fidelity vs z=h.mean(dim=2): round-trip faithfulness (firewall: text-only)
```

The bridge is the **encoder half of a Natural-Language Autoencoder** (latent→text);
§2b adds the **decoder half** (text→latent) whose reconstruction fidelity is a
label-free *faithfulness* metric — does the generated text render the latent, or
hallucinate past it? NLA antecedent: Anthropic, *Natural Language Autoencoders*.

---

## 1. The two interfaces

### Interface A — what latent goes in
The world model emits, per frame, a contextualized **token grid** `[B, L, 11, 512]`
(10 player tokens + 1 global). Two modes of consumption:

- **Single-moment reasoning** ("what's happening here / what should X do"):
  feed ONE frame's 11 tokens. Preserve the per-player structure — do NOT mean-pool
  to one vector; the LLM should be able to attend to individual players. 11 tokens
  in.
- **Round/sequence reasoning** ("narrate this round / why did CT lose"):
  feed the **event-latents** selected by the adaptive clock (surprise-gated,
  `decisions-ledger.md` §2) — ~10–30 event frames × 11 tokens. This is the
  payoff of the clock: it makes a whole round fit in sentence-length input
  (~110–330 tokens) instead of ~hundreds of frames. Until the clock is
  productized, sub-sample uniformly (every k frames) as a stand-in.

**Augment the latent with the predictive-head outputs** (the part that is NOT in
raw features — this is what makes grounding non-circular, see §5): per fed frame,
append `value` logit (P(CT win)) and a compact summary of a K-sample rollout
(e.g. value mean/spread at +2s/+4s, and the top predicted displacement-class per
player). These ride as extra channels concatenated to each frame's tokens before
the resampler.

### Interface B — what text comes out
Free-form tactical reasoning conditioned on a task prompt. The text channel is
ordinary Qwen output; nothing special. Examples of task prompts: "Assess this
moment for the CT side.", "What is the T-side's best play?", "Narrate the key
events of this round."

---

## 2. Architecture (the trainable bridge)

Three trainable pieces; **the world model and the Qwen base are frozen.**

1. **Latent featurizer (tiny MLP).** Per token: `512 (+ predictive channels) →
   d_proj`. LayerNorm. Adds a learned modality/role embedding (which player slot,
   or global) so the resampler keeps agent identity.
2. **Perceiver resampler.** M learned queries (start M=32) cross-attend to the
   variable-length latent-token sequence → **M fixed soft tokens** in `d_llm`
   (Qwen hidden; read from config at build time). Handles both single-moment
   (11 in) and event-sequence (K×11 in) with the same M out. 2–4 layers.
   Standard Flamingo/IDEFICS component; well-trodden.
3. **Injection into Qwen — start LLaVA-style, upgrade to Flamingo if needed.**
   - **v1 (default): soft-prompt prefix.** Prepend the M soft tokens to the text
     embedding sequence; train **QLoRA** (4-bit base — required to fit 35B on one
     80GB pod; the FP8 repo variant exists as an alternative) on Qwen.
     Simplest, fastest to a result, ablate-the-latent is trivial (zero the soft
     tokens). Cost: soft tokens consume context budget and compete in
     self-attention.
   - **v2 (upgrade only if v1's grounding is weak): gated cross-attention.**
     Insert Flamingo-style gated xattn layers that attend to the soft tokens,
     base self-attention untouched (tanh-gate inits at 0 → identity at start).
     Stronger conditioning, keeps text context clean. More params, more plumbing.

**Trainable params (the verbalizer / encoder half):** featurizer + resampler
(~tens of M) + QLoRA adapters (~0.1–1% of 35B). Everything else frozen. The
**reconstructor (decoder half)** is a separate module trained separately — see §2b.

---

## 2b. The reconstructor (decoder half — the NLA bottleneck-is-text)

The bridge above is only the *encoder* of a Natural-Language Autoencoder: latent →
text. Closing the loop means adding a **text-only decoder** `R` that reconstructs
the frozen world-model latent from Qwen's *generated* reasoning. Reconstruction
fidelity then answers the question ablate-the-latent cannot: not "does the bridge
*use* the latent" but "does the output text *faithfully render* it, or embellish
past it?" (the Qwen-embellishment risk the bridge was missing).

**What it reconstructs (the target `z`).** The **pooled-512 latent**
`z = h.mean(dim=2) ∈ R^512` — the *exact* vector `value_head` consumes in
`scripts/train_world_model.py` (`self.value_head(h.mean(dim=2))`). It is provably
value-bearing, low enough dimension to be text-expressible (sidesteps the capacity
risk), and answers the load-bearing question: did the text preserve what determines
the model's read? Targets are **fixed per-dim standardized** (mean/std frozen over
the train set) so MSE isn't dominated by a few high-variance dims. Slots are
identity-fixed by `slot_emb` (slot k is always player k, index 10 global), so
per-token reconstruction needs **no Hungarian matching**.

**The text-only firewall (non-negotiable).** `R` consumes **only the decoded
answer string**, re-tokenized in a fresh context — no soft-token prefix, no prompt,
no KV-cache that ever saw the soft tokens:

```
Qwen generates answer text y  -> detach + decode to STRING -> re-tokenize fresh
   -> re-embed via frozen text encoder (run ONLY on the generated string, never
      the residual stream that saw soft tokens)
   -> reverse-Perceiver (2–4 layers, M_r learned queries cross-attend to H_y)
   -> MLP head -> z_hat in R^512
```

The reverse-Perceiver is a **separate module — do NOT tie weights with the forward
resampler** (reuse-in-reverse leaks latent-side structure). **Hard rule, enforced
in code:** `R` takes `y_ids` only; **zero tensor path** from latent / soft tokens /
the Qwen residual stream that saw soft tokens may reach `R`'s input. Enforce with
`stop_gradient` + an assertion + a unit test: `recon_from_shuffled_text` and
`recon_from_empty_text` must score at the **latent-mean floor**. If they don't, the
firewall is broken and every fidelity number is a lie.

**Loss.** `L_recon = (1 − cos(z_hat, z)) + β·MSE(z_hat, z)`, β ≈ 0.1. Cosine is the
headline metric (scale-invariant, matches how downstream heads use direction); MSE
is secondary, on standardized targets.

**Target fallback ladder** (only if the chosen target fails the capacity probe,
§7 Step 0): (1) **[default]** pooled-512 `z`; (2) **decision-relevant projection** —
top-k PCA of `h` over channels the value+dist heads actually use (head-Jacobian);
(3) **head outputs directly** — value logit, per-player top-1 dist class ∈ {0..96},
rollout value mean/spread at +2s/+4s (low-dim, certainly text-expressible, and
arguably the *most* meaningful target). **Stretch:** full `[11,512]` per-token grid
(target token k = `h[:,:,k,:]`, no matching) — only after pooled-512 shows a clean
latent-on ≫ latent-off gap.

**Trained as an evaluator first, always separate.** The **reported faithfulness
number always comes from a separate decoder trained on frozen, held-out verbalizer
outputs** — no gradient to Qwen / resampler / LoRA. A jointly-trained decoder
co-adapts with the verbalizer into a private cipher (high fidelity, unreadable text)
— the documented NLA failure mode and this project's own circularity scar. The clean
number requires the verbalizer frozen when the decoder is trained. (Optional aux/RL
uses of recon — as a small annealed regularizer in 2a, or a constraint-filter in
GRPO — are in §5; they never source the reported metric.)

---

## 3. The decisive eval — ablate-the-latent (eval #1) AND recon-fidelity (eval #2)

Run identical SFT/inference with the soft tokens **zeroed** (and separately
**shuffled** across examples). Report the delta on a held-out reasoning eval
(value-agreement of stated assessment, or a discriminative probe on the text).

- **latent-on ≫ latent-off** → the bridge grounds in the world model. Thesis holds.
- **latent-on ≈ latent-off** → the bridge is ignoring the latent and pattern-
  matching the prompt/templates — **caster-SFT circularity has returned.** Stop
  and fix grounding before proceeding.

This is the council's "implement ablate-the-latent FIRST" requirement. It is the
single test that distinguishes this bridge from the Era-1 failure.

### eval #2 — recon-fidelity (the missing faithfulness leg)

Ablate and recon are **orthogonal and complementary — recon subsumes nothing.** A
bridge can *pass* ablate (it uses the latent) yet *fail* recon (it uses it but Qwen
embellishes detail not in the latent); ablate alone is blind to exactly that
embellishment. **Keep both.**

| Eval | Question | Role |
|---|---|---|
| **Ablate-the-latent** (#1) | Does the bridge **use** the latent at all? | **FIRST GATE** — the Era-1 circularity tripwire. |
| **Recon-fidelity** (#2) | Is the latent **faithfully rendered** in the text, or hallucinated past? | **SECOND GATE** — the missing faithfulness leg. |

**Eval order for a green light:**
1. **Ablate passes** — grounding exists (latent-on ≫ latent-off on value-agreement).
2. **Recon-fidelity above floor AND beats controls** — grounding is faithful.
3. **Value-agreement / reasoning quality** — grounding is useful.

**Mandatory anti-gaming controls** (report fidelity ONLY against these, never bare):
- **(a) Shuffled-text control** — decode a *different* example's text; fidelity must
  collapse to the floor (else the decoder rode a prior/mean, not the text).
- **(b) Ablated-bridge control** — text generated with soft tokens zeroed must
  reconstruct *worse*. The delta `recon(latent-on text) − recon(latent-off text)`
  is **the single strongest honesty number**: it shows the text carries
  *latent-specific* information, not prompt patterns.
- **(c) Latent-mean baseline** — predicting the corpus-mean `z` sets the floor.
  Report fidelity as **fraction-of-variance-explained over that floor**, not raw
  cosine (raw cosine on a low-rank post-LayerNorm manifold is deceptively high).
- **(d) Empty/scrambled-text invariant** — must score at the floor (firewall audit).

**The faithfulness-gibberish guard (mandatory).** Recon-fidelity must **never** be
reported or used as a gate *alone* — it can be satisfied by latent-encoding
gibberish (a steganographic degenerate code). Always pair it with:
- a **value-head-agreement** number — feed `z_hat` through the **frozen `value_head`**
  and compare `P(CT win)` to the true value (per-example, un-fakeable by a
  mean-predictor); and
- a **readability/perplexity guard** (base-Qwen perplexity, or a human readability
  check on the generated text).

A passing faithfulness claim requires **fidelity-above-controls AND value-agreement
AND readability.** Recon alone is not allowed to carry a grounding claim.

---

## 4. Compute placement (4090 local vs pod)

- **SFT (Phase 2):** precompute latents **locally on the 4090** — encoding is
  ~29 ms/round, the model is 19M params. Cache `(soft-latent-input, predictive
  channels, target text)` to disk, ship to the pod. The pod then trains *only*
  the bridge on cached pairs — no world model needed on the pod for SFT. Cheap,
  decouples the two models, makes the SFT dataset reproducible.
- **GRPO (Phase 3):** the world model must run **on the pod** — rollouts are
  generated live from the policy's chosen states, can't be precomputed. World
  model (19M) + Qwen (QLoRA) co-resident. `scripts/pod_setup_grpo.sh` already
  handles the toolchain (kernel/CUDA matching — the 14h-for-40-steps lesson).

---

## 5. SFT-BOOTSTRAPPING DECISION (the one genuinely open question)

The bridge needs `(latent, text)` pairs. Where does the text come from without
re-importing the caption-SFT circularity? Three sources, evaluated:

| Source | Pro | Con / risk |
|---|---|---|
| **A. Templated from predictive heads** (value bucket, event/surprise tags, rollout outcomes) | free, infinite, exact, controllable, available now | reads templated; needs to lean on *predictive* outputs to be non-circular |
| **B. Caster commentary** (the parked line) | real human reasoning, true target distribution | per-event alignment ~25% (ASR ceiling), sparse, noisy |
| **C. Prediction-conditioned LLM** (Claude given the model's value+rollouts) | fluent, scalable reasoning | circular **iff** the LLM sees only raw features; non-circular iff it sees the predictions |

### Decision

**Phase 2a (bootstrap) — Templated grounding from PREDICTIVE outputs (A′).**
The critical refinement that avoids the Era-1 trap: template the text from what
the world model **knows that raw features don't** —
- the **value head** ("CT is in a losing position, ~32% to win"),
- the **surprise/event signal** ("a fight is imminent near B"),
- the **sampled rollout** ("T's most likely line is an A execute in ~3s; CT5 will
  probably rotate").
Static state description ("3v5, 1:20 left") is allowed as scaffolding but is NOT
the grounding — the grounding is the *foresight*. Text that is a pure function of
the current frame is exactly what ablate-the-latent must punish; text that
requires the forward model is what it must reward. Tens of thousands of pairs,
generated locally, free. **Gate: ablate-the-latent must pass on this set.**

Why this is *not* caption-SFT: caption-SFT asked the model to learn understanding
from text that restated its inputs (zero new info, +0.008). Here (a) the
understanding is already in the frozen latent, and (b) the target text encodes the
*predictive* outputs, which are not computable from the current frame — so the
soft tokens are load-bearing by construction, and ablate-the-latent verifies it.

**NLA complements this decision — it does not replace it.** Reconstruction-faithful
text is *not* the same as good tactical reasoning text: NLA optimizes
information-preservation and can yield faithful-but-stilted prose. So templated
grounding still owns format/vocab, and GRPO still owns reasoning quality; the recon
objective only certifies faithfulness. Two optional, hedged uses of recon as a
*training signal* (never the reported metric, which is always a separate decoder):
- **Phase 2a-0 (optional warm-start, not milestone-critical):** label-free recon
  pretrain of featurizer+resampler+QLoRA against the decoder. Listed for
  completeness; the milestone does **not** depend on it.
- **Phase 2a aux (optional, only after ablate passes):** `L = CE_template + λ·L_recon`,
  λ small (0.05–0.1), annealed down, **decoder frozen** during this (no collusion).
  Recon then acts as a regularizer keeping Qwen on-latent while it learns
  format/vocab. λ is an **unvalidated hyperparameter — measure fluency before/after.**
  Gradients reach the verbalizer only via Qwen's **teacher-forced** hidden states for
  the answer span (no sampling op on the forced path → differentiable, straight-
  through). The literal REINFORCE NLA loop (reward = −L_recon on freely generated
  text) is **research-stretch, gated behind milestone success**, always with the
  GRPO KL-to-SFT anchor and a fluency guard.

**Phase 2b (fluency, optional) — mix in B and/or C.** Once the channel is wired
and grounded, enrich with prediction-conditioned Claude reasoning (C — Claude
shown the value/rollouts, never raw-features-only) and whatever aligned commentary
(B) survives the 4.6σ global / 25% per-event alignment. This buys natural prose;
it is not required for the thesis.

**Phase 3 (GRPO) — the real reasoning quality.** SFT is the warm start only.
Reasoning is made *good* by GRPO. **AMENDED 2026-07-18 (first-principles-plan
CHANGE A): the reward is GROUNDED — verbalized predictions scored against the
ACTUAL demo future** (probabilistic scoring: CRPS/Brier over extracted claims,
not binary match). An earlier version of this section named value-through-rollout
as the sole quality signal; that reward is model-authored (the LLM is graded by
the same frozen value head whose latent it consumes) and uncorrectable without an
environment — reward hacking by construction (adversarial-review F3; red-team
verdict unambiguous). **Value-through-rollout is DEMOTED to the group generator**:
it produces the K sampled futures the group is built from, but the score each
completion earns comes from the realized demo future. No human reasoning labels
in the RL loop. Pre-GRPO feasibility gates (first-principles-plan CHANGE F):
(i) the text→claim checker must reach AUC ≥ ~0.75 on a constructed pseudo-gold
set; (ii) reward-noise ICC under single-draw future noise must exceed ~0.2, or
group advantages are noise; (iii) run one offline ReST/rejection-sampling round
with the validated checker BEFORE any on-policy GRPO.

**Recon in GRPO is a CONSTRAINT, not a summed reward.** Do **not** add `λ·recon` to
the GRPO reward — that lets the policy trade reasoning for info-density (stilted
text). Instead: **hard-floor / zero the advantage** of any of the G=16 group
completions whose recon-fidelity < threshold τ, *before* group-normalizing.
Faithfulness becomes a **feasibility constraint**; the grounded reward stays the
**sole quality signal** that shapes ranking. τ = 25th percentile of the
SFT-passing recon distribution. **Documented fallback:** if the constraint starves
the group (zero-variance-advantage collapse), drop to **recon-as-eval-only.**

---

## 6. Risks & open questions

- **Circularity creep.** If templates lean on static state, the bridge can satisfy
  them without the latent. Mitigation: predictive-output grounding + ablate-the-
  latent as a hard gate, run continuously, not once.
- **35B on one pod.** QLoRA(4-bit) is the plan; confirm it trains with the
  resampler + soft-prompt in ≤80GB before scaling the dataset. FP8 variant is the
  fallback.
- **Soft-token context cost.** M=32 is cheap for single-moment; event-sequences
  (K×11 → M) are the reason the resampler compresses to fixed M. If grounding is
  weak at M=32, raise M or move to gated xattn (v2).
- **Event clock dependency.** Round-level reasoning wants the adaptive clock
  (rescued, AUC 0.698) productized. Single-moment reasoning does not — start there.
- **Eval beyond ablation.** Need a reasoning-quality metric for SFT (value-
  agreement between stated assessment and the value head is the cheapest honest
  one) before GRPO has a reward to optimize.

### NLA-specific kill-criteria (report as a negative result, do not quietly drop)
1. **Capacity floor failure.** If even value+rollout (or top-PCA) cannot be
   reconstructed above the latent-mean floor from ≤256 tokens (§7 Step-0 sweep), the
   metric has no dynamic range — NLA degrades to a value/rollout-only probe, or is
   reported as a negative result. Learned **before any pod spend.**
2. **Firewall failure.** If shuffled/empty-text recon does not collapse to the floor,
   the metric is meaningless — fix or abandon.
3. **Steganographic degenerate code.** If recon-as-reward (Track-2 / aux λ) drives
   readability/perplexity down (fidelity up, prose down), that variant **does not
   ship** — recon stays metric-only.
4. **No latent-specific signal.** If `recon(real) ≈ recon(shuffled/ablated)`, the text
   carries no latent-specific information — same stop-and-fix as a failed ablate.

### NLA honest caveats
- **Recon-faithful ≠ good reasoning.** NLA certifies *information preservation only*;
  reasoning quality is GRPO's job, format/vocab is templated grounding's job.
- **Capacity is real.** A bounded text sequence cannot losslessly carry
  `11×512 = 5632` dims — and should not. Measure on the **decision-relevant
  subspace**, not the raw vector.
- **Double-counting guard.** The bridge is fed value+rollout *channels* (§1); the
  recon target must be the **world-model latent itself (pre-augmentation,
  `z=h.mean`)**, not the appended channels — else "faithfulness" only measures
  whether the text echoes the channels. Report raw-latent vs appended-channel
  contributions separately.
- **Static-derivable confound.** Weight the recon target toward predictive/foresight
  channels over static-derivable ones (Line-in-the-Sand discipline), or recon
  inherits the circularity it was meant to detect.

---

## 7. First concrete milestone (smallest thing that tests the thesis)

0. **(NEW) LOCAL capacity kill-test — no pod.** `scripts/nla_capacity_probe.py`.
   **RESULT (2026-06-14, `wm_3map_dist_v3m`, 13,928 val frames): GO.** The pooled-512
   target is trivially carryable — `z` is extremely low-rank (95% of variance in
   **6 dims**, 99% in 17; value-AUC **fully preserved by the top ~8** PCA components),
   and verbalizable content (value logit + per-player predicted movement) reconstructs
   it at **R²=0.92** (shuffled-content floor −0.93, latent-mean floor 0). So the
   information is provably THERE — the NLA decoder is not capacity-blocked.
   **Important caveat the probe surfaced:** pooled-512 is *so* low-rank it's a **lossy,
   too-easy target** — a faithfulness gate on it would be weak. The richer, harder, more
   meaningful target is the **per-token `[11,512]` grid** (the per-player structure
   pooling discards); make that the real recon target, with pooled-512 as the floor.
   This probe is a ceiling only (no Qwen text, no text-encoder-quality) — the real metric
   (recon from Qwen *generated* text) waits for the bridge. Remaining sweep to run when
   building the decoder: text-budget ∈ {32,64,128,256} tokens × target ∈ {pooled-512,
   per-token grid, value+rollout} with the oracle/templated-text decoder.
1. Single-moment only. Latent = one frame's 11 tokens + value + a 1-step rollout
   summary. Resampler M=32 → soft prefix. QLoRA on Qwen3.6-35B-A3B. Aux `λ·L_recon`
   optional (default λ=0 for the milestone).
2. SFT set: ~20–50k templated-from-predictive-heads pairs, generated locally from
   `wm_3map_dist_v3m`, on the 3 maps.
3. Train on one pod; **run BOTH gates** — ablate-the-latent (latent-on vs off on
   value-agreement) AND recon-fidelity (reconstruct `z` from Qwen's *generated* text,
   vs shuffled-text and ablated-bridge controls, above the capacity floor, with
   value-head-agreement + readability).
4. **Green light = TWO numbers** (was one): `latent-on > latent-off` (grounding
   exists) **AND** `recon(real text) > recon(shuffled/ablated text)` above the
   capacity floor, with value-head-agreement and readability intact (grounding is
   faithful). Failure = fix grounding (more predictive content, gated xattn) before
   spending another pod-hour.

Build order: **Step-0 local decoder capacity sweep (cached latents, no Qwen)** →
featurizer+resampler+soft-prompt wiring (local CPU smoke on a tiny LLM stub) →
latent precompute + templated SFT-set generator → pod QLoRA run → ablate-the-latent
**+ recon-fidelity (separate held-out decoder)**. Keep the world model, SFT-set
generation, and the Step-0 capacity sweep on the 4090; reserve the pod for QLoRA only.
