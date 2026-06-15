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
```

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

**Trainable params:** featurizer + resampler (~tens of M) + QLoRA adapters
(~0.1–1% of 35B). Everything else frozen.

---

## 3. The decisive eval — ablate-the-latent (eval #1, before anything else)

Run identical SFT/inference with the soft tokens **zeroed** (and separately
**shuffled** across examples). Report the delta on a held-out reasoning eval
(value-agreement of stated assessment, or a discriminative probe on the text).

- **latent-on ≫ latent-off** → the bridge grounds in the world model. Thesis holds.
- **latent-on ≈ latent-off** → the bridge is ignoring the latent and pattern-
  matching the prompt/templates — **caster-SFT circularity has returned.** Stop
  and fix grounding before proceeding.

This is the council's "implement ablate-the-latent FIRST" requirement. It is the
single test that distinguishes this bridge from the Era-1 failure.

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

**Phase 2b (fluency, optional) — mix in B and/or C.** Once the channel is wired
and grounded, enrich with prediction-conditioned Claude reasoning (C — Claude
shown the value/rollouts, never raw-features-only) and whatever aligned commentary
(B) survives the 4.6σ global / 25% per-event alignment. This buys natural prose;
it is not required for the thesis.

**Phase 3 (GRPO) — the real reasoning quality.** SFT is the warm start only.
Reasoning is made *good* by GRPO: group = sampled world-model rollouts, reward =
value-through-rollout (gate passed, flat ~0.82 through 8s) + perception grounding.
No human reasoning labels in the RL loop. **This is why we don't over-invest in
2a/2b text quality** — the bridge SFT teaches the mapping; GRPO teaches the reasoning.

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

---

## 7. First concrete milestone (smallest thing that tests the thesis)

1. Single-moment only. Latent = one frame's 11 tokens + value + a 1-step rollout
   summary. Resampler M=32 → soft prefix. QLoRA on Qwen3.6-35B-A3B.
2. SFT set: ~20–50k templated-from-predictive-heads pairs, generated locally from
   `wm_3map_dist_v3m`, on the 3 maps.
3. Train on one pod; **run ablate-the-latent immediately.**
4. Success = latent-on beats latent-off on value-agreement by a clear margin.
   That single number says the bridge grounds in the world model — the green light
   for Phase 2b/3. Failure = fix grounding (more predictive content, gated xattn)
   before spending another pod-hour.

Build order: featurizer+resampler+soft-prompt wiring (local CPU smoke on a tiny
LLM stub) → latent precompute + templated SFT-set generator → pod QLoRA run →
ablate-the-latent. Keep the world model and SFT-set generation on the 4090;
reserve the pod for the QLoRA train only.
