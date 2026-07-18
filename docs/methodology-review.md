# Chimera — Methodology Review (2026-06-21)

Start-to-end scrutiny of the training methodology. One row per finding:
the **claim**, what would **falsify** it, the code-grounded **evidence**, and
**status**. Investigated by 6 read-only agents 2026-06-22 against the actual
code + cached outputs.

> This file tracks the *review*. `decisions-ledger.md` stays the canonical
> rationale; fold resolved findings back into it.

## META-FINDING (the pattern across all six)

The architecture and self-critique discipline are strong, but **several
load-bearing numbers in the canonical docs do not survive contact with the
artifacts**:
- the `0.803/0.807` linear-probe parity (decisions-ledger §2) — **never computed**;
- "88/96 real frames at depth 8" (progress L30) — **miscount**, it is 32/96;
- the `+27.2pp` facing shortcut — cached run shows **+3pp**; +27.2pp unreproduced,
  causal test never run;
- the acceptance gate itself (probe transfer, W03) — **never run**.

The docs assert more certainty than the evidence supports. The good news: almost
every fix is cheap and local. ~40 min of 4090 time settles or sharpens 4 of 6.

---

## F0 — The pivot's central claim is unmeasured (probe transfer vs ceiling)
- **Claim:** next-state pretraining pushes probe transfer above the round
  encoder's ~16-demo saturation ceiling (`world-model-design.md §7`).
- **Evidence (CONFIRMED, severe):** zero probe results exist in any `wm_*`
  output dir; `feature-list.json:24` W03 `passes=false`; `outputs/world_model/h8/
  train.log:49` literally says *"low loss != understanding. The real test is probe
  transfer … not yet executed."* Round-encoder ceiling is real:
  `probe_outcome` 0.759 (saturates ~16 demos, negative slope after); action probes
  near-chance (movement 51.8%, engage 71.4%). `scripts/value_probe.py` is built &
  ready (~5 min local); action/next probes need ~2–3h to adapt from `_archive/`.
- **Status:** grounded · **experiment ready (Tier 0: value_probe; Tier 1: action/next)**

## F1 — "Learned tactics" is dead; surviving win is oracle-selected
- **Claim (README):** model internalizes tactics/geometry/causality.
- **Evidence (CONFIRMED):** `decision_eval.py` (selection-effect fix verified) —
  copy beats model: hard turn −52%, reversal −37%, straight −70%. The coverage
  win is real but `dist_coverage_eval.py:124` is a true oracle `min` over 16
  samples vs single-point copy; the fair best-of-K sampled baseline (progress
  L56–58) was **never added**. Honest reframe: the model learned a **multimodal
  option-set representation**, not tactical *choice* — sufficient for GRPO
  "generate the group," but the README overclaims.
- **Status:** grounded · **fix: add sampled-baseline column (Tier 1) + rewrite claim**

## F2 — Value is the co-trained head, not demonstrably the dense win
- **Evidence (CONFIRMED, worse than doc'd):** value head co-trained, gradients
  flow into trunk (`train_world_model.py:495`, weight 0.3 L364). The `0.803/0.807`
  linear-probe numbers are **phantom — never computed** on any checkpoint. Headline
  `0.856/0.865` = value-*through-rollout* of the co-trained head, not a probe.
  Value AUC **dropped** v2+dist 0.849 → v3+dist 0.841 — v3 perception did not help
  value. Both checkpoints needed for the deconfound already exist
  (`wm_3map`=v2+cotrain, `wm_3map_dist_v3m`=v3+cotrain); `value_probe.py` on both =
  **~2 min**. A `value_weight=0` run (1h pod) tests if next-state alone yields value.
- **Status:** grounded · **experiment ready (Tier 0 + optional Tier 2)**

## F3 — GRPO reward is defined two contradictory ways
- **Claim A** (`world-model-design §9`, progress L217): reward = verbalized
  prediction vs **actual demo future** (grounded).
- **Claim B** (`decisions-ledger §5`, `bridge-design §5`): reward =
  **value-through-rollout**, the model's own estimate (gameable model-based RL).
- **Evidence:** no world-model GRPO exists — `grpo_trainer.py`/`rewards.py` are
  **parked VLM-era** (RECALL/judge/percept); `rollout_eval.py` value-AUC is
  eval-only, never wired to training. "Verbalized-vs-actual" infra: **none**.
  This is a genuine **design fork** to resolve, not doc cleanup. (Note: the demos
  carry real futures + outcomes — the grounded reward is *available* if we track a
  real trajectory; the model's-own-value reward is the gameable one.)
- **Status:** grounded · **needs a design decision (no compute)**

## F4 — Phase-3 viability gate is untested where it matters
- **Evidence (CONFIRMED, reframed):** window L=96, horizon k=8. Depth 8 = 8 steps
  × 8 frames = 64 generated → **32/96 real (33%)**, NOT 88/96 (progress L30
  miscounts steps as frames). So the gate is *stronger* than documented (flat AUC
  with ⅔ imagined) — but **depth 12 = 100% imagined is never run** (`rollout_eval.py:
  199` caps depths at 8). AR loop + gap-scaling verified correct; derived dims
  frozen at decode but value head still reads stale perception dims. Fix:
  `--depths 0,1,2,4,8,12,16`, ~20 min local. 262 val rounds support depth 12.
- **Status:** grounded · **experiment ready (Tier 0)**

## F5 — No velocity input → facing shortcut; bridge inherits a biased latent
- **Evidence (CONFIRMED asymmetry; magnitude UNVERIFIED):** per-player block has
  sin/cos yaw+pitch (dims 3–6) and **0 velocity dims** (`feature_schema` L45–68;
  `build_tick_sequences.py:196–201`), though velocity *is* in the awpy parquet.
  Target = xy displacement (= velocity·dt), so the model must infer velocity from
  history while reading facing for free — the asymmetry is real. **But** the cached
  `facing_bias_check` shows only **+3pp** (truth follows momentum 65% / model 62%);
  the **+27.2pp** headline is unreproduced and the `--corrupt-yaw` causal test is
  implemented but **never run**. Don't commit a 9–13h local (3–4h pod) v4 rebuild
  on an unverified number — run the cheap causal test first.
- **Status:** grounded · **gate the retrain on Tier 0 yaw-shuffle test**

## Minor
- `feature-list.json` stale — marks world model `todo` though trained & measured.
- Several docs cite the phantom `0.803/0.807` and the `88/96` miscount; purge/fix.

---

## Experiment ladder (cheap → expensive)

**Tier 0 — minutes, local 4090, free, additive — settles/sharpens 4 findings:**
1. `value_probe.py` on `wm_3map` + `wm_3map_dist_v3m` (F2/F0) — ~2 min — first real
   probe numbers + v2-vs-v3 deconfound.
2. `rollout_eval.py --depths 0,1,2,4,8,12,16 --value-auc` (F4) — ~20 min — do
   rollouts carry value at 100% imagined?
3. `facing_bias_check.py --corrupt-yaw` (F5) — ~10 min — is the facing shortcut real?

**Tier 1 — hours, local, free:**
4. Adapt `_archive/probe_pro_action.py` → `model.latent()`; build next-state probe
   (F0) — the decisive thesis test vs the round-encoder ceiling.
5. Add fair best-of-K sampled baseline to `dist_coverage_eval.py` (F1).

**Tier 2 — pod, $, only if warranted:**
6. v3 `value_weight=0` run (F2) — does next-state alone yield value? (~1h)
7. v4 schema rebuild + retrain (F5) — only if #3 confirms the shortcut matters.

**Design / docs (no compute):**
8. Resolve the GRPO reward fork (F3) — grounded-vs-actual-future vs model's-own-value.
9. Purge phantom numbers, fix the 88/96 miscount, refresh `feature-list.json`.
