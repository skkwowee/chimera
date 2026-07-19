# Chimera 2 — ⛔ DEPRECATED (2026-07-18)

**This clean-room rebuild is superseded.** The methodology reset it was created
for happened INSIDE the main repo instead: `~/chimera` now carries the locked
pre-registered recipe (`docs/retrain-recipe.md`, Knobs 1–7), the first-principles
plan, the fixed builders, and the ordered runbook (`claude-progress.txt` top).
**All work continues in `~/chimera`. Do not develop here.**

This repo is kept as a DONOR for runbook step [3] in the main repo:
- `tools/gate.py` + `tests/test_rails.py` — the forgery-rejecting green-gate
  mechanism (a stage can't run without its predecessor's *verifiable* green);
- the **gradient-based `test_no_value_leak`** (backprop value BCE alone, assert
  zero grads outside the value head) — required verbatim by Knob 5 for the
  canonical detached-head run.
Port those, then this repo is history.

---

A barebones, peer-review-grade rebuild. **One question:** does a next-state-prediction world model of CS2 — trained with **zero outcome gradient on its trunk** (detached value head, unit-test-proven) — learn a frozen latent that beats *same-pipeline* supervised baselines under a fixed linear probe, and **keep improving with data** where they flatline?

> The contribution is the **pre-registered protocol + saturation/divergence phenomenology + a world-model shortcut audit**, NOT "SSL beats supervised" (textbook — see `docs/related.md`). Read `docs/thesis.md` first, `docs/lessons.md` before touching any eval.

## The chain (each row must go green before the next; run.sh enforces)
```
L0   data       one-pass corpus with the D1-D6 fixes; NO velocity (Knob-3 contingency)
L1e  edges      k=4 dist-edge fit, 5-map clean train only + ring-occupancy assert
L1   trunk      19M dist-head trunk (97-class classify-then-refine, k=4/500ms),
                DETACHED value head, scheduled sampling + SS-off control
L1c  coverage   minADE-K WITH the fair stochastic baseline (CHANGE B) — unreportable without it
L1b  shortcut   --corrupt-yaw audit on the NEW model (non-gating; informs v4 velocity)
L2a  RE         round-encoder retrain, v6 config verbatim, fixture-test gated (Knob 7a)
L2   probe      KEYSTONE (C1-REP): six frozen reps, linear-only gates, Δ≥+0.02 pre-registered
OOD  overpass   Knob 4: zeroed map one-hot + ID-zeroed control + fine-tune anchor
L3   scaling    C1-SCALE: wm's own N_sat→N_full delta ≥ +0.01 (RE slope never gates)
---  gate: C1 green unlocks the rest; dirs below do not exist yet ---
L4   bridge     C2: faithful verbalization (deferred; CHANGE D/E referees first)
L5   grpo       C3: grounded-future GRPO (deferred; CHANGE F feasibility gates first)
```

## Build discipline (hard rules)
1. **No stage N+1 until stage N's row is green.** Green = the ONE contract in `tools/gate.py`: `results/LN.json` with `green==true` + a `script` field naming a checked-in file that exists + `seeds`. Enforced three ways: run.sh, `gate.require_green()` inside every stage script (direct `python LN_*/x.py` is gated identically), and Rail 4 in `tests/test_rails.py` (run.sh runs the rails first, so a forged row dies twice).
2. **Directories are earned.** No `L4_*`/`L5_*` until C1 is green.
3. **Zero outcome gradient into the trunk.** The value head exists but is **detached** (stop-grad, end-phase-masked — locked Knob 5); `test_no_value_leak` backprops the value loss alone and asserts every non-head param grad is None. This is the machine-checkable form of "value_weight=0" — string-absence was retired because it forbids the locked design and proves nothing.
4. **No unhedged number without a regenerating script + committed raw output.** Baselines included. Known phantoms (see `docs/lessons.md` §1, incl. T2's fabricated +19.3pp and the 64-vs-128-tick correction) are recomputed or struck.
5. **Reposition before you write.** No C1 prose until `docs/related.md` states the one-line delta vs DeepMDP/SPR/Dreamer/bisimulation/DIAMOND/**MLMove**.

## Relationship to chimera (v1) — statement of intent
- **ROLE.** chimera2 is a **clean-room rewrite of the C1 keystone experiment** (frozen zero-outcome-gradient trunk → probe transfer → scaling), not a fork and not yet the successor. `/home/soone/chimera` remains **canonical** for corpus tooling, the certified datasheet, the locked Knobs 1–7 retrain recipe, and all pod/GPU runs. Train production models in chimera; cite chimera2 for the C1 protocol once its rows are green.
- **COMPATIBILITY.** chimera2's L1 implements the SAME locked design as chimera (k=4 distributional head + detached value head). The earlier apparent contradiction ("no value head" here vs "detached head" there) is resolved in chimera's favor: detachment already makes the trunk gradient-identical to `value_weight=0`, and chimera2 proves it with the gradient-based rail rather than banning the head.
- **DEPENDENCY.** Corpus FACTS (`EXCLUDED_MAPS`, `clean_blob`, canonical counts) are **imported** from `chimera/scripts/_corpus.py` via `L0_data/corpus_facts.py` (path import; override with `CHIMERA_REPO`), with a parity rail that fails loudly if upstream changes. Logic is **reimplemented** (one-pass builder, seeded-trainer pattern); VLM-era code (grpo_trainer/rewards/recall), value-co-trained trunk code, and `_archive/` scripts are **never ported**. Green results JSONs should record the chimera commit hash (`upstream_commit`) once L0 lands.
- **SUPERSESSION.** When L2+L3 go green here, the keystone protocol and its numbers port back into chimera's recipe as the citable C1 result; chimera2 then either becomes the paper repo (chimera archived as tooling) or is folded back — decided at that gate, not before.

## Layout
```
docs/        thesis.md  lessons.md  related.md
data/        README.md (provenance+release)  manifest/{match_ids,loto,lomo}.json
L0_data/     build_corpus.py  schema.py  inspect.py  corpus_facts.py (imports from chimera)
L1_trunk/    model.py  fit_edges.py  train.py  eval_skill.py  coverage.py  audit_shortcut.py
L2_probe/    probe.py  round_encoder.py  outcome_encoder.py  ood_overpass.py  scaling_curve.py  metrics.py
tools/       gate.py (the ONE green contract + stage gating)
results/     committed raw JSON per row
tests/       rails: leakage, no-value-leak (gradient), metric-identity, gate contract, corpus counts, chimera parity
init.sh      bootstrap: .venv + pytest + health check
run.sh       gated pipeline (rails first; refuses stage N+1 until N green)
```

## Run
```bash
./init.sh           # once: venv + deps + rails health check
./run.sh            # rails -> L0 -> L1e -> L1 -> L1c -> (L1b) -> L2a -> L2 -> OOD -> L3
                    # stops cleanly at the first unimplemented/non-green row
```

## Status
All rows empty. The keystone (L2) is the gate that unlocks everything downstream.

Provenance: distilled from `chimera-v1` via a multi-agent dissect→thesis→adversarial-review→synthesize pass, 2026-06-24 — **superseded 2026-07-18** by alignment to the main repo's methodology reset. Canonical ground truth lives in chimera: `docs/retrain-recipe.md` (+`retrain-recipe-knobs4-7.md`), `docs/first-principles-plan.md`, `docs/adversarial-review.md`, `docs/datasheet.md`, and the ordered runbook at the top of `claude-progress.txt`.
