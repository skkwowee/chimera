# main.tex Disposition — the VLM-Era Draft

This note records what the archived VLM-era draft (`archive-vlm/main.tex`) still contains of
value and what the world-model pivot superseded. Read it before touching
`archive-vlm/main.tex` or when deciding the fate of the standalone RECALL paper. The
world-model paper's skeleton lives in `OUTLINE.md`; the NLA/bridge design that
once drafted here is canonical in `docs/bridge-design.md`.

## Decision state

The pivot made the world model the flagship paper (`OUTLINE.md`). `archive-vlm/main.tex` is
deliberately preserved untouched: it holds the project's two finished artifacts
and is the source for a standalone RECALL methods paper. The recommendation
stands — ship RECALL on its own timeline rather than demoting it to a
lessons-learned subsection of a paper whose flagship results do not exist yet —
but the standalone has not been written. Nothing real in `archive-vlm/main.tex` was
invalidated by the pivot; what died was the connective tissue (the two-phase-VLM
thesis) and the never-run GRPO machinery.

## What stands (publish standalone)

**RECALL and its diagnosis — the crown jewel.** The retrieval-advantage-collapse
finding is real, finished, and honest: the estimator definition (kNN V̂/Q̂,
Â = Q̂ − V̂, k_min gate); the σ_s diagnostic with Tables 1–3 (variance, k-sweep,
positive-rule ablation); the two failure mechanisms (F1 same-trajectory leakage,
F2 outcome-correlated positive labels); and the fix (Layer 0 retrieval-side mask
as the win, Layer 1 counterfactual contrastive honestly reported as a null). Key
numbers: on 2,013 post-plant states, 52.6% saturation under the shipped 19-dim
embedding (median per-query σ_s = 0.000, ceiling 0.49 at p = 0.60); excluding
same-trajectory neighbors lifts σ_s to 0.331; an untrained sentence-transformer
encoder reaches 0.390. The retrieval-RL, bisimulation, zero-variance-GRPO,
supervised-contrastive-collapse, and counterfactual-augmentation related-work
paragraphs support it and survive intact.

## What is salvage (real, but demoted)

- **SFT-for-HUD perception**: 67.3% overall, +17pp over baseline; 11-field
  schema, demo-synchronized ground truth, LoRA r=4; training curve; the
  overfitting-boundary analysis. Orthogonal to the world model (which is
  text-free). Fold into the RECALL standalone as the system it was built inside,
  or spin out as a tech report.
- **Data pipeline**: 4 demos, 4,353 screenshots, 5,309 labels, 3,322 SFT
  examples — reusable, real.
- **Zero-shot VLM baselines** (Opus/Sonnet/Qwen HUD reading, frontier models at
  ~50%) — tied to the SFT story.
- **TRL multimodal-GRPO bug + manual-loop workaround** and the 20-step smoke
  test — a methods note, not a result.

## What is superseded

The title and two-phase-VLM thesis; the problem formulation, GRPO phase, and
reward-design sections (all TODO — full GRPO training never ran); the empty
full-pipeline results table; and the Fig. 1 placeholder depicting the old
pipeline. GRPO returns only in the world-model plan, as a grounded reward
against realized futures — a different setup (`docs/bridge-design.md` §5).

## Citation flags in main.tex

`chen2024game` and `huang2022strategic` (cited in the "VLM training and game AI"
paragraph) and `zhao2023counterstrike` (uncited) could not be verified against
any real publication in the 2026-07-19 audit — flagged in `references.bib`; do
not carry them into any submission without independent verification. All other
entries verified real (see `OUTLINE.md` §8).
