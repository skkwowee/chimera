# Level-2 encoder learning curve: data is not the bottleneck

**Date:** 2026-05-16
**Encoder:** v6 config (d_model=512, n_layers=4, ~13M params; per
`docs/round-encoder-design.md`)
**Probe:** L2-G2 small MLP (128 hidden, 2 layers) → `round_won` binary
**Val split:** 12 demos held out (262 rounds, 0 overlap with train demos)
**Figure:** `outputs/round_encoder/learning_curve/learning_curve.{png,pdf}`
**Script:** `scripts/learning_curve.py`, plot: `scripts/plot_learning_curve.py`

## Result

| Train demos | Train rounds | val_acc_event_only | probe_outcome val_acc |
|---:|---:|---:|---:|
| 16 | 328 | 0.5276 | 0.7592 |
| 32 | 643 | 0.4756 | 0.7680 |
| 48 | 996 | 0.5231 | **0.7748 (peak)** |
| 64 | 1369 | 0.5060 | 0.7723 |
| 69 | 1471 | 0.5586 | 0.7615 |

Top-end slope: **−0.022 probe_acc per +10 demos.** Negative.

## What this means

For the v6 SSL design + the L2-G2 probe, the encoder *saturates by 16
training demos* (~328 rounds). Adding demos beyond that produces no
meaningful improvement and may even hurt slightly (likely interference
from the event-CE objective overfitting at larger scales).

This is a counterintuitive result given the project's earlier history.
Going from 24 → 81 demos *did* close the σ_s gate (median 0.4635 →
0.4496) but that was a clustering metric, not a generalization-via-probe
metric. The L2-G2 probe — which is the strongest gate for "does outcome
information actually live in the embedding" — was already past the 0.65
threshold at the smallest dataset we tried.

In other words: at ~16 demos, the encoder already extracts essentially
all the outcome-relevant structure that this **architecture + SSL
objective combination** can extract from per-tick god-view features.
Throwing more data at the same recipe doesn't unlock more.

## Caveats — what the curve does NOT measure

- **σ_s and clustering quality** (L2-G1, L2-G3) likely keep improving
  with more data, even if probe doesn't — clustering measures need more
  density to fill out the embedding space. We didn't re-run σ_s at each
  subset.
- **Downstream RL utility.** The probe predicts `round_won` from a
  single tick's embedding. RECALL retrieval (the actual L3 reward
  signal) needs a *dense enough* index to find tactically-similar
  neighbors. That benefits from more data even if the per-tick encoder
  is saturated.
- **Tactical diversity.** All 81 demos are from 4 tournaments and
  ~20 pro teams. The "ceiling" we hit may be a ceiling of *tactical
  pattern variety in this slice of pro CS*, not of CS broadly. Demos
  from amateur, MR12, ranked, or older meta could shift the curve.
- **Event prediction (red line)** is noisy run-to-run because event-CE
  overfits hard after epoch 2-3 (see `claude-progress.txt`
  2026-05-15 v5/v6/v7 entries). The peak val_acc captured here is
  unstable; the curve flatness conclusion is for probe_outcome.

## Implications for future work

1. **Data scaling is not the next lever.** Adding more pro-CS Bo3s
   from similar tournaments won't materially improve this encoder.
   Storage cost to skip: ~150 MB/demo if `.dem` files are deleted
   after parsing.

2. **Architectural changes also didn't help** (v7 1024-d + salience,
   same data). Capacity isn't the bottleneck either.

3. **What *might* move the needle:**
   - **Different SSL objectives.** The current 5 (3 forward-pred + CE +
     time-to-event) overfit fast. Contrastive learning over (tactical,
     non-tactical) pairs, or masked-tick reconstruction with heavier
     masking, are unexplored.
   - **Different input representation.** The 582-d god-view feature is
     handcrafted. A learned tokenizer over raw game-state events,
     or per-player attention before round-level aggregation
     (hierarchical), are different design points.
   - **Different probe target.** `round_won` is binary and noisy.
     Probing for `pro_action_next` (multi-class) or
     `next_event_within_4s` may show non-saturation where
     `round_won` saturated.
   - **Distribution shift.** Demos from a different era / skill bracket
     could test generalization rather than in-distribution capacity.

4. **For the paper.** The learning curve is a strong negative result
   that *constrains the design space*: future Level-2 encoder iterations
   should justify themselves by improving probe_acc *at fixed N=16-48
   demos*, not by training on more data. This is the kind of result
   that prevents over-claiming "encoder scales with data" — it doesn't,
   at least not in this regime.

## Reproducing

```bash
# Train + probe across subsets
python scripts/learning_curve.py
# (runs 5 trainings, ~7 min total on RTX 4090, writes JSON)

# Render figure
python scripts/plot_learning_curve.py
```

Subsets are nested-by-demo (seeded shuffle then take first N). Val split
is fixed across all subsets, so the probe accuracies are directly
comparable.
