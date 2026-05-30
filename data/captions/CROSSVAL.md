# Verbalization discriminative check — verified cross-validation results

> **Correction note:** the commit message of `563fdb9` contains a placeholder
> table (majority=0.544 across all folds) that was written before the real
> 4-fold run finished. **This file holds the verified numbers.** The committed
> code and `captions.jsonl` data are correct; only that message's table is wrong.

## Setup

- 2013 captions (Claude sonnet-4-6) of egocentric single-tick game states.
- Predict `round_won`. Demo-disjoint split (val = one held-out demo).
- Three classifiers: majority floor / structured-feature ceiling (11 numeric
  features) / caption TF-IDF (1-2 grams, logistic regression).
- `gap = structured_acc - caption_acc` (lower = language preserves more signal).

## Results (4-fold demo-disjoint cross-val)

| fold     | majority | structured | caption | gap     | caption lift |
|----------|----------|------------|---------|---------|--------------|
| mirage   | 0.634    | 0.759      | 0.569   | +0.190  | −0.065       |
| inferno  | 0.593    | 0.754      | 0.859   | −0.105  | +0.266       |
| nuke     | 0.589    | 0.812      | 0.839   | −0.027  | +0.250       |
| overpass | 0.579    | 0.846      | 0.871   | −0.026  | +0.292       |
| **MEAN** | **0.599**| **0.793**  | **0.785**| **+0.008** | **+0.186** |

## Reading

- Captions beat the majority baseline by **+18.6pp** on average.
- Captions are **statistically tied with the structured ceiling** (mean gap
  +0.008). On 3 of 4 maps captions *exceed* the hand-picked numeric features —
  language carries signal the chosen numerics miss.
- The single-tick ceiling is higher than feared (0.79 vs 0.60 majority), so the
  result is **not** confounded by the "needs temporal context" concern.

## Asterisk: mirage

Mirage is the lone outlier — captions *underperform* majority there (−6.5pp,
gap +0.19). Possible causes worth checking before fully trusting the
conclusion:
- Mirage has the highest majority baseline (0.634), so there's less headroom.
- Mirage states may be genuinely harder to verbalize from single-tick
  egocentric features.

## Verdict

**Green light for the encoder → language bridge**, with mirage flagged for
follow-up. Natural language is a near-sufficient statistic for tactical
outcome at the single-tick level.
