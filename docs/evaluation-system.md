# Native step forecast check

This script asks: **does the model predict players' next positions better than
assuming they stay still or keep moving?** It is a sanity check, not proof that
Chimera understands tactics.

```bash
python scripts/eval_forecasts.py run --corpus PATH_TO_VAL_BLOB \
  --checkpoint PATH_TO_CHECKPOINT --out outputs/forecast.npz
python scripts/eval_forecasts.py score outputs/forecast.npz
```

It samples 16 futures per fixed-stride example and saves them with the actual
positions, baselines, source identities, settings and input/code hashes. Scoring
the saved file needs no model rerun. Use identical corpus, seed, stride and sample
count across checkpoints; check recorded window/horizon before comparing.

Lower is better: `mean_error` averages position error; `best_joint_error` takes
the best whole-team sample; `energy` also accounts for the spread of predictions
using the fair IID ensemble estimator. All are in game units, averaged over
eligible examples. Only players alive at both anchor and target are scored;
empty examples are excluded and counted. Best-of-16 alone is not a quality gate.

Only the checkpoint's native prediction step is evaluated. Frame offsets are
nominal, not certified wall-clock durations. No autoregressive rollout, 64 Hz
truth, confidence intervals, fitted stochastic baseline or canonical gate passes
are claimed. Use trusted checkpoints and one validation blob at a time.

Trainer changes, AUC fixes and the larger evaluation framework are out of scope.
Next substantive work remains resolving rollout cadence before long-horizon
evaluation; the locked Phase 4 controls remain incomplete.
