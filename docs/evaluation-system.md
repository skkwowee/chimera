# Reproducible forecast evaluation

This is the small evaluation foundation, not a completed Phase 4 gate. It answers:
**does a checkpoint predict held-out XY better than copy and smoothed constant
velocity on exactly the same anchors?** No dashboard, judge model, training job,
or new model choice is needed.

## Contract

- `manifest` freezes source-keyed anchors, map filters, window, horizon and
  stride against a SHA-256 of the corpus. Only non-freeze, non-end anchors enter.
- `generate` writes predictions, truth and masks to a versioned NPZ bundle.
  Default K is 16. Checkpoint, code and array hashes, runtime, seed and
  temperature travel with the evidence. Sampling uses an anchor-local RNG.
- `score` needs only the saved bundle. Comparisons require identical manifests,
  K, seed, temperature, truth and masks; there is no silent intersection.
  Existing outputs are never overwritten.

All errors use game XY units and the same truth-defined mask: alive at anchor
and target. Empty support is excluded and counted. This is survivor-conditioned
position evaluation, **not complete-state probabilistic calibration**.

The report contains sample-mean ADE/FDE, joint and per-player minADE-K, and a
joint-path energy score. A selected minADE member stays fixed across time;
joint minADE also keeps it fixed across players. Energy uses masked joint L2
divided by the square root of observed player-times. IID samples use the fair
finite-ensemble estimator; deterministic baselines use their exact point score.
MinADE is a coverage diagnostic, not a standalone model-selection criterion.

Scores are pooled over anchors and broken out by map. Candidate-minus-baseline
deltas include paired match-cluster percentile intervals, preserving all anchors
and multiplicities within each sampled match. These intervals are exploratory,
can undercover with few matches, and are not the locked BCa/percentile-t gates.
One effective match produces no interval. Per-map intervals are deferred.

## Run

Use a trusted, completed **validation** blob and a checkpoint with matching
schema/window/horizon. Commands below are examples, not recorded model results.
Omit `--checkpoint` to generate only the two cheap baselines.

```bash
python scripts/eval_forecasts.py manifest \
  --corpus data/processed/tick_sequences/val_v3m_p2.pt \
  --window 96 --horizon 4 --steps 1 --out outputs/eval/anchors.json
python scripts/eval_forecasts.py generate \
  --corpus data/processed/tick_sequences/val_v3m_p2.pt \
  --manifest outputs/eval/anchors.json --checkpoint PATH_TO_CHECKPOINT \
  --samples 16 --seed 0 --out outputs/eval/forecasts.npz
python scripts/eval_forecasts.py score \
  --bundle outputs/eval/forecasts.npz --out outputs/eval/report.json
```

Generate another checkpoint against the same manifest, then pass its bundle as
`score --compare OTHER.npz`. Both implementations remain traceable. Corpora are
memory-mapped through the existing loader; use one validation blob at a time.
PyTorch corpora/checkpoints must be trusted; bundle loading disables pickle.

## Deliberate limits and next step

Native k=4 supports one-step evaluation only. Appending a k=4 prediction to an
8 Hz history creates mixed cadence, so multi-step generation is rejected unless
the model is k=1 with raw v2 features. V3 derived-feature refresh is not invented.
Because source ticks can be missing, frame offsets are nominal, not certified
elapsed time. Reports explicitly mark 64 Hz truth unavailable and gates
`not_adjudicated`.

Next: resolve the rollout cadence protocol, then add the locked fitted per-bucket
baseline, real 64 Hz comparison, mode-switch coherence and joint-interaction
controls. Keep Phase 4 incomplete until those exist. The optional diagnostic
Gaussian baseline was deliberately omitted to avoid a competing protocol.

Training-monitor integrity fixes accompany the runner: fixed validation crops
and an independent loader RNG; tie-aware AUC; separate prediction/value optimizer,
gradient-clipping and AMP-scaler domains. Detaching the value latent alone did
not prevent outcome labels from changing the shared clipping/overflow decision.
Crop metadata binds starts and lengths, not corpus identity. Neither it nor
anchor-local sampling promises bitwise equality across devices/runtime versions.

## Research refresh, September 2026

[HarnessEval-W](https://arxiv.org/abs/2608.16859), released August 17 and revised
September 1, emphasizes inspectable evidence behind world-model scores. Our
application is saved tensor evidence, not an LLM judge for exact game-state truth.
[From Generation to Simulation](https://arxiv.org/abs/2608.23070), August 24,
distinguishes generation from state feedback and reproducible long-horizon
simulation. Neither paper establishes that Chimera passes those capabilities.

The metric choice follows
[Evaluation of Trajectory Distribution Predictions with Energy Score](https://proceedings.mlr.press/v235/shahroudi24a.html):
min-of-K alone does not properly assess the predicted distribution. These papers
support auditable, complementary measurements; they do not justify changing the
locked thresholds or choosing a new language-model backbone.
