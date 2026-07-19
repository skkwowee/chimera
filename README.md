# Chimera

**A next-state-prediction world model for Counter-Strike 2.**

This README is the repo front door: the thesis, the pipeline at a glance, honest
current status, and pointers into `docs/`. For run state and what to do next,
read the ordered runbook at the top of `claude-progress.txt`.

## Thesis

Chimera learns CS2 by predicting the future. A causal transformer is trained
over sequences of engine-accurate game-state frames — state → state, no text —
like language-model pretraining, but over game states instead of tokens. Three
claims, each gated before the next:

- **C1, the keystone:** dense next-state prediction alone yields strategic
  abstraction — round outcome becomes linearly decodable from the frozen world
  model, beating a measured supervised ceiling under one identical,
  pre-registered probe protocol.
- **C2, faithful translation:** a small trainable bridge can render that frozen
  understanding as tactical language. Language cannot *create* understanding
  from clean symbolic state (measured: +0.008); it can only translate a forward
  model's foresight. Faithfulness is certified by the NLA gate — ablation
  proves the latent is *used*, firewalled reconstruction proves the text
  *renders* it.
- **C3, grounded reasoning:** GRPO with a verifiable reward — the model's
  verbalized predictions checked against the realized future in the demo —
  improves the truthfulness of novel sampled reasoning. Demos give complete
  future state, so predictions have deterministic truth values.

## Pipeline at a glance

```
HLTV demos ─► 8 Hz tick tensors ─► WORLD MODEL ─► LANGUAGE BRIDGE ─► GROUNDED GRPO
              (round-scoped,       (19M, frozen    (frozen Qwen3.6-    (reward = the
               match-split)         after retrain)  35B-A3B + QLoRA,    realized demo
                                                    NLA gate)           future)
```

**World model.** A 19M causal transformer over 11 tokens per frame (10 players
+ 1 global). The prediction head is distributional — 97-class
classify-then-refine over player xy displacement — because futures at decision
points are genuinely multimodal and a point regressor mode-averages into a
ghost-step no player takes. Step size k = 4 frames (500 ms); longer horizons
are reached by sampled autoregressive rollout, scored by coverage (minADE-K),
not point error. The value head is detached (stop-grad), so no outcome gradient
touches the trunk. The model is judged by frozen linear-probe transfer against
a measured floor and ceiling — never by prediction loss.

**Bridge and GRPO (phases 2–3).** A trainable resampler feeds the frozen latent
as soft tokens into Qwen3.6-35B-A3B (base frozen, QLoRA). The bridge is the
encoder half of a Natural-Language Autoencoder: a text-only decoder
reconstructs the latent from the generated text, giving a label-free
faithfulness metric. GRPO then rewards verbalized predictions against actual
demo futures. Full design: [`docs/bridge-design.md`](docs/bridge-design.md).

## Status (2026-07)

The 2026-07-18 methodology reset is the current ground truth:

- An adversarial review confirmed 24 findings (4 critical) — corpus defects,
  an unimplementable recipe, and a compromised measurement layer. Criticals
  and majors were fixed the same day; see
  [`docs/adversarial-review.md`](docs/adversarial-review.md).
- The retrain recipe (Knobs 1–7) is locked and pre-registered:
  [`docs/retrain-recipe.md`](docs/retrain-recipe.md) +
  [`docs/retrain-recipe-knobs4-7.md`](docs/retrain-recipe-knobs4-7.md). No knob
  unlocks without evidence hitting a written switch trigger.
- **The canonical retrain is pending.** All existing checkpoints (`wm_3map*`,
  round-encoder runs) are historical baselines only, never evaluated as final.
  Work proceeds through the ordered runbook in `claude-progress.txt`:
  corpus patch → distributional-head edges → trainer completion → coverage
  harness → local smoke → runs → pre-registered gates.
- The keystone C1 comparison has pre-registered pass/fail criteria (recipe,
  Knob 7), with committed failure branches: a C1-REP failure falsifies C1
  outright — no salvage wording.
- The prior VLM era ("See, Then Think") is archived: scripts in
  `scripts/_archive/`, rationale in the ledger.

## Where to read

| Doc | What it holds |
|---|---|
| `claude-progress.txt` | Run state + ordered runbook — read first every session |
| [`docs/decisions-ledger.md`](docs/decisions-ledger.md) | What was tried, killed, kept; evaluation traps |
| [`docs/retrain-recipe.md`](docs/retrain-recipe.md) (+ knobs 4–7) | The pre-registered training contract |
| [`docs/first-principles-plan.md`](docs/first-principles-plan.md) | Why each stage exists — the causal chain and weak links |
| [`docs/adversarial-review.md`](docs/adversarial-review.md) | Confirmed defects + punch list |
| [`docs/datasheet.md`](docs/datasheet.md) | Corpus certification + defect registry |
| [`docs/bridge-design.md`](docs/bridge-design.md) | Phase 2/3: bridge, NLA gate, GRPO reward |
| [`docs/corpus-strategy.md`](docs/corpus-strategy.md) | Corpus future and expansion triggers |
| [`docs/pod-runbook.md`](docs/pod-runbook.md) | GPU/pod guardrails — read before provisioning |

## Data

Corpus facts live in [`docs/datasheet.md`](docs/datasheet.md). In brief: 92 pro
matches → clean corpus of 3,876 train / 705 val rounds at 8 Hz, round-scoped,
match-split. The canonical arm trains on 5 maps (3,573/641 clean rounds) with
de_overpass held out entirely as the OOD set (367 rounds). Two feature schemas:
v2 (597-d, canonical) and v3 (687-d, +9 derived perception dims/player,
ablation arm).

Demos are ingested by a zero-local-storage HLTV → `.dem` → HuggingFace pipeline
([chimera-demo-pipeline](https://github.com/skkwowee/chimera-demo-pipeline)).
Locally, `scripts/parse_demos.py` produces per-tick parquet (awpy) and
`scripts/build_tick_sequences.py` assembles the tensor blobs.

## History — why next-state prediction

The project began as a vision-language pipeline ("See, Then Think") and pivoted
after consistent, load-bearing negative results: outcome supervision saturated
the round encoder at ~16 demos (round outcome is ~1 bit/round); change-point
losses found statistical, not semantic, boundaries; Claude-generated captions
were circular (+0.008 over the structured-feature ceiling); per-event
commentary grounding topped out at ~25% despite a clean 4.6σ global VOD↔demo
alignment. Every one of these was a sparse, downstream, or circular signal.
Next-state prediction is the opposite — dense, self-supervised, and grounded in
the actual dynamics of the game. Full rationale:
[`docs/decisions-ledger.md`](docs/decisions-ledger.md).

## Standalone repos

- **[chimera-demo-pipeline](https://github.com/skkwowee/chimera-demo-pipeline)** —
  HLTV scrape → demo download → tick-sequence build on HF, zero local storage.
  Hosts the parked commentary-grounding work.
- **[cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)** — Next.js
  demo-replay viewer; dormant (rollout visualization now lives in-repo at
  `viewer/gen_viewer.html`).
- **[cs2-tools](https://github.com/skkwowee/cs2-tools)** — Python utilities for
  demo parsing, viewer data export, and screenshot capture.

## Project structure

```
chimera/
├── claude-progress.txt         # Run state + ordered runbook — read first
├── docs/                       # See "Where to read"
├── feature-list.json           # Harness feature inventory (passes per feature)
├── decisions.md                # VLM-era decision log (D001–D024, historical)
├── paper/                      # NeurIPS 2026 submission
├── data/
│   ├── demos/                  # Local .dem files
│   └── processed/tick_sequences/  # v2m/v3m tensor blobs (see datasheet)
├── scripts/
│   ├── parse_demos.py          # .dem → per-tick parquet (awpy)
│   ├── build_tick_sequences.py # parquet → tensor blobs
│   ├── train_world_model.py    # World-model trainer
│   ├── data.py                 # HF Hub data management (pull/push/clean)
│   └── _archive/               # Superseded VLM / round-encoder scripts
└── src/                        # Parked VLM code
```

## Setup

```bash
git clone https://github.com/skkwowee/chimera.git
cd chimera
uv sync

cp .env.example .env
# Edit .env with your HF_TOKEN
```

## Beyond games

The recipe — learn dynamics from cheap self-supervised next-state prediction,
then read the learned latent for value, events, and language — applies to any
structured multi-agent domain where dense self-supervision beats sparse outcome
labels. CS2 is the controlled environment to prove it: engine-accurate state,
clear outcomes, a rich tactical layer to recover.

## License

MIT
