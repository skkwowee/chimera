# Parsed archive v2

The archive is not the model feature schema. Existing tick columns and the
kills/bomb/damages/rounds/header JSON files remain compatible with the builder.
No training features, splits or frozen corpus files change.

Additional player columns: flash_duration, active_weapon_name,
active_weapon_ammo, total_ammo_left, is_in_reload, zoom_lvl, duck_amount and game_time.
These are native parser values, not inferred tactical labels; nulls remain
null (for example zoom on a knife). Ammo semantics must not be interpreted as
a readiness guarantee. Exact ticks and player IDs remain in the records.

Four additional Parquet tables preserve Awpy's smokes, infernos, shots and
footsteps, including entity/player IDs and source tick boundaries. Smoke/fire
end times may use Awpy's duration fallback; the configured durations are recorded
in the manifest. Sound events do not establish who actually heard them, and
geometric LOS is not player knowledge. Dense grenade trajectories and raw flash/HE
detonation tables are not exported in this version; retain the original demos.

`<stem>_parse.json` records the raw-demo hash, parser script hash, package versions,
tickrate, requested properties and hashes of all ten output files. It is published
last. A missing, outdated or damaged bundle is reparsed; `--force` also reparses.
Use separate match-ID directories to prevent same-stem collisions across matches.
The companion pipeline collector must include these five additional files when
archiving. An older `--from-parsed` bundle remains legacy, not an enriched archive.

Parser revision 3 explicitly configures 64 Hz and checks it against the median
tick/game-time delta over the first 4,096 observed ticks. Revision 2 inherited
Awpy's 128 Hz default; its tickrate metadata and duration-fallback tables must be
regenerated. The script/version hash invalidates those completion markers.

The archive alone does not fix the tensor builder or model visibility.

## Six-map pilot (2026-10-01)

StarSeries matches 2398107 and 2398108: all six Source 2 demos parsed, totaling
117 rounds and 8,719,630 player-tick rows. Each map contained ten distinct players,
nonzero blindness/reload observations and nonempty new event tables. All 66
output files passed manifest/hash verification; reruns skipped all six bundles.
Outputs remain local under `data/staging/2026-09-starseries/<match_id>/parsed-v2/`.

This is an extraction/integrity check, not a tactical-ground-truth certificate.
In particular, native `total_ammo_left` peaked at 4 on these demos, so its meaning
is unvalidated and it must not be consumed as reserve bullet count. Utility
geometry, duration fallback and actual observer knowledge remain unverified.
No tensors were rebuilt. Cache and Anubis remain outside the current map vocab.

## Builder follow-up (2026-10-01)

Fresh builds use `feature_schema_v2.2` (still 597 dimensions); derived builds use
`feature_schema_v3.1` and record their source schema. No frozen corpus, split,
training default or model feature width is changed. Old derived features cannot
be reused across schema/tick-grid changes.

- Slots use the full round's roster. Missing, duplicate, null or non-finite
  sampled player state rejects the round instead of encoding a death. Rejections
  are printed and recorded in demo summaries; unresolvable rosters/scores abort
  the build. This is conservative exclusion, not interpolation or a new mask.
- Sampling uses an exact raw-tick grid, not every eighth *observed* row. Metadata
  retains `raw_ticks`, ordered `player_steamids` and source tickrate. Legacy
  archives without a parse marker still assume 64 Hz; they are not clock-certified.
- Scores follow roster overlap through side switches, including rejected rounds.
- Drop-event coordinates are last-known locations until pickup/plant, with the
  existing `none` bomb-state bit (no new dropped-state dimension). Plant bits,
  not nonzero coordinates, gate derived bomb distance. Post-plant context persists
  after resolution, but bomb age stops at resolution/end and is capped at 40s.

After the clock correction, all six maps were rebuilt **in memory for validation**:
98/117 rounds, 91,139 frames accepted; 19 rounds excluded for sampled tick gaps.
Exact ticks, per-slot alive flags and team scores matched the archived source.
Only Inferno/Mirage are in the current map vocabulary (33 accepted rounds);
Cache/Anubis were diagnostic builds, not training candidates. Three regression
tests cover these fixes. The raw demos and all 117 archived rounds remain intact;
no training blobs were replaced or uploaded.

## Data-readiness audit checkpoints (2026-10-01)

Read-only HF inventory at revision
`69b9cbaafa70b400b1fbe7c371304f004af5e3d6`: 189 unique raw-match records
reference 506 demo paths, all present. The 70 tensor-match records have neither
schema versions nor archived parse bundles; all 179 referenced demo paths exist.
Both newly downloaded match IDs are absent from HF. All 189 raw manifest dates
are null, so this manifest alone cannot certify recency or tournament tier.
No remote artifacts were changed.

Locally, the old corpus has 81 raw demos and 81 parsed maps. The split manifest
contains 92 groups (70 HF match IDs and 22 local team-pair groups). Both P2
validation blobs exist; both P2 training blobs are absent. P2 remains explicitly
`complete=false, canonical=false`. No large training blob was loaded.

The 19 fresh-pilot gaps were checked directly with `DemoParser.parse_ticks`
using every absent tick and the same game-state flags Awpy filters. All 34,727
ticks exist with ten player rows each. Every tick is freeze time and waiting for
resume: 18 gaps of 1,919 ticks are team timeouts, one gap of 185 ticks is a resume
pause. This is intentional Awpy filtering, not damaged downloads. The builder's
98-round result remains unchanged: retaining pauses, cropping freeze time or
splitting sequences would be a separate sampling-policy decision.

Recommendation: preserve P2 as the historical lane and prepare a separately
versioned fresh-data candidate. Do not mark P2 complete or mix schema generations
just to make a training command run. Integration fixes and candidate validation
are the next checkpoint.

### Inventory and handoff result

The HF tree contains 551 raw demos. Its 506 manifest references resolve to 498
unique paths: eight flat paths are claimed by multiple match IDs, and 53 raw
files are unlisted. Five reused paths cross the frozen train/validation split
(four match pairs: 2394156/2393226, 2394174/2393350, 2394148/2391109,
2393042/2394222). This proves ambiguous **re-bake source ownership**, not that
the already-baked tensors are identical; do not silently relabel or delete them.
All 70 remote schema JSONs report `feature_schema_v2`, 597 dimensions, nominal
8 Hz. The manifest's missing versions are not evidence of v2.2 compatibility.
All 81 local raw headers are Source 2. The P2 v2 validation blob has 770 rounds
and 14 match IDs, but no exact raw-tick vectors or raw-demo hashes in round meta.

Checkpoint `176e1e9` (Chimera) and `9a7366f` (pipeline) close the handoff seams:
the CLI requires one match ID, rejects demo-level splitting and nonempty output
directories, and requires verified parse manifests. Unsupported maps are logged
and excluded; an all-excluded build fails. Round metadata keeps match/source
identity. Script hashes survive sandbox copies. The uploader refuses unreadable,
empty or identity/tick-misaligned tensor bundles. Training checks semantic schema,
cadence, match overlap and available raw hashes; checkpoints stamp their source
schema and forecast evaluation rejects schema mismatches. These are compatibility
guards, not changed model architecture, losses or canonical split assignments.

The local-only integration adapter exercised the real pipeline using four raw
demos and a filesystem upload sink, **without any HF writes**. It preserved 44
archive files and produced 33 supported-map rounds / 33,251 frames (80 MB).
`--from-parsed` replay passed; a separate rebuild produced identical tensors,
round metadata, event labels and event times on all 33 rounds.

A historical training-side source (`local-gamerlegion-vs-vitality`, one Mirage
demo) was reparsed separately: 12 accepted rounds / 9,525 frames (23 MB).
That candidate and the fresh 33-round candidate have disjoint match IDs and raw
hashes and pass the actual `RoundWindows` loader. This two-group integration
fixture is **not** a new canonical split or a generalization benchmark.

The existing CPU smoke run completed 30 steps at k=4, with finite evaluations and
schema-stamped checkpoints. Native-step forecast generation/scoring passed on 101
anchors; repeating generation reproduced all saved arrays and provenance exactly.
The smoke deliberately reuses its input as validation, runs only 30 steps, and
does not reach scheduled sampling's ramp. Its metrics are not quality evidence.
No GPU/pod, paid compute, mass download, canonical retrain or merge was performed.
The split manifest and both P2 validation-file hashes still match the committed
corpus manifest. Existing large training blobs were not loaded or modified.

### Reproduction on this workspace

Run from the Chimera checkout. Use fresh output directories when repeating builds;
the local adapter is retained at `data/staging/readiness/run_pipeline_handoff.py`.
It imports the companion pipeline checkout and redirects uploads to disk only.

```bash
../chimera-demo-pipeline/.venv/bin/python data/staging/readiness/run_pipeline_handoff.py data/staging/readiness/pipeline-pass-2
.venv/bin/python scripts/build_tick_sequences.py --match-id 2398108 --demos-dir data/staging/readiness/pipeline-pass-1/parsed/2398108 --out-dir data/staging/readiness/replay-2
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/train_world_model.py --smoke --arch player --dist-head --horizon 4 --val-pt data/staging/readiness/pipeline-pass-1/tick_sequences/2398108/train.pt --out outputs/readiness-repeat --seed 0
.venv/bin/python scripts/eval_forecasts.py run --checkpoint outputs/readiness-repeat/h4_mt/last.pt --corpus data/staging/readiness/pipeline-pass-1/tick_sequences/2398108/train.pt --out outputs/readiness-repeat/forecasts.npz --stride 256 --samples 4 --device cpu
.venv/bin/python scripts/eval_forecasts.py score outputs/readiness-repeat/forecasts.npz
```

The trainer appends `h4_mt/` to `--out`; checkpoints are not at the bare output
root. Existing local artifacts: `data/staging/readiness/pipeline-pass-1/`,
`data/staging/readiness/legacy-train-candidate/`,
`data/staging/readiness/replay-from-parsed/`, and
`outputs/data-readiness-smoke-k4/`. These are ignored local data, not uploads.

Next decision: use a separately versioned fresh corpus once ambiguous raw-source
ownership is resolved. Decide freeze/pause treatment before recovering the 19
excluded rounds; decide Cache/Anubis support before widening the map schema. Do
not complete P2 by quietly substituting new-builder output: its semantics differ.
