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

## Data-cleanup completion (2026-10-03)

The fresh builder now writes `feature_schema_v2.3`: when sampled ticks are absent
before `freeze_end`, discard the prefix through the last such missing tick.
Keep the original grid alignment and every sampled gameplay tick; never join
across a missing interval. Missing player state or missing ticks during play
still reject the round. `pre_live_trim_ticks` records the discarded prefix span;
demo summaries count affected rounds. This changes sampling, not feature width.

All six pilot demos were rechecked against the committed v2.2 builder in memory:
117/117 rounds accepted, all 19 previously excluded rounds recovered, and all
98 previously accepted tensors/event labels/event times remain exactly equal.
Every original sampled gameplay tick is retained. Cache/Anubis were diagnostic
checks only; they remain excluded from actual builds. The supported-map staged
pair contains 41 fresh rounds plus 14 historical rounds, not a canonical split.

HF revision `94405b3cb7107293555141bce9b02a3f7b988809` resolves the eight shared
legacy paths: 16 distinct original CS2 files recovered from match-upload history
into `demos/<match_id>/` paths, with 14 raw-manifest rows updated atomically.
Source 2 headers, historical SHA-256s and destination hashes are verified.
All 189 match records remain. No old demo or tensor was deleted or overwritten;
HF added only the 16 files, their LFS tracking lines and the manifest references.
Full source revisions/hashes are in the companion pipeline's
`reports/2026-10-03-source-recovery.json`.

The pipeline rejects raw paths claimed by multiple matches. Changed source paths
make old tensor entries pending and prevent reuse of their old parse archive.
Historical tensor provenance is intentionally unchanged: this repairs future
re-bake inputs, not a certification of existing tensors or their historical split.

P2 is still incomplete and untouched. Feature selection, Cache/Anubis support
and the next controlled training experiment remain separate decisions. No model
training, feature expansion or full-corpus rebuild was performed for this cleanup.

## Player-view capture recipe (2026-10-05; not yet rendered)

Cleanup PRs Chimera #7–9 and demo-pipeline #1–3 are merged. Both main branches
passed CI. The next perception artifact should be **engine-rendered first-person
video**, not another geometric visibility feature. This is a reconstruction,
not a recording of the original player's screen or evidence of their attention.

Use [CS Demo Manager's CLI](https://cs-demo-manager.com/docs/cli) with CS2,
HLAE and FFmpeg. Its [video guide](https://cs-demo-manager.com/docs/guides/video)
documents direct video encoding with HLAE, avoiding temporary raw-image dumps.
Analyze a demo once, then supply a JSON config containing multiple short sequences
to `csdm video --config-file capture.json`. No repeated GUI clicking is needed
after installation/setup. Keep one renderer worker; batch by demo and player.
Start with a few 8-second, 1080p/64-fps clips. Reduce resolution/frame rate only
after checking that small enemies and brief peeks survive; rendering throughput
has not been measured. Select ordinary live-play intervals as well as smoke,
flash and peek examples, not just kills. Do not render every player for every
tick before this pilot passes.

### Version and safety gates

- Local CS2 exists, Steam build `25687242`, `PatchVersion=1.41.8.8`,
  `ClientVersion=2000924`. The inspected Inferno pilot's demo header reports
  `patch_version=14181`. These are different patches. Valve warns that
  [TrueView is disabled by default on version mismatch](https://www.counter-strike.net/newsentry/578276333072678918)
  and is not identical to the original screen even on a matching build.
  Forcing TrueView on does **not** certify compatibility.
- CS Demo Manager and HLAE were not found in PATH/standard installation locations.
  Source reviewed: CS Demo Manager `v3.20.1`; latest HLAE release observed:
  `v2.192.6`. Their compatibility with this installed game is untested.
- Use a separate HLAE config folder, offline `.dem` playback and `-insecure`;
  never join live servers. Confirm CS2/anti-cheat clients are not in use before
  launching: CS Demo Manager's launcher can stop an existing CS2 process.
  Do not downgrade the user's normal game installation. If needed, use a separate
  [compatible game installation and plugin](https://cs-demo-manager.com/docs/guides/playback).
  Installation/game launch awaits approval; no capture was attempted.

### Smallest executable probe

Source: match `2398108`, `aurora-vs-vitality-m2-inferno.dem`, SHA-256
`fa4973cff27742992e9a8986cd1a42fbdc50ceee7eb21b0f3173f980dfb44873`.
Use ZywOo (`76561198113666193`), round 1, ticks **1715–2227** (8 seconds).
The parsed archive confirms he is alive throughout, including the preceding
64 ticks. Copy that verified demo to a Windows-local capture folder, then:

```powershell
csdm analyze "C:\chimera-capture\aurora-vs-vitality-m2-inferno.dem"
csdm video "C:\chimera-capture\aurora-vs-vitality-m2-inferno.dem" 1715 2227 --focus-player 76561198113666193 --recording-system HLAE --encoder-software FFmpeg --recording-output video --ffmpeg-video-container mp4 --width 1920 --height 1080 --framerate 64 --true-view --no-show-x-ray --no-player-voices --cfg "cl_trueview_show_status 1" --close-game-after-recording
```

This is an **unexecuted diagnostic command**, not a verified capture. Configure
the database, isolated game config and compatible recorder/plugin first. In
batch JSON, set `trueView: true` at the top level and `showXRay: false`,
`playerVoicesEnabled: false`, `cfg: "cl_trueview_show_status 1"` in **every**
sequence. Put the player camera switch before the retained clip (e.g. tick 1651
for this probe). Do not rely on appended CLI flags: the reviewed
[config-file parser returns early](https://github.com/akiver/cs-demo-manager/blob/v3.20.1/src/cli/commands/video-command.ts).
The [recorder hides TrueView status by default](https://github.com/akiver/cs-demo-manager/blob/v3.20.1/src/node/video/generation/create-cs2-video-json-file.ts);
retain the explicit override for audit clips.

Before accepting output, inspect POV identity, X-ray state, TrueView status,
smoke/flash rendering and warm-up artifacts; verify frame-to-demo-tick alignment
against known events rather than assuming frame 0 equals the requested tick.
Cache only checked outputs, keyed by demo hash, player, tick interval, game build,
CS Demo Manager/plugin/HLAE versions and capture settings. Preserve this metadata
beside each video. Keep mismatched/uncertain captures separate from validated
ones. Rendering produces pixels, not automatic enemy-visibility labels: reviewed
labels remain a separate step. No model inputs or training corpus change here.
