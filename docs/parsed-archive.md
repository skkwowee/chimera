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
