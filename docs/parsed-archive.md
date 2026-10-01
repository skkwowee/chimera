# Parsed archive v2

The archive is not the model feature schema. Existing tick columns and the
kills/bomb/damages/rounds/header JSON files remain compatible with the builder.
No training features, splits or frozen corpus files change.

Additional player columns: flash_duration, active_weapon_name,
active_weapon_ammo, total_ammo_left, is_in_reload, zoom_lvl and duck_amount.
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

This does not fix score, missing-observation, time-grid, bomb-state or visibility
issues in the tensor builder. Those require separate versioned changes.

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
