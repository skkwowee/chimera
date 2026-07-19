# Chimera CS2 — Dataset Datasheet (Stage 0)

*Reproducibility + defensibility record for the training corpus. Written 2026-06-29.
This certifies what the corpus IS; it does not decide the feature schema (see §7).*

## 1. Provenance
- **Source:** professional CS2 demos from HLTV (tier-1 events).
- **Skill tier:** professional only. **Scope claim must say so** — results are not
  claimed to generalize to lower skill tiers without an OOD test.
- **Parser:** `demoparser2 >= 0.41.3` (0.41.1 hits `EntityNotFound` on Major demos).
- **Canonical store:** HF `skkwowee/chimera-cs2`, `tick_sequences/<match_id>/{train,val}.pt`.
  CORRECTED 2026-07-19: HF holds 224 raw `.dem` (109.7 GB) under `demos/` covering
  ALL 92 matches (the earlier "~27 of 81 stems / re-bake impossible" claim was false
  — it conflated per-match tick_sequences blobs with raw demos). Split provenance:
  70 HF-pipeline + 22 local matches; local rounds with no HF tick-blob overlap are
  back-filled as `local-<team-pair>` pseudo-matches (mechanism unchanged). NOTE:
  filename is NOT a unique match key (5 proven same-name-different-content
  collisions) — keying is sha256 + HLTV match id.

## 2. Representation
- **Rate:** 8 Hz (downsample 8 from **64-tick** CS2 demos; corrected 2026-07-18 —
  an earlier version said 128-tick, which was wrong and internally inconsistent;
  adversarial-review D5). Round-scoped (no attention across resets).
- **Tokens:** 11 per frame = 10 players + 1 global.
- **Schema v2 (`*_v2m.pt`):** 597-d/frame — 10×56 per-player + 37 global.
- **Schema v3 (`*_v3m.pt`):** 687-d/frame — v2 + 9 derived perception dims/player
  (LOS/FOV/exposure raycasts, input-only). **Confound flagged** — see §7.
- **Target:** next-frame xy displacement (= velocity·dt), per player.
- Full layout: `data/processed/tick_sequences/feature_schema_v1.json` (labeled v2).

## 3. Split (leak audit) — PASS
- **Unit:** match-level. `seed=0`, `val_frac=0.15`.
- **92 matches → 78 train / 14 val. Zero match overlap** (verified).
- Dedup key: `(norm_stem, round_num, first_tick, n_ticks)`. `norm_stem` canonicalizes
  team order (`a-vs-b` == `b-vs-a`) so a renamed duplicate of the same game cannot
  straddle the split. This is the leak class that most often invalidates sports-ML papers; it is handled.
- Manifest: `data/processed/tick_sequences/split_manifest_v2.json`.

## 4. Size & distribution (rounds)
| map | train | val |
|-----|------:|----:|
| de_mirage | 1052 | 173 |
| de_dust2 | 978 | 163 |
| de_nuke | 540 | 145 |
| de_inferno | 538 | 94 |
| de_ancient | 465 | 66 |
| de_overpass | 303 | 64 |
| de_anubis | 151 | 65 |
| de_train | 16 | 0 |
| **total** | **4043** | **770** |

- **Imbalance:** mirage+dust2 ≈ 50%. **Cross-map metrics MUST be reported per-map**,
  never pooled — a pooled average is dominated by mirage and hides transfer failure.

## 4b. Feature-schema review (2026-07-05) — PASS with notes
Dimension-level audit of the encoders (`build_tick_sequences.py`), verified empirically:
- **No target/outcome leakage** — all global dims are past-or-present only.
- **Dead players freeze at body position** (531/531 sampled death transitions; zero
  origin-teleports) — displacement targets are benign without an alive mask, but the
  canonical retrain adds one anyway (see retrain-recipe Knob 5) to stop spending
  capacity on corpses (~13% of player-frames).
- **Known observability gap:** ACTIVE utility (smokes/mollies/flashes in the world) is
  not represented — only carried inventory. A mid-round smoke is invisible to the model
  (and to the v3 LOS raycasts). Legitimate future feature (Line-in-the-Sand pass);
  documented limitation for now.
- Cosmetic: CS:GO-era normalization constants (score/16, round/30, time/115) can
  mildly exceed 1.0 in OT/post-plant; consistent train/eval, left as-is.

## 5. Known defects & exclusions
- **D1 — de_anubis has no map identity (CONFIRMED bug).** `MAP_VOCAB` has 7 maps and
  omits anubis; `build_tick_sequences.py:256,575` leave the map one-hot all-zeros for
  unknown maps. 216 anubis rounds (151/65) are indistinguishable by map. **Action: EXCLUDE**
  (mask `map_idx == -1`) until either dropped or vocab+rebuild. Pollutes cross-map claims.
- **D2 — de_train is unusable.** 16 train / 0 val rounds: too few to learn, impossible to
  evaluate. **Action: EXCLUDE.**
- **Clean corpus after D1+D2 exclusion: 3876 train / 705 val = 4581 rounds.**
  Exclusion is applied as a load-time mask (reversible), NOT a re-bake.
  (Canonical TRAINING corpus is further restricted to 5 maps = 3,573/641 with
  de_overpass held out as OOD — retrain-recipe Knob 4.)

**Defects found by the 2026-07-18 adversarial review (docs/adversarial-review.md)
— code fixed, CORPUS REBUILD REQUIRED before the canonical retrain:**
- **D3 — bomb_state one-hot entirely dead** (vocab string mismatch; all 4 bits zero
  on every frame corpus-wide). Falsifies the earlier "dimension-level audit — PASS"
  claim for this block. [review D1] **Follow-up (2026-07-18): the raw `bomb_site`
  label itself is unreliable** — 873/879 plants labeled "bombsite_b" while plant
  positions form two well-separated clusters (1.5–3k units) on every map, i.e.
  both sites are planted but the label doesn't record it. The rebuild must derive
  site from plant POSITION (xy; z on nuke, whose sites stack vertically), not the
  metadata label. **Refinement (implementer validation, 927 plants/40 demos): the
  per-EVENT bombsite strings are fine (493 A / 434 B, 0 disagreements with derived
  positions) — the broken path was the per-round metadata. The v2.1 builder derives
  from position and logs the event label as a cross-check.**
- **D4 — round_time includes pauses/halftime** (anchored at previous round's
  official_end; ~28% of rounds shifted, up to +2.4 normalized units). [review D2]
- **D5 — 17.5% of frames are freeze-phase**, previously undisclosed; now masked
  from displacement losses at train time and to be disclosed/stratified in evals.
  [review D3]
- **D6 — v3 dist_to_bomb was distance-to-origin on the 84% pre-plant frames**;
  now plant-gated with sentinel. Contaminated the v2-vs-v3 deconfound. [review D4]

## 6. Reproducibility checklist
- [x] Corpus regenerable from script (`merge_hf_tick_sequences.py` + `build_v3_features.py --workers 1`).
- [x] Split deterministic (`seed=0`) and manifested.
- [x] Parser version pinned.
- [x] Exclusion mask (D1/D2) implemented as a shared load-time helper —
      `scripts/_corpus.py::clean_blob` (unit-tested). Wired into `value_probe.py`;
      wire into `rollout_eval.py` / `facing_bias_check.py` as each is run.
- [ ] Sampling-rate (8 Hz) justification written — peeker-advantage windows ~100–250 ms;
      confirm 8 Hz is sufficient or ablate to 16 Hz. TODO.
- [ ] OOD / lower-tier holdout for the generalization claim — TODO (optional, strengthens paper).

## 7. What this datasheet does NOT decide
The feature schema (v2m vs v3m, and any future v4 adding velocity inputs) is **NOT**
settled here. It is settled by experiment:
- **v2 vs v3 (does perception help?)** → Tier-0 `value_probe.py` deconfound on
  `wm_3map` (v2+cotrain) vs `wm_3map_dist_v3m` (v3+cotrain).
- **v4 velocity (facing-shortcut fix)?** → gated on the `facing_bias_check.py --corrupt-yaw`
  causal test. Do NOT rebuild on the unreproduced +27pp number.
Rebuilding tensors on a hunch is the scope-creep trap; schema changes require a passing gate first.
