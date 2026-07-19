# Corpus Strategy — Consolidated Verdict (2026-07-19)

Synthesized from six evaluated + red-teamed dimensions (source, cadence-representation, schema, scale, storage-provenance, split-ood). All recommendations below are the **post-attack amended** versions; the storage-provenance original was overturned on attack and is superseded here.

---

## 1. Verdict on runbook [1]: patch-in-place **STANDS** — with three amendments and one corrected justification

Every dimension independently re-confirmed patch-in-place as the correct immediate move. Nothing below blocks, reorders, or moots it. But the audit changed *why* it is right and *how* it must run:

**Corrected justification (fix the wording in claude-progress.txt, datasheet §1, MEMORY.md):** the parenthetical "full re-bake impossible: ~65/92 matches HF-only" is **false**, verified live. 224 raw `.dem` (109.7 GB) persist on `skkwowee/chimera-cs2/demos/`, covering 171/171 stems for the 70 pipeline matches; all 81 local stems also exist on HF by name (76 byte-identical, 5 same-name-different-content). 92/92 matches are re-bakeable. The split is **70 HF + 22 local**, not ~65/27. Patch-in-place is right because it is ~100× cheaper (~1h CPU vs ~110 GB download + 20–40 CPU-h parse — intermediates were NOT archived) and reopens no knobs — **not** because there is no alternative. This one false fact was mis-pricing every downstream corpus decision.

**Amendments to [1] (all cheap, none reorder it):**
1. **HARD PRECONDITION — pre-patch snapshot.** The merged train/val v2m+v3m blobs (~28 GB) + `split_manifest_v2.json` are the *actual* single-copy irreversible asset (WSL disk only, nothing on HF) — and [1] mutates them in place. Snapshot before patching (local copy minimum; HF/B2 preferred), archive the patched blobs + manifest after.
2. **Patch lineage stamping.** `patch_corpus.py` must write {script sha, transforms, blob sha256 pre/post, date} into the blobs and the new corpus manifest — otherwise the exact tensors R1 trains on become provenance orphans.
3. **Widened validation + join-key assertion.** Validate against ≥1 HF-era match (download, re-bake with v2.1 builder, diff) in addition to ≥1 local-era match — the 143 HF-era stems have never been locally verified. Add the two-line check that every blob meta carries `match_id` and the set equals the 92 split-manifest keys. This diff doubles as the v2.1 builder's correctness certificate — which is why **no stage-2 baking of anything new happens until [1] validates**.

Bonus synergy: [1]'s validation reads the same local 64-tick parquets the aliasing audit needs — run both in one session.

---

## 2. Decisions NOW (this week; all CPU/bandwidth; none block [1]–[7])

| # | Action | Cost | Protects |
|---|--------|------|----------|
| 1 | **File the FACEIT Downloads API application** — TIME-SENSITIVE: approval is discretionary, ~30-day response, and FACEIT demos live in a rolling 30-day window (prospective-only, no backfill). Filing commits to nothing. Fallback if denied/stalled >45 days: HLTV tier-2/3 (stars-0/qualifiers) via existing scraper. | Free, 10 min | The tier-OOD option — the paper's biggest scope objection converted to a measurement |
| 2 | **Pre-patch blob snapshot** (precondition of [1], see §1) | ~28 GB copy, hours | The only truly irreversible corpus asset |
| 3 | **Correct the record everywhere** — re-bake feasible not impossible; 70+22 split; filename is NOT a unique match key (5 proven cross-event collisions; key on sha256 + HLTV id) | ~1h docs | Removes a reviewer-falsifiable false claim from the datasheet; re-prices all triggers |
| 4 | **Split-integrity audit of the 5 divergent stems** (e.g. spirit-vs-vitality-m1-mirage: 945 MB local vs 654 MB HF) — same game re-encoded would defeat the dedup key and put rounds on both sides of the split | 30 min | Datasheet §3 dedup/split PASS |
| 5 | **Upload only the 5 content-distinct local demos** (~2.9 GB) under collision-safe event-qualified names — NEVER overwrite existing `demos/` paths; delete the 4 `Zone.Identifier` junk files. (The "backfill 53 demos" plan is dead: those files aren't missing, and by-name upload would clobber live provenance.) | Hours upload | Completes 92/92 demo redundancy without corrupting it |
| 6 | **Aliasing audit (weak-link #10)** — no re-parse needed: 81 local matches already have full 64-tick parquets. Measure % direction reversals invisible at 8Hz, sub-sample displacement energy, peek onset-to-apex vs 125 ms; map-stratified. **Commit metric definitions and BOTH trigger thresholds before computing.** Write result into datasheet §6. | Hours, $0 | Closes the open TODO with a number; arms the 16Hz trigger with evidence |
| 7 | **Fix `build_tick_sequences.py:918`** — derive `tickrate_hz` from 64/downsample, not hardcoded 8 (plus line 846 print) | Minutes | Any future 16Hz bake would stamp false provenance |
| 8 | **Start (b-lite) archival scrape**: `chimera-demo run --stars 2`, +200–300 CS2-era matches, 25–50/day, stage-1 only — demos to HF, **nothing baked, nothing merged** | ~$0, 6–12 days background | Insures the fragile scrape channel (Cloudflare/JA3), pre-positions plausible scale-out; honest EV: ~14 effective epochs per [H9] makes trigger-fire plausible, not certain |
| 9 | **Write the scale band into the T-SCALE trigger**: one documented step to 300–400 matches (800 = ceiling); Knob 6's 25k steps FIXED at every size; map-mix pinned, overpass held out; **anubis/MAP_VOCAB fix explicitly EXCLUDED from the scale re-bake** (separate defect-fix event) | 30 min | Makes the future scale event pre-registered, single-variable |
| 10 | **Pre-register the tier-OOD eval spec in datasheet §6**: FACEIT via own pipeline, same v2.1 builder; ~30–50 lvl-10 + ~30–50 lvl-4–6; 5 training maps; eval-only; Knob 7 probe protocol; PLUS: contemporaneous HLTV pro control slice (separates tier from patch/meta drift), defined tier statistic (lobby-median level + in-band fraction), predicted direction written before R1 (degradation is EXPECTED and thesis-consistent), FACEIT tensors stay private (redistribution terms unpublished) | 45 min | Pre-registration vs post-hoc rescue |
| 11 | **Write the v5 pre-registration stub** (renamed — "v4" is already pre-registered as the velocity add): smoke/molly volumes (~20 global dims), flash_duration, active-weapon+reload, bomb_z; TTL-noise caveat; per-channel gates; **masked-equivalence gate** (masked-v5 must reproduce patched-v2m or the ablation is within-bake only, +$15–30 masked-baseline arm); persist parse intermediates to HF during the bake | ~1h doc | Kills "post-hoc schema fishing"; retires the re-parse cost class permanently |
| 12 | **Land corpus_manifest v1 spec** (see §4) + pipeline change: upload per-match parquet + event JSON going forward (~12–50 MB/match); explicitly decide props-v2 (grenade + velocity props in archive parse) rather than drift | 1–2h code | NeurIPS reproducibility contract |
| 13 | **HF PRO $9/mo — REQUIRED** (free-tier public storage is officially "best-effort"; ~150 GB rides on it) **plus** one non-HF copy of the irreplaceable set (5 unique demos + blob snapshot + split manifest; B2 full mirror $0.66/mo optional). Not substitutes: PRO covers quota, the mirror covers takedown/account loss | $9/mo | The entire re-bake story |
| 14 | **Build `exam_manifest.json`** (92 matches → teams/event/event_date/stars/source; hand-enter ~12 event dates — `date_unix` is empty 0/70; ~2–3h for the 22 local pseudo-matches via HLTV lookup; flag possible merged-series pseudo-matches) + pre-register the exam rows and numeric triggers (see §5) | ~0.5–1 day, $0 | Team/temporal defensibility at manifest-only cost |
| 15 | **Record negative-source rationales in the datasheet**: PureSkill csds (parser-provenance confound), ESTA/CS:GO-era (wrong game/parser/schema) | 15 min | Stops future "time-saving" shortcuts |
| 16 | **Absorb into runbook [4]**: pre-registered secondary 64Hz-truth-scored column for decision-frame predictions (local val rounds; primary 8Hz metric unchanged — never replaces it) | Near-zero | Kills "your metric is aliased too" |

Training-source strategy itself: **unchanged — pro HLTV only.** The bedrock argument (every downstream stage samples futures from the behavioral prior, so demonstrator quality is load-bearing) survived two rounds of attack intact.

---

## 3. Armed triggers (deferred; do NOT execute now)

| Change | Trigger | Cost class | Knob impact |
|---|---|---|---|
| **Tier-OOD eval bake** (FACEIT lvl-10 + lvl-4–6, eval-only) | Paper assembly or first reviewer-risk assessment; gated on Downloads approval OR HLTV tier-2/3 fallback | New-scrape + bake, eval-only; ~3–6 days + <$10 CPU | None — pre-sanctioned by datasheet §6; tier lives in eval manifest, not tensor |
| **T-SCALE pro scale-out** (→300–400 matches) | Runbook [6] dense curve still improving at 100% of matches | Re-scrape (already pre-positioned by b-lite) + full re-bake of both schemas | Reopens **Knob 4 counts + Knob 7 ceiling** (written re-lock, ceiling retrained on 4090); Knob 6 budget stays fixed; anubis excluded |
| **T-SATURATE** | Curve flat 50%→100% | $0 | Stop scraping, certify 92, retire weak-link #6 |
| **16Hz matched-subset ablation** | Audit shows >20% turn-bucket reversals invisible at 8Hz, OR R1 turn-frame fails baselines AND audit >10% (both thresholds committed pre-compute) | Subset re-bake (local parquets, downsample-4) + 2 runs, ~$20–40 or local 4090 | None as side-by-side; a **canonical** 16Hz switch, if earned (beats R1 3-seed noise band), reopens Knobs 1/5-binding/6 |
| **v5 superset bake** (smoke/molly, flash, weapon, bomb_z + 16Hz sidecar) | Utility-gap diagnostic (post-R1, pre-registered margin) OR the 16Hz ablation trigger — either runs the bake; **which channels get reported is per-channel gated** | One batched re-bake: ~110 GB streamed + 3–6 CPU-days (v3 raycast, --workers 1) + $20–55 pod | Extends Knob 2 via the masked-equivalence gate; velocity stays OUT (Knob 3, blob-derivable, needs no bake ever) |
| **Parquet backfill, 70 HF matches** | 16Hz ablation or velocity gate firing | ~110 GB download + ~1 CPU-day (streamable) | None. Cannot serve weak-link #12 (props-v1 has no grenade props — that needs the demo re-parse above) |
| **Team-holdout retrain arm** | LOTO probe drop CI-separated for ≥3 of top-8 teams, OR frequency slope p<0.01 after win-rate covariate | 1 pod run ~$8–15 | Written, dated **amendment** to Knob 4's ~$20 budget clause (not "executing the lock"); 6-map ID-ceiling stays pinned as rebuttal contingency |
| **Temporal-cutoff retrain arm** | Future-event eval pack (Option F) degrades CI-separated vs ID val row | 1–2 pod runs (size-matched control mandatory — era confounds source) | Same amendment route; canonical split never re-cut |
| **T3: mixed-tier TRAINING** | Scaling curve demands data beyond HLTV supply AND C1 already answered on pro corpus | Full pivot: new ingest, full re-bake, all arms re-run | Formal paper-pivot: reopens Knobs 2/4/7 in writing + fresh ceiling |
| **Synthetic bot data** | Reviewer disputes physics-vs-tactics attribution AND CHANGE B baseline insufficient | ~1–2 days server setup | Never trains. Diagnostic only |
| **Cold-mirror escalation** | Any HF storage/moderation email | $0.66/mo B2 | None |

---

## 4. corpus_manifest v1 — the reproducibility contract

Sidecar to the blobs; written by the bake pipeline and by `patch_corpus.py`. Required fields:

- **`corpus_version`** — semver, e.g. `2.1.0+patch1`
- **Per-match source inventory** — `{match_id, HLTV match id/URL, event, demo filenames + sha256}` (sha256 free from HF LFS OIDs; sha256 + HLTV id are the match key — filename is proven non-unique)
- **Parser record** — demoparser2 version (≥0.41.3 pinned), awpy version, exact `player_props` list
- **Builder record** — `schema_version, builder_commit, builder_dirty, pipeline_commit, downsample, baked_at` (tickrate derived, per the line-918 fix)
- **Patch lineage** — ordered list of `{script, script_sha, transforms, blob sha256 pre/post, date}`
- **Split record** — seed=0, val_frac=0.15, unit=match, dedup key, split_manifest hash
- **Exclusion record** — D1/D2 mask version, clean_blob version, per-map round counts
- **Dist-edges version** + fit-script hash (after runbook [2])
- **Archive-substrate schema version** — parquet tier props-v1 = X/Y/Z/health/armor/helmet/defuser/inventory/equip/balance/yaw/pitch (no grenade, no velocity); props-v2, if adopted, recorded here to prevent a silent heterogeneous substrate
- **Pointer** to the datasheet defect registry

---

## 5. Exam rows to pre-register now (manifest-only cost, all non-gating, frozen Knob-7 probe protocol — no protocol fork)

1. **Leave-one-team-out outcome probe** (top-8 teams) — defends C1 against "probe memorizes team priors"; the strongest no-retrain memorization instrument.
2. **Team-ID linear probe on frozen latents** (chance = 1/23, permutation control) — memorization meter.
3. **Probe-correctness vs team-train-frequency regression** — with team win-rate as covariate (raw slope is strength-confounded).
4. **Leave-one-event-out probe row** (11 HF events) — better-powered and more exogenous than any team slice; near-free from the same manifest.
5. **Era-composition disclosure table** — disclosure only; the "newest-era val row" is dead as a trigger (2 val matches cannot CI-separate anything).
6. **Option F — future-event eval pack**: pre-register NOW; 5–10 matches from events strictly after the corpus freeze (Cologne playoffs land on HLTV within weeks), scraped with the existing pipeline, baked v2.1, never trains, non-gating. A true temporal holdout with 3–6× the power of any val slice, ~$0. **This is the temporal-axis carrier**, replacing the confounded temporal-cutoff design as the first line.

Plus: **cannot-say scope lines** in the datasheet — the probe-level exam does not test pretraining-level team leakage (the trunk saw every team); the paper must not imply team-invariance regardless of exam outcomes. Disclose top-team concentration (Vitality 20/92, NaVi 19/92; val = 14 matches, not 12; G2 = 9).

---

## 6. Explicitly rejected

| Rejected | One-line reason |
|---|---|
| Mixed-tier **training** corpus (FACEIT) as an increment | Corrupts the behavioral prior every downstream stage samples from, and silently invalidates Knobs 2/4/7 — it is a different paper (survives only as the T3 pivot). |
| Scrims / community servers | No access channel exists for a solo researcher; worst-in-class labels; killed, not deferred. |
| Synthetic bot rollouts as training data | Anti-thesis: prediction of bots teaches bot-ness; physics content already densely supervised. |
| 16Hz-now full re-bake | Reopens Knobs 1/5/6 with zero evidence in hand, doubles blobs past the 22 GB workflow, delays R1 1–2 weeks — feasible ≠ justified. |
| Adaptive/event-triggered corpus cadence | Circular (dataset depends on the model) and destroys Knob 1's fixed-dt semantics; surprise clock stays an eval instrument. |
| Event-tokens as model inputs | Contradicts first-principles Step 2 — presupposes exactly the abstractions C1 needs to emerge, and leaks outcome-timed probe targets. |
| Folding v5 channels into the current patch (pre-R1) | Breaks [1]'s dim-preservation, unlocks Knob 2 with no gate fired, delays the existential C1 test. |
| Pre-emptive scrape+bake+**merge** (b-full) | Silently changes pre-registered counts and Knob 2's bit-identical clause before evidence; bakes with an uncertified builder. |
| By-name backfill of "53 missing" local demos | The files aren't missing (81/81 on HF), and by-name upload would overwrite 5 load-bearing same-name-different-content demos. |
| PureSkill csds as the lower-tier set | Parsed-only, raw .dem deleted — tier shift would be confounded with feature-pipeline provenance. |
| ESTA / CS:GO-era datasets | Wrong game, parser, and schema. |
| LAN-vs-online exam axis | Not constructible — the corpus is 100% LAN/studio-LAN. |
| Newest-era val row as a temporal trigger | Dead trigger: 2 val matches; superseded by the future-event pack (Option F). |
| Door/breakable entity states | No awpy support; nuke-niche tactical surface. |
| Bundling anubis/MAP_VOCAB into the scale re-bake | Confounds the pre-registered data-scaling claim with a map-set change; ships as its own defect-fix event. |

---

**The organizing asymmetry** (corrected form): HLTV is a permanent, backfillable archive — every expensive pro-data decision can wait behind measured triggers. FACEIT is a rolling 30-day window behind a discretionary application — the only time-sensitive corpus action is the free application, now with a written denial fallback. And the one genuinely irreversible asset was never the demos: it is the merged training blobs on a single WSL disk, which runbook [1] is about to mutate — snapshot first.