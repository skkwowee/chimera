# Chimera Corpus Audit — Does the Corpus Capture What Its Consumers Need?

**Scope:** the live corpus `train/val_{v2m,v3m}_p1.pt` as loaded by `scripts/train_world_model.py:393-394` via `scripts/_corpus.py:23-55`. All claims code-cited; findings below survived adversarial refutation unless marked otherwise. Date: 2026-07-20.

---

## 1. How the corpus is built

Six stages transform HLTV `.dem` files into the live blobs:

**Stage 0 — scrape** (`chimera-demo-pipeline/pipeline/hltv.py:29-60`, `download.py`): star-tier HLTV matches → `.rar` → `.dem` on HF `skkwowee/chimera-cs2/demos/` (224 demos, 109.7 GB, 92/92 matches re-bakeable). Zero local storage. GOTV recording is already lossy upstream (source of D7).

**Stage 1 — parse** (`scripts/parse_demos.py:41-91`): awpy parses exactly **12 player props** (position, hp/armor/helmet/defuser, inventory, equip/balance, yaw/pitch) → per-demo `_ticks.parquet` (full 64 Hz) + `kills/bomb/damages/rounds/header` JSONs. This is the **largest information loss in the pipeline**: everything not in the 12 props is discarded — and awpy computes much of it anyway before we throw it away (§2).

**Stage 2 — bake** (`scripts/build_tick_sequences.py`, schema v2.1): round-scoped, 8 Hz via `ticks_all[::8]` over *ticks present* (`:512,533` — the D7 mechanism: GOTV gaps silently compress the grid). Players → steamid-sorted slots per round (identity discarded, `:287-303`). Output: per-round `(T, 597)` = 10×56-d player blocks + 37-d global, plus event labels/times.

**Stage 3 — merge** (`scripts/merge_hf_tick_sequences.py`): pools per-match blobs, dedups by 4-tuple, deterministic **match-level split** (seed 0, 15% val, map-stratified). **Strips `event_labels`/`event_times`/`summaries`** (`:126,187,314-317`) — the trained-on corpus has no event targets.

**Stage 4 — v3 derive** (`scripts/build_v3_features.py`): +9 perception dims/player (LOS/FOV/exposure) from the v2 tensors via awpy BVH raycast — **static geometry only**, xy-only FOV, 91.2% kill-agreement.

**Stage 5 — [1] patch** (`scripts/patch_corpus.py` → `*_p1.pt`): in-tensor recompute of D3 bomb bits (bit-exact 97,197/97,197 vs fresh re-bakes, 54/54 sites incl. 22/22 nuke), D4 clock (frame-anchored — sub-frame quantization ≤0.109 s; ≤2.05 s drift on D7 rounds), D6 dim7 (max err 1.79e-7). Validated in `outputs/patch_validation/report.md`; lineage in `corpus_manifest.json`.

**Stage 6 — load** (`scripts/_corpus.py:45-76`): D1 (anubis) + D2 (train) rounds excluded at load time, reversibly → 3876/705 clean rounds; canonical 5-map training 3573/641 + overpass OOD.

### Discarded-information inventory

| Stage | Discarded | Recoverable? |
|---|---|---|
| GOTV | ~19% of rounds missing >1% of ticks (D7, mean ~3%, max 0.66) | No — lost before the .dem exists |
| Parse (1) | velocity, duck/scope/walk flags, flash-blind, **weapon_fire**, **player_sound (20k rows/demo w/ radius)**, **grenade projectiles + smoke/molly lifecycles**, spotted masks, ammo, is_defusing, buy types | Re-parse (all 224 demos on HF) |
| Parse (1) | bomb-explode events *(builder comment "no explode rows" is FALSE — 329 detonate rows in 76/81 local matches)* | Present locally; comment needs fixing |
| Bake (2) | 7/8 of ticks (gap-compressing); **player identity**; exact weapon models; **bomb Z**; **dropped-bomb position** (reads as `none`/origin); kill detail; `damages.json` entirely; warmup/pause ticks | Re-bake from parse archive |
| Merge (3) | **event_labels/event_times/summaries** — no event targets downstream | Per-match HF blobs |
| v3 (4) | smoke/fire occlusion (LOS asserts clear sight through opaque smoke); vertical aim | No — utility dropped at Stage 1 |
| Patch (5) | tick-time clock anchoring (frame proxy); plant-event Z (player-z proxy, 22/22 correct) | Per-demo re-bake (armed) |
| Load (6) | anubis (216 rds) + train (16); overpass from train | Yes — load-time mask |

---

## 2. What it is built with

| Tool | Role | Pin / gotcha |
|---|---|---|
| curl_cffi (chrome124 TLS) | HLTV scrape | Cloudflare fingerprints pure-Python TLS (`hltv.py:29-39`) |
| unrar (shell) | demo extraction | 1 MB streamed chunks |
| demoparser2 **≥ 0.41.3** | tick/event parse (via awpy) | Preflight hard-fail below floor (`process.py:67-69`); **silently drops nonexistent prop names** — no KeyError |
| awpy ≥ 2.0 | Demo/parse, `VisibilityChecker` BVH, `MAP_DATA` | `.bomb` df lacks begin-defuse; local parses DO emit detonate rows |
| polars/parquet | 64 Hz tick storage | 18 cols incl. free extras (below) |
| torch (CPU) | tensor bake/merge/patch | mmap load, 700 MB guardrail respected |

### Free-at-parse-time signals we currently do NOT take

| Signal | Status | Evidence |
|---|---|---|
| grenade projectiles, smoke/inferno lifecycles, **shots (weapon_fire)**, footsteps | **Computed by awpy on every parse, then discarded** — `demo.py:265,287-310,501-522` vs `parse_demos.py:70-82` writing only 4 event JSONs | ~6 write lines in `parse_demos.py` to keep |
| `place` (named callout region/player/tick) | **Already stored** in all 81 local parquets; consumed nowhere (only grep hit: stem string `build_tick_sequences.py:726`) | Free semantic vocabulary for L3/NLA |
| `spotted` / `approximate_spotted_by` (engine's dynamic-occluder-aware visibility mask) | In demoparser2 catalog; planned only as one-off LOS certification input, never a corpus signal | One prop-list line |
| `flash_duration`, `is_defusing`, `is_scoped`, `active_weapon_*`, ammo, `velo_modifier`, `duck_amount`, velocity_X/Y/Z | All verified emittable by pinned parser; none requested | One line each at v5 re-parse |
| `player_sound` (audibility radius), `weapon_fire` events | Parse OK on corpus demo (20,079 / 3,591 rows); mentioned in **zero** docs | Event-table tier |
| usercmd/input props (buttons, moves) | **Unobtainable** — GOTV carries server snapshots, not client input (needs 5-min empirical confirm + datasheet line) | Close the class |

---

## 3. THE VERDICT: does it capture the necessary features?

**Overall: the corpus is a validated, honest substrate for the locked R1 next-state scope, and INSUFFICIENT as-is for two of its five consumers.** The core blocks (position/aim/hp/econ/alive/bomb-post-patch) are literature-validated and patch-verified. The failures are concentrated where the project's own thesis lives: perception-gated decisions and adjudicable language.

| Consumer | Rating | One-line reason |
|---|---|---|
| **1. Dist head** (500 ms displacement) | **SUFFICIENT-WITH-CAVEAT** | Kinematic core is sound and velocity-omission is literature-endorsed (MLMove); caveats: weapon-out speed class invisible (scoped AWP ≡ knife-out), D7 dt-inflation corrupts labels on ~19% of rounds (~0.6% of frames), utility-forced motion is causeless. |
| **2. Value head / probes** | **SUFFICIENT-WITH-CAVEAT pre-plant; INSUFFICIENT post-plant (~16% of frames)** | Econ/HP/alive are exactly the literature's top win-prob features — but defuse-in-progress is unrepresentable ('CT sticking with 8 s left' ≡ 'nobody on bomb'), the bomb_defused event source undercounts 6.6%, and a dropped bomb reads as `none`@origin in ~49% of rounds ≥5 s; an irreducible AUC floor sits exactly at the ~0.02 keystone margin. |
| **3. Surprise clock** (event detection / frame selection) | **SUFFICIENT-WITH-CAVEAT** | AUC 0.698 stands as measured, but its "tactical event detector" interpretation is over-claimed: invisible utility makes NLL spikes ambiguous between tactic and occluder exactly in executes/retakes, and D7 dropout adds transport-artifact false spikes in the 19% of affected rounds — and this bias propagates into bridge frame selection by design (`bridge-design.md:50-54`). |
| **4. Bridge text** (verbalization) | **SUFFICIENT for Phase-2a** (templates use only corpus-supported facts, `gen_bridge_sft.py:172-175`); **INSUFFICIENT for Phase-2b caster-register narration** | The unspeakable set — flashed, smoked-off, molly-forced, defusing, reloading, crouch-clearing, any player name — covers the dominant register of real tactical talk (~60-72% tactical density measured in the commentary pilot). |
| **5. GRPO checker** (grounded reward) | **INSUFFICIENT as materialized** | The merge stripped event ground truth, 13/14 val matches have no kills side-file, and `event_boundary_check.py:84-85` silently substitutes `[]` — kill-imminent frames scored as clean negatives; the recorded AUC 0.519 "falsification" is artifact-suspect, and the pre-registered amendment-F gates (AUC ≥ 0.75, ICC ≥ 0.2) **cannot be honestly run** on val today. Fixable without re-bake (§5). |

---

## 4. Gap register

**Cost classes:** blob-patch (from data on hand) < events-reparse (kilobyte JSONs from HF demos, no tensor re-bake) < re-bake (rides the armed v5 superset bake, repriced ~1-1.5 h post cs2-nav swap).

### 4a. Already planned (v5 stub / armed triggers) — audit adds quantification, not channels

| Gap | Decision-causality | Consumer hit | Cost | Planned where |
|---|---|---|---|---|
| Active utility in effect (smoke/molly volumes, flash_duration) | ≥1 smoke active for the **majority of live seconds** (7.0 smokes + 6.3 flashes + 4.9 mollies/round; 8.63% of kills thru-smoke; 20.78% of damage events from utility); v3 LOS **anti-correct** through smoke | Dist head, clock, value, bridge, v2-vs-v3 deconfound | re-bake | weak-link #12, v5 stub §3a/b, trigger T2 (`wall-detection-plan.md:78-93`, `corpus-strategy.md:35,53`) — **new here: density/coverage numbers → margin anchors** |
| Active-weapon + reload | 18.75% of kills use a non-primary weapon — but only **1.40% are truly invisible** (measured; margin for the per-channel gate) | Dist head (speed class), bridge | re-bake | v5 stub — **but the named `is_reloading` prop does not exist** (§5) |
| Crouch/duck | movement-speed + clearing reads | Dist head, bridge | re-bake | `wall-detection-plan.md:81` |
| bomb_z | nuke site ambiguity | Value | re-bake | v5 stub |
| D7 tick dropout | frame spacing >125 ms on 19.1% of rounds (921 flagged, fully disclosed in `corpus_manifest.json`) | Dist labels, clock | re-bake (per-demo) | datasheet D7, armed options — **new: clause (c), clock false-surprise consumer** |
| Velocity | literature says leave it OUT (MLMove negative ablation) | — | none | Knob 3 locked; confirmed sufficient |

### 4b. NEW findings (in no plan, stub, or TODO — the delta)

| Gap | Decision-causality | Consumer hit | Cost | Status |
|---|---|---|---|---|
| **Val event ground truth hole + silent `[]` fallback** | AUC 0.519 falsification is artifact-suspect; amendment-F gates unrunnable on 93% of val | Clock eval, GRPO checker | **events-reparse** (~13 val matches; HF `parsed/` is empty — archiving started 2026-07-18, prospective only) | NEW, **do now** |
| **Defuse/plant-in-progress** (`is_defusing`, `bomb_begindefuse/beginplant/abort`) | P(CT win) is near-step-function on defuse-start; ~16% of frames post-plant; bomb.json also undercounts even completed defuses 6.6% (199 vs 213) | Value (keystone margin), bridge, event labels | re-bake | NEW — must enter v5 stub |
| **Sound** (`player_sound` w/ radius, `weapon_fire`) | pros act on heard audio in a large share of rotate/retake decisions; corpus is omniscient-in, sound-blind; also corrects Fact 0 ("deciding variables not in observable state" — partially false) | Bridge/GRPO info-asymmetry claim | re-bake (event-table tier) | NEW — zero doc mentions |
| **Weapon-state prop route defect** | v5 stub names a nonexistent prop; parser **silently drops** unknown props → a v5 bake would ship channel-less undetected; `is_scoped` unplanned | v5 bake integrity | doc fix now | NEW |
| **Dropped-bomb location** | bomb loose in ~49% of rounds ≥5 s (~11.5 s/round of live play), encoded as `none`@origin — conflates "unknown" with map origin | Value, bridge | **blob-patch** (carrier position at has_c4 falling edge) + exact coords at v5 | NEW — D8-class datasheet row owed |
| **`damages.json` unconsumed** | checker-side answer key for damage claims; widens amendment-F claim vocabulary + gives clock an "under fire" annotation | GRPO checker, clock | side-file join (81 local); full-corpus = re-bake | NEW |
| **Clock→bridge selection-bias chaining** | utility-blind + dropout-artifact spikes decide **which latents the bridge ever sees**; NLA gate then penalizes the wrong component | Bridge, NLA gate | doc/eval design | NEW — unwritten anywhere |
| **Player identity absent from metas** | names unspeakable; slot-only language | Bridge | **blob-patch** (steamids in local parquets) | NEW |
| **`place` column unused** | free region tokens for L3/NLA grounding + region-stratified evals | Bridge, evals | blob-patch sidecar (81 local) | NEW |
| **awpy free tables discarded every parse** (grenades/smokes/infernos/shots/footsteps) | every future parse re-loses the exact data v5 needs | v5 economics | **~6 lines in `parse_demos.py`, land now** | NEW |
| **`spotted` masks unplanned as signal** | engine's own dynamic-occluder-aware visibility, free, vs CPU-days of smoke-blind raycasts | v3 successor, certification | one prop line at v5 | NEW |
| **`weapon_fire` absent from props-v2 archive spec** | suppression/misses unrecoverable from intermediates forever if omitted | Engagement modeling | one spec line | NEW |
| **v4 velocity aliasing linkage** | if aliasing audit fails, finite-difference v4 inherits it; no doc links audit → parser-velocity fallback | v4 gate | one conditional line | NEW |
| Detonate-row comment false + label dual-sourcing (16 post-round-end injections) | non-material to numbers; misprices future event work | hygiene | blob-patch/doc | NEW, minor |
| usercmd props = unobtainable from GOTV | prevents future mispricing | — | one datasheet line | NEW, minor |

**Confirmations (no action):** econ/HP/alive are the literature's top-value features and are correctly encoded; v3 LOS dims are literature-precedented (MLMove d_i,t, AlphaStar fog) — but precedent does NOT settle v2m-vs-v3m (Knob 2's experiment does); [1] patch verified end-to-end; D7 disclosure integrity holds; every current v5-stub channel is verified emittable by the pinned toolchain — no vaporware.

---

## 5. Recommended changes

### A. v5 stub edits — REQUIRED before it freezes (the stub's own anti-fishing rule makes omissions permanent)

1. **Add channels:** `is_defusing` + begin/abort-defuse + begin-plant events; `is_scoped`; per-player is-firing / time-since-shot (from `weapon_fire`); dropped-bomb state/position (explicit add-or-waive row); per-frame tick-index / true-dt sidecar (D7 fix rides the only planned re-bake); decision row on in-flight grenade projectiles (OpenAI Five precedent); the free combat-state cluster (`active_weapon_ammo`, `total_ammo_left`, `velo_modifier`, `aim_punch_angle`, `flash_max_alpha`).
2. **Fix the reload route:** pin to a derivation (`active_weapon_ammo` increase or `next_attack_time` window) — `is_reloading`/`in_reload` **do not exist** in demoparser2 0.41.3, and add a **prop-existence smoke test** (parse 1 tick, assert columns non-empty) to the bake preflight — the parser's silent-drop defeats name-only pre-registration.
3. **Persisted-intermediates list:** add `player_sound`, `weapon_fire`, grenade/smoke/inferno lifecycle tables, footsteps, `spotted`/`approximate_spotted_by`, and `damages.json` — or the re-parse cost class is not actually retired.
4. **Pre-register margins from this audit:** active-weapon 18.75%/1.40% split; utility density (7.0 smokes/round, 3-5 simultaneous, 20.78% of damage events); extend the masked-equivalence gate to clock behavior (selection composition inside vs outside utility windows).

### B. Do NOT wait for v5

1. **Event ground truth for val** (blocks amendment-F gates): events-only re-parse of the 13 HF-era val matches' kill JSONs (kilobyte-scale, keyed sha256+HLTV-id to defeat the 5 stem collisions); change `event_boundary_check.py:84-85` from `else []` to **hard fail**; recompute the AUC 0.519 verdict — it is currently artifact-suspect and feeds W04.
2. **~6 lines in `parse_demos.py`** to persist the awpy tables it already computes — every parse until this lands re-loses the data.
3. **p2 blob patch (pre-retrain, ~free now, paper-number re-run later):** dropped-bomb bomb_x/y from carrier position at has_c4 falling edge; steamid/name enrichment of metas; `place` sidecar for the 81 local stems.
4. **Datasheet amendments:** D3 wording fix (`:88` — the 'none' bit was live pre-patch); D7 clause (c) clock/selection consumer; new rows: dropped-bomb misrepresentation, defuse-event 6.6% undercount, weapon-state gap, detonate-comment correction, usercmd-unobtainable class; quantified utility limitation (~1/12 kills, ~1/5 damage events mechanism-blind).
5. **Doc paragraphs:** decisions-ledger caveat on clock AUC 0.698 (utility + transport confounds); bridge-design §1 selection-bias paragraph; NLA-gate control for utility-window frames (low fidelity there is a corpus observability defect, not bridge unfaithfulness); paper scoping of "tactical language" (utility-effect register is corpus-unspeakable).
6. **Builder hygiene:** fix false comments at `build_tick_sequences.py:464-467,636-637`; dedupe detonate/synthesis before any future re-bake; 10-min detonate-provenance audit across the 81 local bundles before any event-label sentence enters the paper.

### C. Explicitly do NOT

- Fold any v5 channel into the current corpus pre-R1 (`corpus-strategy.md:106` — breaks [1]'s dim-preservation).
- Reopen Knobs 1-7: every schema extension routes through the pre-registered masked-equivalence gate; velocity stays out (Knob 3, literature-confirmed); v2m-vs-v3m stays experiment-settled.
- Fire the v5 bake early — triggers stay armed as pre-registered; this audit changes what the bake carries, not when it runs.