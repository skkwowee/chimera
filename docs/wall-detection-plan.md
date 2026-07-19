# Wall/Visibility Detection Plan

**Status**: adopted 2026-07-19 (synthesis of 5-dimension review + judge ruling). Owner directive: accuracy is paramount — training must be representative of what a player would actually see.
**Master gate**: the v2-vs-v3 ablation supersedes everything byte-changing here. If v2 wins, all LOS-accuracy work collapses (the physics mesh remains correct for GeoGate; sunk cost = docs + a <1-day checker swap that speeds any future raycast need). No item below violates this.
**Prior decisions respected**: Knob 2 (awpy + triangulated validation) stays locked; perception dims change bytes only at a re-bake; the v5 superset bake (corpus-strategy item 9) is the sole landing spot for byte/schema changes; nothing lands mid-ablation (validity clause: arms bit-identical except schema).

---

## 1. What we now KNOW (measured, this review, $0 / CPU-only)

### Mesh fidelity (brute-force numpy raycaster over raw .tri, validated 5/5 vs awpy's own test_visibility.py before any claim)

| Error class | Sign | Where measured | Corpus exposure |
|---|---|---|---|
| Foliage | **false-VISIBLE** | Ancient top-mid leaf wall (x[-643,-428], y[524,640]): eye-height (z=224) rays pass clean, 0 crossings over 350u; canopy air volume holds 26 tris (one pole prop). Foliage has no physics collision at all. | Continuous, every tick of mid-control on 465/66 train/val rounds (~10% of 5-map corpus) |
| Fence class | **false-HIDDEN** | Overpass B/water: 850–900u-long, 76–164u-tall zero-thickness sheets (paired crossings 1.0u apart) block rays z=170–280, clear z=320; ~90% confident = see-through chain-link. | OOD holdout map disproportionately (fence-dense) → contaminates Knob 4 OOD protocol specifically |
| Doors | **false-HIDDEN (baked closed)** | dust2 mid-doors blocked by 8.9u leaf at y=1609.9–1618.8; 8 door-sized thin slabs on nuke. Doors open constantly in real play. | nuke 540/145 + dust2 978/163 rounds (~30% of train), door sightlines only |

- **Geometric curation is dead**: 136/150 random overpass nav-pair sightlines are blocked; 72% blocked ONLY by ≤4u-thin geometry. The world mesh is mostly thin sheets — thickness heuristics cannot identify see-through surfaces. **Material labels are mandatory** for any systematic fix.
- **Fix enablers verified**: source .vphys carries per-triangle `m_Materials` → `m_surfacePropertyHashes` (VRF PhysAggregateData.cs); awpy's VphysParser discards them. All six corpus .vpks are locally on disk (Steam install) — zero acquisition cost.
- **Community options empty**: VisCheckCS2 = same .vphys paradigm; vvis = over-permissive PVS render culling; no material-aware community visibility mesh exists.
- **Consumer split**: GeoGate (movement feasibility, foot+28u) is CORRECTLY served by the physics mesh — fences block movement, leaves don't. Any see-through-aware mesh must be a separate LOS-only vis-variant artifact, never a GeoGate replacement.
- **Kill-agreement is structurally blind to all of the above**: kills can't happen through surfaces players can't see through; the bias lives in continuous LOS/time_since_los dims, not kill events.

### Ray model (81 matches, 11,462 kills from data/processed/demos/*_kills.json)

- **Distance gate: CLOSED.** 0/11,462 kills exceed 3500u 2D (max 2943u, dust2; p99.9 = 2423u). No change earns anything.
- **Wallbang conflation = 3.95%** of kills (`penetrated>0`) — ~4pp of the 8.8pp kill-agreement gap is kills where LOS=false is physically correct. Pure ray-model residual (shoulder-peek partials + eye height) ≈ 4–5pp.
- **thrusmoke = 8.68%, attackerblind = 1.31%** of kills — the dynamic-occluder blindspot is ~2x the entire ray-model residual.
- **Pitch**: 12.45% of kills have >10° vertical attacker-victim angle (nuke: 22.7% >10°, 10.2% >20°; |attacker_pitch| p90 = 13.3°). Yaw-only cone corrupts n_fov/n_aim/min_aim_error exactly on nuke. sin/cos_pitch already sit in v2 tensors (per-player dims 5,6) — fix needs **zero extra raycasts**.
- **Crouch is invisible today**: origin-z does not move on duck; no duck prop in PLAYER_PROPS or 64-tick parquets. demoparser2 0.41.3 exposes `duck_amount` (continuous), `ducked`, `ducking` — eye = 64 − 18·duck_amount is parseable at the v5 re-parse.
- **Workload shape**: is_exposed rate only 2.2% of alive player-frames; 17.0 alive cross-pairs/frame; ~66M rays per full bake. Per-pair LOS matrices are NOT stored in v3 blobs — FOV/LOS fixes require re-raycasting, i.e. ride a re-bake, not a blob patch.

### Dynamic occlusion (smokes/flashes/mollies)

- Current LOS has **zero smoke awareness**; the input layer asserts full mutual exposure through an opaque smoke wall — a categorical inversion concentrated in executes/retakes/post-plants, exactly the frames the world model exists to understand. Already registered as datasheet known-optimism and first-principles weak-link #12 (with its trigger diagnostic pre-defined).
- **CS2 removed one-way smokes** — all players see the same volume — so a symmetric geometric occluder is now correct-in-principle (it never was in CS:GO). Sphere (r≈144u) + TTL from detonate/expire events is the industry-standard approximation; no offline tool does voxel-accurate occlusion. Mollies partially obscure (not LOS blockers); flash is viewer-state (`flash_duration` prop), not geometry.
- Effective-LOS costs ~nothing: ray-vs-sphere is closed-form and only runs on pairs that already passed the static BVH ray (2.2% of gated pairs).
- **Slot-saturation risk**: pro executes throw 3–5 smokes simultaneously; a ~20-dim slot layout can saturate during the highest-tactical-density windows. Max-simultaneous-actives must be measured before the layout is locked.

### Performance (the accuracy enabler)

- Root cause of days-long bakes: awpy 2.0.2's VisibilityChecker is 100% pure Python (one-tri-per-leaf BVH, no batching); ~2–4ms/ray on real maps × ~66M rays ≈ 73h — the observed "days". The ~1.5GB/map is Python object overhead (~550B/tri).
- **cs2-nav 0.3.18 (Rust, drop-in API mirror; the checker awpy's own PR #382 adopts)**: 5.8µs/ray (~400–700x), **100.0000% agreement** with awpy-baked LOS in val_v3m_p1.pt on 122,627 rays / 62,817 alive player-frames / 3 maps — ray-for-ray identical on this sample. Memory is NOT better (~450–500B/tri; inferno ~1.2–1.4GB) so `--workers 1` stands; it just no longer costs anything (66M rays ≈ 6.4 min of ray time; full v3-class bake ≈ 1–1.5h end-to-end vs 3+ days).
- Consequence: corpus-strategy's v5 trigger cost line "3–6 CPU-days" is mispriced by ~50–100x post-swap, and multi-ray/crouch/smoke upgrades become affordable (5–10x rays still < 1 day).

### Certification state

- The 91.2% kill-agreement number has **no committed script** — it exists only as prose in build_v3_features.py:18 and retrain-recipe.md:71. Unreproducible, no per-map breakdown, conflates wallbangs (3.95%) and smoke-blind kills (8.68%) with geometry error.
- The V1a spotted-flag cross-check (retrain-recipe.md:78) was specced and **never run**. Spotted is radar bookkeeping with hysteresis → per-frame equality is the wrong score; onset-alignment is the right protocol (§6).
- infra-plan.md:47 already REQUIRES a "≥99.9% agreement cert" for the armed embreex swap — the battery below is the missing instrument that trigger presupposes.
- The radar viewer cannot adjudicate 3D eye-lines; a screenshot gold set is not actually possible as specced.

---

## 2. Do-now list (all $0 cloud, no GPU, zero corpus bytes, does not block runbook [1]–[6]; total ~2–3 days CPU-local)

One instrument, one owner: all V1a variants (map-stratified, smoke-window-stratified, kill-class-filtered) are ONE battery run owned by `certify_los.py`. All byte-changing upgrades share ONE landing spot: the v5 stub (§3).

| # | Action | Cost | Done-check |
|---|---|---|---|
| 1 | Commit **docs/los-certification.md** BEFORE any battery compute: metric definitions (§6), owner-locked thresholds, stratification axes {map} × {active-smoke window} × {kill-class filter}, and the pre-registered per-map signatures (ancient LOS-yes/spotted-no excess = foliage; overpass spotted-yes/LOS-no excess = fence; nuke door contribution; smoke-window degradation). | ~1–2h | Doc committed with thresholds, dated, before certify_los.py first runs |
| 2 | Write measured closures into **docs/datasheet.md + decisions ledger**: distance gate CLOSED (0/11,462 >3500u, max 2943u); wallbang 3.95%; thrusmoke 8.68%; attackerblind 1.31%; overpass 136/150 blocked with 72% by ≤4u geometry (kills geometric fence curation). | ~1h | Ledger rows present with source scripts named |
| 3 | **Slot-count measurement**: max simultaneous active smokes/mollies from lifecycle events across the 81 local matches (event-level, no tick parsing, no BVH). Gates the v5 slot layout — the stub cannot lock without it. | ~1 CPU-hour | Distribution + max reported; slot count chosen with overflow-dim rule |
| 4 | **Battery v1**: scripts/certify_los.py against val_v3m_p1.pt baked dims (mmap, <700MB) + kill/spotted extracts; tests/test_los_certification.py (synthetic-fixture scorers in CI, full battery local-only); first dated row in datasheet §6. One run produces: reproducible per-map pure-geometry kill-agreement, the map-stratified fence/foliage signature (mesh trigger input), and the smoke-window gap (weak-link #12 trigger input). | ~0.5–1 day | report.json + datasheet row; thresholds evaluated against §6 pre-registered numbers |
| 5 | **cs2-nav parity certificate** on all 6 corpus maps (~200k rays/map; the ONLY BVH-loading job, one map at a time, --workers 1). On pass: one-line import swap in build_v3_features.py `_get_vc`, pin cs2-nav==0.3.18, stamp checker version in bake lineage, re-run kill-agreement via the battery (now minutes), correct corpus-strategy.md's v5 CPU-days line. On any parity miss: **no swap**, arm embreex. | ~0.5–1 day | Bit-identical on all 6 maps, logged as datasheet §6 row; swap committed only after |
| 6 | Write the **consolidated v5 pre-registration stub** (corpus-strategy item 9) — contents in §3. Blocked on #3. | ~2–3h | Stub committed; every upgrade independently trigger-gated and per-channel droppable |
| 7 | Code the **pitch-aware cone** in build_v3_features.py behind a flag, default OFF until first post-ablation bake (uses v2 dims 5,6; zero extra raycasts; same 9 dims, truer values). Never lands mid-ablation. | hours | Flag exists, defaults OFF, covered by a synthetic test |

Fold-in ruling: the kill-agreement re-report excluding penetrated/thrusmoke/attackerblind is NOT a separate task — it is the battery's kill-class filter (item 4).

---

## 3. The v5-bake visibility upgrade package (pre-registered; each section independently trigger-gated, per-channel droppable; no upgrade gets its own bake event)

Ships in the v5 superset bake stub, in this form:

- **(a) Smoke/molly utility channels**: TTL-ordered slots sized from the do-now #3 measurement + an overflow-count dim; sphere r=144u, TTL from detonate/expire lifecycle events; `flash_duration` as viewer-state dim; mollies as area-channel, NOT LOS-blocker.
- **(b) Per-player effective-visibility dims**: `n_los_clear`, `exposed_clear`, `flash_remaining` — new dims computed as raw-LOS ∧ not-smoke-occluded (closed-form ray-sphere post-BVH, ~zero marginal bake cost). **Raw 9 dims keep their semantics unchanged** — smoke is never folded into raw LOS (preserves the pre-aim-through-smoke signal and the masked-equivalence gate).
- **(c) LOS mesh v2** = material-aware .tri regenerated from .vphys keeping per-tri `m_Materials` (patched VphysParser, ~100 lines) + surfaceprop-hash classification table + pre-registered foliage occluder AABB list (fixed exceptions list with provenance, committed before the bake; dense foliage occludes, sparse is visible — rule documented) + per-door open/closed table + diff-report proving only flagged sheets changed. **Shipped as a separate vis-variant artifact; GeoGate keeps the physics .tri.**
- **(d) Crouch**: `duck_amount`/`ducked`/`ducking` in the v5 parse prop list (one line NOW in the stub; eye = 64 − 18·duck_amount at bake; negligible compute; omission would cost a full re-parse cycle).
- **(e) Pitch-aware cone flag default ON** (first post-ablation bake).
- **(f) Known-optimism list**: sphere-vs-flood-fill boundaries (wall-clipping, vertical pour-through), bloom/dissipate edge fuzz ±1–2s, HE-clears and bullet holes unmodeled, TTL noise caveat.
- **(g) Re-baselined masked-equivalence protocol**: value fixes to the 9 dims (pitch, mesh v2, ANY-of-3) are permitted at a re-bake — "byte-stable" is scoped to semantics/schema, not values. Masked-v5 is compared against a v3-recipe reference re-baked WITH the value fixes on a sampled subset (minutes, post cs2-nav swap).

---

## 4. Armed triggers

| # | Trigger condition | Action | Cost when fired |
|---|---|---|---|
| T1 | v3 wins the v2-vs-v3 ablation **AND** battery shows the pre-registered map signature (ancient foliage / overpass fence / nuke doors) above margin | Build LOS mesh v2 (§3c): patched VphysParser, surfaceprop table, foliage occluder list, door table, diff-report | 2–4 CPU-days local, rides v5, $0 pod |
| T2 | v3 wins **AND** smoke-window battery gap fires weak-link #12 | Effective-LOS + smoke/molly/flash channels in v5 (§3a,b) | ~zero marginal on v5 bake |
| T3 | v3 wins **AND** 20-round pilot vs spotted/kill frames confirms +2–4pp agreement | Multi-point ANY-of-3 LOS (head/chest/pelvis) in v5. Pilot is an **accuracy** gate — the old 3x-cost objection is stale post-swap (3x of minutes) | ~3x ray time = minutes–hours |
| T4 | Battery shows any per-map pure-geometry agreement below threshold, OR disagreements cluster in material-suspect geometry, OR before promoting any non-bit-identical backend | Engine TraceRay anchor: CS2TraceRay/CounterStrikeSharp dedicated server, 1000 stratified rays from battery disagreement frames, MASK_VISIBLE vs MASK_SHOT, standing-only frames first | ~0.5–1 day; signature-breakage risk on game updates |
| T5 | cs2-nav fails the 6-map parity certificate or goes unmaintained | embreex port (Embree 4.4) with its own **stricter** watertightness parity run + full battery + T4 anchor before promotion | ~2–3 days; reward: 2–5x speed, ~5–10x less memory (unlocks --workers 4+) |
| T6 | A future bake with 5–10x rays exceeds ~1 day | --workers >1 (memory math: worst map ~1.3GB/worker under 22GB ceiling; map-sorted ordering makes workers collide — stay at 1 until then) | config only |
| — | Fraction visibility dims (b2) | **Stays unarmed** until v3 wins AND T3 shows the binary ceiling is limiting | — |

---

## 5. Rejected (one line each)

- **Distance-gate changes** — measured exactly zero: 0/11,462 kills beyond 3500u.
- **Fraction visibility dims (b2)** — unproven training value, full-Kx cost, schema-fishing risk; precondition (binary ceiling shown limiting) doesn't exist yet.
- **Full VRF render-mesh occlusion pipeline** — contradicts locked Knob 2; (b')+(c) captures both dominant error classes at ~10% of the cost.
- **Community visibility meshes** — confirmed empty: VisCheckCS2 = same .vphys paradigm, vvis = over-permissive PVS culling.
- **Geometric fence curation** — 72% of blockers are ≤4u-thin sheets; thickness cannot identify see-through surfaces, material labels are mandatory.
- **Effective-LOS replacing raw LOS (option b)** — breaks Knob 2's frozen semantics mid-ablation, destroys the pre-aim signal, bakes unfalsifiable approximation error into canonical dims.
- **Voxel flood-fill smoke occluder (option d)** — no oracle exists to certify the reproduction; unfalsifiable fidelity claim, days of work for boundary-only gains.
- **PVS voxel cell-pair precompute** — its premise (rays are expensive) is falsified at 5.8µs/ray; the only option on the table that would actually spend accuracy.
- **GPU raycast** — 66M rays is ~6 CPU-minutes in Rust; nothing left to accelerate.
- **Viewer-screenshot human gold set** — top-down radar cannot adjudicate 3D eye-lines; false confidence. (In-game demo-spectate kept only as fallback if T4 is blocked.)

---

## 6. Certification battery spec (pre-registered; thresholds locked by owner in docs/los-certification.md BEFORE first compute)

**Instrument**: scripts/certify_los.py — the single shared ruler for all dimensions' trigger questions. Consumes ground-truth agent extracts (spotted flags, kill events with `penetrated`/`thrusmoke`/`attackerblind`) + baked dims from val_v3m_p1.pt (mmap peek, <700MB). No BVH, no demo parsing in this lane.

**Metrics**
1. **Pure-geometry kill-agreement**: kill-moment LOS agreement excluding `penetrated>0`, `thrusmoke`, `attackerblind` kills — decomposes the 8.8pp residual into named error classes; reported per map and per excluded class.
2. **Spotted-onset alignment** (V1a, finally run): LOS∧FOV-dim rising edge within ±2 frames @8Hz (±250ms) of a spotted rising edge; falling edges excluded (hysteresis); scored against baked per-player dims 58/60 (n_los/n_fov onsets); report precision/recall per map plus onset-lag histograms (so hysteresis mismodeling is visible, not hidden in a scalar).

**Stratification axes**: {map} × {active-smoke window vs matched frames} × {kill-class filter}.

**Pre-registered expected signatures** (from §1 geometry; the battery converts direction into corpus-level rates):
- Ancient: worst LOS-yes/spotted-no excess (foliage false-visible).
- Overpass: worst spotted-yes/LOS-no excess (fence false-hidden).
- Nuke: door-sightline LOS-no/spotted-yes contribution.
- Smoke windows: agreement degradation vs matched no-smoke frames (weak-link #12 diagnostic feed).

**Proposed thresholds (owner locks final numbers before compute; mirroring corpus-strategy item 4)**: pure-geometry kill-agreement ≥95% per map; onset recall ≥85% and precision ≥80% per map vs LOS∧FOV dims. Signature margins for T1/T2 set in los-certification.md.

**Regression wiring**: tests/test_los_certification.py follows the test_corpus_invariants.py pattern — pure threshold-check functions over outputs/los_cert/report.json; pytest wrappers skip when blobs absent (ci.yml stays code-only, <5min); synthetic-fixture unit tests of the scorers run in CI. Every future LOS change (cs2-nav swap, embreex, v5 channels, eye-height/FOV/pitch) MUST re-run the battery and append a dated row to datasheet §6. The cs2-nav parity certificate is logged in the same ledger; bit-identical parity is a stricter standard than the battery thresholds, so the import swap may proceed on the 6-map certificate alone — any future non-bit-identical backend requires full battery + T4 anchor before promotion.

**Rulers doctrine (Knob 2 triangulation)**: kill-agreement and spotted-onset are two independent imperfect rulers; the engine TraceRay anchor (T4) is the only ruler that resolves the physics-vs-render material class, and it stays trigger-armed, not default.