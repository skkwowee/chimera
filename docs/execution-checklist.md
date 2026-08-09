# Chimera — Unified Execution Checklist (LIVING)

**Created 2026-07-26.** **THIS FILE IS THE LIVE EXECUTION ORDER.** The audit
reports — preflight-report.md (§5), deep-sweep-r2.md (§7), corpus-audit,
corpus-implementation-audit — are **frozen registers**; their checklists are
historical *inputs* to this file, already merged below (r2 §7 amendments merged
IN PLACE, not appended). Update status HERE (tick + date); never edit the
frozen registers to track status. Done-checks are verbatim from source — do not
weaken them. Numbers/thresholds/gate criteria are owned by the LOCKED recipe
docs + their dated errata; this file only sequences them.

Legend: `[x]` done (date) · `[ ]` open · **[OD-n]** = blocked on an owner
decision (see final section). Source refs: "pf" = preflight-report, "r2" =
deep-sweep-r2, "rb" = claude-progress runbook.

---

## Phase 0 — Ground truth & environment — ALL DONE 2026-07-26

- [x] 0.1 [B1] Recreate/repair both venvs; re-run `init.sh` → all health checks + HF-auth line print, exit 0. *(pf §5 0.1)* — done 2026-07-26
- [x] 0.2 [B8] `uv lock` + commit → `uv lock --check` passes; demoparser2 ≥0.41.3 in lock. *(pf §5 0.2; commit 249e3d5)* — done 2026-07-26
- [x] 0.3 `git push` chimera-demo-pipeline → no ahead; `ls-remote` = 3c55fe8. *(pf §5 0.3 = pf §4.1)* — done 2026-07-26
- [x] 0.4 [B5] claude-progress absorbs Jul-20/22 audits; **R1 corpus decided = _p2** (owner 2026-07-25: [1b] runs before [2]) → runbook names exact R1 blob filenames; date > 2026-07-22 present. *(pf §5 0.4; rb [1b])* — done 2026-07-26
- [ ] 0.5 [B4] Eval-default sweep commit (9 scripts → `*_p1`, no retired-ckpt defaults) → `grep -rn 'v3.pt\|v2m.pt\|train.pt\|best.pt' scripts/*.py` returns only _p1/archive hits. *(pf §5 0.5)* — **PARTIAL 2026-07-26** (earlier same-day "done" tick was premature — verifier repair pass): blob-default half COMPLETE (all 9 B4 scripts + inspect_features/vq_killtest now default to `*_p1`; see runbook [1] correction); retired-ckpt-default half OPEN — `--ckpt` defaults at outputs/world_model/h8 + outputs/wm_3map* survive in 10 scripts pending owner adjudication of whether world_model/h8 is retired-under-B4 or the intended v2m-baseline probe target, and of "make required vs point at R1 dirs" (R1 dirs don't exist yet). Verbatim grep therefore still returns non-_p1 `best.pt` hits. (Note [1b] re-flips blob defaults to _p2 in its own commit.)
- [x] 0.6 Doc corrections batch: [B9] knobs item 8 → `_p1`; [B10] twin line into runbook [6]; falsified ~65/92 fix (plan:167 + MEMORY.md); [B6] 64Hz column into [4] spec → grep confirms each. *(pf §5 0.6; pf §3 row 1)* — done 2026-07-26 *(honesty note, same day: the morning tick was premature — only [B6]-spec + MEMORY.md had landed; the verifier repair pass then actually executed [B9] knobs4-7:337/:517 → `_p1`, [B10] twin line into runbook [6], plan:167 correction, and extended runbook [4]'s DONE-CHECK per B6's exact fix text; all four greps now confirm)*
- [x] 0.7 Phase-0 addendum [R2-B4a]: archive `eval_scorer.py`, `build_pseudo_gold.py`, `data/eval/` (B12 extension — decoy checker instruments) → no live importers. *(r2 §7 [pre-6] last line; commit b421973)* — done 2026-07-26
- [x] 0.8 Security disposition, one sitting *(r2 §7 [security]; r2 §5)* — done 2026-07-26:
  - identity posture RESOLVED: stays as-is, no rewrite (owner decision, r2 §5.1 closed; commit 451619e)
  - scrape log purged from pipeline history (kept local); corpus-implementation-audit:94 register line amended
  - `.env` chmod 600
- [x] 0.9 Blob-guardrail reword in runbook: val-side only, one at a time, `torch.load(mmap=True)`; train blobs (3–10GB) forbidden. *(r2 §7 [guardrail hygiene])* — done 2026-07-26
- [x] 0.10 This round's hygiene passes: R2 doc amendments committed (E1–E8, A-R2.1–6, D8, runbook — commit 9c969cb); language-tightening pass r3 (commit 8ffe297); stale-facts pass (falsified figures incl. pf §4.9 HF size/gating). — done 2026-07-26

## Phase 1 — Pipeline Tier-A/B (chimera-demo-pipeline)

- [ ] 1.1 Execute registered Tier-A then Tier-B items per corpus-implementation-audit §5 (incl. CHANGE-NOW #14 path fix) → **42p/1s → 43p; skip removed**. *(pf §5 1.1)*
- [ ] 1.2 [B5b] Val events-reparse (corpus-audit §5.B.1) so kills side-files exist for 14/14 val matches → **`event_boundary_check.py` finds >0 events on every val match; silent-[] fallback replaced with hard error**. *(pf §5 1.2; gates [7] CHANGE F)* — **[OD-1]** scope (val-only vs all 92) must be recorded BEFORE this runs. *(r2 §3 "decision line needed NOW")*
- [ ] 1.3 corpus-strategy row 3 split-integrity audit + row 5 builder tickrate fix → **audit result recorded in datasheet §3; `grep tickrate_hz: 8` build script = 0 hits**. *(pf §5 1.3; pf §3 row 2)*

## Phase 1b — [1b] P2 blob patch (owner-decided 2026-07-25: BEFORE [2])

- [ ] 1b.1 Run p2 patch on _p1 blobs — full merged scope: (a) dropped-bomb bomb_x/y from carrier position at has_c4 falling edge; (b) steamid/name meta enrichment; (c) `place` sidecar for the 81 local stems; **(d) crop end-phase frames beyond round_end + 7s; (e) clamp bomb_age at explosion/defuse (cap 1.0 normalized)** — [2]'s quantile fit must not ingest the tail frames. Same validation pattern as [1] (sample diff vs fixed-builder re-bake); lineage restamp; flip defaults _p1 → _p2 in the same commit → **DONE-CHECK: _p2 blobs + manifest entry + validation report committed**. *(rb [1b] + r2 §7 [1b] merged; corpus-audit §5.B.3)*
  - [x] D8 datasheet entry (r12 tail + bomb_age overflow + r24 truncation, quantified) — done 2026-07-26 (commit 9c969cb)
  - [x] P2 writer + fixture certificate — `patch_corpus_p2.py` writes new files atomically, source-keys identity/place joins to reject stem collisions, and carries dedicated P2 invariants; full suite green (2026-08-09).
  - [x] `val_v2m_p2.pt` and `val_v3m_p2.pt` local artifacts + manifest lineage — both cover 770/770 rounds and pass post-write legacy/P2 invariants plus manifest/blob/script hash checks; each removes 7,294 excess tail frames across 46 rounds and clamps 10,507 bomb-age values. Drop inference checked against 222 source-keyed local events: precision 0.982, recall 0.987, median xy error 1.22 game units (`validate_p2_drop_inference.py`). **Partial only; not canonical.**
  - [ ] Remaining for 1b.1: generate both train blobs on the quiet machine; run the required fresh-builder local-era + HF-era diff; flip live defaults together; only then set `p2_status.complete/canonical` and tick 1b.1.

## Phase 2 — Runbook [2]: dist edges (QUIET MACHINE — loads 8.7GB train blob)

- [x] 2.1 [B2] Add 5-map filter (`de_ancient,de_dust2,de_inferno,de_mirage,de_nuke`) + `assert kept_rounds == 3573` to fit_dist_edges.py; add argparse so `--help` works. `--help`, filter/default, round-count, and OOD-holdout guards are test-covered. *(pf §5 2.1)* — done 2026-08-09
- [ ] 2.2 Run fit on the **_p2** train blob; commit new `DIST_EDGES_U` **and** [B3] `OPEN_RING_MAG_U` (replacing both 700.0 literals: train_world_model.py + gen_bridge_sft.py; stamp into ckpt meta) in one commit → **DONE: edges + open-ring mag committed, per-map quantiles in datasheet, no overpass row in fit set**. *(pf §5 2.2 + rb [1b] corpus supersession)*

## Phase 3 — Runbook [3]: trainer completion (code, no GPU)

- [x] 3.0 [pre-3] Knobs errata commit before the [3] edit: Knob 5d mask errata (mask = alive(t) ∧ alive(t+k) ∧ ¬freeze(t); edge-fit stationary% stays freeze-inclusive) + stale line anchors voided (match by content). *(r2 §7 [pre-3]; knobs4-7 §R2-ERRATA E1; commit 9c969cb)* — done 2026-07-26
- [x] 3.0b [R2-B1] evaluate() CUDA device fix (`won` indexed via CPU mask, train_world_model.py:357) → repro'd, fixed, suite green 23p/1s/1xf. *(r2 §7 [3] first add; commit b421973)* — done 2026-07-26
- [x] 3.1 Implement detach/SS per locked recipe **applying errata E1's mask, not the verbatim checklist formula**; + fixture-scoped `--no-clean` flag (loud print) so the 7a RE fixture can load old blobs → `test_no_value_leak` now passes, E1/dead-decode rails are covered, and a forced-p>0 sample-and-swap smoke passes; full suite 40p/1s + ruff green. *(pf §5 3.1 + r2 §7 [3] merged; rb [3])* — done 2026-08-09
- [x] 3.2 Write pinned canonical v2 + v3 commands (blobs, `--maps`, `--dist-head`, seed, `--ss-pmax`) into retrain-recipe.md → exact runnable commands and the SS-off delta are in-repo, explicitly gated on P2 train blobs + fitted distance constants. *(pf §5 3.2 = pf §4.2)* — done 2026-08-09
- [x] 3b NEW GATE: 30-step CUDA smoke on the 4090 incl. ≥1 evaluate() pass, BEFORE any pilot/pod run (CI is CPU-only, cannot see device bugs) → 30/30 steps completed on the local RTX 4090 with dist head, evals at steps 15/30, finite val loss (0.0200 → 0.0124), and only `best_ns.pt`/`last.pt` emitted. Smoke metrics are device validation, not quality evidence. *(r2 §7 [3b]; rb [3b])* — done 2026-08-09

## Phase 4 — Runbook [4]: coverage harness

- [ ] 4.1 Extend rollout_eval: (i) minADE-K (frozen K/temperature; doubles as GRPO group generator); (ii) CHANGE-B fair stochastic baseline (per-bucket damped-CV + fitted residual covariance, K=16, scored identically); (iii) trajectory-coherence metric (mode-switch rate, depth-10/20 vs matched real); (iv) **64Hz-truth-scored column** (corpus-strategy §2 row 13); (v) **joint-coherence interaction control** (inference-only: decode with cross-player attention masked or per-player marginal sampler; if gap ≈ 0, soften "10 coupled agents" wording BEFORE review) → **DONE: sampled-coverage smoke prints baseline AND 64Hz columns on the [1b] val blob (_p2)**. *(pf §5 4.1 + r2 §7 [4] merged; rb [4])* — **[OD-2]** MLMove learned-baseline column yes/no decided before build.

## Phase 5 — Runbook [5]: local smoke

- [ ] 5.1 Run the **pinned** canonical v2 command (from 3.2) with `--smoke` → **DONE: completes on CPU/4090, ckpt meta carries edges + open_ring_mag_u + correct blob shas**. *(pf §5 5.1)*

## Phase pre-6 — Pre-registration sitting (MUST precede any [6] launch; unfixable honestly after results exist)

- [x] p6.1 Gate-statistics errata E1–E8 committed in knobs4-7 §R2-ERRATA (C1 cluster-CI conjunction [R2-B2]; C1-REP statistic pinned; v2→v3 promotion rule; SS-vs-TF statistic + branch; OOD zeroed-ID criterion; OOD probe-transfer scope n=303 reported-not-gated; power/MDE requirement) — all before any run they adjudicate. *(r2 §7 [pre-6]; commit 9c969cb)* — done 2026-07-26
- [ ] p6.2 Power/MDE simulation (CPU afternoon): paired-bootstrap MDE under 14-match clustering for every gate margin; record MDE per gate in the gate table; +2-seed escalation pre-committed iff C1-SCALE MDE > 0.01 → **MDE column exists next to each gate**. *(r2 §7 [pre-6]; knobs4-7 E8)*
- [ ] p6.3 Commit the probe-select split manifest (wd-selection metric = pooled val-ns AUC on that split) → **manifest file in-repo, referenced by E3**. *(r2 §7 [pre-6]; knobs4-7 E3)*
- [ ] p6.4 [R2-B4b] Grounded-GRPO claim schema one-pager (claim types, extractor, CRPS distribution semantics, ICC replicate protocol), per amendment F's own sequencing → **schema doc committed; hard-gate before any Phase-3 work**. *(r2 §7 [pre-6]; rb [pre-6])*

## Phase 6 — Pod gate, then Runbook [6]

- [x] 6.1 [B12] Archive VLM trainers (`train_grpo.py`/`train_sft.py`) + 4 wrappers; fix init.sh text + config.yaml → **`grep -rl train_grpo.py scripts/*.sh` = 0 live hits**. *(pf §5 6.1)* — done 2026-07-26 (executed early, this round) *(scope note 2026-07-26: the config.yaml fix covered content only — the legacy `vlm:` key name survives at config.yaml:20; it points at the live bridge LLM (Qwen3.6-35B-A3B), nothing live reads `config["vlm"]`, rename left to owner)*
- [ ] 6.2 [B7] Consolidate pod_setup_grpo.sh to one copy (port `/workspace/venv` creation); pod-runbook + run scripts agree → **fresh-pod dry-read: bootstrap creates exactly what run scripts require**. *(pf §5 6.2)*
- [ ] 6.3 [B11] **[OD-3]** Owner confirms RunPod account identity/balance; API key stored → **explicit go recorded** (no spend before this — standing rule). *(pf §5 6.3)*
- [ ] 6.4 Pod campaign (~$45–85, only on 6.3 go): 3 seeds × {v2,v3} + SS-off = 7 runs **+ co-trained twin (weight 0.3, no detach; secondary row only)** + RE retrain on the patched v2m blobs (`*_p1` per B9, superseded to `*_p2` by [1b]) + keystone WM data-scaling curve + CHANGE-C matched-capacity supervised ceiling (4090, no pod) → **DONE per runbook checks; RE and WM trained on the same corpus version**. *(pf §5 6.4; rb [6])*
- [ ] 6.5 corpus-strategy row 4 aliasing audit (pre-committed thresholds) BEFORE interpreting gates → **result in datasheet §6; 16Hz trigger armed or dismissed on evidence**. *(pf §5 6.5; pf §3 row 2)*

## Phase 7 — Runbook [7]: gates, then bridge/GRPO ladder

- [ ] 7.1 Run gates (+0.02, 4-of-5 maps, +0.01 — per knobs4-7 incl. §R2-ERRATA constructions) + probes (value_probe/facing_bias on current-corpus defaults, R1 ckpts) → **DONE: gate table complete incl. keystone, twin, E1/E2 secondary rows (+ MDE column per p6.2)**. *(pf §5 7.1)*
- [ ] 7.2 CHANGE F checker/ICC gates (AUC ≥ ~0.75 / ICC ≥ ~0.2; CRPS/Brier not binary; one offline ReST round first) — only runs because 1.2 landed → **DONE: computed on 14/14 val matches, no [] fallback fired**. *(pf §5 7.2)*
- [x] 7.3 Bridge R2 amendments PRE-REGISTERED (bridge-design §Amendments-R2, before any bridge spend): state-as-text arm [R2-B3]; head-Jacobian target as math; G=16 group semantics DECIDED; readability protocol; CHANGE-E leg 5; trainer build line; infra-plan killed-row voided (B12 contradiction resolved). *(r2 §7 [7]; commit 9c969cb)* — done 2026-07-26. **Execution of each remains below:**
- [ ] 7.4 State-as-text baseline arm run co-equal with latent-on/off/shuffled (frozen Qwen + templated textual state, scored on value-agreement/fact-audit/CRPS). *(r2 §7 [7]; bridge-design A-R2.1)*
- [ ] 7.5 Extend `nla_capacity_probe.py` to emit the head-Jacobian projection target (A-R2.2 math). *(r2 §7 [7])*
- [ ] 7.6 CHANGE-E leg 5: falsified-text + paraphrase-invariance probes on the trained decoder → **pass: |Δrecon(flip)| ≫ |Δrecon(paraphrase)| ≈ 0**. *(r2 §7 [7]; A-R2.5)*
- [ ] 7.7 Readability leg per A-R2.4 protocol (perplexity band vs base-Qwen + blinded rubric, fixed n). *(r2 §7 [7])*
- [ ] 7.8 NEW LINE between CHANGE F and on-policy GRPO: build grounded-GRPO manual loop (soft-prefix generate + claim scorer + recon-τ + KL-to-SFT), CPU-smokeable → **measured sec/step → $ budget recorded before go**. *(r2 §7 [7]; A-R2.6)*

## Minors batch (pf §4 — one cleanup session, any time; parallel-safe)

- [x] m1 Push 3c55fe8 (= 0.3). — done 2026-07-26
- [ ] m2 Pinned canonical commands — folds into 3.2 (counted once).
- [ ] m3 ruff-format landmine: delete hook OR standalone reformat commit + `ruff format --check` in CI; ruff==0.15.0 into venv.
- [ ] m4 demo-pipeline zero CI — copy chimera's ci.yml (pytest-only, fixture-based).
- [ ] m5 feature-list.json: header vs W00 `passes=true` contradiction; W00 pre-_p1 blob names. Reconcile.
- [ ] m6 tier-OOD three-way incoherence: strike §2 row 8 + §3 trigger (owner-declined per §6); datasheet §6 checkbox → "parked".
- [ ] m7 pyrightconfig.json still points at pre-move `/home/soone/chimera`.
- [ ] m8 **[OD-4]** RunPod ~$14/mo idle volume bp6ccofvnb: owner inspects, keep/delete (referenced by pod-runbook — don't delete blind).
- [x] m9 HF figure stale (470.5GB not ~450GB; public/ungated not "gated") — done 2026-07-26 (stale-facts pass).
- [ ] m10 E1/E2 probe rows: add "(secondary, never gating)" rows to OUTLINE §5 T1.
- [ ] m11 LaTeX compile smoke (`make iclr && make neurips`; `neurips_2026.sty` rejects `[preprint]`).
- [ ] m12 Cosmetic: empty dirs, `scripts_stier_loop.sh` naming, surviving v1-era blobs vs "~15GB deleted" note.
- [ ] m13 Paper meta-sections (pf §3 row 3): Reproducibility/Ethics/NeurIPS-checklist stubs in OUTLINE §4 + body.tex; datasheet HLTV-ToS position (fold into registered Tier-C #18). Before submission window.
- [ ] m14 corpus-strategy row 12: PureSkill/ESTA rationale paragraph in datasheet. Before next bake / paper. *(pf §3 row 2)*

## OPEN OWNER DECISIONS (decisions-ledger 2026-07-26 — record outcome there, tick here)

- [ ] **OD-1 Events-reparse scope** — (a) extend 1.2 reparse to all 92 matches with same tooling, OR (b) pre-register GRPO/ReST prompt-set restriction to event-covered train matches + verify that subset supports the prompt count. Needed NOW (before/at 1.2); a naive val-drawn prompt set would contaminate every gate. *(r2 §3; ledger OPEN (a))*
- [ ] **OD-2 MLMove disposition** — incomparability paragraph into OUTLINE + T3 notes is mandatory paper work; the cheap learned baseline column (per-player marginal head, same corpus) for T3 is yes/no undecided. Decide before [4]/[6]. *(r2 §4; ledger OPEN (b))*
- [ ] **OD-3 RunPod identity/balance + explicit spend go** (= 6.3 / B11). *(pf §2 B11)*
- [ ] **OD-4 RunPod volume keep/delete** (= m8). *(pf §4.8)*
