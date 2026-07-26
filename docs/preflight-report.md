# Chimera Preflight Report — Final Sweep Before Full Execution
**Date:** 2026-07-25 · **Scope:** pipeline Tier-A/B + runbook [2]–[7] + R1 canonical retrain · **Inputs:** 6 hunter sweeps, 24 verified standing findings (0 refuted)

---

## 1. Verdict: **GO-WITH-FIXES**

The program is structurally sound — corpus [1] is genuinely closed and mirrored, both test suites are green, the machine (disk/RAM/GPU/HF) is GO — but the runbook as written fails on first invocation at [2] (edge fit leaks the OOD holdout), and a cluster of stale-default/doc-drift seams would let [4], [6], and [7] silently run on the unpatched corpus or the archived VLM trainer. Every fix is cheap (the entire pre-flight fix budget is ~4–6 hours of local work, no GPU spend); nothing found invalidates the locked recipe or requires re-opening [1].

---

## 2. Blockers — fix before the FIRST step each gates (execution order)

| # | Gates | Finding | Evidence | Exact fix | Cost |
|---|-------|---------|----------|-----------|------|
| B1 | Every session start | `init.sh` dies on its first health check: both repos' `.venv/bin/activate` hardcode pre-move `/home/soone/<repo>` paths, so activation leaves system python live and `import polars` aborts under `set -e`. | `.venv/bin/activate:45` in both repos; `bash init.sh` reproduces | Recreate both venvs in place (`uv venv` / `python3 -m venv --clear` + reinstall) or sed old prefix → `~/projects/<repo>` across `.venv/bin/activate*` + shebangs; re-run `init.sh` until all health checks print. Also rewrite init.sh's echoed workflow text (still VLM-era). | 30–45 min |
| B2 | **[2]** fit dist edges | **THE blocker.** `fit_dist_edges.py` has no 5-map filter (and no argparse — `--help` crashes), so first invocation fits `DIST_EDGES_U` on 6 maps **including the de_overpass OOD holdout**, contradicting retrain-recipe.md:114's "pre-registered in fit_dist_edges.py" claim and knobs4-7:170's launch-blocker requirement (5-map, 3,573 rounds). | `scripts/fit_dist_edges.py:35-37`; `_corpus.py` excludes only anubis/train | Add `maps="de_ancient,de_dust2,de_inferno,de_mirage,de_nuke"` to the `load_corpus` call (or argparse `--maps` defaulting to the Knob-4 set) + `assert kept_rounds == 3573`; keep per-map printout. | 15 min |
| B3 | **[2]→[3]/[5]** seam | [2]'s done-check omits the separate hardcoded `700.0` open-ring magnitude (k=8-era) in `train_world_model.py:163`; fit_dist_edges computes its k=4 replacement. Missing it re-arms the registered teleport hazard in every sampled decode. | `train_world_model.py:163`; `fit_dist_edges.py:58`; same literal duplicated at `gen_bridge_sft.py:115` | In the same commit as the new edges: replace both `700.0` literals with a named `OPEN_RING_MAG_U` constant from the fit output; stamp into ckpt meta per knobs item 3. Amend [2]'s done-check to include it. | 15 min |
| B4 | **[4]** coverage harness + **[7]** probes | "Defaults flipped to _p1" ([1] DONE note) is only ~1/3 true: **9 live instruments** (`rollout_eval` — the exact file [4] extends, `decision_eval`, `dist_coverage_eval`, `value_probe`, `facing_bias_check`, `gen_bridge_sft`, `nla_capacity_probe` — CHANGE-D referee, `event_boundary_check`, `gen_demo`) still default to pre-patch/pre-merge blobs and retired/HISTORY checkpoints, all still on disk → silent wrong-corpus runs. | Commit 801008d touched only trainer/fit/test-fixture; grep `default=` across the 9 scripts | One sweep commit: flip all `--train-pt/--val-pt` defaults to `*_p1.pt`, remove `best.pt`/HISTORY ckpt defaults (make `--ckpt` required or point at R1 output dirs), optionally add a `load_corpus` warning when a blob lacks `patch_lineage`. | 45–60 min |
| B5 | **[5]/[6]** pre-registration + **[7]** CHANGE F | Runbook and the two July-2x audits don't compose: claude-progress (source of truth) is blind to everything post-Jul-19. Two hard seams: (a) **_p1 vs _p2 for R1 is undecided** — corpus-audit §5.B.3's "pre-retrain p2 blob patch (~free now)" is in no runbook step; (b) [7] CHANGE F is un-runnable as sequenced — 13/14 val matches lack kills side-files (`event_boundary_check.py:85` silent `[]` fallback) and the §5.B.1 events-reparse prereq appears nowhere. | corpus-audit.md §3 row 5, §5.B.1, §5.B.3; grep `2026-07-2` claude-progress.txt → empty | Update claude-progress.txt to absorb both audits: (a) decide and record _p1 or _p2 as the R1 corpus (if _p2: run the patch before [5]); (b) insert the val events-reparse as an explicit pre-[7] step. | 1–2 h (decision + doc + optional p2 patch) |
| B6 | **[4]** pre-registration completeness | corpus-strategy §2 row 13 (secondary 64Hz-truth-scored column in the coverage harness) was never absorbed — [4]'s done-check passes while a pre-registered metric silently vanishes. | corpus-strategy.md:39; zero "64Hz" hits in runbook/[4]/recipes | Add the 64Hz column to [4]'s spec and done-check in claude-progress.txt before implementing the harness. | 10 min doc + implement inside [4] |
| B7 | **[6]** pod bring-up | Divergent duplicate `pod_setup_grpo.sh`: doc-canonical root copy creates **no venv**; every `run_grpo_*.sh` hard-aborts without `/workspace/venv`. First fresh-pod bootstrap per pod-runbook.md fails. | root vs `scripts/` copies differ throughout; `run_grpo_pod.sh:82` | Keep one script (newer root version + port the `/workspace/venv` creation in, or fix run-script `VENV_DIR`), delete the other, update pod-runbook.md + run scripts to name the survivor. | 30 min |
| B8 | **[6]** pod env rebuild + manifest v2 | `uv.lock` is stale: pins demoparser2 0.41.1 < the 0.41.3 hard floor `process.py` preflight aborts on; `uv lock --check` fails. Poisons pod rebuilds and would stamp the wrong parser version into the planned exact-pins manifest v2. | uv.lock vs pyproject.toml:30; `process.py:69` | `uv lock` in chimera, verify demoparser2 ≥0.41.3 resolves, commit before any manifest-v2 generation or pod bootstrap. | 10 min |
| B9 | **[6]** RE retrain / **[7]** keystone | knobs4-7 item 8 (the operative spec — `train_round_encoder.py` isn't restored yet) pins RE defaults to unpatched `train_v2m.pt`/`val_v2m.pt`; verbatim implementation trains the RE on a different corpus than the WM, confounding the keystone L2 comparison. | knobs4-7.md:337, :517 (doc predates _p1) | Doc patch: `s/v2m.pt/v2m_p1.pt/` (train+val) in items covering Knob 7a; grep the doc for any other bare v2m/v3m blob names. | 10 min |
| B10 | **[6]** campaign completeness | The co-trained twin run is budgeted in knobs4-7 (~$5, 2–3 GPU-h) and load-bearing in paper T1 + Limitations §6, but absent from runbook [6]'s enumerated 7 runs — the campaign as written can't fill it and would force a second pod spin-up. | knobs4-7:440, :554; grep `twin` claude-progress → empty | Add one line to runbook [6]: "+ co-trained twin (weight 0.3, no detach), 1 run, secondary row only". | 5 min |
| B11 | **[6]** spend authorization | RunPod account's only pod carries **davidzengming@gmail.com** SSH keys and no local `RUNPOD_API_KEY` — account identity/balance unconfirmed. | env-infra live check | Owner confirms account identity + balance before any paid run; export/store the API key. **No GPU spend without explicit go (standing rule).** | 10 min (owner) |
| B12 | **[7]** GRPO ladder safety | Archived VLM era's `train_grpo.py`/`train_sft.py` (screenshot objective, recall/judge rewards — not the committed grounded CHANGE-A reward) remain live-wired to all four `run_*_pod.sh`/`run_grpo_smoke.sh` wrappers, init.sh guidance, and the live `config.yaml`. Wrong-objective hazard on a 35B pod. | `run_grpo_smoke.sh:63`, `run_grpo_pod.sh:113`, `init.sh:46`, `config/config.yaml` | Move `train_grpo.py`/`train_sft.py` + the four wrappers to `scripts/_archive/`; point init.sh at the runbook; gut config.yaml to surviving hub keys. Must land before B7's consolidated pod script is used. | 45 min |

**Total blocker budget: ~4–6 hours, all local, no GPU.**

---

## 3. Majors fixable in parallel with execution (before their consumer, not before start)

| Finding | Deadline | Fix | Cost |
|---|---|---|---|
| Falsified "~65/92 HF-only / not re-bakeable" survives in `first-principles-plan.md:167` (+ :138) and MEMORY.md — §2 row 2 partially executed, and its target list never included first-principles-plan | Before any corpus decision made from that read-first doc | Correct both instances to "all 92 re-bakeable; 70 HF + 22 local"; note row 2 closed-with-expanded-scope | 10 min |
| corpus-strategy §2 rows 3/4/5/12 silently incomplete: row 3 (split-integrity audit, no output anywhere), row 4 (aliasing audit / 16Hz trigger unarmed, datasheet §6 TODO open), row 5 (`build_tick_sequences.py:917` still hardcodes `tickrate_hz: 8` while §4 claims the fix landed), row 12 (no PureSkill/ESTA rationale in datasheet) | Row 3 before [6] (split PASS integrity); row 4 before [7] gate interpretation; rows 5/12 before next bake / paper | Run row-3 audit and record; execute row-4 aliasing audit with pre-committed thresholds; land the one-line row-5 builder fix; add row-12 datasheet paragraph | 2–4 h total |
| Paper meta-sections have no slot and no producer: no Reproducibility Statement, Ethics statement, or NeurIPS checklist anywhere; HLTV licensing/ToS discussed in zero docs (checklist-mandatory) | Before submission window; harmless during [2]–[7] | Append stubs to OUTLINE §4 + body.tex; add datasheet HLTV-ToS position (fold into registered Tier-C #18) | 1–2 h |

---

## 4. Minors batch (one cleanup session, any time)

1. **Push 3c55fe8** — S-tier campaign record is ahead-1, WSL-only. `git push origin main` in chimera-demo-pipeline. *(Do this one today — 1 command, durability.)*
2. **Pinned canonical commands** — write exact v2 + v3 invocations (blobs, `--maps`, `--dist-head`, seed, `--ss-pmax`) into retrain-recipe.md when landing [3]; [5]'s done-check references it. Guards against the v3m_p1-default vs v2-canonical trap.
3. **ruff-format landmine** — committed pre-commit hook vs 41 unformatted files, no hook installed, no local ruff. Either delete the hook from config or run the reformat as a standalone commit + add `ruff format --check` to CI; install ruff==0.15.0 into the venv.
4. **demo-pipeline has zero CI** — copy chimera's ci.yml (pytest-only, fixture-based, 0.21s).
5. **feature-list.json** — header note contradicts W00 `passes=true`; W00 names pre-_p1 blobs. Reconcile.
6. **tier-OOD three-way incoherence** — strike §2 row 8 and the §3 trigger (both gates owner-declined per §6); close the datasheet §6 checkbox as "parked".
7. **pyrightconfig.json** — still points at pre-move `/home/soone/chimera`; lint bar ran blinded.
8. **RunPod ~$14/mo idle bleed** — 200GB volume bp6ccofvnb since May. Owner inspects contents, then delete/keep decision (volume is referenced by pod-runbook.md — don't delete blind).
9. **HF figure stale** — repo is 470.5GB not ~450GB, and public/ungated not "gated"; update where cited.
10. **E1/E2 probe rows** — add "(secondary, never gating)" rows to OUTLINE §5 T1 so the [6]/[7] outputs have a slot.
11. **LaTeX never compiled** — no toolchain; `neurips_2026.sty` will reject the `[preprint]` option. One-time `make iclr && make neurips` smoke in a container/Overleaf.
12. **Cosmetic** — empty dirs, `scripts_stier_loop.sh` naming, surviving v1-era blobs vs "~15GB deleted" note.

---

## 5. ONE-PAGE EXECUTION ORDER

*Linear; each line ends with its done-check. Steps tagged [Bn] are the fixes above.*

**Phase 0 — Ground truth & environment (local, ~half day)**
- [ ] 0.1 [B1] Recreate/repair both venvs; re-run `init.sh` → **all health checks + HF-auth line print, exit 0**
- [ ] 0.2 [B8] `uv lock` + commit → **`uv lock --check` passes; demoparser2 ≥0.41.3 in lock**
- [ ] 0.3 `git push` chimera-demo-pipeline → **`git status -sb` shows no ahead; `ls-remote` = 3c55fe8**
- [ ] 0.4 [B5] Update claude-progress.txt: absorb Jul-20/22 audits; **decide _p1 vs _p2 for R1** (if _p2: run §5.B.3 patch + re-mirror to HF now) → **runbook names the exact R1 blob filenames; date > 2026-07-22 present**
- [ ] 0.5 [B4] Eval-default sweep commit (9 scripts → `*_p1`, no retired-ckpt defaults) → **`grep -rn 'v3.pt\|v2m.pt\|train.pt\|best.pt' scripts/*.py` returns only _p1/archive hits**
- [ ] 0.6 Doc corrections batch: [B9] knobs item 8 → `_p1`; [B10] twin line into runbook [6]; falsified ~65/92 fix (plan:167 + MEMORY.md); [B6] 64Hz column into [4] spec → **grep confirms each**

**Phase 1 — Pipeline Tier-A/B (chimera-demo-pipeline)**
- [ ] 1.1 Execute registered Tier-A then Tier-B items per corpus-implementation-audit §5 (incl. CHANGE-NOW #14 path fix) → **42p/1s → 43p; skip removed**
- [ ] 1.2 [B5b] Val events-reparse (§5.B.1) so kills side-files exist for 14/14 val matches → **`event_boundary_check.py` finds >0 events on every val match; silent-[] fallback replaced with hard error**
- [ ] 1.3 corpus-strategy row 3 split-integrity audit + row 5 builder tickrate fix → **audit result recorded in datasheet §3; `grep tickrate_hz: 8` build script = 0 hits**

**Phase 2 — Runbook [2]: dist edges**
- [ ] 2.1 [B2] Add 5-map filter + `assert rounds == 3573` to fit_dist_edges.py; add argparse so `--help` works
- [ ] 2.2 Run fit; commit new `DIST_EDGES_U` **and** [B3] `OPEN_RING_MAG_U` (replacing both 700.0 literals) in one commit → **DONE: edges + open-ring mag committed, per-map quantiles in datasheet, no overpass row in fit set**

**Phase 3 — Runbook [3]: trainer completion**
- [ ] 3.1 Implement detach/SS per locked recipe → **`test_no_value_leak` flips xfail→pass; suite fully green**
- [ ] 3.2 Write pinned canonical v2 + v3 commands into retrain-recipe.md → **DONE: exact runnable commands with blobs + `--maps` exist in-repo**

**Phase 4 — Runbook [4]: coverage harness**
- [ ] 4.1 Extend rollout_eval (now _p1-defaulted): minADE-K + CHANGE-B baseline + trajectory-coherence + **64Hz-truth-scored column** → **DONE: sampled-coverage smoke prints baseline AND 64Hz columns on `val_v2m_p1` (or _p2 per 0.4)**

**Phase 5 — Runbook [5]: local smoke**
- [ ] 5.1 Run the **pinned** canonical v2 command with `--smoke` → **DONE: completes on CPU/4090, ckpt meta carries edges + open_ring_mag_u + correct blob shas**

**Phase 6 — Pod gate, then Runbook [6]**
- [ ] 6.1 [B12] Archive VLM trainers + wrappers; fix init.sh text + config.yaml → **`grep -rl train_grpo.py scripts/*.sh` = 0 live hits**
- [ ] 6.2 [B7] Consolidate pod_setup_grpo.sh to one copy; pod-runbook + run scripts agree → **fresh-pod dry-read: bootstrap script creates exactly what run scripts require**
- [ ] 6.3 [B11] Owner confirms RunPod account identity/balance; API key stored → **explicit go recorded** *(no spend before this)*
- [ ] 6.4 Pod campaign: 3 seeds × {v2,v3} + SS-off = 7 runs **+ co-trained twin (secondary)** + RE retrain on **`*_v2m_p1`** + scaling curve + CHANGE-C ceiling → **DONE per runbook checks; RE and WM trained on the same corpus version**
- [ ] 6.5 corpus-strategy row 4 aliasing audit (pre-committed thresholds) before interpreting gates → **result in datasheet §6; 16Hz trigger armed or dismissed on evidence**

**Phase 7 — Runbook [7]: gates**
- [ ] 7.1 Run gates (+0.02, 4-of-5 maps, +0.01) + probes (value_probe/facing_bias on _p1 defaults, R1 ckpts) → **DONE: gate table complete incl. keystone, twin, E1/E2 secondary rows**
- [ ] 7.2 CHANGE F (AUC≥0.75/ICC≥0.2) — only runs because 1.2 landed → **DONE: computed on 14/14 val matches, no [] fallback fired**

**Anytime (parallel):** minors batch §4 · paper meta-section stubs + HLTV-ToS position · LaTeX compile smoke · RunPod volume keep/delete decision.

---
*24 standing findings triaged: 12 gate a step (§2), 3 parallel-major (§3), 12 minor (§4; B3/pinned-command counted once each where folded). Nothing re-reports the four registered audits; all registered-status corrections are noted inline.*
