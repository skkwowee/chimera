# Chimera Program Preflight Report — 2026-07-25

Sweep scope: gaps between the four completed audits (adversarial-review, corpus-audit §4, corpus-implementation-audit §5, first-principles-plan §3) + corpus-strategy §2 checklist. Nothing already correctly registered is re-reported below; several items appear because their **registered status is wrong**.

---

## 1. Verdict: GO-WITH-FIXES

The program's foundations are sound — [1] is genuinely closed (all four `*_p1` blobs + manifests live on HF), the test suites are green in the exact registered state, CI is green at HEAD, disk/GPU/HF/tokens are healthy, and the paper scaffold carries no pre-reset claims. However, two blockers sit directly on the first runbook invocations ([2] cannot produce the pre-registered 5-map edge fit as documented, and the [3]/[4]/[7] eval instruments silently score stale corpora), plus a broken session bootstrap and an unverified split-integrity PASS standing between the split and R1 validity — roughly one working day of fixes, none requiring GPU spend, converts this to a clean GO.

---

## 2. Blockers — must fix before the FIRST step each blocks (execution order)

### B0. Venv relocation broke init.sh and every activated-shell command
**Blocks:** session start — every step [2]-[7] run from an activated shell; harness rule "run init.sh at session start."
**Finding:** Both repos' `.venv/bin/activate` scripts and console-script shebangs hardcode the pre-move `/home/soone/chimera[-demo-pipeline]/.venv`. `bash init.sh` fails first invocation (`ModuleNotFoundError: polars` under `set -e`) and silently re-runs `uv pip install` into the training venv every session. `awpy get maps` has no working invocation at all (`python -m awpy` also fails). init.sh also prints the archived VLM-era SFT workflow as guidance.
**Evidence:** `chimera/.venv/bin/activate:45` → `VIRTUAL_ENV=/home/soone/chimera/.venv` (path does not exist); reproduced live: post-activate `which python3` = `/usr/bin/python3`; 52/67 chimera bin shebangs dead.
**Fix:** Recreate both venvs in place (`uv venv` + reinstall from lock — after B1 below), change init.sh to call `.venv/bin/python` explicitly, replace the SFT guidance with the runbook pointer. Also fixes the dead-shebang minor.
**Cost:** ~30 min.

### B1. uv.lock pins demoparser2 0.41.1 against a >=0.41.3 floor
**Blocks:** B0's lock-faithful venv rebuild; any fresh env/pod sync; pipeline Tier-A/B on a rebuilt env (`process.py` aborts below 0.41.3); CHANGE-AT-V5 #3 manifest stamping (would record 0.41.1 — a version never used).
**Evidence:** `uv.lock:591-592` = 0.41.1; `pyproject.toml:30` = `>=0.41.3`; installed 0.41.3 was upgraded out-of-band, never re-locked.
**Fix:** `cd ~/projects/chimera && uv lock`, verify 0.41.3 in lock, commit. Do this BEFORE recreating venvs in B0.
**Cost:** ~10 min.

### B2. Eval/probe instruments default to pre-patch/pre-merge blobs (registered "[1] defaults flipped" status is overstated)
**Blocks:** runbook [3] verification evals, [4] coverage smoke, [7] gates; any post-run eval in [5]/[6].
**Finding:** `rollout_eval.py:199`, `dist_coverage_eval.py:53`, `gen_demo.py:43` default to `val_v3.pt` (Jun 8 pre-merge); `decision_eval.py:109-111` to `val_v3.pt`/`train_v3.pt`; `eval_world_model.py:57` and `value_probe.py:108-109` to v1-era `val.pt`/`train.pt`. All stale blobs still exist on disk, so first invocations silently score the wrong corpus (broken bomb_state, un-anchored clock, origin-distance dist_to_bomb). Commit 801008d flipped only trainer + fit_dist_edges + test fixture. `decision_eval`/`dist_coverage_eval` also default `--maps` to the 3-map era set.
**Fix:** Flip all six scripts' defaults to `*_v{2,3}m_p1.pt` + 5-map set (or make `--val-pt`/`--train-pt` required); optionally move superseded `train.pt/val.pt/train_v3.pt/val_v3.pt` out of the default dir so stale loads fail loudly. Correct the [1] done-note wording in claude-progress.txt.
**Cost:** ~30 min.

### B3. Split-integrity PASS is ahead of its evidence (datasheet §3 status wrong)
**Blocks:** [5]/[6]/[7] — R1 validity end-to-end. If any of the 5 same-name-different-content stem pairs straddles the split, every gate scores on contaminated val.
**Finding:** The 30-min divergent-stem audit (corpus-strategy §2 row 3) never ran, yet datasheet §3 asserts unqualified "Split (leak audit) — PASS". The dedup key `(norm_stem, round, first_tick, n_ticks)` is defeated by a re-encoded copy of the same game (e.g. spirit-vs-vitality-m1-mirage 945MB local vs 654MB HF).
**Fix:** Run the diff of the 5 stem pairs vs both split sides; keep PASS with a dated evidence line, or fix the split. Until run, annotate §3 "pending divergent-stem check".
**Cost:** ~30 min.

### B4. fit_dist_edges.py has no 5-map filter (registered "no code changes needed" status is wrong)
**Blocks:** runbook [2] and everything consuming DIST_EDGES_U ([3]-[6]).
**Finding:** The documented invocation fits DIST_EDGES_U over all 6 maps including ~303 de_overpass OOD-holdout train rounds (~8% of clean train; overpass is the distributional outlier). `fit_dist_edges.py:35-37` takes only a positional path, calls `load_corpus` without `maps=`; `:54-56` pools magnitudes across maps silently — the [2] done-check would not catch it.
**Fix:** Add `--maps` defaulting to the Knob-4 5-map set (or hardcode), plus a zero-overpass assert after filter; note the deviation from "no code changes" in the pre-registration trail (knobs4-7 item 1, retrain-recipe.md:115).
**Cost:** ~15 min.

### B5. Trainer defaults to the forbidden-local v3 arm with no map filter
**Blocks:** runbook [5] flagless smoke and [6] matrix (silent wrong-arm/wrong-map risk; v3 is 10.7+2.0 GB — "never run v3 locally").
**Finding:** `train_world_model.py:393-394` defaults to `train_v3m_p1.pt`/`val_v3m_p1.pt`; `:382` `--maps` defaults to all maps incl. overpass. The planned [3] checklist item 3 asserts round counts that pass identically on v3m, so no guard distinguishes arms even after [3] lands as specced.
**Fix:** Fold into the [3] consolidated trainer edit: defaults → `v2m_p1`, canonical `--maps` baked, two-stage corpus assert (3,876/705 → 3,573/641) + zero-overpass assert; commit the canonical run command as a launcher script.
**Cost:** +15 min on top of the already-planned [3] edit.

### B6. Owner decision: p1-vs-p2 corpus identity and CHANGE-F sequencing (execution-order seam)
**Blocks:** [2] (corpus identity must be fixed before edges are fit) and [7] CHANGE-F (audit says its gates "cannot be honestly run" on val today).
**Finding:** corpus-audit §5.B's pre-retrain items (p2 blob patch "~free now"; val event ground truth + `event_boundary_check.py:84-85` hard-fail + AUC 0.519 recompute) are in NO execution list — not Tier-A/B, not runbook [2]-[7]. "p2" appears in no file outside corpus-audit.md.
**Fix:** Write one decision line into the runbook top: (a) p2 = new step [1b] before [2] with `_p2` defaults + lineage restamp, or (b) p2 explicitly deferred post-R1. Separately add "Tier-B #11 (val kills JSONs) + event_boundary_check hard-fail + AUC recompute" as an explicit precondition line inside [7] CHANGE-F.
**Cost:** ~15 min (decision + doc edit); Tier-B #11 work itself is already scheduled, just unsequenced.

### B7. Pre-pod cluster — must clear before the first [6] pod is created
1. **RunPod SSH identity:** the one existing pod carries third-party (`davidzengming@gmail.com`) keys in PUBLIC_KEY. Verify/rotate before any [6] pod. ~15 min.
2. **VLM GRPO/SFT stack live outside `_archive`:** `train_grpo.py`, `train_sft.py`, `build_sft_dataset.py`, `run_grpo_pod.sh`, `run_grpo_smoke.sh`, `run_grpo_with_auto_stop.sh`, `run_sft_pod.sh`, `src/training/*`, `src/inference/vlm.py` — `run_grpo_smoke.sh` looks exactly like the [7] smoke entry point; claude-progress.txt:327-329's "Scripts are in scripts/_archive/" is false for this stack. Move to `_archive`. ~20 min.
3. **Duplicate `pod_setup_grpo.sh`:** scripts/ copy is the stale VLM-era bootstrap; `bridge-design.md:240` still points at it. Archive scripts/ copy, fix the reference. ~10 min.
4. **Paper Limitations freeze:** OUTLINE §6 predates D7 tick-dropout and the post-plant AUC-floor disclosures; adding them AFTER [6]/[7] numbers land reads as post-hoc softening under the project's own fidelity rule. Append two bullets (D7 cadence non-uniformity; defuse-invisibility floor at the ±0.02 keystone margin) + hedge the abstract's 8Hz clause. Pure docs. ~30 min.
5. **knobs4-7 item 8 blob names:** RE port says defaults → `train_v2m.pt`/`val_v2m.pt` (unpatched — doc predates [1]); verbatim execution splits WM (_p1) and RE (unpatched) across two corpora inside the C1 comparison. Edit `v2m` → `v2m_p1` in items 8 (and blob paths in 6/10/11). ~5 min.

**Total blocker cost: ~3.5-4 hours, all CPU/docs.**

---

## 3. Majors fixable in parallel with execution

| Finding | Evidence | Fix | Deadline |
|---|---|---|---|
| corpus-strategy contradicts itself 3 ways on tier-OOD (row 8 orders what §6 permanently rejected; §3 trigger cites a datasheet §6 sanction that doesn't exist) | corpus-strategy.md:34 vs :49 vs :97; datasheet.md:123 | Strike row 8; mark §3 trigger DEAD (both channels rejected 2026-07-19); fix datasheet.md:123 wording | Before anyone executes §2 as part of the program |
| 64Hz-truth column (corpus-strategy §2 row 13, pre-registered) never absorbed into runbook [4] harness spec | Harness as specced silently drops it | Add the column to the [4] coverage harness before building it | Before [4] build |
| Paper has no ethics/reproducibility slots, no NeurIPS-checklist plan, and zero licensing/PII content in the datasheet to answer them from (109.7GB HLTV demos with real player identities rehosted on HF) | body.tex:13-49; OUTLINE §4; datasheet grep licen/ethic/privacy = 0 | Add statement slots to body.tex + OUTLINE; add datasheet §8 (source terms, redistribution posture, player-identity note) | Before submission strategy hardens; ~45 min |
| Runbook [3] "flips green" wording false under strict-xfail; knobs4-7 [3] checklist has stale line/tracking pointers | test_no_value_leak is strict xfail — passing requires removing the marker | Correct wording + pointers during the [3] session | During [3] |

---

## 4. Minors batch (~90 min, one hygiene session, any time)

1. `git push` in chimera-demo-pipeline — 3c55fe8 (S-tier 40/40 record) is single-copy on WSL. **Do first; 1 min.**
2. MEMORY.md:8 still carries the falsified "~65/92 matches have no local demos" claim (corpus-strategy row 2 was 2/3 done) — re-poisons every session. Edit to: split 70 HF + 22 local; all 92 re-bakeable from 224 raw .dem on HF; patch-in-place chosen for cost, not necessity.
3. feature-list.json W00 stale post-[1]: point at `*_p1.pt` + corpus_manifest.json + 5-map 3,573/641; keep passes=true.
4. Add pytest CI to chimera-demo-pipeline (copy chimera's ci.yml pattern) + `pre-commit install` in chimera — do BEFORE Tier-A/B edits land so they land gated.
5. corpus-strategy §2 status column (rows 4/5/9/12 all silently open): land the 2-line tickrate_hz fix (`build_tick_sequences.py:917/:845`); schedule row 4 aliasing audit + wall-detection #3 as CPU work; write the consolidated v5 stub (row 9 + wall-detection §3 + corpus-audit §5.A incl. is_reloading fix); add PureSkill/ESTA negative-source rationales to datasheet.
6. pyrightconfig.json venvPath/extraPaths → `/home/soone/projects/chimera`.
7. RunPod volume bp6ccofvnb: ~$14/mo while EXITED vs the recorded no-recurring-spend stance — owner decision: keep for [6] (may hold Qwen base weights) or archive-and-delete.
8. nvidia-smi not on PATH (only `/usr/lib/wsl/lib/nvidia-smi`) — add to PATH or symlink.
9. Paper build: placeholder NeurIPS sty is a fabricated approximation, no TeX installed — grab real template + plan texlive/Overleaf before first compile.
10. claude-progress.txt:98 "15GB cap" note stale (now 22GB/8GB swap).

---

## 5. ONE-PAGE EXECUTION ORDER

Each line: `[ ] step — done-check`. Follow top-to-bottom. No GPU spend before line 20.

**PHASE 0 — BOOTSTRAP REPAIR (blockers B1, B0; ~45 min)**
1. `[ ]` `uv lock` in chimera; commit — done: uv.lock shows demoparser2 0.41.3
2. `[ ]` Recreate both venvs from lock; fix init.sh (explicit `.venv/bin/python`, runbook pointer replaces SFT text) — done: `bash init.sh` exits 0; `.venv/bin/pytest --version` works in both repos; chimera suite still 23p/1s/1xf
3. `[ ]` `git push` in chimera-demo-pipeline — done: origin/main == 3c55fe8
4. `[ ]` Owner decision line at runbook top: p1 vs p2 (B6a) — done: decision written in claude-progress.txt

**PHASE 1 — PIPELINE TIER A/B (gated)**
5. `[ ]` Add pytest CI to pipeline repo — done: first Actions run green
6. `[ ]` Tier-A fixes (cli.py:176-185; hltv.py:149-154; parse_demos.py:41-48; cli.py:310-312 stale `--chimera-dir` default incl. test_process.py:54) — done: 43/43 pass, zero skips
7. `[ ]` Tier-B fixes incl. #11 val kills side-files + `event_boundary_check.py:85` silent-`[]` → hard-fail + AUC 0.519 recompute — done: 13/14 val matches have kills JSONs; recomputed AUC logged; [7] CHANGE-F precondition line added to runbook (B6b)

**PHASE 2 — CORPUS VALIDITY GATE (blockers B3, B2; ~1 h)**
8. `[ ]` Divergent-stem audit: diff 5 stem pairs vs both split sides — done: dated evidence line under datasheet §3 PASS (or split fixed + re-baked manifests)
9. `[ ]` Flip 6 eval-script blob defaults to `*_p1` + 5-map sets; relocate superseded blobs; correct [1] done-note — done: grep shows no `val_v3.pt`/`val.pt` defaults; `rollout_eval.py` smoke prints an `_p1` path

**PHASE 3 — RUNBOOK [2] (blocker B4)**
10. `[ ]` Add 5-map filter + zero-overpass assert to fit_dist_edges.py; log the "no code changes" deviation — done: script refuses overpass rounds
11. `[ ]` Run [2] — done: DIST_EDGES_U committed; per-map quantile table has NO overpass row; edges fit on 3,573 rounds

**PHASE 4 — RUNBOOK [3] (blocker B5 folded in)**
12. `[ ]` Consolidated trainer edit: detach/SS completion + defaults → `v2m_p1` + canonical `--maps` + two-stage asserts (3,876/705 → 3,573/641) + zero-overpass assert; commit canonical launcher script; fix "flips green"/pointer wording — done: flagless run REFUSES; launcher run passes asserts
13. `[ ]` Run [3] done-check — done: value-leak sentinel passes with xfail marker removed; full suite green

**PHASE 5 — RUNBOOK [4]**
14. `[ ]` Build coverage harness WITH the 64Hz-truth column (corpus-strategy row 13) — done: column present in output schema
15. `[ ]` Sampled-coverage smoke on val — done: output names `val_v2m_p1.pt`; baseline + 64Hz-truth columns populated

**PHASE 6 — RUNBOOK [5]**
16. `[ ]` Local smoke via committed launcher (v2, seed 0) — done: asserts print 3,573/641; zero overpass; loss curve sane

**PHASE 7 — PRE-POD HYGIENE (blocker B7; ~1.5 h, can start any time after Phase 0)**
17. `[ ]` Archive VLM stack (train_grpo/train_sft/build_sft_dataset + 4 run_*.sh + src/training VLM modules + vlm.py) and scripts/pod_setup_grpo.sh; fix bridge-design.md:240; correct claude-progress.txt:327-329 — done: `grep -r grpo scripts/ --include='*.sh'` finds only the root bootstrap path
18. `[ ]` Verify/rotate RunPod SSH keys (remove davidzengming key); decide volume bp6ccofvnb keep/delete — done: PUBLIC_KEY holds only owner keys; volume decision logged
19. `[ ]` Paper freeze edits: OUTLINE §6 +2 bullets (D7 dropout; defuse-floor at keystone margin), abstract 8Hz hedge, ethics/repro slots, datasheet §8; knobs4-7 items 6/8/10/11 `v2m` → `v2m_p1` — done: committed BEFORE any [6] number exists

**PHASE 8 — RUNBOOK [6] (first GPU spend — explicit go required)**
20. `[ ]` 7 pod runs per matrix, root `pod_setup_grpo.sh` only, pull via `pull_blobs_hf` (_p1 defaults verified correct) — done: per-run artifact + W&B log; pod stopped after each
21. `[ ]` Knob-7 RE retrain (r1-re-v6) from restored train_round_encoder.py with `_p1` defaults per edited checklist item 8 — done: RE and WM provably trained on the same `_p1` corpus

**PHASE 9 — RUNBOOK [7]**
22. `[ ]` Verify CHANGE-F precondition line satisfied (line 7 complete: val ground truth + hard-fail + recomputed AUC) — done: precondition check passes before any gate fires
23. `[ ]` Run gates incl. C1 keystone ±0.02, interpreting marginal results against the pre-committed defuse-floor limitation — done: gate table committed; paper T1-T6 slots filled from producers

**ANY TIME — MINORS BATCH (~90 min):** MEMORY.md correction; feature-list W00; corpus-strategy status column + tickrate fix + v5 stub + row 4/12; pyrightconfig; nvidia-smi PATH; chimera pre-commit install; NeurIPS real template + TeX plan; "15GB cap" note.