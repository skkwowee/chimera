# Corpus Implementation Audit — Synthesis

**Scope:** acquisition (`~/projects/chimera-demo-pipeline`) + bake/merge/patch (`~/projects/chimera`). Measured against a live 36-pass S-tier archival run (PID 31950, started 2026-07-19, cap-exit ~15:30 PDT 2026-07-22). All file:line claims verified against source; all channel numbers measured from `logs/stier_scrape.log` (1.03M lines) and HF API reads of `skkwowee/chimera-cs2`.

---

## 1. Verdict: QUALIFIED-YES

**Did we build the correct data pipeline to fetch more S-tier demos whenever we want?**

The core channel is correct and just proved itself under load: **288/288 matches in 36 passes over ~67 hours, zero terminal HLTV failures**, 5s pacing + 10/20/30s backoff holding well under Cloudflare limits (hltv.py:39/84-96), collision-proof `demos/<match_id>/` namespacing (upload.py:32-35, since ae3170a), and resumable match-id-keyed manifests (cli.py:196). Measured: 161.3 GB at 10.6 MB/s, 114.6-min pass cadence, one transient HF-upload blip in 66.9h that recovered on the next pass. **"Fetch more demos" works today.**

The qualification is that **"whenever we want" is not yet true unattended**:

- The supervisor is a 14-line **untracked** bash loop with zero error handling (scripts_stier_loop.sh:8-13 — no `set -e`, no exit-code check, no lock, no failure counter). Every failure class is swallowed silently, including `ScrapeLayoutError`, which the pipeline deliberately raises to fail loudly (hltv.py:119-123) — the contract is defeated one layer up.
- Cloudflare degradation is **invisible until total failure**: the retry path (hltv.py:93-96) emits no log output, so "zero 403s in 1.03M lines" means zero *terminal* failures, not zero blocks. The chrome124 fingerprint is ~27 months stale.
- Two filters are not enforced client-side: **4 sub-4-star matches leaked** into the S-tier corpus (cli.py:176-185 never checks `summary.stars`), and **≥97 of 288 matches (~34%) are CS:GO-era** — an explicitly rejected source class (corpus-strategy.md:110) — because no date gate exists and `date_unix` is null on **358/358** manifest entries (dead selector, hltv.py:149-154).
- A fetched demo is still ~3–6 min/map of **un-archived parsing** away from being training data: `parsed/` on HF contains **zero files**, the corpus-audit's "land now, ~6 lines" props-v2 fix (corpus-audit.md:106) never landed, and only 22/92 canonical matches (~24%) are re-bakeable without a ~110 GB + 20–40 CPU-h re-parse.

The change-now items in §5 (items 1–8 are hard gates) convert QUALIFIED-YES to YES, and none of them touch the parts that just proved themselves.

---

## 2. The ideal implementation (first principles)

An **archive-centric evolution** of the existing hardened pipeline — not a rewrite. Seven invariants:

1. **Each demo crosses the Cloudflare channel exactly once, ever.** The parsed props-v2 bundle (superset: *all* non-empty awpy tables + extended player props + `parse_meta.json` with exact pins and dem sha256) is the canonical re-bake substrate, backfilled HF→HF at zero channel cost — 92 canonical matches first, CS2-era archive opportunistically, CS:GO-era never.
2. **No write is observable without its provenance.** Manifest lines fold into the *same* `create_commit` as the content they describe (demo side and parsed-bundle side); sha256 is computed at download and verified against the HF LFS OID; a read-only `reconcile` job adopts historical orphans from server-side truth.
3. **The reliability boundary lives in tested, git-tracked Python.** The bash loop becomes a `chimera-demo campaign` subcommand: flock single-instance, entrypoint preflight, per-pass exit-code check, 2-consecutive-failure halt, `ScrapeLayoutError` = immediate abort, STOP sentinel checked *between matches* (~1 min latency vs ~2 h), end-of-campaign tally to claude-progress.txt / NEEDS_ATTENTION.
4. **Filters are never trusted server-side.** Client-side stars gate; CS2-era gate on a *fixed* date_unix; config-driven impersonation target bumped chrome124→chrome142+ between campaigns only.
5. **Observability is minimal but sufficient:** one retry-log line per non-200 attempt, one health script, a NEEDS_ATTENTION sentinel printed by init.sh. No dashboards, no push infra, no canary cron (a canary spends the scarce resource to monitor it; the breaker bounds fingerprint death to ≤2 wasted passes).
6. **Storage stays one gated HF repo at $0.** Legacy flat demos frozen, bridged by `legacy_matches.csv`; CS:GO-era demos kept-but-era-tagged (first-to-prune on the pre-armed storage-warning trigger); the ~5 GB irreplaceable set (81 parquets, manifests, patch scripts, 5 divergent demos) mirrored to free Google Drive.
7. **Bakes become pure functions of `parsed/`** with a `REQUIRED_PROPS ⊆ parse_meta` preflight that refuses loudly — landed with the v5 bake, mechanically preventing silent props-v1/v2 heterogeneity.

**Explicitly rejected:** event-ledger manifest rewrite (migration risk to just-proven resume/skip logic), SourceAdapter/Faceit/Inbox frameworks (no equivalent S-tier source exists), repo split (highest blast radius anywhere, pre-empting an already-trigger-armed risk), weekly canary, paid mirrors (owner's recorded no-spend stands), proxy rotation (escalation trains Cloudflare on us), legacy-demo migration (110 GB churn for a CSV's worth of benefit), `.dem` validation in `run` (the parse backfill surfaces corruption where it's cheapest).

**Sequencing constraint: nothing touches the channel or the venv until PID 31950 exits (~15:30 PDT today), and `scripts_stier_loop.sh` is committed as-is for the historical record first.**

---

## 3. Current vs. ideal — delta table

| # | Area | Current state (evidence) | Ideal | Tag |
|---|---|---|---|---|
| 1 | Scraper + pacing | curl_cffi results scrape, 5s pacing, 10/20/30s backoff, separate unpaced binary session (hltv.py:39/84-96, cli.py:155). 288/288, zero terminal failures in 66.9h | Same | **KEEP** |
| 2 | Layout-drift contract | `ScrapeLayoutError` raised instead of silent-empty (hltv.py:119-123, :194-196); drift fixtures tested (test_hltv.py:27-79) | Same — the defect is one layer up | **KEEP** |
| 3 | Download integrity | `.part` + Content-Length short-read guard, 3 retries, all-or-nothing tempdir (download.py:46-87) | Same; note no-Content-Length truncation is accepted (download.py:68) and pushed to extract-stage — acceptable, caught by parse backfill | **KEEP** |
| 4 | Extraction | 7z/unar/unrar shell-out, unrar-free excluded (download.py:134-170) — zero test coverage | Same code + tests at v5 | **KEEP** (tests: v5) |
| 5 | Namespacing + atomic demo commit | `demos/<match_id>/`, single create_commit per match (upload.py:14-51) | Extended: manifest line folded into the same commit | **CHANGE-NOW #7** |
| 6 | Manifest format | Three JSONL manifests, shrink guard, attempts≥3 cap (manifest.py) — proved over 36 passes | Same format forever; no DB, no event ledger | **KEEP / NEVER (ledger)** |
| 7 | Supervisor | 14-line untracked bash, no error handling, silent pass-40 exit, STOP_SCRAPE latency up to ~2h (scripts_stier_loop.sh:8-14) | `chimera-demo campaign` subcommand, git-tracked, tested | **CHANGE-NOW #6** |
| 8 | Stars filter | Server `?stars=` trusted; `matches_filters()` never checks stars (cli.py:176-185); 4 leaks confirmed on HF | Client-side gate | **CHANGE-NOW #1** |
| 9 | Era filter / dates | No CS2-era gate; `date_unix` null 358/358 (dead selector hltv.py:149-154); ~34% of campaign is CS:GO-era | Fixed selector (parse match page already fetched) + era gate + all-null-column test | **CHANGE-NOW #2, #3** |
| 10 | Fingerprint | chrome124 hardcoded (hltv.py:33, download.py:41); curl_cffi 0.15.0 ships chrome142/145/146 | Config-driven; bump post-campaign; quarterly runbook item | **CHANGE-NOW #5** |
| 11 | Retry observability | Backoff path emits nothing (hltv.py:93-96); decay invisible until total failure | One structured line per non-200 attempt | **CHANGE-NOW #4** |
| 12 | Dual-commit windows | Demo upload vs manifest push are separate commits (cli.py:254-261 vs :271-274); same shape bake-side (process.py:387-398 vs :439-457) — orphans + 1-2 GB re-downloads + silent blacklist | Same-commit atomicity + `reconcile` for historical orphans | **CHANGE-NOW #7, #17** |
| 13 | sha256 | Nowhere in pipeline; datasheet.md:24 claim falsifiable; backfill dedups by bare filename (cli.py:405-410) — the exact key 5 proven collisions defeat | Stream-hash at download; verify vs LFS OID; backfill 969 files from OIDs; dedup on match_id+sha256 | **CHANGE-NOW #8** |
| 14 | Failure ledger | Append-only; 2349630 ghost (attempts=1 persists after success); 3 transient blips = permanent blacklist (manifest.py:33, :239-260) | Clear-on-success; exclusion *tags* for sub-4★/CS:GO matches (never deletion) | **CHANGE-NOW #9** |
| 15 | Parse substrate | `parsed/` EMPTY on HF; props-v1 only (parse_demos.py:41-48); archives drop grenades/smokes/infernos/shots/weapon_fire/footsteps (process.py:296-302); 0/70 entries carry parsed_files | props-v2 superset + `parse_meta.json`, then HF→HF backfill, 92 canonical first | **CHANGE-NOW #10, #11** |
| 16 | Archive-before-build + rebake | Correct ordering and stale-schema logic (process.py:372-398, 580-622) | Same; e2e tests at v5 | **KEEP** |
| 17 | Legacy 224 flat demos | Split namespace; tools globbing `demos/<match_id>/` miss them | Frozen in place; bridged by `legacy_matches.csv` | **KEEP + CHANGE-NOW #15 / NEVER (migration)** |
| 18 | CS:GO-era + sub-4★ demos on HF | ~100+ GB best-effort weight on free tier (~450 GB total) | Retain (channel cost sunk, deletion irreversible); era/exclusion-tag; first-to-prune on storage email | **KEEP (tagged)** |
| 19 | Patched blobs | tick_sequences_v2.1 `_p1` blobs + patch_lineage sha256s + split manifest dual-homed — audit-confirmed good | Same | **KEEP** |
| 20 | corpus_manifest | ~Half of §4 spec: zero match_ids/HLTV ids/sha256s; loose parser pins ('>=0.41.3') | v2 with per-match inventory generated mechanically; exact pins from uv.lock | **CHANGE-AT-V5 #3** |
| 21 | Merge dedup key | `(norm_stem, round_num, first_tick, n_ticks)` (merge_hf_tick_sequences.py:119-134) — works | Add match_id once every row can carry one | **CHANGE-AT-V5 #4** |
| 22 | exam_manifest | Does not exist; blocked on dates + legacy ids | ~30-line generator over processed_manifest + legacy_matches.csv | **CHANGE-AT-V5 #5** |
| 23 | Chimera-dir default | `/home/soone/chimera` (nonexistent) at cli.py:310-312; mirrored in test_process.py:53-57 → only real-builder preflight test silently skips | Fix to `/home/soone/projects/chimera` | **CHANGE-NOW #14** |
| 24 | Single-copy artifacts | WSL-only: 81 parquets (1.1 GB, sole aliasing-audit input), pre-patch blobs (23.7 GB), 5 divergent demos (2.9 GB). HF-only: 412 GB of demos, no mirror | Irreplaceable ~5 GB to free Drive; divergent demos uploaded event-qualified; Zone.Identifier junk deleted | **CHANGE-NOW #12, #13, #18** |
| 25 | Alternate sources / mirrors / infra | None | None — adapters, canary cron, repo split, paid mirrors, 2nd HF account, proxies, systemd/ntfy all rejected with named reversal triggers | **NEVER** |

---

## 4. Failure-mode register

| Failure mode | Likelihood (12 mo) | Blast radius | Current mitigation | Detection time |
|---|---|---|---|---|
| **Cloudflare fingerprint aging** (chrome124 ~27 mo stale) | **HIGH (~60–80%)** — a cliff, not a slope | Total channel death: `_get()` exhausts retries, pass fails with processed=0, loop burns remaining passes at 90-min pace | None proactive; one-line bump available, must not land mid-campaign | **Days-to-never.** Retry path silent by design (hltv.py:93-96); 0 signals in 1.03M lines proves nothing about sub-terminal blocks |
| **HLTV layout drift** | MEDIUM (~30–50%) | Fails loudly in Python (`ScrapeLayoutError`) but loop has no exit-code check → re-hits listing endpoint every 90 min with a known-broken selector | Fail-loud contract + drift fixtures (test_hltv.py:27-79) | Manual log read only; contract defeated at scripts_stier_loop.sh:8-13 |
| **HLTV/Cloudflare ban** | LOW (<10%) at current posture | Total channel death; no alternative S-tier source exists | Conservative pacing; fetch-once-ever. Caveat: dual-commit window converts HF flakes into 1–2 GB re-downloads — the one path that *spends* the channel needlessly | Exhausted-retry RuntimeErrors in an unwatched log |
| **HF quota / moderation** (free tier ~450 GB; ~34% CS:GO dead weight) | MEDIUM (~20–40%) | 76% of corpus becomes re-scrape-dependent; load-bearing 109.7 GB legacy demos share the repo with 302 GB best-effort archive | Cold-mirror trigger armed on storage email (reactive); irreplaceable ~5 GB is WSL-single-copy | HF sends an email — the only failure mode with a built-in pager |
| **venv / environment drift** (fired Jul 19: loop launched ~30s before pip created the entrypoint) | MEDIUM-HIGH (~40%) — already happened | All remaining passes burn ("cannot execute" every 90 min) | **None.** Recovery was operator kill+edit+relaunch (script mtime 10:11:16 = pass-1 timestamp), not self-healing; launcher untracked, pre-fix version unrecoverable | ~97s *only because the operator was watching*. Unattended: until next manual read |
| **Machine off during tournament window** (WSL2, no cron, no remote trigger) | NEAR-CERTAIN eventually | Soft — HLTV archives demos indefinitely; hard only against a deadline. Current loop exits silently at pass 40 with no tally | None | Never signaled |
| **RAR/extractor drift** | LOW (<5%) | Per-match extract failures; after 3 attempts **permanent blacklist** with no clear-on-success (manifest.py:254-260) — a systemic break looks like scattered per-match failures | Multi-tool fallback (download.py:134-170); **zero test coverage** | failures.jsonl banner on manual read |
| **Filter trust failure** | **ALREADY FIRED** | 4 sub-4★ leaks + ~34% CS:GO-era contamination; date_unix null 358/358 | None client-side | Detected only by this audit; no test catches an all-null column |
| **Silent data loss (dual-commit windows)** | MEDIUM per-campaign | Upload-vs-manifest gap: re-download → 3 flakes → blacklist + orphaned HF storage invisible to `process` (iterates demo_mf.entries only, process.py:580). Bake-side twin: archive committed pre-build but parsed_files persisted only on success (process.py:332-334) | Per-match manifest pushes limit loss to one match | Only via a reconcile that doesn't exist yet |

**Cross-cutting problem:** every row except HF quota shares one answer — *we would not notice*. No health check, no NEEDS_ATTENTION sentinel, no end-of-campaign report, and the single best early-warning signal (per-attempt non-200s) is deliberately silent. The meta-lesson runs deeper: a fix flagged "land now, ~6 lines" never landed, and §2-row-1 carried actions went silently incomplete. The structural answer is machine-checked numbers (substrate-coverage %, date-null-rate) in health output — a slipped fix must show up as a red line, not a memory.

---

## 5. Prioritized change list, with costs

**Sequencing gate: nothing lands until PID 31950 exits (~15:30 PDT today). First action after exit: `git add scripts_stier_loop.sh logs/` and commit as-is for the historical record.**

### Tier A — protects data / prevents silent loss (this week; items 1–8 are hard gates before any next campaign)

| # | Change | Cost | What it prevents |
|---|---|---|---|
| 1 | Client-side stars gate in `matches_filters` (cli.py:176-185) | ~1 line + test | Corpus contamination (4 leaks proved `?stars=` untrustworthy) |
| 2 | CS2-era gate on date_unix (`--allow-csgo` escape hatch) | ~5 lines + test | Repeat of the 34%-wasted campaign |
| 3 | Fix dead date_unix selector (hltv.py:149-154) via the already-fetched match page; add non-null fixture + all-null-column guard tests | ~1 h | 358-entry null column surviving 36 passes unnoticed, again |
| 4 | One structured log line per non-200 retry attempt (hltv.py:93-96) | ~3 lines | Fingerprint decay invisible until cliff |
| 5 | Impersonation target → config/env; bump chrome124→chrome142+; quarterly runbook item; pre-campaign preflight fetch | ~30 min | The HIGH-likelihood register row |
| 6 | Replace bash loop with `chimera-demo campaign` (flock, entrypoint preflight, exit-code checks, 2-failure halt, ScrapeLayoutError abort, STOP between matches, end tally + NEEDS_ATTENTION) | ~1 day + tests | Silent pass-burn, double-launch, the exact Jul-19 incident, silent completion |
| 7 | Atomic provenance: manifest line in the same create_commit as content — both windows (cli.py:254-274; process.py:387-398 vs :439-457) | ~half day + mock-HfApi tests | Orphans, 1–2 GB channel re-spends, silent blacklists |
| 8 | sha256: stream-hash at download → ManifestEntry; verify vs LFS OID post-upload; backfill 969 files from OIDs (API reads only); backfill dedup filename → match_id+sha256 (cli.py:405-410) | ~1 day | Makes datasheet.md:24 true; defeats the 5 proven stem collisions |
| 9 | failures.jsonl clear-on-success (`should_skip_failed` ignores failures preceding last success); exclusion tags for 4 sub-4★ + CS:GO matches | ~2 h | 2349630 ghost; 3-blips-permanent-blacklist trap; falsified failure ledger |
| 12 | **Today-safe:** tar + upload the 81 props-v1 parquets (1.1 GB) to HF | ~15 min | Sole aliasing-audit input is a WSL single copy |
| 13 | Upload 5 divergent demos as `demos/<match_id>/<event-qualified-stem>.dem`; delete 4 Zone.Identifier files | ~1 h | Silent tensor changes on disk-loss rebake; completes §2 row 1 |

### Tier B — retires the re-parse cost class (this week, after Tier A gates)

| # | Change | Cost | Effect |
|---|---|---|---|
| 10 | **Land props-v2** (parse_demos.py:41-48: +velocity XYZ, flash_duration, is_scoped, is_defusing, is_walking, spotted, active_weapon) + persist *every* non-empty awpy table generically (parse_demos.py:70-87, process.py:296-302) + `parse_meta.json` per bundle + parse-time column assertion | ~half day | The audit-mandated fix that slipped; **must precede backfill** — otherwise the archive is heterogeneous-by-omission forever |
| 11 | Parse-backfill campaign (`process --parse-only`), HF→HF only, zero HLTV traffic, resumable, nightly: 92 canonical first (~110 GB down, 20–40 CPU-h), then CS2-era; never CS:GO | machine-time | "Re-parse cost class retired": 0% true → 100% true |

### Tier C — provenance and observability (next 2 weeks)

| # | Change | Cost |
|---|---|---|
| 14 | Fix `--chimera-dir` default → `/home/soone/projects/chimera` (cli.py:310-312, test_process.py:53-57) — un-skips the only real-builder preflight test | ~5 min |
| 15 | `legacy_matches.csv`: {stem, match_id, sha256-from-OID, event, date_unix} for the 92 corpus matches (manual HLTV-id lookup for the 22 local-era) | ~half day |
| 16 | Date backfill for 358 dateless entries: Liquipedia event→date first ($0 channel); optional single paced HLTV pass only if per-match precision proves necessary | ~2 h – half day |
| 17 | Health check + NEEDS_ATTENTION sentinel (cron daily during campaigns): heartbeat age, retry-line rate, attempts≥3, manifest-vs-storage orphan diff, date-null-rate, parsed-coverage %, disk/HF GB; init.sh prints sentinel. Plus `reconcile` adopting orphans from LFS OIDs | ~1 day |
| 18 | Gate the HF repo ($0) + datasheet ToS-honesty section; rclone the ~5 GB irreplaceable set to free Google Drive monthly | ~2 h |
| 19 | Tests with all new code: campaign supervisor, commit atomicity, stars/date gates; shellcheck anything shelled out; all launchers/configs in git | rolled into above |

### Tier D — change-at-v5 (land with the next bake, not before)

1. Bake reads ONLY `parsed/` bundles (`--from-parsed` becomes the sole input path) — blocked until canonical backfill coverage is 100%.
2. `REQUIRED_PROPS`/`REQUIRED_TABLES ⊆ parse_meta.json` preflight; refuse loudly with the insufficient-match list and exact re-parse command.
3. corpus_manifest v2: the missing §4 per-match inventory, generated mechanically from processed_manifest + legacy_matches.csv; exact pins from uv.lock; blob meta stamps.
4. Add match_id to the merge dedup key.
5. `exam_manifest.json` generator (~30 lines) — unblocked by Tier C #15/#16.
6. Orchestration coverage: process_one e2e incl. rebake replacement (process.py:616-618) and from_parsed fallback; extractor tool-matrix tests.

### Never-worth-it (with reversal triggers)

Event-ledger rewrite · SourceAdapter/Faceit (reverse: actual HLTV ban) · InboxAdapter now (~50 reactive lines if ever needed) · weekly canary cron · repo split (reverse: actual HF storage email) · paid mirror (reverse: explicit ~$3/yr approval — worth one conversation, not an assumption) · second free HF account (ToS-theater) · proxy rotation/headless browser · deleting CS:GO/sub-4★ demos (tags, never deletion) · migrating 224 legacy demos · systemd/ntfy/dashboards · `.dem` validation in `run`.

---

## 6. Stress test: "50 new S-tier matches in 3 weeks for a rebuttal"

**Raw throughput is not the constraint** — at 114.6 min/pass × 8 matches, 50 matches is ~7 passes ≈ 13–14 wall-clock hours. The strain is elsewhere:

- **Supply (the real bottleneck):** latest pass shows `processed=8 skipped=291 scanned=300` — the back catalog is nearly exhausted within the scan window. S-tier supply is ~10–25 matches/month and HLTV releases demos days after matches; if the window lacks a Major/IEM/BLAST, **50 genuinely new S-tier matches may not exist**, and the pipeline just logs processed=0 and sleeps.
- **Launch:** no lock file (double launch double-hits the channel), hand-edit an untracked script, replay the Jul-19 incident if the venv drifted.
- **Acquisition:** 1.4% stars-leak rate needs manual verification; if chrome124 has aged out, the first symptom is a completed campaign with processed=0; STOP_SCRAPE latency ~2h forecloses fast reaction to Cloudflare pushback.
- **Demos → tensors:** `parsed/` is empty, so 50 matches ≈ ~55–60 GB re-download + 9–18 CPU-h of awpy parsing, landing on a free-tier repo already at ~450 GB — and the archives written would still be props-v1.

**Timeline verdict:** babysat, ~5–8 days end-to-end — comfortably inside 3 weeks, *if the matches exist on HLTV*. Unattended, any register row fires silently and the deadline is gone. That gap is precisely what Tier A closes.