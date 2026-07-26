# Chimera — Round-2 Deep Sweep: Delta Report
**Date:** 2026-07-26 · **Scope:** second adversarial sweep (stats-validity, code-vs-recipe, data-empirical, reviewer-2, security, bridge-path) · **Baseline:** the five completed registers (adversarial-review, corpus-audit, corpus-implementation-audit, first-principles-plan §3+A–G, preflight-report) — nothing below re-reports a registered item. All findings were independently verified; refuted candidates are excluded.

---

## 1. Headline verdict

**GO-WITH-FIXES stands. Phase 0–9 execution order is unchanged.** Nothing found invalidates the locked recipe's *structure* — the internal-controls story (probe firewall, seed pairing, committed failure branches, corpus asserts) re-verified clean at a deeper level than round 1. But the fix list grows by **4 new blockers and ~15 majors**, and they cluster at exactly two seams:

1. **The gate statistics are less pre-registered than the pre-registration claims.** Five of the six decision rules that adjudicate the pod campaign (C1 CI unit, C1-REP statistic, v2→v3 promotion, SS-vs-TF, OOD confound control) have unpinned or self-contradictory adjudication text. All fixes are doc-level and must land in the knobs docs **before runbook [6]** — after results exist, they are unfixable honestly.
2. **Phase 3 (bridge/GRPO) is design-rich but decision-poor.** The reward has no executable spec, the group semantics contradict across docs, no step builds the trainer, and the decisive external control (state-as-text) is absent. These must close **before runbook [7]** and the 35B spend.

Two state-corrections to the record:

- **The [1] DONE status was never GPU-validated end-to-end** — the post-patch trainer crashes at its first CUDA eval (see R2-B1). "DONE" claims involving the trainer should require a CUDA smoke, not just CPU CI.
- **The >700MB blob guardrail is stale as written**: every val-side `_p1` blob is 1.77–2.04GB. Round 2 followed the guardrail's intent (val-only, one at a time, mmap streaming, peak RSS 2.4GB). Recommend rewording the runbook guardrail to "val-side blobs only, one at a time, `torch.load(mmap=True)`; train blobs (3–10GB) forbidden."

---

## 2. NEW blockers and insertion points

| ID | Finding | Insertion point |
|----|---------|-----------------|
| **R2-B1** | `evaluate()` crashes every CUDA run: `won` stays on CPU and is indexed with a CUDA mask (`train_world_model.py:357`, introduced by the O3-fix commit 2f7ebe7; empirically reproduced). Pilot and all 7 pod runs die at step 500. `--smoke` forces CPU, CI never calls `evaluate()` — invisible until pod money is burning. | Fold the one-line fix into the **[3] consolidated trainer edit**; add a **30-step CUDA smoke** to the pre-pilot checklist (new gate before the Phase-B local pilot). |
| **R2-B2** | C1 gating CI is at the wrong correlation unit: primary bootstrap resamples 641 rounds, but the project's own doctrine makes **match** the correlation length and val holds **14 matches**. The match-cluster CI is demoted to "robustness" with no disagreement rule — the keystone gate can pass on match-level noise. | Doc edit to knobs4-7 §7c/§7e **before [6] launch** (must be locked before any run its gate adjudicates). Promote the cluster CI to gating, or pre-register the conjunction "both CIs must exclude 0" + a 14-cluster coverage caveat (BCa/percentile-t). |
| **R2-B3** | No **state-as-text baseline** anywhere in C2/C3: no arm gives frozen Qwen the same state as plain text and scores it on value-agreement/fact-audit/CRPS. The latent-off arm gives Qwen *nothing*, so "latent-on ≫ latent-off" can pass while "paste the parsed state into the prompt" matches the whole bridge. Cheapest decisive control; prompting-only. | Pre-register as a co-equal eval arm in bridge-design §3 + runbook **[7]**, before any bridge pod spend. |
| **R2-B4** | Grounded-GRPO reward has **no executable spec** (no claim schema, no extractor, no CRPS distribution semantics, no ICC replicate protocol) while same-named VLM-era decoy instruments (`eval_scorer.py` GATE_THRESHOLD=0.70, `build_pseudo_gold.py`, `data/eval/pseudo_gold_stub.jsonl`) survive outside B12's archive list — a naive session can "pass" the checker gate with the wrong instrument. | Two parts: (a) **Phase 0 addendum** — extend B12's archive list with the two scripts + `data/eval/`; (b) write the one-page pre-registered claim schema **before [6]** (per amendment F's own sequencing), hard-gate before any Phase-3 work. |

---

## 3. Spec holes to close before the step that consumes them

### Before [1b]→[2] (p2 patch → edge fit)
- **D8 datasheet entry + end-phase cap** (data-empirical): r12 half-boundary rounds carry up to 60s of dead halftime frames; `bomb_age` overflows its /40 bound to 2.27× on exactly those rounds; r24 rounds truncated to 0s end phase (non-boundary rounds: exact 7.0s pad). `bomb_age` appears in zero registers. Fold the crop (round_end + 7s) and bomb_age clamp into the **[1b] p2 patch** — [2]'s quantile fit otherwise ingests the tail frames. Add D8 to the datasheet either way.

### Before [3] (consolidated trainer edit)
- **Knob 5d mask collision errata** (major): the locked text pins the dist-loss mask to "same as the edge fit" (alive-only, freeze-inclusive) while D3 and current code require freeze exclusion — the checklist formula applied verbatim silently re-admits 17.5% freeze frames into the dist CE. One-line pre-registration errata: mask = alive(t) ∧ alive(t+k) ∧ ¬freeze(t); note the edge fit's stationary% is freeze-inclusive (interior edges unaffected).
- **R2-B1 fix** (above) in the same edit.
- Hygiene: all eight line anchors in the knobs4-7 edit checklist are stale post-[1]; refresh before an implementer follows them.

### Before [6] (pod campaign) — all doc-level, one sitting
- **C1-REP statistic pinning** (major): one paragraph in 7e — gate value = mean over 3 model × 5 probe seeds of pooled AUC; bootstrap recomputes that mean; reconcile the Knob-6 "both baselines, same sign, 3 seeds" clause (12.5% floor FPR — state it) with 7e's max-of-four rule; name the wd-selection metric; commit the probe-select split manifest.
- **v2→v3 canonical-promotion rule** (major): the adversarial review's T3 fix ("gate promotion on CI excluding zero") was never written into either recipe doc — "if v3 wins" has no metric/threshold. Add to Knob 2/6: promoted iff per-seed paired delta > 0 in all 3 pairs AND seed-mean ≥ +0.02 with cluster-paired CI excluding 0; else v2 stays canonical regardless.
- **SS-vs-TF gate** (major): currently 1 control seed vs 3 SS-on seeds, 10 uncorrected per-map cells, no statistic — and the "drop SS" branch contradicts the pre-declared SS-on canonical checkpoint with no budgeted TF rerun. Pre-register the paired per-map statistic + aggregation, and pin the branch (SS-off applies to reporting; canonical stays as declared; TF matrix listed as budgeted contingency).
- **OOD zeroed-ID control criterion** (major): "degrades beyond bootstrap CI" is 25 CIs with no aggregation rule — fires trivially or never. One sentence: paired per-round minADE-16 at depths {10,20}; confounded iff ≥2/5 maps' paired CIs lie entirely on the degradation side.
- **OOD probe-transfer scope** (major): 64 val rounds from ~2–3 series ⇒ SE(AUC) ≈ 0.06; +0.02–0.05 deltas are invisible, and the 6-point curve has no rule for which point carries the claim while the claim-scope paragraph already asserts the win. Pin the claim to n=303, paired delta with CI printed, labeled reported-not-gated; soften the claim-scope sentence (or buy the budgeted second holdout arm).
- **Power/MDE afternoon** (minor, CPU-cheap): no power analysis exists for any margin; C1-SCALE's +0.01+CI construction has an effective bar of ~2×SE ≈ 0.02+, so its failure branch likely fires from underpower. Simulate paired-bootstrap MDE at 14-match clustering; record MDE next to each gate; pre-commit the +2-seed escalation if C1-SCALE MDE > 0.01.
- **RE fixture bypass** (minor): the 7a fixture must reproduce v6 on the OLD blobs, but unconditional `clean_blob` drops ~14% of their rounds — fixture fails spuriously and blocks all clean RE runs. Add an explicitly fixture-scoped `--no-clean` (printed loudly) + one-line amendment.

### Before/at Phase 1.2 (events reparse) — decision line needed NOW
- **Train-side event ground truth** (major): kills side-files exist for ~22/92 matches; the sequenced reparse is val-only, but GRPO/ReST prompts must be train-side (and post-1.2, val would be the *only* split with event files — a naive implementation would draw prompts from val and contaminate every gate). Decide and record: (a) extend the 1.2 reparse to all 92 matches with the same tooling, or (b) pre-register that prompts come only from event-covered train matches and verify that subset supports the prompt count.

### Before [7] (bridge/gates/GRPO)
- **R2-B3, R2-B4** (above).
- **Head-Jacobian recon target** (major): amendment D's PRIMARY target is a phrase, not a spec — no tensor, no k, no loss, no tap-layer interaction; nothing computes it. Write it as math (per-sample value-head Jacobians + dist-head row space, stacked over a fixed val draw, SVD, pre-registered k; cosine+MSE on projected coords; recomputed at the tap-swept layer) and extend `nla_capacity_probe.py` to emit it.
- **GRPO group semantics** (major): first-principles says "the 16 rollouts are the group"; bridge-design says the group is G=16 completions. One-paragraph decision (recommended: G=16 completions from an identical prompt; K=16 rollouts enter only as foresight channels + the reward's CRPS reference set) → decisions-ledger.
- **Phase-3 trainer build line + budget** (major): no runbook step builds the grounded-GRPO trainer/ReST harness; infra-plan's killed-row ("existing scripts stand") flatly contradicts preflight B12 (archive them); the vLLM throughput hope is dep-blocked and G=16 quadruples measured generation cost. Add an explicit runbook line (manual loop: soft-prefix generate + claim scorer + recon-τ + KL-to-SFT, CPU-smokeable, with measured sec/step → $ budget before go); resolve the infra-plan/B12 contradiction in whichever doc loses.
- **Readability leg** (major, reviewer-2): the only gate leg in the program with no metric/threshold/protocol, and steganography kill-criterion #3 depends on it. Pre-register: perplexity ratio vs base-Qwen with a numeric band + a small blinded rubric (fixed n, majority rule).
- **Decoder-side semantic-sensitivity probes** (major, reviewer-2): falsified-text (fact-flip must move recon) + paraphrase-invariance (must not) exist only as an unfireable trigger phrase; schedule as a fifth co-equal CHANGE-E leg (CPU-only, reuses the trained decoder), pass rule |Δrecon(flip)| ≫ |Δrecon(paraphrase)| ≈ 0.

---

## 4. Reviewer-2 open holes: runbook line vs consciously-accepted risk

**Recommend a runbook line (cheap, decision-changing):**
- **State-as-text baseline** — blocker, [7] (above). Non-negotiable.
- **Decoder-side probes + readability pre-registration** — [7] CHANGE E additions (above).
- **MLMove disposition** — decide before [4]/[6]: write the incomparability paragraph (action-conditioned retake control in CS:GO vs open-loop multi-agent forecasting in CS2) into OUTLINE + T3 notes, and *optionally* add one cheap learned baseline column (per-player marginal head on the same corpus) so T3's coverage headline doesn't beat only straw physics. Recommend: paragraph mandatory, learned column decided (yes/no) and recorded now.
- **Joint-coherence interaction control** — arm the cheap trigger alongside [4]: evaluate the trained trunk with cross-player attention masked (or per-player marginal sampler) on coverage; if gap ≈ 0, soften the "10 coupled agents" wording before a reviewer finds it. Given the confirmed facing-shortcut, this risk is live, and the control is inference-only.

**Consciously-accepted as paper-writing-phase work (ledger note, no runbook line):**
- **Othello-GPT / Chess-GPT citations + C1 delta statement** — pure related-work; but treat as mandatory writing work: without the "continuous multi-agent, measured-ceiling extension" sentence, C1 invites a one-citation novelty reject.
- **C3 repositioning vs arXiv 2505.17989** — strike "rare/unique" wording in first-principles :31/:113/:114; restate the delta (world-model-generated groups, dense 8Hz simulatable future, recon-τ constraint). Do it when amending the knobs docs anyway, since :113 is falsified *as worded* in a locked doc.
- **Patchscopes/LatentQA/SelfIE paragraph** — minor; fold into the same related-work pass; position the gate battery (not the bridge mechanism) as the contribution.

---

## 5. Security disposition

No secret has ever leaked into either public repo or the committed scrape log (full-history scans clean, both repos, all refs). Three posture items, none registered:

1. **Pseudonym break (major, act before visibility grows):** `davidzengming@gmail.com` is author/committer on ~all commits in both PUBLIC repos, and `git config user.email` still points at it — the leak recurs every commit. Decide identity posture explicitly. If pseudonymous (everything else says yes): set user.email to the GitHub noreply, enable GitHub's block-exposed-email push protection, `git filter-repo --email-callback` both histories. This compounds with item 2: a real-name-shaped identity attached to a repo publicly documenting an HLTV scraping campaign.
2. **Committed 54MB scrape log (minor):** `logs/stier_scrape.log.gz` publicly documents 40 scrape passes against hltv.org (+ local username paths). Note: corpus-implementation-audit.md:94 *prescribed* this commit without weighing repo publicity — amend that register line rather than treating it as settled. `git rm --cached` + gitignore `*.log.gz` + filter-repo purge; keep the log locally.
3. **Live Anthropic key in world-readable `.env` (minor, local hygiene):** never committed (verified), but perms are 644 in a directory operated on by automated agents and upload tooling. `chmod 600`, move out of the repo tree, rotate if ever pasted elsewhere.

Do items 1–2 in one history-rewrite session (both require filter-repo; one force-push event, coordinated).

---

## 6. Coverage map — what round 2 checked and found CLEAN

*"Nothing found" claims below are auditable against the hunter clean-bills.*

**Statistics/pre-registration:** bootstrap construction coherent as far as it goes (paired percentile, within-resample deltas); linear-probe protocol pinned and reproducible (fit code, wd grid, feature layers, manifests, pooled-metric rule); ≥4/5-maps clause is conservative, secondary rows carry no silent alpha; G0∧C1-REP∧C1-SCALE firewall + selection-on-val-ns-only is a genuine anti-circularity design; no proper-scoring misuse on the distributional head (CRPS/Brier correctly scoped to the future checker; cross-arm edge-set comparability anticipated); Knob-6 budget/escalation rules crisp; OOD decode mechanics exact (zeroed dims by index, loader asserts); seed machinery sound; corpus counts consistent everywhere (3,876/705 → 3,573/641; 367 = 303+64).

**Code-vs-recipe:** Knob 5d loss book conforms (weights, classify-then-refine, 97 classes, edge-fit rule); end-phase mask in both value BCE and eval AUC with verified column arithmetic; `test_no_value_leak` tests the real gradient mechanism; `clean_blob` lockstep covers every per-round parallel list in every blob schema; freeze-mask sign/broadcast correct; D4 plant-gating landed and derived dims input-only/zeroed in both residual branches; SS building blocks in place (generator hook, cv-only gap correction); seed wiring conforms; all registered-pending [3] items confirmed still pending, none silently divergent.

**Data (empirical, val blobs via mmap):** round counts exact (770 raw = 705 clean = 641+64); overpass holdout present, de_train absent; map one-hots clean; bomb bits strictly one-hot, monotone, 0-frame aligned with post_plant; clock re-anchoring exact (sinusoids to 3e-5); v3 dim7 plant-gating recomputes exactly; zero NaN/Inf; feature ranges as documented; D7 flag rule reproduces 147/770 exactly; 0 resurrection rounds, monotone phase sequences, physically coherent spot-checks; bomb_age internally consistent; split hygiene clean (val = exactly the 14 manifest matches).

**Reviewer-2:** C1's internal baseline battery complete (six representations, matched-capacity ceiling, twin, rand_wm); physics baselines + amendment-B fairness registered; BC-as-approach defense recorded; novelty positioning already sound on the world-model/distributional/probing-mechanics/bridge-mechanics axes; generalization scope in OUTLINE §6 matches evidence exactly, no over-claim in live docs; NLA encoder-side + anti-gaming controls comprehensive (except the decoder-side gap reported); human-relevance claims correctly scoped; pre-registration discipline (failure branches, venue rule, anti-fabrication) reviewer-proof as written.

**Security:** full git-history secret scans clean (both repos, all refs, all patterns); scrape log secret-free; `.env` never tracked; pod scripts/config/CI/fixtures clean; manifests leak nothing; tracked progress file clean (dead pod IPs only).

**Bridge-path:** Qwen3.6-35B-A3B availability verified in-world; TRL-bug manual-loop workaround actionably recorded; VRAM arithmetic sound *conditional on 4-bit QLoRA working* (note: that path has never been run — cheap load-smoke it before relying on it); NLA firewall precisely implemented; recon-τ constraint fully specified with starvation fallback; Interface A matches design (foresight channel genuine); anti-gaming battery implementable; NLA kill-criteria pre-registered; non-circular SFT generator exists; amendment-D referees correctly sequenced pre-pod; Knob-7 probes do not hidden-depend on missing kills files.

---

## 7. Amended phase checklist lines (inserts/changes only)

```
[1b] p2 patch — ADD to patch scope:
     + crop end-phase frames beyond round_end + 7s (r12 halftime tail; r24 truncation disclosed)
     + clamp bomb_age at explosion/defuse (cap at 1.0 normalized)
     + add D8 to docs/datasheet.md (r12 tail + bomb_age overflow + r24 truncation, quantified)

[pre-3] KNOBS ERRATA (one commit, before the [3] edit):
     + Knob 5d mask errata: dist CE/refine mask = alive(t) ∧ alive(t+k) ∧ ¬freeze(t);
       note edge-fit stationary% is freeze-inclusive (interior edges unaffected)
     + refresh stale line anchors in knobs4-7 edit checklist

[3]  consolidated trainer edit — ADD:
     + fix evaluate() device bug: move `won` to device before `won[keep]` (train_world_model.py:357)
     + fixture-scoped --no-clean flag (loud print) so the 7a RE fixture can load old blobs; one-line amendment

[3b] NEW GATE (before Phase-B pilot): 30-step CUDA smoke incl. one evaluate() pass — CI is CPU-only and
     cannot catch device bugs

[4]  ADD (cheap, inference-only): cross-player-attention-masked (or per-player marginal) decode control on
     coverage + trajectory-coherence; if gap ≈ 0, soften "joint coherence" wording in recipe + paper

[pre-6] PRE-REGISTRATION EDITS (all doc-level, one sitting, before any pod run):
     + 7c/7e: promote match-cluster CI to gating (or pre-register round∧cluster conjunction) + 14-cluster
       coverage caveat                                                              [R2-B2]
     + 7e: pin C1-REP statistic (seed aggregation, probe-seed entry, comparator set, sign clause, wd
       metric); commit probe-select split manifest
     + Knob 2/6: v2→v3 promotion rule (all-3-pairs sign + seed-mean ≥ +0.02 + cluster CI excludes 0)
     + Knob 6: SS-vs-TF paired per-map statistic + aggregation; pin drop-SS branch (reporting-only for R1;
       TF matrix = budgeted contingency)
     + Knob 4: zeroed-ID control criterion (paired minADE-16, depths {10,20}, ≥2/5 maps degrade)
     + Knob 4: OOD probe claim rides n=303 point only, reported-not-gated; soften claim-scope sentence
     + power/MDE simulation (CPU afternoon); record MDE per gate; pre-commit +2-seed escalation if
       C1-SCALE MDE > 0.01
     + write grounded-GRPO claim schema (types, extractor, CRPS semantics, ICC protocol)   [R2-B4]
     + Phase-0 addendum: archive eval_scorer.py, build_pseudo_gold.py, data/eval/ (B12 extension) [R2-B4]

[1.2/Phase 1] DECISION LINE: extend events-reparse to all 92 matches OR pre-register GRPO/ReST prompt-set
     restriction to event-covered train matches (with count check) — record in decisions-ledger

[7]  bridge/gates — ADD:
     + state-as-text baseline arm: frozen Qwen + templated textual state (same frames/history), scored on
       value-agreement / fact-audit / CRPS, co-equal with latent-on/off/shuffled          [R2-B3]
     + head-Jacobian target written as math + nla_capacity_probe.py extended to emit it
     + GRPO group-semantics decision paragraph in bridge-design §5 (recommended: G=16 completions,
       identical prompt; K=16 rollouts = foresight channels + CRPS reference set) → decisions-ledger
     + CHANGE E leg 5: falsified-text + paraphrase-invariance probes on the trained decoder
       (pass: |Δrecon(flip)| ≫ |Δrecon(paraphrase)| ≈ 0)
     + readability leg pre-registration: perplexity band vs base-Qwen + blinded rubric (fixed n)
     + NEW LINE between CHANGE F and on-policy GRPO: build grounded-GRPO manual loop (soft-prefix
       generate + claim scorer + recon-τ + KL-to-SFT), CPU-smokeable; measured sec/step → $ budget
       before go; resolve infra-plan §3 killed-row vs preflight B12 contradiction

[security, any time before repos gain visibility — one history-rewrite session]:
     + decide identity posture; if pseudonymous: noreply user.email + GitHub email-block +
       filter-repo email rewrite (both repos)
     + git rm --cached logs/stier_scrape.log.gz; gitignore *.log.gz; purge from history; amend
       corpus-implementation-audit.md:94
     + chmod 600 chimera/.env; move key out of tree; rotate if ever pasted elsewhere

[guardrail hygiene]: reword blob guardrail — "val-side blobs only, one at a time, torch.load(mmap=True);
     train blobs forbidden" (all val _p1 blobs exceed the old 700MB figure)
```