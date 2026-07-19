# Infra Plan (2026-07-19)

This is the canonical home for infrastructure decisions taken at the runbook-[1] boundary: what gets built before the corpus patch, what waits behind named triggers, and what is rejected. It synthesizes five red-teamed dimensions (blob-format, tracking, native-accel, config-runs, pod-automation/test-ci) and the sheriff's verdicts of 2026-07-19. The bar every item passed: *materially harder to add later* or *accelerates a known runbook step* — never "nice to have."

**One cross-cutting correction before the lists:** three dimensions independently claimed lineage/manifest work inside [1]. It is ONE artifact. The sha256 pre/post is computed once, written once into `corpus_manifest.json` by `patch_corpus.py`, and consumed three ways: by the HF push, by the provenance stamp, and by `check_lineage()`. Same for tests — the patch validator and the permanent invariant suite are the same functions in the same file. Budget it once.

**The organizing asymmetry:** the container format on disk is *not* materially harder to change later (mmap'd `.pt` already solves the only measured pain, and `load_corpus()` makes any future swap a 1-file change) — but metadata written into the blobs, provenance stamped into checkpoints, and metrics emitted during the $45–85 pod runs **cannot be recreated after the fact**. So: change the loading discipline and the embedded metadata now; change the bytes-on-disk never — until a rewrite happens for other reasons.

---

## 1. Pre-[1] setup (execution order — one focused day, ~5–6h + one ~19 GB upload)

Sequencing is load-bearing: the validator and `load_corpus()` are written FIRST, then `patch_corpus.py` imports both. The validator must pre-exist the mutation (it defines "patched correctly" against pre-patch state — written afterward it can only tautologically bless whatever the patch produced); `load_corpus()` must pre-exist so post-patch verification reads blobs the same way every consumer will.

| # | Item | What / why-now | Cost | Done-check |
|---|------|----------------|------|------------|
| 1 | **`_corpus.load_corpus()` + convert 8 non-mmap readers** | Add `load_corpus(path)` to `scripts/_corpus.py` — `torch.load(map_location="cpu", weights_only=False, mmap=True)` + `clean_blob` — and convert the 8 readers that load fully into RAM. Priority order: `fit_dist_edges.py:37` (step [2] is next), `train_world_model.py:412–413` (steps [5]/[6]), then the six eval scripts (`eval_world_model.py:78`, `rollout_eval.py:222`, `dist_coverage_eval.py:72`, `inspect_features.py:76`, `value_probe.py:132`, `vq_killtest.py:57`). Kills the 22 GB WSL2 ceiling with ~10 one-line edits; 6 scripts already prove mmap works on these exact blobs. `--maps` filter and `clean_blob` only rebuild python lists, never mutate tensor storage — mmap-safe. Centralizing makes any future format swap a 1-file change instead of 14. | ~2h | `fit_dist_edges` and a trainer smoke run without the quiet-machine ritual; grep shows no remaining bare `torch.load` on corpus blobs outside `_corpus.py` and `patch_corpus.py` |
| 2 | **`tests/test_corpus_invariants.py` + three cheap tests + strict xfail + CI witness prep** | The six check functions, written BEFORE the patch: `check_bomb_bits_consistent` (bomb bits agree with per-player has_c4 dim 14 / bomb_x,y; site from position, never the broken label), `check_round_time` ([0,1], re-anchored at freeze→live), `check_dim7_plant_gated` (v3), `check_no_nan_inf`, `check_match_ids` (set == 92 `split_manifest_v2.json` keys, every meta stamped), `check_lineage` (script sha + pre/post sha256 + date). Plus a toy-blob patch→mmap-load round-trip (absorbs blob-format's pytest — same file). Plus: `test_clean_blob.py` (synthetic blob, lockstep list filtering, ms), `test_schema_dims.py` (`feature_schema_v1.json` vs code constants, 597 == 10×56+37 — the test that would have caught D1/anubis), `test_model_smoke.py` (build at real dims, dist=True, one forward+backward, ~1–2 min CPU — the regression net under [3]'s detach/SS surgery). Mark `test_no_value_leak` `@pytest.mark.xfail(strict=True)` so the [3a] red→green flip mechanically forces marker removal in a CI-witnessed commit. Blob-dependent tests skip via fixture when `data/processed/tick_sequences/*.pt` absent. | ~1.5h | `pytest -q` green locally (xfail expected); invariant functions importable by name from `patch_corpus.py` |
| 3 | **`scripts/patch_corpus.py`** — runbook [1] itself, carrying the once-counted artifact | Loads WITHOUT mmap (it mutates dims), writes to NEW filenames (e.g. `train_v2m_p1.pt`, default torch.save zipfile so mmap keeps working) — satisfying the snapshot precondition structurally instead of by discipline. Stamps lineage + `match_id` into blob/metas, **writes `corpus_manifest.json`** per the spec already at corpus-strategy §4 (the unimplemented half of a locked requirement, not new scope), then imports and runs the invariant checks post-write. | Runbook scope | All six invariant checks pass on the patched blobs; manifest contains pre/post sha256, patch lineage, split-manifest hash |
| 4 | **`.github/workflows/ci.yml` + delete stale `config/config.yaml`** | On push/PR: setup-uv, CPU torch wheel, `uv run pytest -q` + `ruff check`, ~3–5 min, $0 forever (public repo). This project's specific documented pathology is checks-that-exist-but-never-run — pre-commit was already configured when the bomb-bit bug shipped, so local hooks alone fail the project's own history test. CI green certifies the *code*, not the corpus (runners can't see 22 GB blobs) — do not read it as more. Archive the VLM-era `config/config.yaml` so it can't be mistaken for live config. | ~35 min | First green run visible on GitHub; `config/config.yaml` gone or moved to `archive/` |
| 5 | **`scripts/push_blobs_hf.py` / `pull_blobs_hf.py`** | Push patched v2m/v3m train+val + `corpus_manifest.json` to `skkwowee/chimera-cs2` (e.g. `tick_sequences_v2.1/`); pull side asserts sha256 from the manifest so a pod refuses to train on a stale/truncated blob. One motion = offsite copy of the single-copy asset + canonical pod input for all 7 [6] runs + integrity check. Push runs as [1]'s final done-check line. | ~1h + upload wall-clock | `pull_blobs_hf.py` on a clean dir reproduces local sha256s |
| 6 | **Checkpoint naming paragraph** in `retrain-recipe.md` | HF model repo `skkwowee/chimera-wm`, layout `runs/r1-{v2\|v3}-s{0,1,2}/` plus `r1-v2-ssoff-s0`, `r1-re-v6`, `r1-sup-ceiling`; each dir: `best.pt`, `best_ns.pt`, `train.log`, `run_meta.json`. Naming-in-doc costs nothing; naming-after-the-fact means renaming artifacts mid-analysis. | ~10 min | Paragraph committed |

Nothing else is pre-[1]. In particular: metrics emit, provenance stamping, pod scripts, and the W&B mirror were all argued into [1] by their respective dimensions and all belong to the free [2]–[5] window (the only [1]-coupled field any of them needs — blob sha256 — comes from the manifest).

---

## 2. Scheduled and deferred (with triggers; none execute now)

**Scheduled — the [2]–[5] window** (GPU-bill-free; building these during [6] at per-second billing is the documented anti-pattern):

| Item | Trigger / deadline | Cost | Notes |
|---|---|---|---|
| **jsonl metrics emit + `results/` registry + scp-before-stop-pod** | Before first [5] smoke run | ~2h | `outputs/<run>/run.json` (argv, args, seed, git sha, config hash, blob sha256 from manifest) at start; one json line per eval block to `metrics.jsonl` (same numbers as the existing print f-string, ~20 lines stdlib); copy both to `results/<config8>-s<seed>-<date>/` and commit — kilobytes, this IS the paper's run registry. Add the scp of these two files to `run_*_pod.sh`/`wait_and_stop_pod.sh` BEFORE stop-pod (verified: today nothing is pulled back — pod stdout dies with the pod, and the [6] curves ARE the paper). |
| **`scripts/_provenance.py` + `--require-clean-git` + env-freeze** | Before first [5] smoke, so every [6] ckpt carries it | ~1.5h | {git sha, dirty flag, cmdline, config_sha256, manifest version + blob sha256, torch/cuda, seed} into ckpt meta + `outputs/<run>/config.json`; `uv export` freeze into the run dir (the 15-minute docker substitute). A git sha absent from a checkpoint cannot be attached retroactively — the retrofit is re-running the pods. |
| **justfile — paired-run targets only** | Before [6] | ~45 min | `train-v2`/`train-v3`/`train-ss-off` differing by exactly one flag, SEED the only parameter — makes "bit-identical pairs" a property of tooling, not attention. Encoding [1]–[2] as targets is transcription; add only if free. |
| **`pod_setup_wm.sh` + `run_wm_pod.sh` + `push_ckpt_hf.py`** | [3] trainer flags exist | ~0.5 day | Copy the GRPO stack (CUDA-match check minus bnb; `wait_and_stop_pod.sh` unchanged; nohup launcher); final steps = upload run dir to HF then watchdog-terminate. SSH dry-run on the existing EXITED pod before any paid run. |
| **Optional `--wandb` mirror** | First [5] smoke; jsonl stays sole source of truth | ~1h | Scalars only (MBs vs 5 GB free cap, $0, no-paid-services intact); `WANDB_MODE=offline` + sync on pods. The connected MCP makes cross-run query conversational — but connectedness is not a forcing function, and W&B's 2026 tier restructure makes "never the only copy" mandatory. |
| **Local cron cost-guard** (kill any pod alive > N h) | First "user go" on pod spend ([5]→[6] gate) | ~30 min | Covers the one failure the pod-side watchdog can't: pod provisioned, run never started. Fails "harder later" as a now-item — trivially addable at the gate. |
| **Batched-K rollout design** | Design constraint on writing [4] — not a task | ~$0 | Shape buffers `[K, L, F]` with K in the batch dimension from line one. This code becomes the GRPO group generator; batch=1 written today is the one loop that WOULD get retrofitted. |

**Deferred behind evidence triggers:**

| Change | Trigger | Cost class | Notes |
|---|---|---|---|
| **safetensors / per-match shards + manifest** | Next corpus write that is already a full rewrite (v4/v5 bake, full re-parse, or routine streaming to pods) | ~2h at that point | Mechanical, not archaeological, precisely because [1] stamped `match_id` into every meta and `load_corpus()` is the single call site to swap. Not before: a 14-consumer format migration is maximum-subtle-bug surface at the minimum-tolerance moment. |
| **embreex raycaster swap** (awpy `is_visible` → batched embree) | Any full re-bake trigger firing (v5 bake, anubis rebuild, 16Hz ablation) | ~0.5 day + ≥99.9% agreement cert vs awpy on sample matches | Days→minutes for the raycast phase. NOT before [1]: a changed LOS backend would contaminate the patch-vs-rebake diff that certifies the v2.1 builder. Arm it now with one line on the datasheet's re-bake TODO so a future session doesn't default to days of pure-Python raycasts. |
| **Trackio swap-in** | W&B free-tier terms tighten | ~0 (one-line import; API-compatible) | HF-native, local-first, zero ToS ambiguity — loses only the MCP conversational layer. |
| **Gate-thresholds-as-tests** ([7] keystone/OOD) | [4]'s minADE-K output schema stabilizes | ~1h | Encode the pre-registered knobs-4–7 thresholds as a test reading `outputs/`, skip-if-absent. |
| **Weave / trace logging** | [7] GRPO debugging needs per-generation reward inspection | ~1h | Not before. |
| **Docker image for pods** | `pod_setup` fails again, or setup repeatedly >~30 min | 0.5–1 day | Hardened script + template pin + per-run env freeze covers the known 14h failure mode. |
| **Hydra/OmegaConf or any config framework** | Second contributor, or run count >~30 with composed configs | 1–2 days | 7 pre-registered runs from locked knobs need argparse + a config hash, not a sweep engine. |

---

## 3. Killed (with one-line reasons)

| Killed | Reason |
|---|---|
| Arrow / WebDataset / zarr corpus migration | All three fight ragged random-crop access; multi-node streaming formats at 10 GB single-box scale; trainer rewrite before the existential test. |
| Custom Rust/PyO3 raycaster | Reimplements embree worse; textbook instance of this project's documented scope-creep failure mode. |
| Rust/native rollout work | The loop is GPU-kernel-launch-bound; PyTorch batching solves it for free. |
| Rust/wasm viewer | 10 canvas dots at 8 Hz is nowhere near any limit; static HTML+JS is already right. |
| MLflow | A long-running server on a RAM-capped WSL2 box that already kills CLI processes; dominated by jsonl + optional W&B. |
| W&B Artifacts / model registry | HF hub is the artifact store (decided); 10 GB blobs blow the 5 GB free cap immediately. |
| W&B Sweeps | 7 pre-registered runs from locked knobs is the opposite of a sweep. |
| typer rewrite | Purely cosmetic. |
| SkyPilot | Its RunPod backend does not support autostop — the single feature this project most needs from it. |
| `requirements.txt` torch pin as a work item | uv.lock is the resolver of record and pod_setup pins independently; a 2-minute cosmetic edit, not infrastructure. |
| Nightly CI, coverage %, codecov, hypothesis, Python matrices | No failure mode in [1]–[7] that these catch. |
| GPU / integration CI | Impossible on free runners; redundant with [5]'s local smoke. |
| Local pre-push hooks as the sole guardrail | Pre-commit was configured when the bomb-bit bug shipped — skippable "automatic" is not automatic. |
| `weights_only=False` remediation | Self-produced blobs; pickle security is not a forcing function here. |
| Pod-side fixes for `weights_only`, GRPO script refresh, RunPod MCP auto-provisioning | [7] is gated behind [6]; existing incident-hardened scripts stand. |

---

## 4. Field practice notes (what actually transferred)

**Transferred cleanly:**

- **The Anthropic harness pattern** (engineering blog; already adopted here as `init.sh`/`feature-list.json`/`claude-progress.txt`) transferred twice more: the justfile is "canonical commands as executable artifacts," and CI is the mechanical form of "verify a feature before marking passes=true" — verification made non-optional. The pod scripts' codify-each-incident-as-a-check style is the same ethos.
- **torch.load(mmap=True)** is the documented low-memory path in PyTorch's serialization docs and exactly what six scripts in this repo already do — the recommendation is standardization, not adoption.
- **jsonl-per-run + pandas** as the solo empirical-research tracking baseline (Alignment Forum "Tips and Code for Empirical Research Workflows") transferred verbatim: plain-text run records, git as the registry, dashboards optional.
- **Commit-hash + full-config in every run record** (standard reproducibility checklists; MLXP builds exactly this) transferred as `_provenance.py` — extending the `vars(args)` pattern the trainers already have.
- **sha-pinned data manifests + HF-as-artifact-store** — the repo already stages via HF to escape the AP-JP-1 volume lock; the manifest just makes the pull verifiable.

**Did not transfer (and why the field's default was wrong here):**

- **safetensors everywhere.** The HF/community case (BLOOM 10 min→45 s) is about *distributed model weights*. For a private, single-machine, ragged intermediate corpus, practitioners keep whatever mmaps and version the loader. Adopting it now would be format churn at the pre-registered-recipe moment.
- **WebDataset/FFCV-style sharding** earns its keep at 50 TB multi-node streaming; at 10 GB single-box it is overhead without payoff.
- **W&B as default tracker.** Dominant in small labs, but the 2026 tier restructure (Free/Pro $60/Enterprise, metered storage) plus known-clunky full export means it can be a mirror, never the source of truth. The connected MCP server is a genuine convenience — and still not a forcing function.
- **Hydra.** The big-lab standard assumes configs that compose across many experiments; MLXP's own motivation notes Hydra is not experiment management. Solo scale with locked knobs: argparse + config hash.
- **SkyPilot.** Canonical for multi-cloud labs, but its documented RunPod caveat (no autostop) disqualifies it for a RunPod-only project whose worst incident class is idle billing.
- **Garcon-style remote-model tooling** (Anthropic interpretability): the concept — never debug on the expensive machine, build and dry-run tooling before the run — transferred as *discipline* (SSH dry-runs on the EXITED pod in the [2]–[5] window), not as software; a 19M model needs no remote-introspection layer.

---

**Bottom line.** Pre-[1] is ~5–6h + one upload, and every item either is [1]'s own locked scope (`patch_corpus.py`, manifest, validator) or removes an irreversibility (mmap for the 22 GB ceiling, CI witness, offsite blob copy, naming convention). The packet's own real creep — caught and cut — was triple-counting the manifest, front-loading tracking/provenance into [1], and the W&B mirror riding in on "the MCP is already connected." Everything expensive waits behind a written trigger, same as the corpus.