# RunPod runbook (Chimera)

Operational guide for provisioning/driving RunPod GPUs for the Phase-2 bridge SFT
and Phase-3 GRPO runs. Read this **before** touching the RunPod MCP. Cost and the
JP region-lock are the two things that bite — both are covered below.

## TL;DR guardrails (do these every time)

- **Cost is per-second and silent.** Always `stop-pod` when stepping away, then
  `get-pod` to **confirm `desiredStatus == "EXITED"`** — a stop request that didn't
  take is a pod billing all night.
- **Never `create-pod` to "check availability."** It bills immediately. To probe
  AP-JP-1 stock, the only honest signal is an actual create attempt — treat it as a
  real (billable) action and confirm with the user first.
- **One 80 GB GPU is the target** for 35B-A3B QLoRA (4-bit). Don't over-provision.
- **Match CUDA to torch before training** — the "14h-for-40-steps" lesson (see
  `pod_setup_grpo.sh`, below). A mismatched kernel silently runs 10–100× slow.

## The accounts/IDs that matter

| Thing | ID | Notes |
|---|---|---|
| Existing pod | `rfr4dd6aimlw7y` (`damaged_gray_narwhal`) | currently EXITED; `start-pod` to resume |
| Network volume | `bp6ccofvnb` (`improved_gold_capybara`, 200 GB) | **pinned to AP-JP-1 (Japan)** |
| SSH | `~/.ssh/id_ed25519` | `ssh -T -o StrictHostKeyChecking=no -p PORT -i ~/.ssh/id_ed25519 root@IP` |

## The JP region-lock — the real cause of the GPU-stockout pain

Network volumes **cannot move regions**. `bp6ccofvnb` lives in **AP-JP-1**, so any
pod that mounts it must also be in AP-JP-1 — a single, high-demand region. That is
why GPUs are often "unavailable": the constraint is the *region*, not the GPU type.

**The fix: decouple storage from the region.** Most work does **not** need the JP
volume:

- **Phase-2 SFT** (`bridge-design.md` §4): the pod trains *only* on cached
  `(latent-input, predictive-channels, target-text)` pairs precomputed on the local
  4090. Ship those pairs + checkpoints via **HuggingFace** (the demo pipeline
  already uses HF, zero-local-storage) or any object store. Then **any region's GPU
  works** — pick whichever has stock, ignore AP-JP-1 entirely.
- **Phase-3 GRPO**: world model (19M) + Qwen co-resident; checkpoints can likewise
  live on HF. The JP volume is convenient persistence, not a requirement.

So: reserve the AP-JP-1 volume for things that genuinely need fast local persistence;
otherwise stage through HF and provision in whatever region has the GPU.

## GPU selection for 35B-A3B QLoRA

`list-gpu-types` `stockStatus` is **GLOBAL, not per-data-center.** It tells you what
to *try*; it does not promise AP-JP-1 has it. Use it to pick a fallback ladder, then
let `create-pod` resolve actual availability at deploy time.

Snapshot (global stock, 2026-06-18 — re-query before relying on it):

| GPU | VRAM | Secure | Global stock | Fit for 35B QLoRA |
|---|---|---|---|---|
| **H100 SXM 80GB** (`NVIDIA H100 80GB HBM3`) | 80 | yes | **High** | **default pick** — well-trodden CUDA, fits 4-bit QLoRA + resampler |
| **RTX PRO 6000 Blackwell** (`...Server Edition`) | 96 | yes | **High** | cheap + roomy, BUT Blackwell (sm_120) needs CUDA ≥12.8 + Blackwell-built torch/kernels — a CUDA-matching risk; validate the toolchain first |
| H200 SXM (`NVIDIA H200`) | 141 | yes | Medium | headroom for bf16/FP8 or longer sequences; pricier |
| A100 80GB (PCIe / SXM4) | 80 | yes | Low | works; lower global stock |
| B200 (`NVIDIA B200`) | 180 | yes | Low | overkill + Blackwell toolchain risk |

**Recommendation:** default to **H100 SXM 80GB** (high stock, no Blackwell toolchain
surprises). Keep **RTX PRO 6000 (96GB)** and **A100 80GB** as ladder fallbacks. Only
reach for H200/B200 if a run actually needs >80 GB. The FP8 Qwen repo variant
(`bridge-design.md` §2.3) is the memory fallback if 4-bit QLoRA won't fit.

## Intent → MCP tool (+ guardrail)

| Intent | Tool(s) | Guardrail |
|---|---|---|
| See what's running / costing | `list-pods` | check `desiredStatus`, `costPerHr` |
| Inspect a pod | `get-pod` (`includeMachine`, `includeNetworkVolume`) | — |
| Find a GPU | `list-gpu-types --minMemoryGb 80 --secureCloudOnly` | stock is **global**, not JP |
| Find a region | `list-data-centers` | volume-bound work → AP-JP-1 only |
| Resume existing box | `start-pod rfr4dd6aimlw7y` | cheaper than a fresh create |
| New training box | `create-pod` | **confirm GPU id, region, costPerHr with user first**; attach `bp6ccofvnb` only if JP is required |
| Done / stepping away | `stop-pod` → `get-pod` | **verify `desiredStatus == EXITED`** |
| Tear down for good | `delete-pod` | irreversible; volume survives separately |

## Toolchain — `pod_setup_grpo.sh` (⚠ currently MISSING from the repo)

`bridge-design.md` §4 and the decisions ledger reference `scripts/pod_setup_grpo.sh`
for CUDA/kernel matching, but it is **not in the repo**. Before any GRPO pod spend,
recover or rewrite it. It must pin:
- torch / CUDA versions matched to the pod image (causal-conv1d / Mamba kernels go
  10–100× slow on a mismatch — the documented gotcha),
- `demoparser2 >= 0.41.3` (older crashes `EntityNotFound` on Major demos),
- the Qwen3.6-35B-A3B + QLoRA deps.

Until it exists, treat any pod GRPO run as blocked on toolchain reproducibility.
