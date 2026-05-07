# Cloud Claude Code session — chimera setup

One-time configuration so every new cloud session at `claude.ai/code` lands
ready-to-orchestrate. Cloud sessions are the **orchestrator** here — they
SSH into RunPod for any heavy work and pull small audit logs back. Models,
training, and large data stay on the pod's network volume.

## One-time setup (in claude.ai/code)

### 1. Connect the chimera repo

In claude.ai/code → Environments → New environment → connect
`skkwowee/chimera`. Name the environment `chimera-prod` (the name appears
in the pre-fill URL below).

### 2. Set environment variables / secrets

Add these in the environment's **Environment Variables** panel. They persist
for every session that uses this environment.

| Variable | Required | Purpose |
|---|:---:|---|
| `ANTHROPIC_API_KEY` | yes | Judge reward, baseline_eval, any Claude API calls |
| `RUNPOD_SSH_KEY` | yes | ed25519 private key (full file contents incl. headers and newlines) for `ssh root@<pod>` |
| `RUNPOD_API_KEY` | yes | Auto-stop wrappers, pod-state queries via the GraphQL API |
| `HF_TOKEN` | optional | Only needed if you push/pull from a private HF repo |

The bootstrap (`scripts/cloud_session_setup.sh`) reads these on session
start — `RUNPOD_SSH_KEY` becomes `~/.ssh/id_ed25519`, `HF_TOKEN` lands at
`~/.cache/huggingface/token`. Sandboxes are ephemeral, so this happens
every session — but the env-var values themselves are permanent.

### 3. Network allowlist

Cloud sessions default to a "Trusted" allowlist (PyPI, npm, GitHub, HF).
RunPod is not in the default list. In the environment's **Network Access**
panel switch to **Custom** and add:

```
*.runpod.io
runpod.io
api.runpod.io
```

If you'll connect to a specific pod IP rather than its `*.runpod.proxy`
hostname, also add the IP literal. Pod IPs change on restart, so prefer
the proxy hostname when possible.

### 4. SessionStart hook

Already in the repo at `.claude/settings.json` — runs
`scripts/cloud_session_setup.sh` on every session start when
`CLAUDE_CODE_REMOTE=true`. Local sessions are unaffected.

The hook:

- Installs the orchestration-only Python deps (anthropic, polars, etc).
  Skips torch/peft/trl/bitsandbytes on purpose — those run on the pod.
- Materializes the SSH key from `RUNPOD_SSH_KEY` into `~/.ssh/id_ed25519`.
- Materializes the HF token from `HF_TOKEN` into the standard cache path.
- Trusts `github.com` host key for git pushes.
- Prints a connectivity summary (RunPod API reachable? HF reachable?
  Anthropic key set?).

## Pre-fill URL

Bookmark this so every new session opens the chimera environment without
clicking through the picker:

```
https://claude.ai/code?repositories=skkwowee/chimera&environment=chimera-prod
```

Replace `chimera-prod` with whatever you named the environment.

## What you should see on first session

After the SessionStart hook runs (~30s cold, <5s warm), the log should show:

```
[chimera-cloud-setup] installing orchestration deps
[chimera-cloud-setup] writing RunPod SSH key from RUNPOD_SSH_KEY
[chimera-cloud-setup] writing HF token from HF_TOKEN env
[chimera-cloud-setup] connectivity check:
  RunPod API:    reachable
  HuggingFace:   reachable
  Anthropic key: set
[chimera-cloud-setup] repo state:
  branch:        main
  HEAD:          77b85d6 Add measurement-methodology framework + 2 diagnostic scripts
  uncommitted:   0 files
[chimera-cloud-setup] ready.
```

If RunPod shows unreachable, fix the network allowlist (step 3).
If the SSH key is missing, the warn line will fire — set `RUNPOD_SSH_KEY`.

## What this sandbox is NOT for

The cloud sandbox is small (~10–30GB ephemeral disk). Do not:

- Download the merged Qwen3.5-35B-A3B model (~70GB).
- Pull `data/captures/` (GBs of screenshots — they live on the pod and
  the public HF dataset; cloud doesn't need them for orchestration).
- Train anything locally — there's no GPU here.

Workflow: SSH to RunPod → `bash scripts/run_grpo_with_auto_stop.sh ...` →
`scp` the small JSONL audit files back → run `scripts/passer_spread_audit.py`
locally on the cloud sandbox. That's it.

## Local vs cloud — when to use which

| Task | Local WSL | Cloud session |
|---|:---:|:---:|
| Heavy iteration on trainer code with `tmux` | ✓ | |
| Debugging a hung pod, log diving | ✓ | |
| One-shot "kick off F08v5" while away from desk | | ✓ |
| Reading the repo from another device | | ✓ |
| Running `passer_spread_audit.py` on a fresh `useful_jumps.jsonl` | ✓ | ✓ |
| Hand-authoring `data/eval/pseudo_gold_advices.jsonl` | ✓ | ✓ |

Both can coexist — the SessionStart hook is gated on `CLAUDE_CODE_REMOTE`
so the local environment is never bootstrapped twice.

## Updating the bootstrap

`scripts/cloud_session_setup.sh` is idempotent and fast. To change what
gets installed, edit it and commit; the next cloud session picks up the
change. To debug it, run it locally with `CLAUDE_CODE_REMOTE=true bash
scripts/cloud_session_setup.sh` — it's safe on WSL too (writes to your
real `~/.ssh` only if the env vars are set, otherwise warns and exits).

## Caveats and gotchas

- **No persistent disk between sessions.** Anything you `wget` or
  `huggingface-cli download` is wiped. Push it back to git or HF if you
  need it next session.
- **Env vars are visible to anyone with edit access** to the environment.
  Don't add other people's secrets here. For solo use this is fine.
- **Setup script caching ~7 days.** Anthropic-side setup scripts
  (different from our SessionStart hook) cache for ~7 days; our hook
  runs every session and is the right place for things that need to
  happen at session start vs once-per-environment.
