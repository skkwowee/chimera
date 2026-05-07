#!/usr/bin/env bash
# Cloud Claude Code SessionStart bootstrap for chimera.
#
# Runs every cloud session via .claude/settings.json's SessionStart hook
# (gated on CLAUDE_CODE_REMOTE=true so local sessions are untouched).
#
# This sandbox is the ORCHESTRATOR — heavy training lives on RunPod. We
# install only what's needed to drive the pod and run offline analysis
# on the small JSONL audit logs pulled back. Torch/bitsandbytes/peft/trl
# are deliberately skipped; if you need them, you're on the wrong host.
#
# Idempotent: each step short-circuits if already done. Cold start ~30s,
# warm start <5s.

set -euo pipefail

CHIMERA_ROOT="${CHIMERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$CHIMERA_ROOT"

log() { printf "\033[1;36m[chimera-cloud-setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[chimera-cloud-setup WARN]\033[0m %s\n" "$*" >&2; }

# --- 1. Lightweight Python deps (orchestration only) ---
# We install a curated subset of requirements.txt. The heavy ML stack
# (torch, peft, trl, bitsandbytes, faiss-cpu) lives on the pod.
ORCH_DEPS=(
    "anthropic>=0.25.0"
    "python-dotenv>=1.0.0"
    "requests>=2.31.0"
    "polars>=0.19.0"
    "pandas>=2.0.0"
    "pyarrow>=10.0.0"
    "tqdm>=4.66.0"
)

if ! python3 -c "import anthropic, polars, dotenv" 2>/dev/null; then
    log "installing orchestration deps"
    python3 -m pip install --quiet --upgrade "${ORCH_DEPS[@]}"
else
    log "orchestration deps already installed"
fi

# --- 2. SSH key for RunPod ---
# RUNPOD_SSH_KEY env var holds the ed25519 private key (literal contents,
# newlines preserved). Set it once in the cloud environment config.
mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [[ -n "${RUNPOD_SSH_KEY:-}" ]]; then
    if [[ ! -f ~/.ssh/id_ed25519 ]]; then
        log "writing RunPod SSH key from RUNPOD_SSH_KEY"
        printf '%s\n' "$RUNPOD_SSH_KEY" > ~/.ssh/id_ed25519
        chmod 600 ~/.ssh/id_ed25519
    else
        log "SSH key already in place"
    fi
else
    warn "RUNPOD_SSH_KEY not set — pod SSH will fail until you configure it"
fi

# Trust the GitHub host key so 'git push' over SSH works without prompting.
if ! grep -q "github.com" ~/.ssh/known_hosts 2>/dev/null; then
    log "adding github.com to known_hosts"
    ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
    chmod 644 ~/.ssh/known_hosts
fi

# --- 3. HuggingFace auth (optional, for any direct HF access) ---
if [[ -n "${HF_TOKEN:-}" ]]; then
    if [[ ! -f ~/.cache/huggingface/token ]]; then
        log "writing HF token from HF_TOKEN env"
        mkdir -p ~/.cache/huggingface
        printf '%s' "$HF_TOKEN" > ~/.cache/huggingface/token
        chmod 600 ~/.cache/huggingface/token
    fi
else
    warn "HF_TOKEN not set — only public HF resources accessible"
fi

# --- 4. Quick connectivity sanity ---
# Don't fail the session on these — just surface state to the log.
log "connectivity check:"

if curl -fsS --max-time 5 https://api.runpod.io/graphql -o /dev/null 2>&1; then
    printf "  RunPod API:    \033[1;32mreachable\033[0m\n"
else
    printf "  RunPod API:    \033[1;31munreachable\033[0m (check network allowlist for *.runpod.io)\n"
fi

if curl -fsS --max-time 5 https://huggingface.co -o /dev/null 2>&1; then
    printf "  HuggingFace:   \033[1;32mreachable\033[0m\n"
else
    printf "  HuggingFace:   \033[1;31munreachable\033[0m\n"
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    printf "  Anthropic key: \033[1;32mset\033[0m\n"
else
    printf "  Anthropic key: \033[1;33mnot set\033[0m (judge_reward will fail)\n"
fi

# --- 5. Repo state summary ---
log "repo state:"
printf "  branch:        %s\n" "$(git branch --show-current)"
printf "  HEAD:          %s\n" "$(git log -1 --oneline)"
printf "  uncommitted:   %s files\n" "$(git status --porcelain | wc -l)"

log "ready. See docs/cloud-session.md if you need to add more env vars."
