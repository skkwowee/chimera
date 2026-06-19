#!/usr/bin/env bash
# pod_setup_grpo.sh — RunPod toolchain bootstrap for the Phase-2 bridge QLoRA SFT
# and Phase-3 GRPO. See docs/pod-runbook.md and docs/bridge-design.md §4.
#
# THE LESSON THIS SCRIPT EXISTS FOR ("14h-for-40-steps"): a torch/CUDA/kernel
# mismatch silently runs 10-100x slow. So the discipline here is:
#   1. NEVER reinstall torch — the image ships a torch matched to its CUDA driver.
#      Pin every other dep AGAINST that torch instead.
#   2. Health-check loudly BEFORE any long run: CUDA match + a bitsandbytes 4-bit
#      timing sanity. Bail on mismatch rather than burn pod-hours on a slow kernel.
#
# Idempotent. Usage:  bash pod_setup_grpo.sh   (optionally: QWEN_MODEL=<id> bash ...)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/chimera}"
REPO_URL="${REPO_URL:-git@github.com:skkwowee/chimera.git}"
PY="${PY:-python}"

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. CUDA match
say "GPU / CUDA / torch consistency check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — not a GPU pod"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
SMI_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo 0)
$PY - <<PY || die "torch cannot see the GPU — driver/CUDA mismatch, FIX BEFORE TRAINING"
import torch, sys
print("torch", torch.__version__, "compiled-CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
tc = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
smi = int("${SMI_CUDA}" or 0)
print("driver CUDA major", smi, "| torch CUDA major", tc)
if smi and tc and smi < tc:
    sys.exit(f"driver CUDA {smi} < torch CUDA {tc}: kernels will fall back / run slow")
print("CUDA match OK")
PY

# ---------------------------------------------------------------- 2. deps
# Install everything EXCEPT torch, against the image's torch. --no-deps where a
# package would otherwise try to drag in a different torch.
say "Installing QLoRA / bridge stack (torch left untouched)"
$PY -m pip install -q --upgrade pip
$PY -m pip install -q \
    "transformers>=4.45" "peft>=0.13" "accelerate>=1.0" "bitsandbytes>=0.44" \
    "datasets>=3.0" "huggingface_hub>=0.25" "safetensors" "sentencepiece" \
    "demoparser2>=0.41.3"   # <0.41.3 crashes EntityNotFound on Major demos (memory gotcha)

# ---------------------------------------------------------------- 3. health checks
say "bitsandbytes 4-bit health check (the slow-kernel guard)"
$PY - <<'PY' || die "bitsandbytes 4-bit path is broken/slow — do NOT start a run"
import torch, time, bitsandbytes as bnb
print("bitsandbytes", bnb.__version__)
dev = "cuda"
lin = bnb.nn.Linear4bit(4096, 4096, bias=False, compute_dtype=torch.bfloat16).to(dev)
x = torch.randn(32, 4096, dtype=torch.bfloat16, device=dev)
for _ in range(3): lin(x)          # warmup
torch.cuda.synchronize(); t = time.time()
for _ in range(50): lin(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 50 * 1000
print(f"4-bit 4096x4096 matmul: {dt:.2f} ms/iter")
assert dt < 50, f"4-bit matmul {dt:.1f}ms is implausibly slow — CUDA/kernel mismatch"
print("bitsandbytes 4-bit OK")
PY

# ---------------------------------------------------------------- 4. repo
say "Repo"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only || echo "(repo pull skipped)"
fi
mkdir -p "$REPO_DIR/data/processed/bridge_sft" "$REPO_DIR/outputs/bridge"

# ---------------------------------------------------------------- 5. SFT pull (optional)
# bridge SFT trains on cached pairs from HF — no world model needed on the pod (§4).
if [ "${PULL_SFT:-1}" = "1" ]; then
    say "Pulling bridge SFT cache from HF"
    $PY - <<PY || echo "(SFT pull skipped — run scripts/push_sft_hf.py locally first)"
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download("skkwowee/chimera-cs2", "bridge_sft/train_single.pt", repo_type="dataset")
dst = "$REPO_DIR/data/processed/bridge_sft/train_single.pt"
shutil.copy(p, dst); print("SFT cache ->", dst, os.path.getsize(dst)//1_000_000, "MB")
PY
fi

say "DONE — pod ready. Train: python scripts/train_bridge.py --llm qwen --model \$QWEN_MODEL"
