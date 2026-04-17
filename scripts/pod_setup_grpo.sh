#!/usr/bin/env bash
# Pod bootstrap for GRPO training with Qwen3.5-35B-A3B MoE.
#
# Why this exists: the previous run took 14h for 40 steps because torch's CUDA
# build did not match the pod's CUDA runtime. causal-conv1d and flash-linear-attention
# failed to build, transformers fell back to a Python ref impl, torch inductor
# burned 30 min compiling the fallback path, then ran ~5-10x slower per step.
#
# This script verifies the toolchain BEFORE installing anything heavy, then
# verifies kernels import cleanly BEFORE starting training. Fails fast.
#
# Usage on pod:
#   bash scripts/pod_setup_grpo.sh
#
# Recommended pod template (verify torch/CUDA pair before launching):
#   runpod/pytorch:2.6.0-py3.11-cuda12.4.1
#
# Anti-recommended: any "latest" template — they ship mismatched CUDA builds
# roughly every other week.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/chimera}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "=== Chimera GRPO Pod Setup ==="
echo

# ---------------------------------------------------------------------------
# 1. Verify CUDA / torch pairing BEFORE installing anything
# ---------------------------------------------------------------------------
echo "--- Toolchain check ---"
RUNTIME_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 || echo "unknown")
TORCH_INFO=$(python3 -c "import torch; print(torch.__version__, torch.version.cuda)" 2>&1 || echo "torch-import-failed")
TORCH_VERSION=$(echo "$TORCH_INFO" | awk '{print $1}')
TORCH_CUDA=$(echo "$TORCH_INFO" | awk '{print $2}')

echo "  nvidia-smi CUDA runtime: $RUNTIME_CUDA"
echo "  torch:                   $TORCH_VERSION"
echo "  torch.version.cuda:      $TORCH_CUDA"

if [ "$TORCH_CUDA" = "None" ] || [ "$TORCH_CUDA" = "torch-import-failed" ]; then
    echo "ABORT: torch is CPU-only or not installed. Pick a CUDA pod template."
    exit 1
fi

# The nvidia-smi "CUDA Version" line is the MAX CUDA the driver supports, not the
# installed runtime. NVIDIA drivers are backward-compatible, so torch built for
# CUDA X.Y works as long as driver CUDA >= X.Y. Warn-only on the check; the
# post-install import test is authoritative.
RUNTIME_MAJOR=$(echo "$RUNTIME_CUDA" | cut -d. -f1)
TORCH_MAJOR=$(echo "$TORCH_CUDA" | cut -d. -f1)
if [ "$TORCH_MAJOR" -gt "$RUNTIME_MAJOR" ] 2>/dev/null; then
    echo
    echo "WARN: torch built for CUDA $TORCH_CUDA but driver only supports up to $RUNTIME_CUDA."
    echo "      This was the failure mode on the 14h/40-step run. Kernels may not link."
    echo "      Consider switching to a pod template with driver CUDA >= $TORCH_CUDA."
    echo "      Continuing — will verify via import test after install."
else
    echo "  driver CUDA $RUNTIME_CUDA >= torch CUDA $TORCH_CUDA — backward compat. OK."
fi
echo

# ---------------------------------------------------------------------------
# 2. Repo + system tools
# ---------------------------------------------------------------------------
if [ -d "$REPO_DIR" ]; then
    echo "Repo exists at $REPO_DIR, pulling..."
    git -C "$REPO_DIR" pull
else
    echo "Cloning repo..."
    git clone git@github.com:skkwowee/chimera.git "$REPO_DIR"
fi
cd "$REPO_DIR"

if ! git lfs version >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 3. Base deps (matches requirements.txt)
# ---------------------------------------------------------------------------
echo "--- Installing base deps ---"
uv pip install --system -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Fast-path kernels — install in this order, verify each
# ---------------------------------------------------------------------------
echo
echo "--- Installing fast-path kernels ---"

# FlashAttention-2 first; ships as a prebuilt wheel for common torch/cuda combos.
# Independent of causal-conv1d. If this fails, fall back to attn_implementation=sdpa.
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn..."
    pip install --no-build-isolation flash-attn==2.7.4.post1 || {
        echo "  flash-attn install failed — will use --attn-impl sdpa as fallback"
    }
fi

# causal-conv1d. CAUSAL_CONV1D_FORCE_BUILD=1 skips the prebuilt-wheel cache and
# compiles against the local torch ABI — slower install but the only reliable path.
if ! python3 -c "import causal_conv1d" 2>/dev/null; then
    echo "Installing causal-conv1d (building from source against local torch)..."
    CAUSAL_CONV1D_FORCE_BUILD=1 pip install --no-build-isolation causal-conv1d
fi

# flash-linear-attention. PyPI name is `fla`.
if ! python3 -c "import fla" 2>/dev/null; then
    echo "Installing flash-linear-attention (fla)..."
    pip install --no-build-isolation fla
fi

# ---------------------------------------------------------------------------
# 5. Hard verify — fail fast if anything is broken
# ---------------------------------------------------------------------------
echo
echo "--- Verifying kernels ---"
python3 - <<'PY'
import sys
problems = []
try:
    import causal_conv1d
    print(f"  causal_conv1d {getattr(causal_conv1d, '__version__', '?')} OK")
except Exception as e:
    problems.append(f"causal_conv1d: {e}")
try:
    import fla
    print(f"  fla (flash-linear-attention) {getattr(fla, '__version__', '?')} OK")
except Exception as e:
    problems.append(f"fla: {e}")
try:
    import flash_attn
    print(f"  flash_attn {getattr(flash_attn, '__version__', '?')} OK")
except Exception as e:
    print(f"  flash_attn missing ({e}) — pass --attn-impl sdpa to train_grpo.py")
if problems:
    print()
    print("ABORT: required kernels failed to import:")
    for p in problems:
        print(f"  - {p}")
    print()
    print("Do not start training. The fallback path is 5-10x slower.")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# 6. Pull data
# ---------------------------------------------------------------------------
echo
echo "--- Pulling data from Hub ---"
python3 scripts/data.py pull --all

echo
echo "=== Setup complete ==="
echo
echo "Recommended GRPO command:"
echo "  TORCHDYNAMO_DISABLE=1 PYTHONUTF8=1 PYTHONPATH=$REPO_DIR \\"
echo "    python3 scripts/train_grpo.py --manual \\"
echo "      --model-name <SFT-checkpoint-or-merged-path> \\"
echo "      --data data/training/grpo/smoke_test.jsonl \\"
echo "      --reward-mode recall --kl-coef 0.02 \\"
echo "      --num-generations 4 --max-tokens 256 \\"
echo "      --max-steps 100 --save-steps 10"
