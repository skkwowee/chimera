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
VENV_DIR="${VENV_DIR:-/workspace/venv}"
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

# ---------------------------------------------------------------------------
# 3. Venv at $VENV_DIR with system-site-packages
# ---------------------------------------------------------------------------
# Why a venv: ubuntu 24.04 templates mark system Python externally-managed
# (PEP 668), so `uv pip install --system` is blocked. A uv venv with
# --system-site-packages keeps fast install (uv) AND sees the system torch
# (no 70GB redownload) AND keeps our installs isolated from /usr.
# All later invocations must use $VENV_PY, not /usr/bin/python3.
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# If venv exists with a different torch than system, nuke it. The kernel
# build chain is fundamentally tied to torch's CUDA version, and the system
# torch is what matches the pod's installed nvcc.
SYSTEM_TORCH=$(/usr/bin/python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [ -d "$VENV_DIR" ]; then
    VENV_TORCH=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    if [ "$SYSTEM_TORCH" != "$VENV_TORCH" ]; then
        echo "Venv torch ($VENV_TORCH) differs from system torch ($SYSTEM_TORCH). Recreating venv..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR (--system-site-packages, sees system torch)..."
    uv venv --system-site-packages "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"

# Sanity: the venv must see the system torch, not its own.
"$VENV_PY" -c "import torch; print(f'  venv-visible torch: {torch.__version__} cuda={torch.version.cuda}')" \
    || { echo "ABORT: venv cannot see torch"; exit 1; }

# ---------------------------------------------------------------------------
# 4. Base deps into the venv
# ---------------------------------------------------------------------------
# Pre-pin torch to the system version+CUDA inside the venv. uv's resolver does
# NOT count system-site-packages as "installed", so without this, transitive
# torch deps from transformers/trl/peft pull the latest wheel (cu130) and
# shadow the system torch. The kernels then can't find a matching nvcc.
SYSTEM_TORCH_FULL=$(/usr/bin/python3 -c "import torch; print(torch.__version__)")
SYSTEM_TORCH_VERSION="${SYSTEM_TORCH_FULL%+*}"          # e.g. "2.8.0"
SYSTEM_TORCH_CUDA_TAG="${SYSTEM_TORCH_FULL##*+}"        # e.g. "cu128"
echo "--- Pinning venv torch to system: $SYSTEM_TORCH_VERSION+$SYSTEM_TORCH_CUDA_TAG ---"
VIRTUAL_ENV="$VENV_DIR" uv pip install \
    "torch==$SYSTEM_TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$SYSTEM_TORCH_CUDA_TAG" \
    --extra-index-url "https://pypi.org/simple"

# Now install requirements.txt — uv sees torch is satisfied and won't upgrade.
echo "--- Installing remaining base deps into $VENV_DIR ---"
VIRTUAL_ENV="$VENV_DIR" uv pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 5. Fast-path kernels — install in this order, verify each (all into venv)
# ---------------------------------------------------------------------------
echo
echo "--- Setting up CUDA toolchain for kernel builds ---"

# nvcc must match torch's CUDA version. System torch is cu128, pod has CUDA 12.8
# toolkit at /usr/local/cuda-12.8 (symlinked from /usr/local/cuda) but it's not
# on PATH by default. Wire it up explicitly.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ABORT: nvcc not found at $CUDA_HOME/bin/nvcc. Install CUDA toolkit:"
    echo "       apt-get install cuda-toolkit-12-8"
    exit 1
fi
NVCC_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
TORCH_CUDA_VENV=$("$VENV_PY" -c "import torch; print(torch.version.cuda)")
echo "  nvcc CUDA: $NVCC_CUDA"
echo "  torch CUDA: $TORCH_CUDA_VENV"
if [ "$NVCC_CUDA" != "$TORCH_CUDA_VENV" ]; then
    echo "ABORT: nvcc ($NVCC_CUDA) != torch CUDA ($TORCH_CUDA_VENV). Kernels won't build."
    exit 1
fi
echo "  toolchain matches. OK."

echo
echo "--- Installing fast-path kernels ---"

# `uv venv` does not install pip into the venv — use `uv pip install --python $VENV_PY`
# instead. --no-build-isolation lets the build see torch (required for causal-conv1d
# and fla setup.py to detect ABI flags).
UV_PIP=(uv pip install --python "$VENV_PY" --no-build-isolation)

# FlashAttention-2. Builds from source if no matching prebuilt wheel.
# CRITICAL: limit GPU archs or the build compiles for sm_80, sm_90, sm_100, sm_120
# (~30-60 min on H200). FLASH_ATTN_CUDA_ARCHS=9.0 limits to H200's arch only
# (~10-15 min). MAX_JOBS=8 parallelizes the per-arch nvcc compiles.
# Override with FLASH_ATTN_CUDA_ARCHS in the env if running on a different GPU.
# Independent of causal-conv1d. If install fails, --attn-impl sdpa is the fallback
# (~80% of FA2's speed).
if ! "$VENV_PY" -c "import flash_attn" 2>/dev/null; then
    echo "Installing flash-attn (sm_${FLASH_ATTN_CUDA_ARCHS:-9.0} only, MAX_JOBS=${MAX_JOBS:-8})..."
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-9.0}" \
    MAX_JOBS="${MAX_JOBS:-8}" \
    "${UV_PIP[@]}" flash-attn==2.7.4.post1 || {
        echo "  flash-attn install failed — will use --attn-impl sdpa as fallback"
    }
fi

# causal-conv1d. CAUSAL_CONV1D_FORCE_BUILD=1 skips the prebuilt-wheel cache and
# compiles against the local torch ABI — slower install but the only reliable path.
if ! "$VENV_PY" -c "import causal_conv1d" 2>/dev/null; then
    echo "Installing causal-conv1d (building from source against local torch)..."
    CAUSAL_CONV1D_FORCE_BUILD=1 "${UV_PIP[@]}" causal-conv1d
fi

# flash-linear-attention. PyPI name is `flash-linear-attention`; imports as `fla`.
if ! "$VENV_PY" -c "import fla" 2>/dev/null; then
    echo "Installing flash-linear-attention..."
    "${UV_PIP[@]}" flash-linear-attention
fi

# ---------------------------------------------------------------------------
# 6. Hard verify — fail fast if anything is broken
# ---------------------------------------------------------------------------
echo
echo "--- Verifying kernels (in venv) ---"
"$VENV_PY" - <<'PY'
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
# 7. Pull data
# ---------------------------------------------------------------------------
echo
echo "--- Pulling data from Hub ---"
"$VENV_PY" scripts/data.py pull --all

echo
echo "=== Setup complete ==="
echo
echo "Venv: $VENV_DIR (use $VENV_PY for all training commands)"
echo
echo "Recommended GRPO command:"
echo "  TORCHDYNAMO_DISABLE=1 PYTHONUTF8=1 PYTHONPATH=$REPO_DIR \\"
echo "    $VENV_PY scripts/train_grpo.py --manual \\"
echo "      --model-name <SFT-checkpoint-or-merged-path> \\"
echo "      --data data/training/grpo/smoke_test.jsonl \\"
echo "      --reward-mode recall --kl-coef 0.02 \\"
echo "      --num-generations 4 --max-tokens 256 \\"
echo "      --max-steps 100 --save-steps 10"
