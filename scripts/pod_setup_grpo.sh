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

# Find the system python that has torch installed. uv venv defaults to uv's
# managed python (often a different minor version than system), and
# --system-site-packages can't bridge interpreter versions. Reuse this for
# venv creation AND for reading system torch version / recreation check.
SYSTEM_PY=""
for candidate in /usr/bin/python3 /usr/local/bin/python3 python3; do
    if "$candidate" -c "import torch" 2>/dev/null; then
        SYSTEM_PY="$candidate"
        break
    fi
done
if [ -z "$SYSTEM_PY" ]; then
    echo "ABORT: no system python with torch found. Tried /usr/bin/python3, /usr/local/bin/python3, python3."
    exit 1
fi

# If venv exists with a different torch than system, nuke it. The kernel
# build chain is tied to torch's CUDA version, and the system torch is what
# matches the pod's installed nvcc.
SYSTEM_TORCH=$("$SYSTEM_PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [ -d "$VENV_DIR" ]; then
    VENV_TORCH=$("$VENV_DIR/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    if [ "$SYSTEM_TORCH" != "$VENV_TORCH" ]; then
        echo "Venv torch ($VENV_TORCH) differs from system torch ($SYSTEM_TORCH). Recreating venv..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR (-p $SYSTEM_PY, --system-site-packages)..."
    uv venv --python "$SYSTEM_PY" --system-site-packages "$VENV_DIR"
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
SYSTEM_TORCH_FULL=$("$SYSTEM_PY" -c "import torch; print(torch.__version__)")
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

# Drop venv-local torch/torchvision so --system-site-packages reveals the
# SYSTEM torch (which was installed by the pod template against the pod's
# nvcc). Required when pytorch.org/whl/<cuda_tag> doesn't have a wheel for
# the exact (version, python, cuda) triple we need — uv then falls through
# to the plain PyPI wheel which ships cu121 universal, shadowing the system
# cu124 install and breaking the nvcc/torch match check below.
echo "--- Dropping venv-local torch (fall through to system cu$SYSTEM_TORCH_CUDA_TAG) ---"
rm -rf "$VENV_DIR"/lib/python*/site-packages/torch \
       "$VENV_DIR"/lib/python*/site-packages/torchvision \
       "$VENV_DIR"/lib/python*/site-packages/torch-*.dist-info \
       "$VENV_DIR"/lib/python*/site-packages/torchvision-*.dist-info
VENV_TORCH_POST=$("$VENV_PY" -c "import torch; print(torch.__version__)")
if [ "$VENV_TORCH_POST" != "$SYSTEM_TORCH_FULL" ]; then
    echo "ABORT: after cleanup venv torch is $VENV_TORCH_POST, expected $SYSTEM_TORCH_FULL."
    echo "       System torch at $SYSTEM_PY must be importable for fallback to work."
    exit 1
fi
echo "  venv torch now: $VENV_TORCH_POST (via system-site-packages)"

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

# Kernel installs:
#  --no-build-isolation lets the build see venv's (system) torch during setup.py
#  --no-deps prevents uv from re-pulling torch (or any other dep) into the venv.
#    Without this, flash-attn's install resolves its torch dep and installs
#    the latest (e.g., 2.11.0+cu130), shadowing the system torch we just
#    wired up and breaking the nvcc match.
UV_PIP=(uv pip install --python "$VENV_PY" --no-build-isolation --no-deps)

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

# flash-linear-attention. PyPI's `flash-linear-attention==0.4.2` is missing the
# `fla.modules` submodule that transformers' modeling_qwen3_5_moe.py imports.
# Use the git main version (0.5.0+) which has fla.modules.FusedRMSNormGated.
# Verify with: python -c "from fla.modules import FusedRMSNormGated"
if ! "$VENV_PY" -c "from fla.modules import FusedRMSNormGated" 2>/dev/null; then
    echo "Installing flash-linear-attention from git main (PyPI 0.4.2 lacks fla.modules)..."
    "${UV_PIP[@]}" "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@main"
fi

# tilelang. fla's gated_delta_rule chunk_bwd_dqkwg falls back to tilelang on
# Hopper GPUs (H100/H200) because Triton >= 3.4.0 has a known correctness bug
# there (fla-org/flash-linear-attention#640). Without tilelang, the GRPO
# backward pass crashes with RuntimeError on the first optimizer step.
if ! "$VENV_PY" -c "import tilelang" 2>/dev/null; then
    echo "Installing tilelang (required by fla on Hopper GPUs)..."
    "${UV_PIP[@]}" tilelang
fi

# Patch tilelang's bundled TVM. tilelang 0.1.7/0.1.8 ship a TVMDerivedObject
# wrapper whose dynamic subclass inherits from a fully-slotted CObject MRO
# (every base has __slots__=()). The wrapper tries `self._inst = ...` which
# fails with AttributeError: '_NestedLoopCheckVisitor' object has no attribute
# '_inst'. Fix: declare __slots__ on the wrapper itself so _inst (plus the
# weakref the wrapper takes of self) actually have storage. Idempotent.
TVM_SUPPORT_PY=$("$VENV_PY" -c "import tilelang, os; print(os.path.join(os.path.dirname(tilelang.__file__), '3rdparty/tvm/python/tvm/runtime/support.py'))")
if [ -f "$TVM_SUPPORT_PY" ] && ! grep -q '__slots__ = ("_inst", "key", "handle", "__weakref__")' "$TVM_SUPPORT_PY"; then
    echo "Patching tilelang TVM support.py to add __slots__ to TVMDerivedObject..."
    "$VENV_PY" - "$TVM_SUPPORT_PY" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'class TVMDerivedObject(metadata["cls"]):  # type: ignore\n        """The derived object to avoid cyclic dependency."""\n\n        _cls = cls'
new = 'class TVMDerivedObject(metadata["cls"]):  # type: ignore\n        """The derived object to avoid cyclic dependency."""\n\n        __slots__ = ("_inst", "key", "handle", "__weakref__")\n        _cls = cls'
if old not in s:
    print("ABORT: tilelang support.py pattern not found — upstream may have changed it")
    sys.exit(1)
open(p, 'w').write(s.replace(old, new))
print("  patched", p)
PY
fi

# ---------------------------------------------------------------------------
# 6. Hard verify — fail fast if anything is broken
# ---------------------------------------------------------------------------
echo
echo "--- Re-verify torch is still the system cu tag (kernels sometimes re-pull) ---"
VENV_TORCH_CHECK=$("$VENV_PY" -c "import torch; print(torch.__version__)")
if [ "$VENV_TORCH_CHECK" != "$SYSTEM_TORCH_FULL" ]; then
    echo "  venv torch drifted to $VENV_TORCH_CHECK; cleaning again..."
    rm -rf "$VENV_DIR"/lib/python*/site-packages/torch \
           "$VENV_DIR"/lib/python*/site-packages/torchvision \
           "$VENV_DIR"/lib/python*/site-packages/torch-*.dist-info \
           "$VENV_DIR"/lib/python*/site-packages/torchvision-*.dist-info
    VENV_TORCH_CHECK=$("$VENV_PY" -c "import torch; print(torch.__version__)")
    if [ "$VENV_TORCH_CHECK" != "$SYSTEM_TORCH_FULL" ]; then
        echo "  ABORT: after re-cleanup venv torch is $VENV_TORCH_CHECK, expected $SYSTEM_TORCH_FULL"
        exit 1
    fi
fi
echo "  venv torch: $VENV_TORCH_CHECK"

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
    import tilelang
    print(f"  tilelang {getattr(tilelang, '__version__', '?')} OK")
except Exception as e:
    problems.append(f"tilelang: {e}")
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
