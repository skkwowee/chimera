#!/usr/bin/env bash
set -eu

REPO_DIR="/workspace/chimera"
export HF_HOME="/workspace/.cache/huggingface"

echo "=== Chimera RunPod Setup ==="

# Clone repo
if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists at $REPO_DIR, pulling latest..."
    git -C "$REPO_DIR" pull
else
    echo "Cloning repo..."
    git clone git@github.com:skkwowee/chimera.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Install git-lfs (needed for HF dataset clone)
if ! git lfs version >/dev/null 2>&1; then
    echo "Installing git-lfs..."
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install
fi

# Install uv for fast dependency management
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies (RunPod templates have torch/CUDA pre-installed)
echo "Installing dependencies..."
uv pip install --system -r requirements.txt

# Pull data (labels + captures from public Hub repo)
echo "Pulling data from Hub..."
python scripts/data.py pull --all

echo ""
echo "=== Setup complete ==="
echo "Run inference with:"
echo "  cd $REPO_DIR"
echo "  python scripts/compare_models.py --samples 5 --models qwen"
