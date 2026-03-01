#!/usr/bin/env bash
set -eu

REPO_DIR="/workspace/chimera"

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

# Install dependencies (RunPod templates have torch/CUDA pre-installed)
echo "Installing dependencies..."
pip install -r requirements.txt

# Pull data (labels + captures from public Hub repo)
echo "Pulling data from Hub..."
python scripts/data.py pull

echo ""
echo "=== Setup complete ==="
echo "Run inference with:"
echo "  cd $REPO_DIR"
echo "  python scripts/compare_models.py --samples 5 --models qwen"
