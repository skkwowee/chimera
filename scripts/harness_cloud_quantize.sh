#!/usr/bin/env bash
# ============================================================
# Harness: Quantize Qwen3.5-27B VLM on cloud GPU
# ============================================================
#
# Self-contained script to run on a rented cloud GPU.
# SCP this + cloud_quantize.py to the machine, then run.
#
# WHAT THIS DOES:
#   1. Installs Python dependencies
#   2. Logs into HuggingFace Hub
#   3. Quantizes Qwen3.5-27B (full VLM) to BnB NF4
#   4. Pushes checkpoint to HF Hub (skkwowee/chimera-cs2-qwen3.5)
#   5. Verifies the upload
#
# REQUIREMENTS:
#   - GPU with >= 48GB VRAM (A100 80GB ideal)
#   - Python 3.10+, pip, CUDA drivers (pre-installed on cloud VMs)
#   - HF_TOKEN env var set (write access to Hub repo)
#
# ---- PROVIDER SETUP ----
#
# LAMBDA LABS ($1.25/hr A100 80GB):
#   - https://cloud.lambdalabs.com
#   - Add SSH key in dashboard, launch 1x A100 instance
#   - SSH in: ssh ubuntu@<ip>
#   - Python, CUDA, PyTorch pre-installed
#   - scp scripts/cloud_quantize.py scripts/harness_cloud_quantize.sh ubuntu@<ip>:~/
#   - HF_TOKEN=hf_xxx bash harness_cloud_quantize.sh
#   - Terminate instance when done
#
# RUNPOD ($1.64/hr community, $2.49/hr secure):
#   - https://runpod.io
#   - Deploy > GPU Pod > A100 80GB > PyTorch template
#   - Use web terminal or SSH
#   - scp scripts/cloud_quantize.py scripts/harness_cloud_quantize.sh root@<ip>:~/
#   - HF_TOKEN=hf_xxx bash harness_cloud_quantize.sh
#   - Stop pod when done
#
# ESTIMATED TIME:
#   ~20-40 min total (download ~55GB + quantize + upload ~18GB)
#
# ESTIMATED COST:
#   Lambda: ~$1-2 total
#   RunPod: ~$1-3 total
#
# ============================================================
set -e

echo "============================================================"
echo "Cloud Quantization: Qwen3.5-27B VLM → BnB NF4"
echo "============================================================"
echo ""

# --- Step 1: Check GPU ---
echo "=== Step 1: GPU check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$GPU_MEM" -lt 40000 ]; then
    echo "WARNING: ${GPU_MEM}MB VRAM detected. Recommended: 48GB+"
    echo "The full bf16 model is ~55GB. May OOM on this GPU."
    read -p "Continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# --- Step 2: Install dependencies ---
echo "=== Step 2: Install dependencies ==="
pip install --quiet transformers bitsandbytes accelerate huggingface-hub psutil
echo "Dependencies installed."
echo ""

# --- Step 3: HuggingFace login ---
echo "=== Step 3: HuggingFace login ==="
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "Usage: HF_TOKEN=hf_xxx bash harness_cloud_quantize.sh"
    echo ""
    echo "Get a write token at: https://huggingface.co/settings/tokens"
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"
echo ""

# --- Step 4: Run quantization ---
echo "=== Step 4: Quantize and push ==="
echo ""
python cloud_quantize.py

echo ""
echo "=== Step 5: Verify upload ==="
pip install --quiet huggingface-hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('skkwowee/chimera-cs2-qwen3.5')
safetensors = [f for f in files if f.endswith('.safetensors')]
print(f'Files on Hub: {len(files)} total, {len(safetensors)} safetensors')
if 'quantize_meta.json' in files:
    import json
    from huggingface_hub import hf_hub_download
    meta = json.load(open(hf_hub_download('skkwowee/chimera-cs2-qwen3.5', 'quantize_meta.json')))
    print(f'Parameters: {meta[\"parameter_count_B\"]}')
    print(f'Vision encoder: {meta[\"includes_vision_encoder\"]}')
    print(f'Model class: {meta[\"model_class\"]}')
print('Verification OK.')
"

echo ""
echo "============================================================"
echo "SUCCESS! Checkpoint pushed to HuggingFace Hub."
echo ""
echo "On your local machine, pull it with:"
echo "  python scripts/data.py pull --model"
echo ""
echo "You can now terminate this cloud instance."
echo "============================================================"
