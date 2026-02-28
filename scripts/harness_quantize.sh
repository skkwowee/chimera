#!/usr/bin/env bash
# ============================================================
# Harness: Pull pre-quantized Qwen3.5-27B VLM from Hub
# ============================================================
#
# Downloads the BnB NF4 checkpoint created by cloud_quantize.py.
# Run this locally on your RTX 4090 machine.
#
# The checkpoint was quantized on a cloud GPU (A100 80GB) because
# local quantization OOMs — BnB materializes bf16 weights on GPU
# before quantizing, ignoring max_memory constraints.
#
# PREREQUISITES:
#   1. Cloud quantization already done (see harness_cloud_quantize.sh)
#   2. ~20GB disk for the checkpoint
#
# RUN:
#   bash scripts/harness_quantize.sh
#
set -e

echo "============================================================"
echo "Pull pre-quantized Qwen3.5-27B VLM from Hub"
echo "============================================================"
echo ""

# Disk space check
AVAIL_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
echo "Available disk: ${AVAIL_GB}GB (need ~20GB)"
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "ERROR: Not enough disk space."
    exit 1
fi
echo ""

# Pull checkpoint
uv run python scripts/quantize_base_model.py

echo ""
echo "Next steps:"
echo "  1. Update config.yaml model_name to: models/Qwen3.5-27B-bnb-4bit"
echo "  2. Test inference: uv run python scripts/test_qwen35_inference.py"
