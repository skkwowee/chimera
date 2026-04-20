#!/usr/bin/env bash
# Run a 5-step GRPO smoke test on the pod with the validated working config.
# Use this AFTER pod_setup_grpo.sh has succeeded.
#
# This is the exact config that landed at ~3.8 min/step (Round 3, 2026-04-18).
# If you change G, max-tokens, or attn-impl and the per-step time blows up,
# this script is the comparison baseline.
#
# Run from /workspace/chimera on the pod:
#   bash scripts/run_grpo_smoke.sh
#
# Output to /workspace/outputs/grpo/smoke and tail the speed_test.log.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/chimera}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
VENV_PY="$VENV_DIR/bin/python"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/grpo/smoke}"
LOG_FILE="${LOG_FILE:-/workspace/grpo_smoke.log}"

# Use FA2 if installed, sdpa otherwise.
if "$VENV_PY" -c "import flash_attn" 2>/dev/null; then
    ATTN_IMPL="flash_attention_2"
else
    ATTN_IMPL="sdpa"
    echo "NOTE: flash_attn not installed — falling back to sdpa (~80% of FA2 speed)"
    echo "      Install with: FLASH_ATTN_CUDA_ARCHS=9.0 MAX_JOBS=8 uv pip install \\"
    echo "        --python $VENV_PY --no-build-isolation flash-attn==2.7.4.post1"
fi

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "START: $(date -Iseconds)"
echo "ATTN_IMPL: $ATTN_IMPL"
echo "Logging to: $LOG_FILE"
echo

# nohup so SSH drops don't kill it. Tail the log to follow.
# Extra CLI args after the script invocation are forwarded to train_grpo.py
# (e.g. --sft-adapter /path/to/adapter to bootstrap from an SFT LoRA).
nohup "$VENV_PY" scripts/train_grpo.py --manual \
    --data data/training/grpo/smoke_test.jsonl \
    --reward-mode recall \
    --num-generations 4 --max-tokens 256 \
    --max-steps 5 --save-steps 100 \
    --logging-steps 1 \
    --max-eval-samples 50 \
    --perception-only \
    --attn-impl "$ATTN_IMPL" \
    --kl-coef 0.02 \
    --output "$OUTPUT_DIR" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "${LOG_FILE%.log}.pid"
sleep 3

if kill -0 "$PID" 2>/dev/null; then
    echo "Launched PID $PID. Tail with:"
    echo "  tail -f $LOG_FILE"
    echo "Expected: ~3.8 min/step → ~20 min total for 5 steps."
else
    echo "ERROR: process died immediately. Last 20 lines of log:"
    tail -20 "$LOG_FILE"
    exit 1
fi
