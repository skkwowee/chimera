#!/usr/bin/env bash
# Run a GRPO training command, then auto-stop the pod on completion.
# Use this when kicking off a long training run before walking away.
#
# Usage on pod:
#   bash scripts/run_grpo_with_auto_stop.sh stop \
#       /workspace/venv/bin/python scripts/train_grpo.py --manual \
#         --data data/training/grpo/smoke_test.jsonl \
#         --reward-mode recall --max-steps 100 ...
#
# First arg is "stop" or "terminate" (passed to auto_stop_pod.sh).
# Rest is the training command + args, run via nohup.
#
# Pod stops itself when training exits — success OR failure. No idle billing
# while you sleep. Logs go to /workspace/grpo_run.log; pod-stop result to
# /workspace/auto_stop.log.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 {stop|terminate} <training command...>"
    exit 1
fi

STOP_ACTION="$1"
shift

REPO_DIR="${REPO_DIR:-/workspace/chimera}"
LOG="${LOG:-/workspace/grpo_run.log}"
STOP_LOG="${STOP_LOG:-/workspace/auto_stop.log}"

cd "$REPO_DIR"

# Wrap the user command + auto-stop in a single bash that nohup keeps alive.
# The auto-stop runs regardless of whether training succeeds (|| true), so a
# crash still releases the pod.
#
# Env vars needed by train_grpo.py (matching run_grpo_smoke.sh):
#   PYTHONPATH=$REPO_DIR  — so `from src.training import ...` resolves
#   CUDA_HOME, PATH, LD_LIBRARY_PATH — for any kernel/CUDA lookups at runtime
#   TORCHDYNAMO_DISABLE=1 — skip the inductor compile storm on Qwen3.5 MoE
#   PYTHONUTF8=1 — TRL model card writer is locale-fragile
#   PYTHONUNBUFFERED=1 — so stdout shows up in $LOG live
nohup bash -c "
    cd $REPO_DIR
    export PYTHONPATH=$REPO_DIR
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:\$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
    export TORCHDYNAMO_DISABLE=1
    export PYTHONUTF8=1
    export PYTHONUNBUFFERED=1
    export HF_HOME=\${HF_HOME:-/workspace/.cache/huggingface}
    # Source /workspace/.env for ANTHROPIC_API_KEY (judge reward) etc.
    if [ -f /workspace/.env ]; then
        set -a
        source /workspace/.env
        set +a
    fi
    echo '[START]' \$(date -Iseconds) >> '$LOG'
    $* >> '$LOG' 2>&1
    EXIT=\$?
    echo '[END]' \$(date -Iseconds) exit=\$EXIT >> '$LOG'
    bash $REPO_DIR/scripts/auto_stop_pod.sh '$STOP_ACTION' >> '$STOP_LOG' 2>&1 || true
" > /dev/null 2>&1 &

PID=$!
echo "$PID" > /workspace/grpo_run.pid
sleep 2

if kill -0 "$PID" 2>/dev/null; then
    echo "Launched. PID $PID. Pod will $STOP_ACTION when training exits."
    echo "  Training log: $LOG"
    echo "  Auto-stop log: $STOP_LOG"
    echo "  Tail with:   tail -f $LOG"
    echo "  Cancel:      kill \$(cat /workspace/grpo_run.pid)  # then manually stop pod"
else
    echo "ERROR: process died immediately. Last log lines:"
    tail -20 "$LOG" 2>/dev/null
    exit 1
fi
