#!/usr/bin/env bash
# Run a GRPO training command under nohup.
# Optionally auto-stop/terminate the pod on completion (opt-in).
#
# Usage on pod (keep pod alive, default):
#   bash scripts/run_grpo_with_auto_stop.sh none \
#       /workspace/venv/bin/python scripts/train_grpo.py --manual ...
#
# Usage on pod (auto-stop when done):
#   bash scripts/run_grpo_with_auto_stop.sh stop \
#       /workspace/venv/bin/python scripts/train_grpo.py --manual ...
#
# First arg is "none" (default, keep pod), "stop", or "terminate".
# Rest is the training command + args, run via nohup.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 {none|stop|terminate} <training command...>"
    exit 1
fi

STOP_ACTION="$1"
shift

case "$STOP_ACTION" in
    none|stop|terminate) ;;
    *) echo "ERROR: first arg must be {none|stop|terminate}, got '$STOP_ACTION'"; exit 1 ;;
esac

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
    if [ '$STOP_ACTION' != 'none' ]; then
        bash $REPO_DIR/scripts/auto_stop_pod.sh '$STOP_ACTION' >> '$STOP_LOG' 2>&1 || true
    fi
" > /dev/null 2>&1 &

PID=$!
echo "$PID" > /workspace/grpo_run.pid
sleep 2

if kill -0 "$PID" 2>/dev/null; then
    if [ "$STOP_ACTION" = "none" ]; then
        echo "Launched. PID $PID. Pod will stay alive when training exits."
    else
        echo "Launched. PID $PID. Pod will $STOP_ACTION when training exits."
    fi
    echo "  Training log: $LOG"
    echo "  Auto-stop log: $STOP_LOG"
    echo "  Tail with:   tail -f $LOG"
    echo "  Cancel:      kill \$(cat /workspace/grpo_run.pid)  # then manually stop pod"
else
    echo "ERROR: process died immediately. Last log lines:"
    tail -20 "$LOG" 2>/dev/null
    exit 1
fi
