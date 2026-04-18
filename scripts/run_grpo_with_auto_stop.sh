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
nohup bash -c "
    cd $REPO_DIR
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
