#!/usr/bin/env bash
# Launch GRPO training on a RunPod pod with the right env vars and survives SSH drops.
#
# Sets:
#   TORCHDYNAMO_DISABLE=1   — skip the 30 min inductor compile storm. With the fast
#                             kernels present, inductor is not winning anything for
#                             Qwen3.5 MoE (graph breaks on dynamic expert routing).
#   PYTHONUTF8=1            — TRL model card writer chokes on default locale
#   PYTHONUNBUFFERED=1      — see logs in real time
#
# Usage:
#   ./scripts/run_grpo_pod.sh 'ssh root@x.x.x.x -p 12345 -i ~/.ssh/key' \
#       --model-name /workspace/outputs/sft/final_model \
#       --data data/training/grpo/smoke_test.jsonl \
#       --reward-mode recall --max-steps 100
#
# Monitor:
#   ssh ...host... 'tail -f /workspace/outputs/grpo/train_*.log'

set -euo pipefail

usage() {
    echo "Usage: $0 '<ssh-string>' [extra train_grpo.py args...]"
    echo
    echo "Pre-flight: scripts/pod_setup_grpo.sh must have run successfully on the pod."
    exit 1
}

[ $# -lt 1 ] && usage

SSH_STRING="$1"
shift
EXTRA_ARGS=("$@")

# --- Parse SSH string ---
CONN="${SSH_STRING#ssh }"
USER_HOST=""
PORT=""
IDENTITY=""
tokens=($CONN)
i=0
while [ $i -lt ${#tokens[@]} ]; do
    case "${tokens[$i]}" in
        -p) PORT="${tokens[$((i+1))]}"; i=$((i+2)) ;;
        -i) IDENTITY="${tokens[$((i+1))]}"; i=$((i+2)) ;;
        *)  [ -z "$USER_HOST" ] && USER_HOST="${tokens[$i]}"; i=$((i+1)) ;;
    esac
done

if [ -z "$USER_HOST" ]; then
    echo "Error: could not parse user@host from SSH string"
    exit 1
fi

[ -n "$IDENTITY" ] && IDENTITY="${IDENTITY/#\~/$HOME}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
[ -n "$PORT" ] && SSH_OPTS+=(-p "$PORT")
[ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")

ssh_cmd() { ssh "${SSH_OPTS[@]}" "$USER_HOST" "$@"; }

REPO_DIR="/workspace/chimera"
OUTPUT_DIR="/workspace/outputs/grpo"

echo "Checking SSH..."
if ! ssh_cmd "echo ok" >/dev/null 2>&1; then
    echo "Error: cannot connect to $USER_HOST"
    exit 1
fi
echo "Connected."

echo
echo "=== Pulling latest code ==="
ssh_cmd "cd $REPO_DIR && git pull origin main 2>&1 | tail -3"

# --- Re-verify kernels before launching (cheap, catches a stale env) ---
echo
echo "=== Re-verifying fast-path kernels on pod ==="
ssh_cmd "python3 -c '
import sys
ok = True
for mod, label in [(\"causal_conv1d\", \"causal_conv1d\"), (\"fla\", \"flash-linear-attention\")]:
    try:
        __import__(mod)
        print(f\"  {label}: OK\")
    except Exception as e:
        print(f\"  {label}: FAIL ({e})\")
        ok = False
sys.exit(0 if ok else 1)
'" || {
    echo
    echo "ABORT: kernels not loadable on pod. Re-run scripts/pod_setup_grpo.sh."
    echo "       Override with --allow-slow-fallback if you really mean to (you don't)."
    exit 1
}

# --- Build training command ---
ARGS_STR=""
[ ${#EXTRA_ARGS[@]} -gt 0 ] && ARGS_STR="${EXTRA_ARGS[*]}"

TRAIN_CMD="cd $REPO_DIR && \
    TORCHDYNAMO_DISABLE=1 \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=$REPO_DIR \
    python3 scripts/train_grpo.py --manual $ARGS_STR"

echo
echo "=== Launching GRPO (nohup) ==="
echo "Command: $TRAIN_CMD"
echo

ssh_cmd "mkdir -p $OUTPUT_DIR && nohup bash -c '$TRAIN_CMD' > /workspace/grpo_stdout.log 2>&1 &
    PID=\$!
    echo \$PID > /workspace/grpo_train.pid
    echo \"Training started with PID \$PID\"
    sleep 3
    if kill -0 \$PID 2>/dev/null; then
        echo 'Process is running.'
    else
        echo 'ERROR: Process died immediately. Last 30 lines of log:'
        tail -30 /workspace/grpo_stdout.log
        exit 1
    fi"

echo
echo "=== Launched ==="
echo "Monitor:        ssh ${SSH_OPTS[*]} $USER_HOST 'tail -f /workspace/grpo_stdout.log'"
echo "GPU:            ssh ${SSH_OPTS[*]} $USER_HOST 'nvidia-smi'"
echo "Still running?: ssh ${SSH_OPTS[*]} $USER_HOST 'kill -0 \$(cat /workspace/grpo_train.pid) && echo running || echo stopped'"
echo "Kill:           ssh ${SSH_OPTS[*]} $USER_HOST 'kill \$(cat /workspace/grpo_train.pid)'"
