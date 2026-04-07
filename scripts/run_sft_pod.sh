#!/usr/bin/env bash
set -euo pipefail

# Run SFT training on a RunPod instance with proper process management.
# Uses nohup so training survives SSH disconnects.
#
# Usage:
#   ./scripts/run_sft_pod.sh 'ssh root@x.x.x.x -p 12345 -i ~/.ssh/key'
#   ./scripts/run_sft_pod.sh 'ssh root@x.x.x.x -p 12345' --resume /workspace/outputs/sft/checkpoint-150
#
# Monitor:
#   ssh root@x.x.x.x -p 12345 'tail -f /workspace/outputs/sft/train_*.log'

usage() {
    echo "Usage: $0 '<ssh-string>' [extra train_sft.py args...]"
    echo ""
    echo "Run SFT training on a RunPod pod."
    echo "Training runs under nohup so it survives SSH disconnects."
    echo ""
    echo "Examples:"
    echo "  $0 'ssh root@x.x.x.x -p 12345 -i ~/.ssh/key'"
    echo "  $0 'ssh root@x.x.x.x -p 12345' --resume /workspace/outputs/sft/checkpoint-150"
    echo ""
    echo "Monitor training:"
    echo "  ssh root@x.x.x.x -p 12345 'tail -f /workspace/outputs/sft/train_*.log'"
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
DATA_DIR="/workspace/chimera-data"
OUTPUT_DIR="/workspace/outputs/sft"

# --- Check connectivity ---
echo "Checking SSH connectivity to $USER_HOST..."
if ! ssh_cmd "echo ok" >/dev/null 2>&1; then
    echo "Error: cannot connect to $USER_HOST"
    exit 1
fi
echo "Connected."

# --- Pull latest code ---
echo ""
echo "=== Pulling latest code ==="
ssh_cmd "cd $REPO_DIR && git pull origin main 2>&1 | tail -3"

# --- Check data ---
echo ""
echo "=== Data check ==="
ssh_cmd "for d in $DATA_DIR/captures/*/; do
    name=\$(basename \$d)
    imgs=\$(find \$d/raw -name '*.jpg' 2>/dev/null | wc -l)
    labs=\$(find \$d/labels -name '*.json' 2>/dev/null | wc -l)
    echo \"  \$name: \$imgs screenshots, \$labs labels\"
done"

# --- Build SFT dataset if needed ---
echo ""
echo "=== Building SFT dataset ==="
ssh_cmd "cd $REPO_DIR && PYTHONPATH=$REPO_DIR python3 scripts/build_sft_dataset.py \
    --captures-dir $DATA_DIR/captures \
    --output data/sft_dataset.json \
    --report data/sft_coverage_report.txt 2>&1 | tail -5"

# --- Build training command ---
ARGS_STR=""
[ ${#EXTRA_ARGS[@]} -gt 0 ] && ARGS_STR="${EXTRA_ARGS[*]}"

TRAIN_CMD="cd $REPO_DIR && PYTHONPATH=$REPO_DIR PYTHONUNBUFFERED=1 python3 scripts/train_sft.py \
    --dataset data/sft_dataset.json \
    --output $OUTPUT_DIR \
    --epochs 1 \
    --lora-r 4 \
    --lora-alpha 8 \
    --lora-dropout 0.1 \
    --lr 1e-5 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --save-steps 50 \
    --logging-steps 5 \
    --seed 42 \
    $ARGS_STR"

# --- Launch training with nohup ---
echo ""
echo "=== Launching SFT training (nohup) ==="
echo "Command: $TRAIN_CMD"
echo ""

ssh_cmd "nohup bash -c '$TRAIN_CMD' > /workspace/sft_stdout.log 2>&1 &
    PID=\$!
    echo \$PID > /workspace/sft_train.pid
    echo \"Training started with PID \$PID\"
    echo \"Logs: $OUTPUT_DIR/train_*.log (structured) and /workspace/sft_stdout.log (raw)\"
    sleep 2
    if kill -0 \$PID 2>/dev/null; then
        echo 'Process is running.'
    else
        echo 'ERROR: Process died immediately. Check /workspace/sft_stdout.log'
        tail -20 /workspace/sft_stdout.log
        exit 1
    fi"

echo ""
echo "=== Training launched ==="
echo ""
echo "Monitor with:"
echo "  ssh ${SSH_OPTS[*]} $USER_HOST 'tail -f $OUTPUT_DIR/train_*.log'"
echo ""
echo "Check GPU usage:"
echo "  ssh ${SSH_OPTS[*]} $USER_HOST 'nvidia-smi'"
echo ""
echo "Check if still running:"
echo "  ssh ${SSH_OPTS[*]} $USER_HOST 'kill -0 \$(cat /workspace/sft_train.pid) && echo running || echo stopped'"
echo ""
echo "Kill training:"
echo "  ssh ${SSH_OPTS[*]} $USER_HOST 'kill \$(cat /workspace/sft_train.pid)'"
