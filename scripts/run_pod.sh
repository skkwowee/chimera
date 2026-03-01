#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 '<ssh-string>' [compare_models.py args...]"
    echo ""
    echo "Automate RunPod H200 inference: setup, run, download results."
    echo ""
    echo "Examples:"
    echo "  $0 'ssh root@x.x.x.x -p 12345 -i ~/.ssh/key' --samples 10 --models qwen"
    echo "  $0 'ssh root@x.x.x.x -p 12345' --samples 5"
    exit 1
}

[ $# -lt 1 ] && usage

SSH_STRING="$1"
shift
EXTRA_ARGS=("$@")

# --- Parse SSH string ---
# Expected format: ssh [user@host] [-p port] [-i keypath]

# Strip leading "ssh " if present
CONN="${SSH_STRING#ssh }"

# Extract user@host (first non-flag token)
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

# --- Build SSH/SCP helpers ---
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10)
[ -n "$PORT" ] && SSH_OPTS+=(-p "$PORT")
[ -n "$IDENTITY" ] && SSH_OPTS+=(-i "$IDENTITY")

ssh_cmd() { ssh "${SSH_OPTS[@]}" "$USER_HOST" "$@"; }
scp_cmd() {
    local scp_opts=("${SSH_OPTS[@]}")
    # scp uses -P not -p for port
    for i in "${!scp_opts[@]}"; do
        [ "${scp_opts[$i]}" = "-p" ] && scp_opts[$i]="-P"
    done
    scp "${scp_opts[@]}" "$@"
}

REPO_DIR="/workspace/chimera"

# --- Check connectivity ---
echo "Checking SSH connectivity to $USER_HOST..."
if ! ssh_cmd "echo ok" >/dev/null 2>&1; then
    echo "Error: cannot connect to $USER_HOST"
    exit 1
fi
echo "Connected."

# --- Setup ---
echo ""
echo "=== Setting up pod ==="
ssh_cmd "bash -lc '
    if [ -d $REPO_DIR ]; then
        echo \"Repo exists, pulling latest...\"
        git -C $REPO_DIR pull
    else
        echo \"Cloning repo...\"
        git clone git@github.com:skkwowee/chimera.git $REPO_DIR
    fi
    cd $REPO_DIR && bash scripts/pod_setup.sh
'"

# --- Inference ---
echo ""
echo "=== Running inference ==="
ARGS_STR=""
[ ${#EXTRA_ARGS[@]} -gt 0 ] && ARGS_STR="${EXTRA_ARGS[*]}"

ssh_cmd "bash -lc 'cd $REPO_DIR && python scripts/compare_models.py $ARGS_STR'"

# --- Download results ---
echo ""
echo "=== Downloading results ==="
mkdir -p data/predictions

# Get the latest comparison file from the pod
REMOTE_FILE=$(ssh_cmd "ls -t $REPO_DIR/data/predictions/comparison_*.json 2>/dev/null | head -1")

if [ -z "$REMOTE_FILE" ]; then
    echo "Error: no comparison results found on pod"
    exit 1
fi

BASENAME=$(basename "$REMOTE_FILE")
LOCAL_PATH="data/predictions/$BASENAME"
scp_cmd "$USER_HOST:$REMOTE_FILE" "$LOCAL_PATH"

echo ""
echo "=== Done ==="
echo "Results: $LOCAL_PATH"
