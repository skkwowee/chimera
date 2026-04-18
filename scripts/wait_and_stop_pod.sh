#!/usr/bin/env bash
# Watchdog: wait for named background jobs to finish, then stop/terminate the pod.
# Detects two completion conditions:
#   (a) all watched processes are gone (normal completion or crash)
#   (b) any watched log file goes stale for too long (hang detection)
# When either fires, calls auto_stop_pod.sh with the configured action.
#
# Use to walk away from a pod with confidence — pod will shut down whether
# the work finishes, crashes, or silently hangs.
#
# Usage:
#   bash scripts/wait_and_stop_pod.sh [stop|terminate] \
#       --watch-process pod_setup_grpo.sh \
#       --watch-process snapshot_download \
#       --watch-log /workspace/pod_setup.log \
#       --watch-log /workspace/model_download.log \
#       --hang-minutes 30
#
# Defaults: action=stop, hang-minutes=30, log-file=$LOG_FILE if set.
# Polls every 60s. Logs to /workspace/watchdog.log by default.

set -u

ACTION="stop"
WATCH_PROCS=()
WATCH_LOGS=()
HANG_MINUTES=30
POLL_SECONDS=60

while [ $# -gt 0 ]; do
    case "$1" in
        stop|terminate) ACTION="$1"; shift ;;
        --watch-process) WATCH_PROCS+=("$2"); shift 2 ;;
        --watch-log)     WATCH_LOGS+=("$2"); shift 2 ;;
        --hang-minutes)  HANG_MINUTES="$2"; shift 2 ;;
        --poll-seconds)  POLL_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ ${#WATCH_PROCS[@]} -eq 0 ] && [ ${#WATCH_LOGS[@]} -eq 0 ]; then
    echo "ABORT: nothing to watch. Pass --watch-process and/or --watch-log."
    exit 1
fi

HANG_SECONDS=$((HANG_MINUTES * 60))
echo "[$(date -Iseconds)] Watchdog starting. action=$ACTION poll=${POLL_SECONDS}s hang=${HANG_MINUTES}m"
echo "  watch processes: ${WATCH_PROCS[*]:-(none)}"
echo "  watch logs:      ${WATCH_LOGS[*]:-(none)}"

SELF_PID=$$
PARENT_PID=$PPID

while true; do
    # Liveness check: any watched process (other than us) still alive?
    # pgrep -f matches the FULL command line, and our own argv contains the
    # watched patterns. Exclude ourselves and our parent shell.
    ALIVE_PROCS=()
    for pat in "${WATCH_PROCS[@]}"; do
        FOUND=""
        for pid in $(pgrep -f "$pat" 2>/dev/null); do
            if [ "$pid" != "$SELF_PID" ] && [ "$pid" != "$PARENT_PID" ]; then
                FOUND="$pid"
                break
            fi
        done
        if [ -n "$FOUND" ]; then
            ALIVE_PROCS+=("$pat")
        fi
    done

    # Hang check: any watched log file stale for too long?
    STALE_LOGS=()
    NOW=$(date +%s)
    for f in "${WATCH_LOGS[@]}"; do
        if [ -f "$f" ]; then
            MTIME=$(stat -c %Y "$f" 2>/dev/null || echo 0)
            AGE=$((NOW - MTIME))
            if [ "$AGE" -gt "$HANG_SECONDS" ]; then
                STALE_LOGS+=("$f(${AGE}s)")
            fi
        fi
    done

    if [ ${#ALIVE_PROCS[@]} -eq 0 ]; then
        echo "[$(date -Iseconds)] All watched processes gone. Triggering $ACTION."
        REASON="processes-finished"
        break
    fi

    if [ ${#STALE_LOGS[@]} -gt 0 ]; then
        echo "[$(date -Iseconds)] Hang detected: log(s) stale > ${HANG_MINUTES}m: ${STALE_LOGS[*]}"
        echo "  Killing live watched processes: ${ALIVE_PROCS[*]}"
        # CAREFUL: pkill -f matches the FULL command line. Our own command
        # line contains the watched patterns as --watch-process arguments,
        # so pkill -f "$pat" would kill US too. Resolve PIDs first, exclude
        # our own PID and our parent shell, then kill by PID.
        SELF_PID=$$
        PARENT_PID=$PPID
        for pat in "${ALIVE_PROCS[@]}"; do
            for pid in $(pgrep -f "$pat" 2>/dev/null); do
                if [ "$pid" != "$SELF_PID" ] && [ "$pid" != "$PARENT_PID" ]; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        done
        REASON="hang-detected"
        break
    fi

    echo "[$(date -Iseconds)] alive=${ALIVE_PROCS[*]} sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

# Brief pause so any cleanup writes settle on the network volume
sleep 5

REPO_DIR="${REPO_DIR:-/workspace/chimera}"
echo "[$(date -Iseconds)] reason=$REASON  invoking auto_stop_pod.sh $ACTION"
bash "$REPO_DIR/scripts/auto_stop_pod.sh" "$ACTION"
