#!/usr/bin/env bash
# Stop or terminate the current RunPod pod via the REST API.
# Use as the last line of a training wrapper so the pod auto-shuts down
# when training finishes (success OR failure) — no idle billing.
#
# Required env vars (read from /workspace/.env if present):
#   RUNPOD_API_KEY   — get from RunPod dashboard → Settings → API Keys.
#                      DO NOT commit. Lives in /workspace/.env (gitignored,
#                      on network volume so persists across pods).
#   RUNPOD_POD_ID    — auto-set by RunPod inside every pod.
#
# Usage:
#   bash scripts/auto_stop_pod.sh           # stop (resumable, storage cost continues)
#   bash scripts/auto_stop_pod.sh terminate # terminate (deletes pod, only volume persists)
#
# Stop vs terminate:
#   stop      — pod definition saved, can resume later. ~$0.10/hr ongoing for the
#               attached storage. GPU charge stops.
#   terminate — pod definition deleted. Network volume persists separately.
#               Cheapest when walking away. Spin up a fresh pod next time.
#
# For our network-volume workflow, terminate is usually right: everything we
# care about (venv, repo, HF cache, data) is on /workspace which survives.

set -euo pipefail

ACTION="${1:-stop}"
if [ "$ACTION" != "stop" ] && [ "$ACTION" != "terminate" ]; then
    echo "Usage: $0 [stop|terminate]"
    exit 1
fi

# Source .env from the network volume if it has one. This is where the
# API key should live — NEVER in a committed file.
if [ -f /workspace/.env ]; then
    set -a
    source /workspace/.env
    set +a
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ABORT: RUNPOD_API_KEY not set."
    echo "       Put it in /workspace/.env on the pod:"
    echo "         echo 'RUNPOD_API_KEY=rpa_...' >> /workspace/.env"
    echo "         chmod 600 /workspace/.env"
    exit 1
fi
if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "ABORT: RUNPOD_POD_ID not set. This script must run inside a RunPod pod."
    exit 1
fi

case "$ACTION" in
    stop)
        URL="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"
        ;;
    terminate)
        # DELETE on the pod resource terminates it.
        URL="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
        ;;
esac

echo "[$(date -Iseconds)] $ACTION pod $RUNPOD_POD_ID via $URL"

if [ "$ACTION" = "terminate" ]; then
    HTTP_METHOD="DELETE"
else
    HTTP_METHOD="POST"
fi

RESPONSE=$(curl -sS -w "\n%{http_code}" -X "$HTTP_METHOD" "$URL" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json")
BODY=$(echo "$RESPONSE" | head -n -1)
CODE=$(echo "$RESPONSE" | tail -n 1)

echo "HTTP $CODE: $BODY"

if [ "$CODE" -ge 200 ] && [ "$CODE" -lt 300 ]; then
    echo "Pod $ACTION request accepted. The pod may take a few seconds to actually shut down."
    exit 0
else
    echo "Pod $ACTION FAILED. You may need to stop the pod manually from the dashboard."
    exit 1
fi
