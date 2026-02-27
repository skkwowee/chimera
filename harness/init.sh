#!/bin/bash
# Unified harness for chimera repo extractions.
# Covers both cs2-demo-viewer and cs2-tools extraction.
# Run at the start of each session to verify state and orient.
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== Chimera Extraction Harness ==="
echo ""

# ─── Health checks ───

echo "=== Health checks ==="

# Python deps
python3 -c "import polars" 2>/dev/null && echo "  polars: ok" || echo "  polars: NOT INSTALLED"
python3 -c "import awpy" 2>/dev/null && echo "  awpy: ok" || echo "  awpy: NOT INSTALLED"

# cs2-tools installed
python3 -c "import cs2_tools" 2>/dev/null && echo "  cs2-tools: installed" || echo "  cs2-tools: NOT INSTALLED (pip install cs2-tools[parse])"

# Published repos
echo "  cs2-demo-viewer: https://github.com/skkwowee/cs2-demo-viewer"
echo "  cs2-tools: https://github.com/skkwowee/cs2-tools"


# ─── Feature status ───

echo ""
echo "=== Viewer extraction (cs2-demo-viewer) ==="
python3 -c "
import json
with open('harness/viewer-features.json') as f:
    features = json.load(f)
done = sum(1 for f in features if f['passes'])
total = len(features)
print(f'  {done}/{total} features passing')
for f in features:
    status = 'PASS' if f['passes'] else 'TODO'
    print(f\"  [{status}] {f['id']} — {f['name']}\")
"

echo ""
echo "=== Tools extraction (cs2-tools) ==="
python3 -c "
import json
with open('harness/tools-features.json') as f:
    features = json.load(f)
done = sum(1 for f in features if f['passes'])
total = len(features)
print(f'  {done}/{total} features passing')
for f in features:
    status = 'PASS' if f['passes'] else 'TODO'
    print(f\"  [{status}] {f['id']} — {f['name']}\")
"

# ─── Decision log ───

echo ""
echo "=== Decision log ==="
DECISION_COUNT=$(grep -c '^## D[0-9]' "$ROOT/docs/decisions.md" 2>/dev/null || echo 0)
LAST_ID=$(grep -o '^## D[0-9]*' "$ROOT/docs/decisions.md" 2>/dev/null | tail -1 | sed 's/## //')
NEXT_NUM=$((10#${LAST_ID#D} + 1))
NEXT_ID=$(printf "D%03d" "$NEXT_NUM")
echo "  $DECISION_COUNT decisions logged"
echo "  Next ID: $NEXT_ID"

echo ""
echo "=== Ready ==="
echo "Extraction complete — both repos published."
echo "Log decisions in docs/decisions.md (next: $NEXT_ID)."
echo ""
