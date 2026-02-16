#!/bin/bash
# Harness initializer for CS2 Demo Viewer extraction.
# Run at the start of each session to verify state and orient.
set -e

cd "$(dirname "$0")"

echo "=== CS2 Demo Viewer Extraction Harness ==="
echo ""

# Check we're in the right place
if [ ! -f feature-list.json ]; then
  echo "ERROR: feature-list.json not found. Run from viewer-harness/"
  exit 1
fi

# Check chimera site/ exists and builds
SITE_DIR="../site"
if [ ! -d "$SITE_DIR" ]; then
  echo "ERROR: ../site/ not found"
  exit 1
fi

echo "=== Health checks ==="

# Node deps
if [ ! -d "$SITE_DIR/node_modules" ]; then
  echo "Installing Node deps..."
  cd "$SITE_DIR" && npm install && cd ..
fi
echo "  node $(node --version)"
echo "  site/ exists: yes"

# Check if subtree branch exists already
if git rev-parse --verify demo-viewer >/dev/null 2>&1; then
  echo "  demo-viewer branch: exists"
else
  echo "  demo-viewer branch: not yet created"
fi

# Check if standalone repo exists
if [ -d "$HOME/cs2-demo-viewer" ]; then
  echo "  ~/cs2-demo-viewer/: exists"
else
  echo "  ~/cs2-demo-viewer/: not yet created"
fi

# TypeScript check on site/
echo ""
echo "=== TypeScript check ==="
cd "$SITE_DIR"
./node_modules/.bin/tsc --noEmit && echo "  tsc: OK" || echo "  tsc: FAILED"
cd ..

# Feature status
echo ""
echo "=== Feature status ==="
python3 -c "
import json
with open('viewer-harness/feature-list.json') as f:
    features = json.load(f)
done = sum(1 for f in features if f['passes'])
total = len(features)
print(f'  {done}/{total} features passing')
for f in features:
    status = 'PASS' if f['passes'] else 'TODO'
    print(f\"  [{status}] {f['id']} — {f['name']}\")
"

echo ""
echo "=== Ready ==="
echo "Read viewer-harness/claude-progress.txt for context."
echo "Work on the first feature where passes=false."
echo ""
