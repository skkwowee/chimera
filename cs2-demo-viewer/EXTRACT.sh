#!/bin/bash
# Extract cs2-demo-viewer from chimera's site/ directory.
#
# Run from the chimera repo root:
#   bash cs2-demo-viewer/EXTRACT.sh
#
# Prerequisites:
#   - Clean working tree (commit or stash changes first)
#   - GitHub repo 'cs2-demo-viewer' already created (empty)
#
# What this does:
#   1. git subtree split --prefix=site into a temp branch
#   2. Clone into a new directory
#   3. Apply rebranding (package.json, layout.tsx, page.tsx)
#   4. Copy docs and README
#   5. Push to the new repo
set -e

REPO_URL="${1:-git@github.com:skkwowee/cs2-demo-viewer.git}"
WORK_DIR="../cs2-demo-viewer-repo"

echo "=== Extracting cs2-demo-viewer ==="
echo "Target repo: $REPO_URL"
echo "Work dir: $WORK_DIR"
echo ""

# Step 1: Subtree split
echo "Step 1: Subtree split..."
git subtree split --prefix=site -b _demo-viewer-split

# Step 2: Clone into new directory
echo "Step 2: Creating new repo directory..."
rm -rf "$WORK_DIR"
mkdir "$WORK_DIR"
cd "$WORK_DIR"
git init
git pull ../chimera _demo-viewer-split

# Step 3: Apply rebranding
echo "Step 3: Applying rebranding..."
cp ../chimera/cs2-demo-viewer/package.json package.json
cp ../chimera/cs2-demo-viewer/layout.tsx src/app/layout.tsx
cp ../chimera/cs2-demo-viewer/page.tsx src/app/page.tsx

# Step 4: Copy docs and README
echo "Step 4: Copying docs and README..."
mkdir -p docs
cp ../chimera/cs2-demo-viewer/docs/data-format.md docs/
cp ../chimera/cs2-demo-viewer/README.md README.md

# Step 5: Add .gitkeep for data directories
mkdir -p public/viewer-data public/maps
touch public/viewer-data/.gitkeep
touch public/maps/.gitkeep

# Step 6: Commit and push
echo "Step 5: Committing..."
git add -A
git commit -m "Initial extraction from chimera repo

Standalone CS2 demo viewer with radar visualization, vision cones,
kill lines, shot tracers, and timeline scrubbing.

Extracted from: https://github.com/skkwowee/chimera"

echo ""
echo "Step 6: Push to remote..."
git remote add origin "$REPO_URL"
git branch -M main
git push -u origin main

echo ""
echo "=== Done ==="
echo "Repo at: $WORK_DIR"
echo "Remote: $REPO_URL"

# Cleanup temp branch in chimera
cd ../chimera
git branch -D _demo-viewer-split
