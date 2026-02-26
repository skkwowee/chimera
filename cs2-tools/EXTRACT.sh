#!/bin/bash
# Push cs2-tools to its own repo.
#
# Run from the chimera repo root:
#   bash cs2-tools/EXTRACT.sh
#
# Prerequisites:
#   - GitHub repo 'cs2-tools' already created (empty)
set -e

REPO_URL="${1:-git@github.com:skkwowee/cs2-tools.git}"
WORK_DIR="../cs2-tools-repo"

echo "=== Extracting cs2-tools ==="
echo "Target repo: $REPO_URL"
echo "Work dir: $WORK_DIR"
echo ""

# Copy package to new directory
echo "Step 1: Copying package..."
rm -rf "$WORK_DIR"
cp -r cs2-tools "$WORK_DIR"
cd "$WORK_DIR"

# Remove the extraction script from the published repo
rm -f EXTRACT.sh

# Init git and commit
echo "Step 2: Initializing git..."
git init
git add -A
git commit -m "Initial extraction from chimera repo

Python utilities for CS2 demo parsing, viewer data export,
and automated screenshot capture.

Extracted from: https://github.com/skkwowee/chimera"

# Push
echo "Step 3: Pushing to remote..."
git remote add origin "$REPO_URL"
git branch -M main
git push -u origin main

echo ""
echo "=== Done ==="
echo "Repo at: $WORK_DIR"
echo "Remote: $REPO_URL"
