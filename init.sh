#!/bin/bash
# Quick environment setup for chimera project
# Run this at the start of a new session to verify everything works.
set -e

cd "$(dirname "$0")"

echo "=== Chimera project init ==="

# Python venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# Python deps (pip may not exist in venv — skip if already installed)
python3 -c "import polars, pyarrow, awpy" 2>/dev/null || {
  echo "Installing Python deps..."
  python3 -m pip install -q -r requirements.txt
}

# Ensure output dirs exist
mkdir -p data/processed/demos

# Quick health checks
echo ""
echo "=== Health checks ==="
python3 -c "import polars; print(f'  polars {polars.__version__}')"
python3 -c "import pyarrow; print(f'  pyarrow {pyarrow.__version__}')"
python3 -c "import awpy; print(f'  awpy {awpy.__version__}')"
python3 -c "import cs2_tools; print(f'  cs2-tools ok')" 2>/dev/null || echo "  cs2-tools: NOT INSTALLED"
ls data/processed/demos/*_ticks.parquet 2>/dev/null | wc -l | xargs -I{} echo "  {} parquet files in data/processed/demos/"

echo ""
echo "=== Ready ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Parse demos:    cs2-parse-demos data/demos/ --output data/processed/demos"
echo "Export viewer:  cs2-export-viewer --input data/processed/demos --output viewer-data"
echo ""
