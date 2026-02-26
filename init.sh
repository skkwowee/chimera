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
  python3 -m pip install -q polars pyarrow "awpy>=2.0.0"
}

# Node deps (site)
if [ ! -d site/node_modules ]; then
  echo "Installing Node deps..."
  cd site && npm install && cd ..
fi

# Ensure awpy maps are downloaded
if [ ! -f ~/.awpy/maps/map-data.json ]; then
  echo "Downloading radar maps..."
  awpy get maps
fi

# Ensure output dirs exist
mkdir -p data/processed/demos
mkdir -p site/public/viewer-data
mkdir -p site/public/maps

# Quick health checks
echo ""
echo "=== Health checks ==="
python3 -c "import polars; print(f'  polars {polars.__version__}')"
python3 -c "import pyarrow; print(f'  pyarrow {pyarrow.__version__}')"
python3 -c "import awpy; print(f'  awpy {awpy.__version__}')"
echo "  node $(node --version)"
ls data/processed/demos/*_ticks.parquet 2>/dev/null | wc -l | xargs -I{} echo "  {} parquet files in data/processed/demos/"
ls site/public/viewer-data/*/round_*.json 2>/dev/null | wc -l | xargs -I{} echo "  {} round JSONs in site/public/viewer-data/"

echo ""
echo "=== Ready ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Parse demos:    python scripts/parse_demos.py data/demos/"
echo "Export viewer:  python scripts/export_viewer_data.py"
echo "Dev server:     cd site && npm run dev -- -H 0.0.0.0"
echo "Prod server:    cd site && npm run build && npm run start -- -H 0.0.0.0"
echo ""
echo "Next task: read feature-list.json, work on first feature where passes=false"
echo "Extraction harness: bash harness/init.sh"
