#!/bin/bash
# Chimera session bootstrap. Run at the start of a new session.
# Uses .venv/bin/python explicitly (never `source activate` — activate scripts
# hardcode the venv's creation path and break when the repo moves; preflight B0).
set -e

cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== Chimera project init ==="

# Venv: must exist and be import-healthy. We do NOT silently reinstall —
# a broken venv should be rebuilt deliberately from the lock (see below).
if [ ! -x "$PY" ]; then
  echo "  .venv missing/broken. Rebuild from lock:"
  echo "    uv venv && uv sync --frozen && uv add --dev pytest"
  exit 1
fi
"$PY" -c "import polars, pyarrow, awpy, torch" 2>/dev/null || {
  echo "  venv imports failing. Rebuild from lock: uv venv && uv sync --frozen"
  exit 1
}

mkdir -p data/processed/demos

echo ""
echo "=== Health checks ==="
"$PY" - <<'EOF'
import polars, pyarrow, awpy, torch
from importlib.metadata import version as v
print(f"  polars {polars.__version__} | pyarrow {pyarrow.__version__} | awpy {awpy.__version__}")
print(f"  torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
dp = v("demoparser2")
assert tuple(map(int, dp.split("."))) >= (0, 41, 3), f"demoparser2 {dp} < 0.41.3 floor"
print(f"  demoparser2 {dp} (>=0.41.3 floor OK)")
EOF
ls data/processed/demos/*_ticks.parquet 2>/dev/null | wc -l | xargs -I{} echo "  {} parquet files in data/processed/demos/"
"$PY" -c "from huggingface_hub import HfApi; HfApi().whoami(); print('  HF Hub: authenticated')" 2>/dev/null || echo "  HF Hub: not logged in (run: huggingface-cli login)"

echo ""
echo "=== Where to start ==="
echo "READ FIRST: top of claude-progress.txt — the ORDERED RUNBOOK with done-checks."
echo "Then docs/preflight-report.md §5 — the merged Phase 0-9 execution checklist."
echo "Recipe is LOCKED (docs/retrain-recipe.md, Knobs 1-7); no GPU/pod spend without explicit owner go."
