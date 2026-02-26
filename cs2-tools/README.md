# cs2-tools

Python utilities for working with Counter-Strike 2 demo files. Parse `.dem` files into structured data, export viewer-ready JSON, and automate screenshot capture from demo playback.

Extracted from the [chimera](https://github.com/skkwowee/chimera) research project.

## Tools

| Tool | Description |
|------|-------------|
| `cs2_tools.netcon` | TCP client for CS2's network console (`-netconport`) |
| `cs2_tools.parse_demos` | Parse `.dem` files into Parquet tick data + metadata JSONs via [awpy](https://github.com/pnxenopoulos/awpy) |
| `cs2_tools.export_viewer_data` | Export parsed data as compact per-round JSON for [cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer) |
| `cs2_tools.plan_captures` | Generate capture plans (which ticks + POVs to screenshot) |
| `cs2_tools.capture_screenshots` | Drive CS2 demo playback via netcon to capture JPEGs |

## Install

```bash
# Core only (netcon + capture_screenshots — no heavy deps)
pip install -e .

# With parsing support (awpy, polars, pyarrow)
pip install -e ".[parse]"
```

## Usage

### Parse demos into Parquet

```bash
cs2-parse-demos data/demos/
cs2-parse-demos data/demos/match.dem --dry-run
cs2-parse-demos data/demos/ --output data/processed/demos
```

Produces per-demo: `{stem}_ticks.parquet`, `{stem}_rounds.json`, `{stem}_kills.json`, `{stem}_damages.json`, `{stem}_shots.json`, `{stem}_bomb.json`, `{stem}_header.json`.

### Export viewer data

```bash
cs2-export-viewer --input data/processed/demos --output viewer-data --maps-output maps
```

Produces `index.json`, per-demo `meta.json`, and per-round `round_XX.json` files consumable by cs2-demo-viewer.

### Plan screenshot captures

```bash
cs2-plan-captures data/processed/demos/furia-vs-vitality-m1-mirage
cs2-plan-captures data/processed/demos/ --all --dry-run
```

Samples every ~3 seconds during active rounds plus event ticks (first kill, bomb plant). For each tick, selects 1 T-side + 1 CT-side player to spectate.

### Capture screenshots

Requires CS2 running on Windows with `-netconport 2121 -console -novid`.

```bash
cs2-capture data/captures/furia-vs-vitality-m1-mirage/capture_plan.json
cs2-capture capture_plan.json --resume --limit 100
```

### Netcon client (library use)

```python
from cs2_tools.netcon import CS2Netcon

with CS2Netcon(port=2121) as con:
    con.playdemo("match-demo")
    con.goto_tick(5000)
    con.spec_player("ZywOo")
    con.screenshot("round01_tick5000_zywoo")
```

## License

MIT
