# cs2-demo-viewer

Interactive browser-based viewer for Counter-Strike 2 demo replays. Renders player positions on radar maps with vision cones, kill lines, shot tracers, and timeline scrubbing.

Extracted from the [chimera](https://github.com/skkwowee/chimera) research project.

## Features

- Radar canvas with player positions (T amber / CT blue)
- Wall-clipped vision cones via raycasting
- Kill lines and damage lines
- Shot tracers with trajectory visualization
- Multi-level map support (upper/lower for maps like de_nuke)
- Timeline slider with kill event markers
- Demo and round selection

## Tech Stack

- Next.js 16
- React 19
- Tailwind CSS v4
- HTML Canvas for radar rendering

## Quickstart

```bash
git clone https://github.com/skkwowee/cs2-demo-viewer.git
cd cs2-demo-viewer
npm install
```

Add demo data to `public/viewer-data/` and radar images to `public/maps/` (see [Data Format](#data-format) below), then:

```bash
npm run dev
```

Open http://localhost:3000/viewer.

## Data Format

The viewer expects JSON data in `public/viewer-data/` and radar PNGs in `public/maps/`. See [docs/data-format.md](docs/data-format.md) for the full schema.

**Quick summary:**

```
public/
├── viewer-data/
│   ├── index.json                    # [{stem, map_name, rounds}]
│   └── {demo}/
│       ├── meta.json                 # rounds, kills, damages, shots, bomb events
│       └── round_XX.json             # frames with player positions per tick
└── maps/
    ├── map-data.json                 # coordinate transforms
    ├── de_mirage.png                 # radar images
    └── de_mirage_lower.png
```

### Generating data from .dem files

Use [cs2-tools](https://github.com/skkwowee/cs2-tools):

```bash
pip install cs2-tools[parse]
cs2-parse-demos data/demos/ --output data/processed/demos
cs2-export-viewer --input data/processed/demos --output public/viewer-data --maps-output public/maps
```

## License

MIT
