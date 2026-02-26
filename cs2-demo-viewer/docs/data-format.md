# Data Format

The viewer loads JSON data from `public/viewer-data/` and radar images from `public/maps/`.

## Directory Structure

```
public/
├── viewer-data/
│   ├── index.json                          # List of available demos
│   ├── {demo-stem}/
│   │   ├── meta.json                       # Demo metadata
│   │   ├── round_01.json                   # Per-round frame data
│   │   ├── round_02.json
│   │   └── ...
│   └── {another-demo}/
│       └── ...
└── maps/
    ├── map-data.json                       # Coordinate transforms
    ├── de_mirage.png                       # Radar images (upper level)
    ├── de_mirage_lower.png                 # Radar images (lower level)
    └── ...
```

## index.json

Array of demo entries. Used by the demo selector dropdown.

```json
[
  {
    "stem": "furia-vs-vitality-m1-mirage",
    "map_name": "de_mirage",
    "rounds": 25
  }
]
```

## meta.json

Per-demo metadata including round info, kills, damages, shots, and bomb events.

```json
{
  "stem": "furia-vs-vitality-m1-mirage",
  "map_name": "de_mirage",
  "header": {
    "map_name": "de_mirage",
    "server_name": "...",
    "tickrate": 64
  },
  "rounds": [
    {
      "round_num": 1,
      "start": 1234,
      "end": 5678,
      "freeze_end": 2000,
      "winner": "ct",
      "reason": "ct_killed",
      "bomb_plant": null,
      "bomb_site": "not_planted"
    }
  ],
  "kills": [
    {
      "tick": 3456,
      "round_num": 1,
      "attacker_name": "ZywOo",
      "attacker_side": "ct",
      "attacker_X": -1200.5,
      "attacker_Y": 300.2,
      "attacker_Z": -100.0,
      "victim_name": "KSCERATO",
      "victim_side": "t",
      "victim_X": -900.0,
      "victim_Y": 400.1,
      "victim_Z": -100.0,
      "weapon": "ak47",
      "headshot": true
    }
  ],
  "damages": [
    {
      "tick": 3450,
      "round_num": 1,
      "attacker_name": "ZywOo",
      "attacker_side": "ct",
      "attacker_X": -1200.5,
      "attacker_Y": 300.2,
      "attacker_Z": -100.0,
      "victim_name": "KSCERATO",
      "victim_side": "t",
      "victim_X": -900.0,
      "victim_Y": 400.1,
      "victim_Z": -100.0,
      "weapon": "ak47",
      "dmg_health": 111,
      "dmg_health_real": 100,
      "hitgroup": 1
    }
  ],
  "shots": [
    {
      "tick": 3449,
      "round_num": 1,
      "player_name": "ZywOo",
      "player_side": "ct",
      "player_X": -1200.5,
      "player_Y": 300.2,
      "player_Z": -100.0,
      "weapon": "ak47",
      "yaw": 45.3
    }
  ],
  "bomb": [
    {
      "tick": 4000,
      "round_num": 2,
      "event": "plant",
      "player_name": "KSCERATO",
      "site": "A"
    }
  ]
}
```

## round_XX.json

Per-round frame data with downsampled player positions at each tick.

```json
{
  "round_num": 1,
  "start_tick": 1234,
  "end_tick": 5678,
  "freeze_end": 2000,
  "winner": "ct",
  "reason": "ct_killed",
  "bomb_plant_tick": null,
  "bomb_site": null,
  "frames": [
    {
      "tick": 2000,
      "players": [
        {
          "name": "ZywOo",
          "side": "CT",
          "x": -1200.5,
          "y": 300.2,
          "z": -100.0,
          "yaw": 45.3,
          "hp": 100,
          "alive": true
        },
        {
          "name": "KSCERATO",
          "side": "T",
          "x": -900.0,
          "y": 400.1,
          "z": -100.0,
          "yaw": 180.0,
          "hp": 100,
          "alive": true
        }
      ]
    }
  ]
}
```

## maps/

### Radar images

PNG files named `{map_name}.png` (upper level) and `{map_name}_lower.png` (lower level, e.g. for de_nuke). Source: [awpy](https://github.com/pnxenopoulos/awpy) map data.

### map-data.json

Coordinate transformation metadata for converting game coordinates (X, Y) to radar pixel coordinates. Keyed by map name:

```json
{
  "de_mirage": {
    "pos_x": -3230,
    "pos_y": 1713,
    "scale": 5.0,
    "z_cutoff": null
  },
  "de_nuke": {
    "pos_x": -3453,
    "pos_y": 2887,
    "scale": 7.0,
    "z_cutoff": -500
  }
}
```

**Coordinate conversion:**

```
pixel_x = (game_x - pos_x) / scale
pixel_y = (pos_y - game_y) / scale
```

For maps with `z_cutoff` (e.g. de_nuke), players with `z < z_cutoff` are rendered on the lower level radar image.

## Generating data

Use [cs2-tools](https://github.com/skkwowee/cs2-tools) to generate viewer data from CS2 demo files:

```bash
# Parse .dem files into Parquet + metadata JSONs
cs2-parse-demos data/demos/ --output data/processed/demos

# Export viewer-ready JSON
cs2-export-viewer --input data/processed/demos --output public/viewer-data --maps-output public/maps
```
