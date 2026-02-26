# Chimera

**See, Then Think: Two-Phase VLM Training for Game Understanding**

Vision-language models struggle with domain-specific visual grounding in competitive gaming, where rapid scene understanding is critical for strategic decision-making. Chimera is a two-phase training paradigm: first, supervised fine-tuning on screenshot–game state pairs teaches the model to read the HUD accurately (visual grounding); then, Group Relative Policy Optimization with demo-derived rewards teaches strategic reasoning scored against pro play and round outcomes. We evaluate on Counter-Strike 2, demonstrating improved performance over single-phase approaches.

## Motivation

Frontier VLMs fail at basic CS2 visual grounding (e.g., weapon identification, player counts), and they lack the strategic knowledge to advise on play. Both problems have cheap data sources: demo files provide engine-accurate game state for grounding, and pro match outcomes provide a reward signal for reasoning. The key insight: perception and reasoning are separable training objectives — teach them in sequence rather than asking RL to learn both at once.

**Hypothesis:** A VLM that first learns accurate visual grounding via SFT on screenshot–demo pairs, then learns strategic reasoning via GRPO scored against pro decisions and round outcomes, produces better tactical advice than either phase alone — and the two-phase approach is more data-efficient than end-to-end training.

## Method

### Phase 1: Visual Grounding (SFT on Screenshots)

Supervised fine-tuning on screenshot–game state pairs. Ground truth comes from demo data synchronized to VOD frames (engine-accurate health, armor, weapons, player counts, etc. — not model-labeled). The model learns to read the HUD correctly. LoRA on both vision and language layers.

### Phase 2: Strategic Reasoning (GRPO on Demo-Derived Rewards)

Refine with Group Relative Policy Optimization using 7 reward signals. Vision rewards (format, hard/soft field accuracy) prevent SFT regression. Reasoning rewards (decision alignment vs pro play, outcome weighting by round result, consistency, reasoning quality) are the RL training signal. Vision layers are frozen; only language layers train. Outcome reward gets the highest weight (0.30).

## Experiment

| Model | Description |
|-------|-------------|
| A | Qwen3-VL zero-shot (no training) |
| B | Qwen3-VL + SFT only (visual grounding, no GRPO) |
| C | Qwen3-VL + SFT + GRPO (visual grounding + strategic reasoning) |
| D | Qwen3-VL + GRPO only (no SFT, tests whether SFT phase is necessary) |

If C > B, GRPO improves reasoning beyond what SFT alone provides. If C > D, SFT visual grounding is a necessary foundation for effective GRPO.

### Pipeline

Each step is isolated: it reads from defined inputs, writes to defined outputs, and validates its own results. Intermediate artifacts are cleaned up.

- [x] **Step 1 — Data schema & manifest.** Unified data manifest (`src/data/manifest.py`, JSONL append-only) for tracking screenshot provenance, source, timestamps, and transcript context. Collection scripts write to `data/manifest.jsonl`, `ScreenshotDataset` loads/filters by manifest fields, training utils accept manifest filtering.
- [x] **Step 2 — Demo data pipeline.** Parse pro demos with awpy into full-tick Parquet + metadata JSONs (`scripts/parse_demos.py`). Export downsampled viewer data (`scripts/export_viewer_data.py`). Interactive demo viewer at `/viewer` with radar canvas, vision cones (wall-clipped via raycasting), kill/damage lines, shot tracers, timeline scrubbing, and split upper/lower rendering for multi-level maps. 4 demos parsed (Furia vs Vitality, maps: Mirage/Inferno/Nuke/Overpass, 83 rounds, 563 kills).
- [ ] **Step 3 — Screenshot-demo synchronization.** Sync VOD frames to demo ticks to produce (screenshot, exact_game_state) pairs. Figure out time offset between broadcast and demo file. Extract frames at intervals, pair with ground truth game state from Parquet data.
- [ ] **Step 4 — Phase 1: Visual grounding (SFT).** SFT on Qwen3-VL with screenshot–game state pairs. LoRA on vision + language layers. Model learns to read the HUD accurately. Run zero-shot eval (Model A) to establish baseline.
- [ ] **Step 5 — GRPO dataset from demos.** Convert demo snapshots into decision training format. Each sample: game state → pro_action (categorized via ACTION_TAXONOMY) + round_won outcome.
- [ ] **Step 6 — Phase 2: Strategic reasoning (GRPO).** Train Models B, C, D. 7 reward signals. Compare SFT-only vs SFT+GRPO vs GRPO-only.
- [ ] **Step 7 — Evaluation + analysis.** Per-field accuracy, consistency scores, reasoning quality across all models. Write up findings.

### Resuming work

Start a session by reading this pipeline checklist. Each step lists its inputs, outputs, and validation criteria. Validate the last completed step before starting the next.

## Output Format

All models produce structured JSON with three sections:

```json
{
  "game_state": {
    "map_name": "de_dust2",
    "round_phase": "playing",
    "player_side": "CT",
    "player_health": 100,
    "player_armor": 100,
    "player_money": 4750,
    "team_money_total": 20000,
    "weapon_primary": "M4A1-S",
    "weapon_secondary": "USP-S",
    "utility": ["smoke", "flashbang", "HE grenade"],
    "alive_teammates": 4,
    "alive_enemies": 5,
    "bomb_status": "carried",
    "site": "A",
    "visible_enemies": 0
  },
  "analysis": {
    "situation_summary": "Early round, full team alive, holding A site",
    "economy_assessment": "full-buy",
    "round_importance": "medium",
    "immediate_threats": ["Potential A execute"],
    "opportunities": ["Strong utility for retake"]
  },
  "advice": {
    "primary_action": "Hold crossfire position with teammate",
    "reasoning": "2v5 retake is difficult, better to get picks early",
    "fallback": "Fall back to site if smoked off",
    "callout": "Two holding A, need info mid"
  }
}
```

## Standalone Repos

Two components have been extracted as standalone repositories:

- **[cs2-demo-viewer](https://github.com/skkwowee/cs2-demo-viewer)** — Interactive Next.js viewer for CS2 demo replays (extracted from `site/`)
- **[cs2-tools](https://github.com/skkwowee/cs2-tools)** — Python utilities for demo parsing, viewer data export, and screenshot capture (extracted from `src/netcon.py` and `scripts/`)

The original files remain in this repo and continue to work as part of the chimera pipeline. The standalone repos are for independent use without the training infrastructure.

## Project Structure

```
chimera/
├── paper/                      # NeurIPS 2026 submission
│   ├── main.tex                # Main paper
│   ├── references.bib          # Bibliography
│   └── figures/                # Paper figures
├── config/config.yaml          # Configuration settings
├── data/
│   ├── manifest.jsonl          # Data provenance tracking
│   ├── raw/                    # Raw screenshots
│   ├── labeled/                # Claude-labeled ground truth
│   ├── processed/              # Processed data
│   └── predictions/            # VLM predictions
├── src/
│   ├── data/                   # Data loading + manifest utilities
│   ├── inference/              # VLM inference (Qwen3-VL-8B)
│   ├── labeling/               # Claude API labeling
│   ├── training/               # SFT + GRPO training modules
│   └── prompts.py              # Shared prompts for all models
├── scripts/
│   ├── collect_youtube.py      # Download CS2 gameplay from YouTube
│   ├── collect_with_transcript.py  # Collect with transcript context
│   ├── label_screenshots.py    # Generate labels with Claude
│   ├── run_inference.py        # Run VLM inference
│   ├── evaluate.py             # Evaluate predictions
│   ├── train_sft.py            # Phase 1: SFT fine-tuning
│   ├── train_grpo.py           # Phase 2: GRPO fine-tuning
│   ├── upload_to_hub.py        # Upload datasets/models to HF Hub
│   └── generate_review.py      # Generate HTML viewer for review
├── site/                       # Project website
└── notebooks/                  # Jupyter notebooks
```

## Setup

```bash
git clone https://github.com/skkwowee/chimera.git
cd chimera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Usage

### Collect Data

```bash
# YouTube screenshots (extracts frame every 5 seconds)
python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID"

# With transcript context (writes to manifest)
python scripts/collect_with_transcript.py "https://youtube.com/watch?v=VIDEO_ID"
```

### Label with Claude

```bash
python scripts/label_screenshots.py --single path/to/screenshot.png
python scripts/label_screenshots.py --input data/raw --output data/labeled
```

### Run Inference

```bash
python scripts/run_inference.py --single screenshot.png
python scripts/run_inference.py --input data/raw --output data/predictions
python scripts/run_inference.py --labeled-only --input data/raw --output data/predictions
```

### Evaluate

```bash
python scripts/evaluate.py --predictions data/predictions --labels data/labeled
```

### Train

```bash
# Phase 1: SFT (run first)
python scripts/train_sft.py --screenshots data/raw --labels data/labeled
python scripts/train_sft.py --dry-run  # check VRAM

# Phase 2: GRPO (uses SFT output)
python scripts/train_grpo.py \
    --model-name outputs/sft/final_model/merged_16bit \
    --screenshots data/raw --labels data/labeled
python scripts/train_grpo.py --dry-run  # check VRAM
```

### Review

```bash
python scripts/generate_review.py --images data/raw --labels data/labeled --embed
python scripts/generate_review.py --compare --embed
```

## Hardware Requirements

| Task | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (Qwen3-VL-8B) | ~18 GB | RTX 4090, A100 |
| Inference (4-bit) | ~6 GB | RTX 3080+ |
| SFT Training | ~18 GB | RTX 4090 (24GB) |
| GRPO Training | ~20 GB | RTX 4090 (24GB) |

## Why This Matters Beyond Games

The general principle — *learn reasoning from cheap structured data, then ground in vision with minimal labels* — applies to robotics (sim logs → real-world), medical imaging (patient records → radiology), and autonomous driving (telemetry → camera perception). Games are the controlled environment to prove the paradigm.

## License

MIT
