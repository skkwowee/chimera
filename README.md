# Chimera

**Think Before You See: VLMs as Game Agents Without Reinforcement Learning from Scratch**

Vision-language models struggle with domain-specific visual grounding in competitive gaming, where rapid scene understanding is critical for strategic decision-making. Chimera is a two-phase training paradigm that performs supervised fine-tuning on structured game replay data *before* visual fine-tuning, enabling the model to learn strategic reasoning independently from visual perception. We evaluate on Counter-Strike 2 screenshot analysis, demonstrating improved performance over vision-only fine-tuning.

## Motivation

Frontier VLMs fail at basic CS2 visual grounding (e.g., weapon identification), yet the strategic knowledge they need already exists in structured replay data. The key insight: humans learn game strategy from replays and theory first, then apply that knowledge when watching live. We test whether VLMs benefit from the same curriculum.

**Hypothesis:** A VLM fine-tuned on structured game replay data (strategy) before visual fine-tuning (perception) needs fewer labeled screenshots to achieve the same accuracy — and produces better strategic reasoning — than one trained on screenshots alone.

## Method

### Phase 1: Strategy Pre-training (SFT on Structured Data)

Train on text-only structured game state data parsed from pro match demos. The model learns XvX win rates, economy decisions, and positional reasoning from real pro match outcomes — no vision layers involved. LoRA on the language backbone only.

### Phase 2: Visual Grounding (GRPO on Screenshots)

Refine with Group Relative Policy Optimization on screenshot-advice pairs. Multi-signal reward function evaluates format correctness, tactical accuracy, and reasoning quality. The strategy knowledge from Phase 1 transfers to visual scene understanding.

## Experiment

| Model | Description |
|-------|-------------|
| A | Qwen3-VL zero-shot (no training) |
| B | Qwen3-VL + SFT on 100 screenshots (vision only) |
| C | Qwen3-VL + demo pre-training + SFT on 100 screenshots (strategy + vision) |
| D | Qwen3-VL + demo pre-training + SFT on 20 screenshots (data efficiency test) |

If C > B, pre-training helps. If D ≈ B, pre-training reduces data requirements.

### Pipeline

Each step is isolated: it reads from defined inputs, writes to defined outputs, and validates its own results. Intermediate artifacts are cleaned up.

- [x] **Step 1 — Data schema & manifest.** Unified data manifest (`src/data/manifest.py`, JSONL append-only) for tracking screenshot provenance, source, timestamps, and transcript context. Collection scripts write to `data/manifest.jsonl`, `ScreenshotDataset` loads/filters by manifest fields, training utils accept manifest filtering.
- [ ] **Step 2 — Demo data pipeline.** Download pro demos, parse with awpy, extract key-moment snapshots (pre-round, first contact, post-plant) into structured JSON with full game state + round outcome.
- [ ] **Step 3 — Strategy pre-training dataset.** Convert snapshots into text-based SFT format. Game state as input, analysis + outcome as target. Train/val split with balance checks.
- [ ] **Step 4 — Phase 1: Strategy pre-training.** SFT on Qwen3-VL language backbone (no vision layers). LoRA. Model learns XvX win rates, economy decisions, positional reasoning from real pro match outcomes.
- [ ] **Step 5 — Screenshot labeling + baseline.** Label ~100 screenshots with Claude. Run zero-shot eval (Model A) to establish the floor.
- [ ] **Step 6 — Phase 2: Visual grounding.** Train Models B, C, D. Compare strategy-pretrained vs vision-only vs few-shot.
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
