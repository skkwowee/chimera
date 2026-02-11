# Chimera

CS2 gaming copilot that analyzes screenshots and provides real-time strategic advice using vision-language models.

## Why This Problem?

1. Domain gap in frontier VLMs (weapon identification failure)
2. Strategic knowledge exists but isn't grounded to visuals
3. Generalization across games is the real goal
4. Games as a proxy for embodied AI

To test if the task is learnable: try overfitting on 10 examples, probe zero-shot visual grounding (e.g., "what weapon is this?"), and check if errors are systematic vs random.

## Features

- **Screenshot Analysis**: Extract game state (health, armor, money, weapons, player counts, bomb status)
- **Strategic Analysis**: Economy assessment, round importance, threats and opportunities
- **Tactical Advice**: Primary actions, reasoning, fallback plans, team callouts
- **Multiple VLM Support**: Qwen3-VL, Qwen2-VL, DeepSeek-VL2
- **GRPO Training**: Fine-tune models on your own labeled data using Unsloth

## Project Structure

```
chimera/
├── config/config.yaml       # Configuration settings
├── data/
│   ├── raw/                 # Raw screenshots
│   ├── labeled/             # Claude-labeled ground truth
│   ├── processed/           # Processed data
│   └── predictions/         # VLM predictions
├── src/
│   ├── data/                # Data loading utilities
│   ├── inference/           # VLM inference (Qwen, DeepSeek)
│   ├── labeling/            # Claude API labeling
│   ├── training/            # GRPO training module
│   └── prompts.py           # Shared prompts for all models
├── scripts/
│   ├── collect_youtube.py   # Download CS2 gameplay from YouTube
│   ├── label_screenshots.py # Generate labels with Claude
│   ├── run_inference.py     # Run VLM inference
│   ├── evaluate.py          # Evaluate predictions
│   ├── train_grpo.py        # GRPO fine-tuning
│   └── generate_review.py   # Generate HTML viewer for review
└── notebooks/               # Jupyter notebooks
```

## Setup

```bash
# Clone the repository
git clone https://github.com/skkwowee/chimera.git
cd chimera

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Usage

### 1. Collect Screenshots from YouTube

Download CS2 gameplay videos and extract frames at 1080p:

```bash
# Basic usage (extracts frame every 5 seconds)
python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID"

# Custom interval (every 10 seconds)
python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID" --interval 10
```

### 2. Label Screenshots with Claude

Generate ground truth labels using Claude's vision capabilities:

```bash
# Label a single screenshot
python scripts/label_screenshots.py --single path/to/screenshot.png

# Label all screenshots in a directory
python scripts/label_screenshots.py --input data/raw --output data/labeled
```

### 3. Run VLM Inference

Analyze screenshots with local vision-language models:

```bash
# Single image with Qwen3-VL (recommended)
python scripts/run_inference.py --single screenshot.png --model qwen3

# Batch inference
python scripts/run_inference.py --input data/raw --output data/predictions

# Only process images that have Claude labels (for comparison)
python scripts/run_inference.py --labeled-only --input data/raw --output data/predictions

# Available models: qwen3, qwen3-moe, qwen2, deepseek
```

### 4. Evaluate Predictions

Compare VLM predictions against ground truth:

```bash
python scripts/evaluate.py --predictions data/predictions --labels data/labeled
```

### 5. Review Labels

Generate a standalone HTML viewer to inspect images and labels side-by-side:

```bash
# Generate with embedded images (works offline, ~2MB per 5 images)
python scripts/generate_review.py --images data/samples --labels data/labeled --embed

# Generate with file references (smaller, requires local server)
python scripts/generate_review.py --images data/raw --labels data/labeled
python -m http.server 8000  # Then open http://localhost:8000/review.html

# Compare Claude labels vs Qwen predictions side-by-side
python scripts/generate_review.py --compare --embed
```

Navigate with arrow keys, press F to flag items for review.

### 6. Fine-tune with GRPO

Train Qwen3-VL on your labeled data using Group Relative Policy Optimization:

```bash
# Basic training
python scripts/train_grpo.py --screenshots data/raw --labels data/labeled

# Dry run (check VRAM usage)
python scripts/train_grpo.py --dry-run

# Custom settings
python scripts/train_grpo.py \
    --epochs 5 \
    --lr 1e-5 \
    --accuracy-weight 0.6 \
    --lora-r 32

# Resume from checkpoint
python scripts/train_grpo.py --resume outputs/grpo/checkpoint-500
```

## Output Format

All analysis outputs follow this JSON structure:

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

## Hardware Requirements

| Task | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (Qwen3-VL-8B) | ~18 GB | RTX 4090, A100 |
| Inference (4-bit) | ~6 GB | RTX 3080+ |
| GRPO Training | ~20 GB | RTX 4090 (24GB) |

## Configuration

Edit `config/config.yaml` to customize:

- Model selection and parameters
- Training hyperparameters
- Reward function weights
- Data paths

## Research: Strategy-First Visual Grounding

**Hypothesis:** A VLM pretrained on structured game replay data (strategy) before visual fine-tuning (perception) needs fewer labeled screenshots to achieve the same accuracy — and produces better strategic reasoning — than one trained on screenshots alone.

Like humans: learn the game from replays and theory, then apply that knowledge when watching live.

### The experiment

| Model | Description |
|-------|-------------|
| A | Qwen3-VL zero-shot (no training) |
| B | Qwen3-VL + SFT on 100 screenshots (vision only) |
| C | Qwen3-VL + demo pretraining + SFT on 100 screenshots (strategy + vision) |
| D | Qwen3-VL + demo pretraining + SFT on 20 screenshots (data efficiency test) |

If C > B, pretraining helps. If D ≈ B, pretraining reduces data requirements.

### Pipeline

Each step is isolated: it reads from defined inputs, writes to defined outputs, and validates its own results. Intermediate artifacts are cleaned up. See `experiment-state.json` for current progress and machine-readable state.

- [ ] **Step 1 — Demo data pipeline.** Download pro demos, parse with awpy, extract key-moment snapshots (pre-round, first contact, post-plant) into structured JSON with full game state + round outcome.
- [ ] **Step 2 — Strategy pretraining dataset.** Convert snapshots into text-based SFT format. Game state as input, analysis + outcome as target. Train/val split with balance checks.
- [ ] **Step 3 — Phase 1: Strategy pretraining.** SFT on Qwen3-VL language backbone (no vision layers). LoRA. Model learns XvX win rates, economy decisions, positional reasoning from real pro match outcomes.
- [ ] **Step 4 — Screenshot labeling + baseline.** Label ~100 screenshots with Claude. Run zero-shot eval (Model A) to establish the floor.
- [ ] **Step 5 — Phase 2: Visual grounding.** Train Models B, C, D. Compare strategy-pretrained vs vision-only vs few-shot.
- [ ] **Step 6 — Evaluation + analysis.** Per-field accuracy, consistency scores, reasoning quality across all models. Write up findings.

### Why this matters beyond games

The general principle — *learn reasoning from cheap structured data, then ground in vision with minimal labels* — applies to robotics (sim logs → real-world), medical imaging (patient records → radiology), and autonomous driving (telemetry → camera perception). Games are the controlled environment to prove the paradigm.

### Resuming work

Read `experiment-state.json` for current step, what's done, and what's next. Each step lists its inputs, outputs, and validation criteria. Start a session by checking the state file and running validation on the last completed step to confirm nothing is broken.

## License

MIT
