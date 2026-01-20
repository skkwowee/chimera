# Chimera

CS2 gaming copilot that analyzes screenshots and provides real-time strategic advice using vision-language models.

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
│   └── training/            # GRPO training module
├── scripts/
│   ├── label_screenshots.py # Generate labels with Claude
│   ├── run_inference.py     # Run VLM inference
│   ├── evaluate.py          # Evaluate predictions
│   └── train_grpo.py        # GRPO fine-tuning
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

### 1. Label Screenshots with Claude

Generate ground truth labels using Claude's vision capabilities:

```bash
# Label a single screenshot
python scripts/label_screenshots.py --single path/to/screenshot.png

# Label all screenshots in a directory
python scripts/label_screenshots.py --input data/raw --output data/labeled
```

### 2. Run VLM Inference

Analyze screenshots with local vision-language models:

```bash
# Single image with Qwen3-VL (recommended)
python scripts/run_inference.py --single screenshot.png --model qwen3

# Batch inference
python scripts/run_inference.py --input data/raw --output data/predictions

# Available models: qwen3, qwen3-moe, qwen2, deepseek
```

### 3. Evaluate Predictions

Compare VLM predictions against ground truth:

```bash
python scripts/evaluate.py --predictions data/predictions --labels data/labeled
```

### 4. Fine-tune with GRPO

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

## License

MIT
