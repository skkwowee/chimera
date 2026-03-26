"""
Configuration utilities.
"""

from pathlib import Path
from typing import Any

import yaml

# Single source of truth for the default model name.
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"


def load_config(config_path: Path | str = "config/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)
