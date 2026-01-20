"""
Configuration utilities.
"""

from pathlib import Path

import yaml


def load_config(config_path: Path | str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)
