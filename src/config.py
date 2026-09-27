from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration."""
    config_path = path or DEFAULT_CONFIG_PATH

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {config_path}")

    return config