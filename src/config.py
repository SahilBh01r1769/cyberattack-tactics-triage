from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    """Load an experiment config, resolving a relative path from the project root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path: str | Path) -> Path:
    """Resolve a project-relative path without relying on the current directory."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

