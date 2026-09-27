from __future__ import annotations

from pathlib import Path
from typing import Any


def cache_path(context: Any, name: str) -> Path | None:
    value = (context.config.stage_cache or {}).get(name)
    if value is not None and not value.exists():
        raise FileNotFoundError(f"configured cache does not exist: {name}={value}")
    return value
