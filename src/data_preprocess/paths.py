"""Relative workspace layout under the repository root."""

from __future__ import annotations

import os
from pathlib import Path

# src/data_preprocess/paths.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent

_WORKSPACE_ENV = "GOOSEOMNI_PREPROCESS_WORKSPACE"
WORKSPACE = Path(
    os.environ.get(_WORKSPACE_ENV, str(REPO_ROOT / "data_preprocess_workspace"))
).expanduser()

RAW_ROOT = WORKSPACE / "raw"
EPISODES_ROOT = WORKSPACE / "episodes"
GAMES_ROOT = WORKSPACE / "games"
ALIGNED_ROOT = WORKSPACE / "aligned"
RUNS_ROOT = WORKSPACE / "runs"
ASSETS_ROOT = PACKAGE_ROOT / "assets"
START_TEMPLATE = ASSETS_ROOT / "start_template.png"


def ensure_workspace() -> Path:
    for path in (RAW_ROOT, EPISODES_ROOT, GAMES_ROOT, ALIGNED_ROOT, RUNS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    return WORKSPACE
