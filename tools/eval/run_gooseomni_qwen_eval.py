#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Deprecated compatibility entry for the unified GooseOmni evaluator."""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from gooseomni.cli import main  # noqa: E402


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]


if __name__ == "__main__":
    warnings.warn(
        "run_gooseomni_qwen_eval.py is deprecated; use `gooseomni eval run`.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(["eval", "run", *sys.argv[1:]]))
