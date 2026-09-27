from __future__ import annotations

import os
from typing import Any

from gooseomni.config.settings import CONFIG


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def level1_include_asr() -> bool:
    raw = os.getenv("GOOSEOMNI_LEVEL1_INCLUDE_ASR")
    if raw is None:
        raw = os.getenv("GOOSEOMNI_INCLUDE_ASR")
    if raw is None:
        raw = CONFIG.benchmark("level1.include_asr", False)
    return parse_bool(raw, default=False)


def level1_asr_tag() -> str:
    return "with-asr" if level1_include_asr() else "no-asr"


def add_level1_experiment_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    payload["include_asr"] = level1_include_asr()
    payload["asr_tag"] = level1_asr_tag()
    experiment = payload.setdefault("experiment", {})
    experiment["include_asr"] = payload["include_asr"]
    experiment["asr_tag"] = payload["asr_tag"]
    return payload


def add_level1_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    row["include_asr"] = level1_include_asr()
    row["asr_tag"] = level1_asr_tag()
    return row
