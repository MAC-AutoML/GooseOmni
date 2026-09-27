from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gooseomni.config import PATHS


@dataclass(frozen=True)
class TrackDefinition:
    name: str
    public_trials: str
    hidden_gold: str | None
    gold: str | None = None
    probe_groups: str | None = None
    default_modalities: tuple[str, ...] = ("text",)


TRACKS = {
    "leaderboard_core": TrackDefinition(
        "leaderboard_core",
        "public/leaderboard_core/trials.jsonl",
        "private/leaderboard_core/hidden_gold.jsonl",
        "private/leaderboard_core/gold.jsonl",
        "public/leaderboard_core/probe_groups_public.jsonl",
    ),
    "raw_video_smoke": TrackDefinition(
        "raw_video_smoke",
        "public/raw_video_smoke/raw_video_smoke.jsonl",
        "private/leaderboard_core/hidden_gold.jsonl",
        "private/leaderboard_core/gold.jsonl",
        "public/leaderboard_core/probe_groups_public.jsonl",
        ("av", "v", "a", "text"),
    ),
    "agentic_midgame_prediction": TrackDefinition(
        "agentic_midgame_prediction",
        "public/agentic_midgame_prediction/trials.jsonl",
        "private/agentic_midgame_prediction/hidden_gold.jsonl",
    ),
}


def resolve_benchmark_root(value: str | Path) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PATHS.root / path, PATHS.root / "benchmark" / path])
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"benchmark root not found: {value}")


def get_track(name: str) -> TrackDefinition:
    try:
        return TRACKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation track: {name}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_for_trial(row: dict[str, Any]) -> str:
    schema = row.get("expected_output_schema", {})
    return (
        f"{row.get('prompt', '')}\n\nOUTPUT_REQUIREMENTS:\n"
        "Return strict JSON only. Do not include markdown fences. Use only evidence "
        "available under the stated input condition and never infer hidden scorer data.\n"
        f"EXPECTED_OUTPUT_SCHEMA_JSON:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def resolve_media_path(benchmark_root: Path, row: dict[str, Any]) -> Path | None:
    value = row.get("video_file")
    if not value:
        return None
    path = Path(str(value))
    resolved = path if path.is_absolute() else benchmark_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"benchmark media is missing: {resolved}")
    return resolved.resolve()
