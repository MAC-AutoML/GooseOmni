from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gooseomni.config.paths import PATHS

from .schema import POVRef, Segment

DEFAULT_DATASET_ROOT = Path(
    os.getenv(
        "GOOSEOMNI_DATASET_ROOT",
        PATHS.root / "data" / "gooseomni",
    )
)


def load_segments_jsonl(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                segments.append(Segment.model_validate(json.loads(line)))
    return segments


def load_segment(segment_id: str, dataset_root: Path | None = None) -> Segment:
    root = dataset_root or DEFAULT_DATASET_ROOT
    for segment in load_segments_jsonl(root / "segments.jsonl"):
        if segment.segment_id == segment_id:
            return segment
    raise FileNotFoundError(f"segment_id not found: {segment_id}")


def iter_povs(segment: Segment) -> Iterable[POVRef]:
    return iter(segment.povs)


def resolve_video_path(dataset_root: Path, video_file: str) -> Path:
    path = Path(video_file)
    if path.is_absolute():
        return path
    return dataset_root / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(obj), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, rows: Iterable[Any], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_to_jsonable(row), ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")


def safe_json_loads(raw_text: str) -> Any:
    text = _extract_json_text(raw_text)
    return json.loads(text)


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        fenced = re.match(
            r"```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL | re.IGNORECASE
        )
        if fenced:
            return fenced.group(1).strip()
        if text.lower().startswith("```json"):
            return text[7:].strip()
        return text[3:].strip()
    return text


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    return obj
