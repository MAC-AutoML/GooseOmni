from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text


def parse_json(raw_text: str) -> Any:
    text = extract_json_text(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        recovered = recover_json_array(text)
        if recovered is not None:
            return recovered
        raise


def recover_json_array(text: str) -> list[dict[str, Any]] | None:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    index = 1
    items: list[dict[str, Any]] = []
    while index < len(stripped):
        while index < len(stripped) and stripped[index] in " \n\r\t,":
            index += 1
        if index >= len(stripped) or stripped[index] == "]":
            break
        try:
            item, next_index = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
        index = next_index
    return items if items else None


def validate_model(model_type: type[BaseModel], payload: Any) -> BaseModel:
    return model_type.model_validate(payload)


def validate_list(item_type: type[T], payload: Any) -> list[T]:
    return TypeAdapter(list[item_type]).validate_python(payload)


def validation_error_payload(exc: Exception, raw_response: str) -> dict[str, Any]:
    details: Any = str(exc)
    if isinstance(exc, ValidationError):
        details = exc.errors()
    return {"error": details, "raw_response": raw_response}
