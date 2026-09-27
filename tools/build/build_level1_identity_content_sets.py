#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEECH_VERBS = (
    "said",
    "asked",
    "felt",
    "hopes",
    "hope",
    "explains",
    "wants",
    "thinks",
    "says",
)


def parse_option(option_text: str) -> tuple[str, str] | None:
    text = re.sub(r"^([A-Da-d])\s*[.)]\s*", "", option_text or "").strip()
    lowered = text.lower()
    matches = []
    for verb in SPEECH_VERBS:
        match = re.search(rf"\b{re.escape(verb)}\b", lowered)
        if match:
            matches.append(match.start())
    if not matches:
        return None
    split_at = min(matches)
    speaker = text[:split_at].strip()
    content = text[split_at:].strip()
    if not speaker or not content:
        return None
    return speaker, content


def choice_prefix(index: int) -> str:
    return "ABCD"[index]


def build_unique_options(parsed: list[tuple[str, str]], target: str, correct_choice: str) -> tuple[list[str], str] | None:
    idx = "ABCD".index(correct_choice)
    values = []
    for speaker, content in parsed:
        values.append(speaker if target == "speaker" else content)
    correct_value = values[idx]
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    if correct_value not in unique_values or len(unique_values) < 2:
        return None
    options = [f"{choice_prefix(i)}. {value}" for i, value in enumerate(unique_values)]
    return options, choice_prefix(unique_values.index(correct_value))


def build_sets(rows: list[dict], max_samples: int, seed: int) -> tuple[list[dict], list[dict]]:
    valid = []
    for row in rows:
        parsed = [parse_option(opt) for opt in row.get("options", [])]
        if len(parsed) != 4 or any(item is None for item in parsed):
            continue
        valid.append((row, parsed))

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:max_samples]

    identity_rows = []
    content_rows = []
    for row, parsed_raw in selected:
        parsed = [(speaker, content) for speaker, content in parsed_raw if speaker and content]
        correct = str(row.get("correct_answer", "")).strip().upper()
        if correct not in {"A", "B", "C", "D"}:
            continue

        identity_built = build_unique_options(parsed, "speaker", correct)
        content_built = build_unique_options(parsed, "content", correct)
        if identity_built is None or content_built is None:
            continue
        identity_options, identity_answer = identity_built
        content_options, content_answer = content_built

        common = {
            **row,
            "metadata": {
                **(row.get("metadata") or {}),
                "source_task": "level1",
                "ablation_source_id": row.get("id"),
            },
        }
        identity_rows.append({
            **common,
            "question": "Who is speaking in the specified video segment?",
            "options": identity_options,
            "correct_answer": identity_answer,
            "metadata": {**common["metadata"], "ablation": "identity-unique"},
        })
        content_rows.append({
            **common,
            "question": "Which utterance is spoken in the specified video segment?",
            "options": content_options,
            "correct_answer": content_answer,
            "metadata": {**common["metadata"], "ablation": "content-unique"},
        })

    return identity_rows, content_rows


def write_dataset(rows: list[dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/level_1/dataset.json")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-root", default="data/level_1_ablation")
    args = parser.parse_args()

    rows = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    identity_rows, content_rows = build_sets(rows, args.max_samples, args.seed)
    out_root = ROOT / args.output_root
    write_dataset(identity_rows, out_root / "identity_unique.json")
    write_dataset(content_rows, out_root / "content_unique.json")
    print(f"identity_unique={len(identity_rows)} {out_root / 'identity_unique.json'}")
    print(f"content_unique={len(content_rows)} {out_root / 'content_unique.json'}")


if __name__ == "__main__":
    main()
