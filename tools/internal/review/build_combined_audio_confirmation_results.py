from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def result_task_id(path: Path) -> str | None:
    if path.name.endswith(".error.json"):
        return path.name.removesuffix(".error.json")
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return path.stem
    task_id = payload.get("review_task_id")
    return task_id if isinstance(task_id, str) and task_id else path.stem


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def clear_result_files(output_root: Path) -> int:
    removed = 0
    output_root.mkdir(parents=True, exist_ok=True)
    for path in output_root.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


def build_combined_results(
    queue: Path, result_roots: list[Path], output_root: Path
) -> dict[str, Any]:
    tasks = read_jsonl(queue)
    expected_ids = [
        row["review_task_id"]
        for row in tasks
        if isinstance(row.get("review_task_id"), str)
    ]
    expected_set = set(expected_ids)
    stale_results_removed = clear_result_files(output_root)

    indexed: dict[str, Path] = {}
    duplicate_results: list[dict[str, str]] = []
    ignored_results: list[str] = []
    for root in result_roots:
        for path in sorted(root.glob("*.json")):
            task_id = result_task_id(path)
            if not task_id:
                ignored_results.append(path.as_posix())
                continue
            if task_id not in expected_set:
                ignored_results.append(path.as_posix())
                continue
            if task_id in indexed:
                duplicate_results.append(
                    {
                        "review_task_id": task_id,
                        "kept_result_file": indexed[task_id].as_posix(),
                        "duplicate_result_file": path.as_posix(),
                    }
                )
                continue
            indexed[task_id] = path

    materialized = 0
    materialization_modes: dict[str, int] = {}
    for task_id in expected_ids:
        src = indexed.get(task_id)
        if src is None:
            continue
        suffix = ".error.json" if src.name.endswith(".error.json") else ".json"
        mode = link_or_copy(src, output_root / f"{task_id}{suffix}")
        materialized += 1
        materialization_modes[mode] = materialization_modes.get(mode, 0) + 1

    missing_ids = [task_id for task_id in expected_ids if task_id not in indexed]
    summary = {
        "ok": True,
        "queue": queue.as_posix(),
        "result_roots": [root.as_posix() for root in result_roots],
        "output_root": output_root.as_posix(),
        "expected_tasks": len(expected_ids),
        "combined_results": materialized,
        "missing_results": len(missing_ids),
        "missing_review_task_ids": missing_ids,
        "duplicate_results": duplicate_results,
        "ignored_results": ignored_results,
        "materialization_modes": dict(sorted(materialization_modes.items())),
        "stale_results_removed": stale_results_removed,
        "promotion_to_human_verified_gold": False,
    }
    write_json(output_root.parent / "combined_results_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine audio-confirmation result roots into one queue-ordered result root."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = build_combined_results(args.queue, args.result_root, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
