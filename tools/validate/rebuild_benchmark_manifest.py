#!/usr/bin/env python3
"""Rebuild the GooseOmni benchmark integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LINE_COUNT_PATHS = (
    "public/leaderboard_core/trials.jsonl",
    "private/leaderboard_core/hidden_gold.jsonl",
    "public/raw_video_smoke/raw_video_smoke.jsonl",
    "public/leaderboard_core/probe_groups_public.jsonl",
    "public/agentic_midgame_prediction/trials.jsonl",
    "private/agentic_midgame_prediction/hidden_gold.jsonl",
)

PRIVATE_MARKERS = ('"hidden_gold":', '"forbidden_event_ids":')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def public_leak_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted((root / "public").rglob("*")):
        if path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(root)}:{marker}")
    return hits


def build_manifest(root: Path) -> dict[str, object]:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    return {
        "name": "gooseomni_v1",
        "source_pass": "pass248_death_skill_scope_audit",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "layout": {
            "public": "model-facing benchmark inputs and public reports",
            "private": "gold and scorer-private answers for local scoring only",
            "tables": "human-readable and CSV summaries",
        },
        "validation": {
            "line_counts": {
                relpath: line_count(root / relpath) for relpath in LINE_COUNT_PATHS
            },
            "public_file_leak_hits": public_leak_hits(root),
        },
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("benchmark/gooseomni_v1")
    )
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))
    return 1 if manifest["validation"]["public_file_leak_hits"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
