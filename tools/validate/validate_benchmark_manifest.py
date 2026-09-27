#!/usr/bin/env python3
"""Validate the published GooseOmni benchmark without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.validate.rebuild_benchmark_manifest import LINE_COUNT_PATHS, build_manifest


EXPECTED_LINE_COUNTS = dict(
    zip(LINE_COUNT_PATHS, (889, 889, 48, 276, 160, 160), strict=True)
)


def validate_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "issues": ["missing manifest.json"]}

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_manifest(root)
    issues: list[str] = []
    if recorded.get("name") != "gooseomni_v1":
        issues.append("manifest name is not gooseomni_v1")
    if actual["validation"]["line_counts"] != EXPECTED_LINE_COUNTS:
        issues.append("benchmark line counts do not match the release contract")
    if actual["validation"]["public_file_leak_hits"]:
        issues.append("private markers found in public benchmark files")
    if recorded.get("files") != actual["files"]:
        issues.append("manifest file inventory or SHA-256 does not match")
    if recorded.get("file_count") != actual["file_count"]:
        issues.append("manifest file_count does not match")
    if recorded.get("total_bytes") != actual["total_bytes"]:
        issues.append("manifest total_bytes does not match")
    return {
        "ok": not issues,
        "line_counts": actual["validation"]["line_counts"],
        "public_file_leak_hits": actual["validation"]["public_file_leak_hits"],
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("benchmark/gooseomni_v1")
    )
    report = validate_manifest(parser.parse_args().root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
