from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(run_root: str | Path, scores: dict[str, Any] | None = None) -> Path:
    root = Path(run_root)
    payload = scores or json.loads((root / "scores.json").read_text(encoding="utf-8"))
    lines = ["# GooseOmni Evaluation Report", "", f"Benchmark: `{payload['benchmark_root']}`", ""]
    lines.append(
        "| track / model / modality | teacher bias | ok | errors | skipped | parse | schema | latency(s) |"
    )
    lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|")
    for key, row in sorted(payload.get("groups", {}).items()):
        teacher_bias = row.get("teacher_bias", {}).get(
            "same_model_as_data_generator", False
        )
        lines.append(
            f"| {key} | {'yes' if teacher_bias else 'no'} | {row['ok']} | "
            f"{row['errors']} | {row['skipped']} | "
            f"{row['json_parse_success']:.3f} | {row['schema_validation_success']:.3f} | "
            f"{row['mean_latency_sec']:.3f} |"
        )
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
