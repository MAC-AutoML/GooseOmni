from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from gooseomni.benchmark.agentic_midgame import score_agentic_midgame_prediction
from gooseomni.benchmark.decrypto_scoring import score_decrypto_diagnostics

from .tracks import get_track, read_jsonl, resolve_benchmark_root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_filtered(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _basic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row.get("status") != "skipped"]
    ok = [row for row in attempted if row.get("status") == "ok"]
    return {
        "rows": len(rows),
        "attempted": len(attempted),
        "ok": len(ok),
        "errors": sum(row.get("status") == "error" for row in rows),
        "skipped": sum(row.get("status") == "skipped" for row in rows),
        "json_parse_success": (
            sum(bool(row.get("parse_ok")) for row in attempted) / len(attempted)
            if attempted
            else 0.0
        ),
        "schema_validation_success": (
            sum(bool(row.get("schema_ok")) for row in attempted) / len(attempted)
            if attempted
            else 0.0
        ),
        "mean_latency_sec": (
            sum(float(row.get("latency_sec", 0.0)) for row in ok) / len(ok)
            if ok
            else 0.0
        ),
    }


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {name: _basic_metrics(items) for name, items in sorted(groups.items())}


def score_evaluation(
    run_root: str | Path, benchmark: str | Path
) -> dict[str, Any]:
    run_path = Path(run_root)
    benchmark_root = resolve_benchmark_root(benchmark)
    benchmark_manifest = json.loads(
        (benchmark_root / "manifest.json").read_text(encoding="utf-8")
    )
    generation_models = list(benchmark_manifest.get("review_models", []))
    attempts = read_jsonl(run_path / "responses.jsonl")
    latest_by_key: dict[str, dict[str, Any]] = {}
    for row in attempts:
        key = str(row.get("task_key") or "")
        if not key:
            continue
        previous = latest_by_key.get(key)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            latest_by_key[key] = row
    responses = list(latest_by_key.values())
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        groups[(str(row.get("track")), str(row.get("model")), str(row.get("modality")))].append(row)
    reports: dict[str, Any] = {}
    for (track_name, model, modality), rows in sorted(groups.items()):
        key = f"{track_name}/{model}/{modality}"
        work = run_path / "scoring" / track_name / model / modality
        work.mkdir(parents=True, exist_ok=True)
        filtered = work / "responses.jsonl"
        _write_filtered(filtered, rows)
        metrics = _basic_metrics(rows)
        track = get_track(track_name)
        if track.hidden_gold:
            hidden_rows = read_jsonl(benchmark_root / track.hidden_gold)
            hidden_by_trial = {
                str(row.get("trial_id")): row
                for row in hidden_rows
                if row.get("trial_id") is not None
            }
            for row in rows:
                hidden = hidden_by_trial.get(str(row.get("trial_id")), {})
                nested = hidden.get("hidden_gold")
                if isinstance(nested, dict):
                    hidden = {**hidden, **nested}
                row["gold_source"] = hidden.get("gold_source", "unknown")
        metrics["by_probe_type"] = _group_metrics(rows, "probe_type")
        metrics["by_gold_source"] = _group_metrics(rows, "gold_source")
        metrics["teacher_bias"] = {
            "same_model_as_data_generator": model in generation_models,
            "data_generation_models": generation_models,
        }
        if track_name in {"leaderboard_core", "raw_video_smoke"} and track.hidden_gold:
            score_path = work / "scores.json"
            metrics["track_metrics"] = score_decrypto_diagnostics(
                filtered, benchmark_root / track.hidden_gold, score_path
            )
        elif track_name == "agentic_midgame_prediction" and track.hidden_gold:
            score_path = work / "scores.json"
            metrics["track_metrics"] = score_agentic_midgame_prediction(
                filtered, benchmark_root / track.hidden_gold, score_path
            )
        reports[key] = metrics
    result = {
        "benchmark_root": str(benchmark_root),
        "attempts": len(attempts),
        "unique_tasks": len(responses),
        "groups": reports,
    }
    _write_json(run_path / "scores.json", result)
    return result
