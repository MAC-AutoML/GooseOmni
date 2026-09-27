from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gooseomni.config import PATHS
from gooseomni.data_pipeline.provenance import (
    append_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from gooseomni.models.model_server.clients import create_client
from gooseomni.models.pipeline.model_client import ModelClient
from gooseomni.models.pipeline.types import InferenceRequest
from gooseomni.models.registry import get_model_spec, list_model_specs

from .tracks import (
    get_track,
    prompt_for_trial,
    read_jsonl,
    resolve_benchmark_root,
    resolve_media_path,
)

MODALITIES = ("text", "v", "a", "av")


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark: str | Path
    run_root: Path
    models: tuple[str, ...]
    tracks: tuple[str, ...]
    modalities: tuple[str, ...] | None = None
    prompt_version: str = "gooseomni_eval_v2"
    limit: int | None = None
    skip: int = 0
    stride: int = 1
    resume: bool = False
    workers: int = 4


def parse_json_answer(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {}, isinstance(value, dict))
    except Exception:
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if not match:
            return {}, False
        try:
            value = json.loads(match.group(0))
            return (value if isinstance(value, dict) else {}, isinstance(value, dict))
        except Exception:
            return {}, False


def schema_matches(parsed: dict[str, Any], schema: dict[str, Any]) -> bool:
    return bool(parsed) and all(key in parsed for key in schema)


def modality_flags(modality: str) -> tuple[bool, bool]:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality: {modality}")
    return modality in {"v", "av"}, modality in {"a", "av"}


def supports_modality(model_name: str, modality: str) -> tuple[bool, str | None]:
    if model_name == "gpt4o" and modality not in {"text", "v"}:
        return False, "gpt4o evaluation supports Text/V only"
    if model_name == "gpt_audio" and modality != "a":
        return False, "gpt_audio evaluation supports A only"
    spec = get_model_spec(model_name)
    required = {"text"}
    if modality in {"v", "av"}:
        required.add("video")
    if modality in {"a", "av"}:
        required.add("audio")
    missing = required - set(spec.pipeline_inputs)
    if missing:
        return False, f"model does not support: {','.join(sorted(missing))}"
    return True, None


def _selected_rows(
    rows: list[dict[str, Any]], config: EvaluationConfig
) -> list[dict[str, Any]]:
    selected = [
        row
        for index, row in enumerate(rows)
        if index >= config.skip and (index - config.skip) % config.stride == 0
    ]
    return selected[: config.limit] if config.limit is not None else selected


def _task_key(
    row: dict[str, Any], model: str, modality: str, prompt_version: str
) -> str:
    return "|".join(
        [
            str(row.get("trial_id") or row.get("probe_id")),
            model,
            modality,
            prompt_version,
        ]
    )


def _completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row.get("task_key"))
        for row in read_jsonl(path)
        if row.get("status") in {"ok", "skipped"} and row.get("task_key")
    }


def _config_payload(config: EvaluationConfig, benchmark_root: Path) -> dict[str, Any]:
    manifest = benchmark_root / "manifest.json"
    model_config = PATHS.config_dir / "gooseomni.yaml"
    return {
        "benchmark_root": str(benchmark_root),
        "benchmark_manifest_hash": sha256_file(manifest),
        "model_config_hash": sha256_file(model_config),
        "models": list(config.models),
        "tracks": list(config.tracks),
        "modalities": list(config.modalities) if config.modalities else None,
        "prompt_version": config.prompt_version,
        "limit": config.limit,
        "skip": config.skip,
        "stride": config.stride,
    }


def _prepare_run(config: EvaluationConfig, benchmark_root: Path) -> dict[str, Any]:
    payload = _config_payload(config, benchmark_root)
    config_hash = stable_hash(payload)
    manifest_path = config.run_root / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise ValueError(
                "evaluation config or benchmark input changed; use a new run directory"
            )
    elif (config.run_root / "responses.jsonl").exists():
        raise ValueError(
            "responses exist without run_manifest.json; use a new run directory"
        )
    manifest = {
        "version": "gooseomni_eval_v2",
        "config_hash": config_hash,
        "config": payload,
        "status": "running",
    }
    write_json(manifest_path, manifest)
    for name in ("events.jsonl", "errors.jsonl"):
        (config.run_root / name).touch(exist_ok=True)
    return manifest


def plan_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    root = resolve_benchmark_root(config.benchmark)
    models = (
        tuple(spec.name for spec in list_model_specs())
        if config.models == ("all",)
        else config.models
    )
    tasks: list[dict[str, Any]] = []
    for track_name in config.tracks:
        track = get_track(track_name)
        rows = _selected_rows(read_jsonl(root / track.public_trials), config)
        modalities = config.modalities or track.default_modalities
        for model in models:
            get_model_spec(model)
            for modality in modalities:
                supported, reason = supports_modality(model, modality)
                tasks.append(
                    {
                        "track": track_name,
                        "model": model,
                        "modality": modality,
                        "rows": len(rows),
                        "status": "ready" if supported else "skipped",
                        "reason": reason,
                    }
                )
    return {"benchmark_root": str(root), "task_groups": tasks}


def _run_one(
    benchmark_root: Path,
    track_name: str,
    row: dict[str, Any],
    model_name: str,
    modality: str,
    prompt_version: str,
    client_factory: Callable[[str], ModelClient],
) -> dict[str, Any]:
    key = _task_key(row, model_name, modality, prompt_version)
    supported, reason = supports_modality(model_name, modality)
    base = {
        "task_key": key,
        "trial_id": str(row.get("trial_id") or row.get("probe_id") or ""),
        "probe_group_id": row.get("probe_group_id"),
        "probe_type": row.get("probe_type"),
        "probe_family": row.get("probe_family"),
        "input_condition": row.get("input_condition"),
        "track": track_name,
        "model": model_name,
        "modality": modality,
        "prompt_version": prompt_version,
    }
    if not supported:
        return {**base, "status": "skipped", "skip_reason": reason}
    try:
        media = resolve_media_path(benchmark_root, row)
        use_video, use_audio = modality_flags(modality)
        if (use_video or use_audio) and media is None:
            return {**base, "status": "skipped", "skip_reason": "trial has no media"}
        started = time.monotonic()
        prompt = prompt_for_trial(row)
        result = client_factory(model_name).predict(
            InferenceRequest(
                prompt=prompt,
                video_path=media,
                use_video=use_video,
                use_audio=use_audio,
                metadata={"track": track_name, "trial_id": base["trial_id"]},
            )
        )
        parsed, parse_ok = parse_json_answer(result.text)
        expected_schema = row.get("expected_output_schema", {})
        return {
            **base,
            "status": "ok",
            "raw_response": result.text,
            "parsed": parsed,
            "parse_ok": parse_ok,
            "schema_ok": schema_matches(parsed, expected_schema),
            "latency_sec": round(
                result.latency_sec
                if result.latency_sec is not None
                else time.monotonic() - started,
                3,
            ),
            "error": None,
            "model_info": {"adapter": result.model or model_name},
            "input_asset_hash": (
                sha256_file(media)
                if media is not None and (use_video or use_audio)
                else stable_hash({"trial_id": base["trial_id"], "prompt": prompt})
            ),
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "raw_response": "",
            "parsed": {},
            "parse_ok": False,
            "schema_ok": False,
            "latency_sec": 0.0,
            "error": str(exc),
        }


def run_evaluation(
    config: EvaluationConfig,
    client_factory: Callable[[str], ModelClient] = create_client,
) -> dict[str, Any]:
    benchmark_root = resolve_benchmark_root(config.benchmark)
    config.run_root.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_run(config, benchmark_root)
    responses_path = config.run_root / "responses.jsonl"
    events_path = config.run_root / "events.jsonl"
    errors_path = config.run_root / "errors.jsonl"
    models = (
        tuple(spec.name for spec in list_model_specs())
        if config.models == ("all",)
        else config.models
    )
    completed = _completed_keys(responses_path) if config.resume else set()
    pending: list[tuple[str, dict[str, Any], str, str]] = []
    for track_name in config.tracks:
        track = get_track(track_name)
        rows = _selected_rows(read_jsonl(benchmark_root / track.public_trials), config)
        for row in rows:
            for model in models:
                modalities = config.modalities or track.default_modalities
                for modality in modalities:
                    key = _task_key(row, model, modality, config.prompt_version)
                    if key not in completed:
                        pending.append((track_name, row, model, modality))
    local_models = {spec.name for spec in list_model_specs() if spec.kind == "local"}
    workers = (
        1
        if any(model in local_models for _, _, model, _ in pending)
        else max(1, config.workers)
    )
    append_jsonl(
        events_path,
        {
            "event": "evaluation_started",
            "time_unix": time.time(),
            "pending": len(pending),
            "workers": workers,
        },
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                _run_one,
                benchmark_root,
                track,
                row,
                model,
                modality,
                config.prompt_version,
                client_factory,
            ): index
            for index, (track, row, model, modality) in enumerate(pending)
        }
        ordered: dict[int, dict[str, Any]] = {}
        for future in as_completed(future_map):
            ordered[future_map[future]] = future.result()
        results = [ordered[index] for index in sorted(ordered)]
    with responses_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            append_jsonl(
                events_path,
                {
                    "event": "task_finished",
                    "time_unix": time.time(),
                    "task_key": result["task_key"],
                    "status": result["status"],
                },
            )
            if result["status"] == "error":
                append_jsonl(errors_path, result)
    summary = {
        "benchmark_root": str(benchmark_root),
        "responses": str(responses_path),
        "submitted": len(results),
        "ok": sum(row["status"] == "ok" for row in results),
        "error": sum(row["status"] == "error" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "workers": workers,
    }
    manifest.update({"status": "complete", "summary": summary})
    write_json(config.run_root / "run_manifest.json", manifest)
    write_json(config.run_root / "summary.json", summary)
    append_jsonl(
        events_path,
        {"event": "evaluation_finished", "time_unix": time.time(), **summary},
    )
    return summary
