from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends.qwen_omni import ModelBackend
from .json_utils import parse_json, validation_error_payload, write_json
from .perception_quality import normalize_pass_event, validate_pass_events
from .prompts import (
    audio_perception_prompt,
    pov_event_prompt,
    round_boundary_prompt,
    visual_perception_prompt,
)
from .schemas import Clip, POVEvent, RoundBoundaryCandidate


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_event_time_window(normalized: dict, clip: Clip) -> None:
    start_sec = _coerce_float(normalized.get("start_sec"), clip.start_sec)
    end_sec = _coerce_float(normalized.get("end_sec"), clip.end_sec)

    start_sec = min(max(start_sec, clip.start_sec), clip.end_sec)
    end_sec = min(max(end_sec, clip.start_sec), clip.end_sec)
    if end_sec <= start_sec:
        end_sec = min(clip.end_sec, start_sec + 1.0)
        if end_sec <= start_sec:
            start_sec = clip.start_sec
            end_sec = clip.end_sec

    normalized["start_sec"] = start_sec
    normalized["end_sec"] = end_sec


def load_manifest(path: Path) -> list[Clip]:
    clips: list[Clip] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                clips.append(Clip.model_validate_json(line))
    return clips


def filter_clips(
    clips: list[Clip],
    game_id: str | None = None,
    player_id: str | None = None,
    limit: int | None = None,
) -> list[Clip]:
    selected = [
        clip
        for clip in clips
        if (game_id is None or clip.game_id == game_id)
        and (player_id is None or clip.player_id == player_id)
    ]
    return selected[:limit] if limit is not None else selected


def normalize_pov_event_payload(item: dict, clip: Clip) -> dict:
    normalized = dict(item)
    normalized["clip_id"] = clip.clip_id
    normalized["game_id"] = clip.game_id
    normalized["player_id"] = clip.player_id
    normalized["episode_id"] = clip.episode_id
    normalized["phase_index"] = clip.phase_index
    normalized["phase_type"] = clip.phase_type
    normalized["source_clip_id"] = clip.source_clip_id
    normalized.setdefault("start_sec", clip.start_sec)
    normalized.setdefault("end_sec", clip.end_sec)
    normalized.setdefault("event_type", "observation")
    normalized.setdefault("description", "")
    normalized.setdefault("visible_players", [])
    normalized.setdefault("mentioned_players", [])
    normalized.setdefault("location", None)
    normalized.setdefault("evidence", None)
    normalized.setdefault("speaker_id", None)
    normalized.setdefault("speaker_confidence", None)
    normalized.setdefault("utterance", None)
    normalized.setdefault("dual_pass_consistent", False)
    normalized.setdefault("visual_text_evidence_id", None)
    normalized.setdefault("heard_by", [])
    normalized.setdefault("meeting_public", False)
    normalized.setdefault("public_ui", False)
    if isinstance(normalized["location"], list):
        normalized["location"] = "、".join(str(item) for item in normalized["location"])
    _normalize_event_time_window(normalized, clip)

    confidence = normalized.get("confidence", 0.0)
    if isinstance(confidence, str):
        confidence_map = {
            "高": 0.9,
            "中": 0.6,
            "低": 0.3,
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }
        normalized["confidence"] = confidence_map.get(confidence.strip().lower(), 0.0)
    return normalized


def normalize_round_boundary_payload(item: dict, clip: Clip) -> dict:
    normalized = dict(item)
    normalized["clip_id"] = clip.clip_id
    normalized["game_id"] = clip.game_id
    normalized["player_id"] = clip.player_id
    normalized.setdefault("aligned_start_sec", clip.start_sec)
    normalized.setdefault("aligned_end_sec", clip.end_sec)
    normalized.setdefault("boundary_type", "uncertain_boundary")
    normalized.setdefault("evidence", "")
    normalized.setdefault("confidence", 0.0)
    normalized.setdefault("source_povs", [clip.player_id])
    normalized.setdefault("needs_review", False)

    start_sec = _coerce_float(normalized.get("aligned_start_sec"), clip.start_sec)
    end_sec = _coerce_float(normalized.get("aligned_end_sec"), clip.end_sec)
    start_sec = min(max(start_sec, clip.start_sec), clip.end_sec)
    end_sec = min(max(end_sec, clip.start_sec), clip.end_sec)
    if end_sec <= start_sec:
        if start_sec >= clip.end_sec:
            start_sec = max(clip.start_sec, clip.end_sec - 1.0)
        end_sec = min(clip.end_sec, start_sec + 1.0)
        normalized["needs_review"] = True
    normalized["aligned_start_sec"] = start_sec
    normalized["aligned_end_sec"] = end_sec

    if not normalized["source_povs"]:
        normalized["source_povs"] = [clip.player_id]
    if clip.player_id not in normalized["source_povs"]:
        normalized["source_povs"].append(clip.player_id)

    confidence = normalized.get("confidence", 0.0)
    if isinstance(confidence, str):
        confidence_map = {
            "高": 0.9,
            "中": 0.6,
            "低": 0.3,
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }
        normalized["confidence"] = confidence_map.get(confidence.strip().lower(), 0.0)
    return normalized


def annotate_pov_events(
    clips: list[Clip],
    backend: ModelBackend,
    output_dir: Path,
    error_dir: Path,
    resume: bool,
    pass_kind: str = "combined",
    canonical_players: set[str] | None = None,
    speaker_confidence_min: float = 0.85,
) -> dict[str, int]:
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for clip in clips:
        output_path = output_dir / f"{clip.clip_id}.json"
        if resume and output_path.exists():
            stats["skipped"] += 1
            continue

        prompt_factory = {
            "combined": pov_event_prompt,
            "visual": visual_perception_prompt,
            "audio": audio_perception_prompt,
        }.get(pass_kind)
        if prompt_factory is None:
            raise ValueError(f"unknown perception pass: {pass_kind}")
        prompt = prompt_factory(clip)
        try:
            raw_response = backend.generate(prompt, clip.clip_path)
            payload = parse_json(raw_response)
            normalized_rows = [
                normalize_pass_event(
                    normalize_pov_event_payload(item, clip),
                    pass_kind,
                    speaker_confidence_min,
                )
                for item in payload
            ]
            quality_issues = validate_pass_events(
                normalized_rows, pass_kind, canonical_players
            )
            if quality_issues:
                raise ValueError("; ".join(quality_issues))
            events = []
            for index, item in enumerate(normalized_rows):
                item["evidence_id"] = f"{clip.clip_id}:{pass_kind}:{index:02d}"
                events.append(POVEvent.model_validate(item))
            write_json(
                output_path,
                {
                    "clip": clip.model_dump(),
                    "events": [event.model_dump() for event in events],
                    "raw_response": raw_response,
                    "perception_pass": pass_kind,
                },
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            raw = locals().get("raw_response", "")
            error_payload = validation_error_payload(exc, raw)
            try:
                write_json(
                    error_dir / f"{clip.clip_id}.json",
                    {
                        "clip": clip.model_dump(),
                        **error_payload,
                    },
                )
            except TypeError:
                write_json(
                    error_dir / f"{clip.clip_id}.json",
                    {
                        "clip": clip.model_dump(),
                        "error": str(error_payload.get("error", exc)),
                        "raw_response": raw,
                    },
                )
    return stats


def annotate_round_boundaries(
    clips: list[Clip],
    backend: ModelBackend,
    output_dir: Path,
    error_dir: Path,
    resume: bool,
) -> dict[str, int]:
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for clip in clips:
        output_path = output_dir / f"{clip.clip_id}.json"
        if resume and output_path.exists():
            stats["skipped"] += 1
            continue

        prompt = round_boundary_prompt(clip)
        try:
            raw_response = backend.generate(prompt, clip.clip_path)
            payload = parse_json(raw_response)
            boundaries = [
                RoundBoundaryCandidate.model_validate(
                    normalize_round_boundary_payload(item, clip)
                )
                for item in payload
            ]
            write_json(
                output_path,
                {
                    "clip": clip.model_dump(),
                    "round_boundary_candidates": [
                        boundary.model_dump() for boundary in boundaries
                    ],
                    "raw_response": raw_response,
                },
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            raw = locals().get("raw_response", "")
            write_json(
                error_dir / f"{clip.clip_id}.json",
                {
                    "clip": clip.model_dump(),
                    **validation_error_payload(exc, raw),
                },
            )
    return stats


def reprocess_error_files(error_dir: Path, output_dir: Path) -> dict[str, int]:
    stats = {"ok": 0, "error": 0}
    for error_path in sorted(error_dir.glob("*.json")):
        try:
            import json

            record = json.loads(error_path.read_text(encoding="utf-8"))
            clip = Clip.model_validate(record["clip"])
            raw_response = record.get("raw_response", "")
            payload = parse_json(raw_response)
            events = [
                POVEvent.model_validate(normalize_pov_event_payload(item, clip))
                for item in payload
            ]
            write_json(
                output_dir / f"{clip.clip_id}.json",
                {
                    "clip": clip.model_dump(),
                    "events": [event.model_dump() for event in events],
                    "raw_response": raw_response,
                    "reprocessed_from_error": str(error_path),
                },
            )
            error_path.unlink()
            stats["ok"] += 1
        except Exception:  # noqa: BLE001
            stats["error"] += 1
    return stats
