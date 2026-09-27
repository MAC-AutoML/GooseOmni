from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_utils import write_json
from .splitting import load_sync_offsets


@dataclass(frozen=True)
class EventAlignedExportConfig:
    trials_path: Path
    global_events_path: Path
    sync_offsets_path: Path
    raw_dir: Path
    output_dir: Path
    pre_context_sec: float = 8.0
    post_context_sec: float = 4.0
    max_duration_sec: float = 120.0
    limit: int | None = None
    overwrite: bool = False
    dry_run: bool = False
    reencode: bool = True


def export_event_aligned_omni_eval(config: EventAlignedExportConfig) -> dict[str, int]:
    trials = _load_json_list(config.trials_path)
    global_events = {
        item["global_event_id"]: item
        for item in _load_json_list(config.global_events_path)
    }
    offsets = load_sync_offsets(config.sync_offsets_path)
    selected_trials = trials[: config.limit] if config.limit is not None else trials

    video_dir = config.output_dir / "videos"
    annotations: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    skipped = 0

    for trial in selected_trials:
        support_events = _support_events(trial, global_events)
        if not support_events:
            skipped += 1
            continue

        primary = support_events[0]
        source_player_id = _source_player(primary, trial)
        game_id = primary["game_id"]
        offset_sec = offsets.get((game_id, source_player_id))
        if offset_sec is None:
            skipped += 1
            continue

        aligned_start, aligned_end = _aligned_window(
            [primary],
            pre_context_sec=config.pre_context_sec,
            post_context_sec=config.post_context_sec,
            max_duration_sec=config.max_duration_sec,
        )
        raw_start = offset_sec + aligned_start
        raw_end = offset_sec + aligned_end
        if raw_end <= raw_start:
            skipped += 1
            continue

        source_video = config.raw_dir / game_id / f"{source_player_id}.mp4"
        if not source_video.exists():
            skipped += 1
            continue

        sample_id = _sample_id(trial, primary, source_player_id)
        rel_video_path = Path(game_id) / source_player_id / f"{sample_id}.mp4"
        output_video = video_dir / rel_video_path
        if config.dry_run:
            print(
                f"{source_video} -> {output_video} "
                f"raw=[{raw_start:.3f},{raw_end:.3f}] aligned=[{aligned_start:.3f},{aligned_end:.3f}]"
            )
        else:
            cut_video(
                source_video=source_video,
                output_video=output_video,
                raw_start_sec=raw_start,
                duration_sec=raw_end - raw_start,
                overwrite=config.overwrite,
                reencode=config.reencode,
            )

        metadata = _metadata(
            trial=trial,
            support_events=support_events,
            source_player_id=source_player_id,
            source_video=source_video,
            aligned_start=aligned_start,
            aligned_end=aligned_end,
            raw_start=raw_start,
            raw_end=raw_end,
            offset_sec=offset_sec,
        )
        annotation = _level2_annotation(
            trial=trial,
            sample_id=sample_id,
            video_file=rel_video_path.as_posix(),
            event_timestamp=_event_timestamp(primary, aligned_start),
            metadata=metadata,
        )
        annotations.append(annotation)
        samples.append(_event_qa_sample(annotation, metadata))

    if not config.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.output_dir / "annotations.json", annotations)
        _write_jsonl(config.output_dir / "samples.jsonl", samples)
        write_json(
            config.output_dir / "manifest.json",
            {
                "format": "gooseomni_event_aligned_omni_eval_v1",
                "annotations": "annotations.json",
                "samples": "samples.jsonl",
                "video_dir": "videos",
                "pre_context_sec": config.pre_context_sec,
                "post_context_sec": config.post_context_sec,
                "max_duration_sec": config.max_duration_sec,
                "reencode": config.reencode,
                "count": len(annotations),
                "skipped": skipped,
            },
        )

    return {"samples": len(annotations), "skipped": skipped}


def cut_video(
    *,
    source_video: Path,
    output_video: Path,
    raw_start_sec: float,
    duration_sec: float,
    overwrite: bool,
    reencode: bool,
) -> None:
    if output_video.exists() and not overwrite:
        return
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = _ffmpeg_cut_command(
        source_video=source_video,
        output_video=output_video,
        raw_start_sec=raw_start_sec,
        duration_sec=duration_sec,
        overwrite=overwrite,
        reencode=reencode,
    )
    subprocess.run(command, check=True)


def _ffmpeg_cut_command(
    *,
    source_video: Path,
    output_video: Path,
    raw_start_sec: float,
    duration_sec: float,
    overwrite: bool,
    reencode: bool,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{max(0.0, raw_start_sec):.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{max(0.001, duration_sec):.3f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
    ]
    if reencode:
        command.extend(["-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac"])
    else:
        command.extend(["-c", "copy"])
    command.extend(["-movflags", "+faststart", str(output_video)])
    return command


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return payload


def _support_events(
    trial: dict[str, Any], global_events: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event_id in trial.get("supporting_global_event_ids", []):
        event = global_events.get(event_id)
        if event:
            events.append(event)
    return events


def _source_player(event: dict[str, Any], trial: dict[str, Any]) -> str:
    source_players = event.get("source_player_ids") or []
    if source_players:
        return str(source_players[0])
    return str(trial["target_player_id"])


def _aligned_window(
    events: list[dict[str, Any]],
    *,
    pre_context_sec: float,
    post_context_sec: float,
    max_duration_sec: float | None = None,
) -> tuple[float, float]:
    event_start = min(float(item["start_sec"]) for item in events)
    event_end = max(float(item["end_sec"]) for item in events)
    start = max(0.0, event_start - max(0.0, pre_context_sec))
    end = event_end + max(0.0, post_context_sec)
    if (
        max_duration_sec is not None
        and max_duration_sec > 0
        and end - start > max_duration_sec
    ):
        start = max(0.0, event_end - max_duration_sec)
        end = start + max_duration_sec
    return start, end


def _event_timestamp(event: dict[str, Any], aligned_window_start: float) -> float:
    return max(0.0, float(event["end_sec"]) - aligned_window_start)


def _metadata(
    *,
    trial: dict[str, Any],
    support_events: list[dict[str, Any]],
    source_player_id: str,
    source_video: Path,
    aligned_start: float,
    aligned_end: float,
    raw_start: float,
    raw_end: float,
    offset_sec: float,
) -> dict[str, Any]:
    return {
        "trial_id": trial["trial_id"],
        "question_type": trial["question_type"],
        "target_player_id": trial["target_player_id"],
        "source_player_id": source_player_id,
        "source_video": str(source_video),
        "sync_raw_start_sec": offset_sec,
        "aligned_window_start_sec": aligned_start,
        "aligned_window_end_sec": aligned_end,
        "raw_window_start_sec": raw_start,
        "raw_window_end_sec": raw_end,
        "crop_primary_global_event_id": support_events[0].get("global_event_id"),
        "supporting_global_event_ids": [
            item.get("global_event_id")
            for item in support_events
            if item.get("global_event_id")
        ],
        "supporting_events": [
            {
                "global_event_id": item.get("global_event_id"),
                "event_type": item.get("event_type"),
                "global_start_sec": item.get("start_sec"),
                "global_end_sec": item.get("end_sec"),
                "clip_start_sec": max(0.0, float(item["start_sec"]) - aligned_start),
                "clip_end_sec": max(0.0, float(item["end_sec"]) - aligned_start),
                "included_in_clip": (
                    float(item["start_sec"]) >= aligned_start
                    and float(item["end_sec"]) <= aligned_end
                ),
                "source_player_ids": item.get("source_player_ids", []),
                "source_clip_ids": item.get("source_clip_ids", []),
                "description": item.get("description", ""),
            }
            for item in support_events
        ],
    }


def _level2_annotation(
    *,
    trial: dict[str, Any],
    sample_id: str,
    video_file: str,
    event_timestamp: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video_id": sample_id,
        "video_file": video_file,
        "full_asr": "",
        "question_1": {
            "timestamp": f"{event_timestamp:.3f}",
            "question": (
                "At the marked timestamp, is the cropped video prefix sufficient "
                "to answer the event-grounded question?"
            ),
            "option_A": "YES",
            "option_B": "NO",
            "correct_answer": "A",
        },
        "question_2": {
            "question": trial["question"],
            "answer": trial["answer"],
        },
        "metadata": metadata,
    }


def _event_qa_sample(
    annotation: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": annotation["video_id"],
        "video_path": annotation["video_file"],
        "question": annotation["question_2"]["question"],
        "answer": annotation["question_2"]["answer"],
        "event_timestamp_sec": float(annotation["question_1"]["timestamp"]),
        "metadata": metadata,
    }


def _sample_id(
    trial: dict[str, Any],
    primary_event: dict[str, Any],
    source_player_id: str,
) -> str:
    return (
        f"{trial['game_id']}_{trial['trial_id']}_{trial['question_type']}_"
        f"{source_player_id}_{primary_event['global_event_id']}"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
