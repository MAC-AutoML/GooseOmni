from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .backends.qwen_omni import ModelBackend
from .json_utils import parse_json, validation_error_payload, write_json
from .prompts import sync_start_prompt
from .schemas import SyncOffset
from .splitting import probe_duration


@dataclass(frozen=True)
class RawVideo:
    game_id: str
    player_id: str
    path: Path
    duration_sec: float | None = None


@dataclass(frozen=True)
class SyncConfig:
    raw_dir: Path
    output_path: Path
    review_dir: Path
    error_dir: Path
    backend: ModelBackend
    game_id: str | None = None
    player_id: str | None = None
    review_window_sec: int = 600
    overwrite_review: bool = False
    resume: bool = True
    limit: int | None = None


def iter_raw_video_records(
    raw_dir: Path,
    game_id: str | None = None,
    player_id: str | None = None,
) -> list[RawVideo]:
    records: list[RawVideo] = []
    for video_path in sorted(raw_dir.glob("*/*.mp4")):
        current_game_id = video_path.parent.name
        current_player_id = video_path.stem
        if game_id and current_game_id != game_id:
            continue
        if player_id and current_player_id != player_id:
            continue
        duration = probe_duration(video_path)
        records.append(
            RawVideo(
                game_id=current_game_id,
                player_id=current_player_id,
                path=video_path,
                duration_sec=duration,
            )
        )
    return records


def load_existing_offsets(path: Path) -> dict[tuple[str, str], SyncOffset]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("offsets", payload)
    offsets: dict[tuple[str, str], SyncOffset] = {}
    for item in records:
        offset = SyncOffset.model_validate(item)
        offsets[(offset.game_id, offset.player_id)] = offset
    return offsets


def make_review_clip(
    video: RawVideo,
    review_dir: Path,
    review_window_sec: int,
    overwrite: bool,
) -> Path:
    output_path = review_dir / video.game_id / f"{video.player_id}_sync_review.mp4"
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video.path),
        "-t",
        str(review_window_sec),
        "-vf",
        "scale=640:-2",
        "-r",
        "1",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def normalize_sync_payload(payload: object, video: RawVideo) -> dict[str, object]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("sync response is an empty list")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("sync response must be a JSON object or one-item array")

    normalized = dict(payload)
    normalized["game_id"] = video.game_id
    normalized["player_id"] = video.player_id
    normalized.setdefault("evidence", None)
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


def save_offsets(path: Path, offsets: list[SyncOffset]) -> None:
    payload = {
        "offsets": [
            offset.model_dump()
            for offset in sorted(offsets, key=lambda item: (item.game_id, item.player_id))
        ]
    }
    write_json(path, payload)


def infer_sync_offsets(config: SyncConfig) -> dict[str, int]:
    existing = load_existing_offsets(config.output_path)
    records = iter_raw_video_records(config.raw_dir, config.game_id, config.player_id)
    if config.limit is not None:
        records = records[: config.limit]

    final_offsets = dict(existing)
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for video in records:
        key = (video.game_id, video.player_id)
        if config.resume and key in final_offsets and final_offsets[key].confidence > 0:
            stats["skipped"] += 1
            continue

        raw_response = ""
        try:
            review_clip = make_review_clip(
                video=video,
                review_dir=config.review_dir,
                review_window_sec=config.review_window_sec,
                overwrite=config.overwrite_review,
            )
            prompt = sync_start_prompt(video)
            raw_response = config.backend.generate(prompt, str(review_clip))
            payload = parse_json(raw_response)
            offset = SyncOffset.model_validate(normalize_sync_payload(payload, video))
            final_offsets[key] = offset
            write_json(
                config.review_dir / video.game_id / f"{video.player_id}_sync_result.json",
                {
                    "raw_video": {
                        "game_id": video.game_id,
                        "player_id": video.player_id,
                        "path": str(video.path),
                        "duration_sec": video.duration_sec,
                    },
                    "review_clip": str(review_clip),
                    "offset": offset.model_dump(),
                    "raw_response": raw_response,
                },
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            write_json(
                config.error_dir / f"{video.game_id}_{video.player_id}_sync.json",
                {
                    "raw_video": {
                        "game_id": video.game_id,
                        "player_id": video.player_id,
                        "path": str(video.path),
                        "duration_sec": video.duration_sec,
                    },
                    **validation_error_payload(exc, raw_response),
                },
            )

    save_offsets(config.output_path, list(final_offsets.values()))
    return stats
