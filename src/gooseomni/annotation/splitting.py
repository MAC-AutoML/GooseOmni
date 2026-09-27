from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schemas import Clip, SyncOffset


@dataclass(frozen=True)
class SplitConfig:
    raw_dir: Path
    output_dir: Path
    manifest_path: Path
    segment_sec: int = 90
    overlap_sec: int = 10
    overwrite: bool = False
    dry_run: bool = False
    sync_offsets_path: Path | None = None
    game_id: str | None = None
    player_id: str | None = None
    limit_clips: int | None = None


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def load_sync_offsets(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("offsets", payload)
    offsets: dict[tuple[str, str], float] = {}
    for item in records:
        offset = SyncOffset.model_validate(item)
        offsets[(offset.game_id, offset.player_id)] = offset.raw_start_sec
    return offsets


def iter_raw_videos(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*/*.mp4"))


def build_clip_plan(
    video_path: Path,
    raw_dir: Path,
    output_root: Path,
    duration: float,
    offset_sec: float,
    segment_sec: int,
    overlap_sec: int,
) -> list[Clip]:
    if overlap_sec >= segment_sec:
        raise ValueError("overlap_sec must be smaller than segment_sec")

    game_id = video_path.parent.name
    player_id = video_path.stem
    usable_duration = max(0.0, duration - offset_sec)
    step_sec = segment_sec - overlap_sec
    clips: list[Clip] = []
    start = 0.0

    while start < usable_duration:
        end = min(start + segment_sec, usable_duration)
        if end <= start:
            break
        start_i = int(round(start))
        end_i = int(round(end))
        clip_id = f"{game_id}_{player_id}_{start_i}_{end_i}"
        clip_path = output_root / game_id / player_id / f"{clip_id}.mp4"
        clips.append(
            Clip(
                game_id=game_id,
                player_id=player_id,
                clip_id=clip_id,
                clip_path=str(clip_path),
                start_sec=start_i,
                end_sec=end_i,
            )
        )
        if end >= usable_duration:
            break
        start += step_sec

    return clips


def split_clip(
    source_path: Path,
    clip: Clip,
    raw_start_sec: float,
    overwrite: bool,
    dry_run: bool,
) -> None:
    output_path = Path(clip.clip_path)
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seek_sec = raw_start_sec + clip.start_sec
    duration_sec = clip.end_sec - clip.start_sec
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{seek_sec:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_sec:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def run_split(config: SplitConfig) -> list[Clip]:
    offsets = load_sync_offsets(config.sync_offsets_path)
    planned: list[tuple[Path, float, Clip]] = []
    for video_path in iter_raw_videos(config.raw_dir):
        game_id = video_path.parent.name
        player_id = video_path.stem
        if config.game_id and game_id != config.game_id:
            continue
        if config.player_id and player_id != config.player_id:
            continue
        duration = probe_duration(video_path)
        offset_sec = offsets.get((game_id, player_id), 0.0)
        clips = build_clip_plan(
            video_path=video_path,
            raw_dir=config.raw_dir,
            output_root=config.output_dir,
            duration=duration,
            offset_sec=offset_sec,
            segment_sec=config.segment_sec,
            overlap_sec=config.overlap_sec,
        )
        planned.extend((video_path, offset_sec, clip) for clip in clips)

    planned.sort(key=lambda item: (item[2].game_id, item[2].start_sec, item[2].player_id, item[2].end_sec))
    if config.limit_clips is not None:
        planned = planned[: config.limit_clips]

    all_clips = [clip for _, _, clip in planned]
    for video_path, offset_sec, clip in planned:
        split_clip(video_path, clip, offset_sec, config.overwrite, config.dry_run)

    if not config.dry_run:
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with config.manifest_path.open("w", encoding="utf-8") as handle:
            for clip in all_clips:
                handle.write(clip.model_dump_json() + "\n")
    return all_clips
