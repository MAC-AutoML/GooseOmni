from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .aligned_clips import probe_duration, valid_clip
from .alignment import raw_interval


def phase_windows(
    episodes: Iterable[dict[str, Any]],
    valid_range: tuple[float, float | None],
    excluded_ranges: tuple[tuple[float, float], ...],
    max_chunk_sec: float,
) -> list[dict[str, Any]]:
    """Split audited phases without ever crossing a phase or excluded range."""
    if max_chunk_sec <= 0:
        raise ValueError("max_chunk_sec must be positive")
    valid_start, valid_end = valid_range
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for phase in episode.get("phases", []):
            start = max(float(phase["abs_start_sec"]), valid_start)
            end = float(phase["abs_end_sec"])
            if valid_end is not None:
                end = min(end, valid_end)
            for allowed_start, allowed_end in _subtract_ranges(
                start, end, excluded_ranges
            ):
                cursor = allowed_start
                while cursor < allowed_end:
                    chunk_end = min(cursor + max_chunk_sec, allowed_end)
                    rows.append(
                        {
                            "episode_id": str(episode["episode_id"]),
                            "phase_index": int(phase["phase_index"]),
                            "phase_type": str(phase["phase_type"]),
                            "start_sec": cursor,
                            "end_sec": chunk_end,
                        }
                    )
                    cursor = chunk_end
    return rows


def materialize_perception_clips(
    games: Iterable[Any],
    alignment: dict[str, Any],
    episodes: list[dict[str, Any]],
    source_manifest: Path,
    output_dir: Path,
    manifest_path: Path,
    max_chunk_sec: float,
) -> dict[str, int]:
    """Create short AV clips directly from raw POV videos and the affine mapping."""
    parents = _parent_clips(source_manifest)
    records: list[dict[str, Any]] = []
    created = skipped = 0
    by_game: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_game.setdefault(str(episode["game_id"]), []).append(episode)
    for game in games:
        windows = phase_windows(
            by_game.get(game.game_id, []),
            game.valid_range,
            game.excluded_ranges,
            max_chunk_sec,
        )
        durations = {
            player.player_id: probe_duration(player.video_path)
            for player in game.players
        }
        for player in game.players:
            mapping = alignment["mappings"][player.player_id]
            for window in windows:
                start = float(window["start_sec"])
                end = float(window["end_sec"])
                raw_start, raw_end = raw_interval(
                    mapping, start, end, durations[player.player_id]
                )
                clip_id = (
                    f"{game.game_id}_{player.player_id}_{window['episode_id']}_"
                    f"p{window['phase_index']}_{window['phase_type']}_"
                    f"{start:.3f}_{end:.3f}"
                )
                target = (
                    output_dir
                    / game.game_id
                    / player.player_id
                    / str(window["episode_id"])
                    / f"phase_{window['phase_index']}"
                    / f"{clip_id}.mp4"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                expected_duration = raw_end - raw_start
                if valid_clip(target, expected_duration) and _has_av_streams(target):
                    skipped += 1
                else:
                    _extract_av_clip(
                        player.video_path, target, raw_start, expected_duration
                    )
                    created += 1
                parent = _find_parent(
                    parents, game.game_id, player.player_id, start, end
                )
                records.append(
                    {
                        "game_id": game.game_id,
                        "player_id": player.player_id,
                        "clip_id": clip_id,
                        "clip_path": str(target),
                        "start_sec": start,
                        "end_sec": end,
                        "episode_id": window["episode_id"],
                        "phase_index": window["phase_index"],
                        "phase_type": window["phase_type"],
                        "source_clip_id": parent,
                        "raw_start_sec": raw_start,
                        "raw_end_sec": raw_end,
                        "alignment_mapping": {
                            "scale": mapping["scale"],
                            "offset": mapping["offset"],
                        },
                        "media_sha256": _sha256_file(target),
                    }
                )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"records": len(records), "created": created, "skipped": skipped}


def _subtract_ranges(
    start: float,
    end: float,
    excluded: tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    if end <= start:
        return []
    allowed = [(start, end)]
    for left, right in sorted(excluded):
        next_allowed = []
        for current_start, current_end in allowed:
            if right <= current_start or left >= current_end:
                next_allowed.append((current_start, current_end))
                continue
            if current_start < left:
                next_allowed.append((current_start, min(left, current_end)))
            if right < current_end:
                next_allowed.append((max(right, current_start), current_end))
        allowed = next_allowed
    return [(left, right) for left, right in allowed if right > left]


def _parent_clips(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _find_parent(
    rows: list[dict[str, Any]],
    game_id: str,
    player_id: str,
    start: float,
    end: float,
) -> str | None:
    for row in rows:
        if (
            row.get("game_id") == game_id
            and row.get("player_id") == player_id
            and float(row["start_sec"]) <= start
            and end <= float(row["end_sec"])
        ):
            return str(row["clip_id"])
    return None


def _extract_av_clip(
    source: Path, target: Path, raw_start: float, duration: float
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{raw_start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-y",
            str(target),
        ],
        check=True,
    )
    if not _has_av_streams(target):
        raise RuntimeError(f"perception clip lacks expected AV streams: {target}")


def _has_av_streams(path: Path) -> bool:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return False
    stream_types = {
        str(row.get("codec_type"))
        for row in json.loads(completed.stdout or "{}").get("streams", [])
    }
    return {"video", "audio"}.issubset(stream_types)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
