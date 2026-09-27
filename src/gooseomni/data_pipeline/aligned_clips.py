from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .alignment import raw_interval


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def valid_clip(path: Path, expected_duration: float) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        duration = probe_duration(path)
    except (subprocess.CalledProcessError, ValueError):
        return False
    return abs(duration - expected_duration) <= 2.0


def materialize_aligned_clips(
    games: Any,
    alignment: dict[str, Any],
    output_dir: Path,
    manifest_path: Path,
    window_sec: float = 90.0,
    limit_clips: int | None = None,
) -> dict[str, int]:
    """Cut clips from one audited affine mapping; never applies ad-hoc corrections."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    created = skipped = 0
    for game in games:
        durations = {player.player_id: probe_duration(player.video_path) for player in game.players}
        global_end = min(
            (
                durations[player.player_id]
                - float(alignment["mappings"][player.player_id]["offset"])
            )
            / float(alignment["mappings"][player.player_id]["scale"])
            for player in game.players
        )
        configured_end = game.valid_range[1]
        if configured_end is not None:
            global_end = min(global_end, configured_end)
        start = max(
            float(game.valid_range[0]),
            float(alignment.get("common_coverage_start_sec", 0.0)),
        )
        while start < global_end:
            end = min(start + window_sec, global_end)
            if any(left < end and right > start for left, right in game.excluded_ranges):
                start = end
                continue
            for player in game.players:
                mapping = alignment["mappings"][player.player_id]
                raw_start, raw_end = raw_interval(
                    mapping, start, end, durations[player.player_id]
                )
                clip_id = f"{game.game_id}_{player.player_id}_{start:.3f}_{end:.3f}"
                target = output_dir / game.game_id / player.player_id / f"{clip_id}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                clip_duration = raw_end - raw_start
                if valid_clip(target, clip_duration):
                    skipped += 1
                else:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-ss",
                            f"{raw_start:.6f}",
                            "-i",
                            str(player.video_path),
                            "-t",
                            f"{clip_duration:.6f}",
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
                    created += 1
                records.append(
                    {
                        "game_id": game.game_id,
                        "player_id": player.player_id,
                        "clip_id": clip_id,
                        "clip_path": str(target),
                        "start_sec": start,
                        "end_sec": end,
                        "raw_start_sec": raw_start,
                        "raw_end_sec": raw_end,
                        "alignment_mapping": {
                            "scale": mapping["scale"],
                            "offset": mapping["offset"],
                        },
                    }
                )
                if limit_clips is not None and len(records) >= limit_clips:
                    break
            if limit_clips is not None and len(records) >= limit_clips:
                break
            start = end
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"records": len(records), "created": created, "skipped": skipped}
