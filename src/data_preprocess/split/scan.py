"""Continuous per-POV episode boundary scan (standalone)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_preprocess.common.backend import ModelBackend
from data_preprocess.common.json_utils import parse_json, validation_error_payload, write_json
from data_preprocess.common.media import cut_interval, probe_duration
from data_preprocess.split.evidence import (
    END_TYPES,
    START_TYPES,
    boundary_evidence_ok,
    end_result_label,
    export_episode_filename,
)
from data_preprocess.split.prompts import (
    WindowClip,
    episode_boundary_end_prompt,
    episode_boundary_start_prompt,
)

WINDOW_SEC = 90.0
WINDOW_OVERLAP_SEC = 5.0
END_SEARCH_SKIP_SEC = 180.0
POST_END_GAP_SEC = 5.0
MIN_EPISODE_SEC = 300.0


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_boundary(item: dict, clip: WindowClip) -> dict[str, Any]:
    normalized = dict(item)
    normalized["clip_id"] = clip.clip_id
    normalized["game_id"] = clip.game_id
    normalized["player_id"] = clip.player_id
    start_sec = _coerce_float(normalized.get("aligned_start_sec"), clip.start_sec)
    end_sec = _coerce_float(normalized.get("aligned_end_sec"), clip.end_sec)
    start_sec = min(max(start_sec, clip.start_sec), clip.end_sec)
    end_sec = min(max(end_sec, clip.start_sec), clip.end_sec)
    if end_sec <= start_sec:
        end_sec = min(clip.end_sec, start_sec + 1.0)
        normalized["needs_review"] = True
    normalized["aligned_start_sec"] = start_sec
    normalized["aligned_end_sec"] = end_sec
    normalized.setdefault("boundary_type", "uncertain_boundary")
    normalized.setdefault("evidence", "")
    normalized.setdefault("confidence", 0.0)
    normalized.setdefault("source_povs", [clip.player_id])
    normalized.setdefault("needs_review", False)
    return normalized


def _annotate_window(
    clip: WindowClip,
    backend: ModelBackend,
    output_dir: Path,
    error_dir: Path,
    *,
    phase: str,
) -> list[dict[str, Any]]:
    prompt = (
        episode_boundary_start_prompt(clip)
        if phase == "start"
        else episode_boundary_end_prompt(clip)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{clip.clip_id}.json"
    try:
        # Ensure window media exists (copy/cut from full VOD if needed).
        clip_path = Path(clip.clip_path)
        if not clip_path.is_file():
            raise FileNotFoundError(f"window clip missing: {clip_path}")
        raw = backend.generate(prompt, str(clip_path))
        payload = parse_json(raw)
        if not isinstance(payload, list):
            payload = []
        kept: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            btype = str(item.get("boundary_type") or "")
            if phase == "start" and btype not in START_TYPES:
                continue
            if phase == "end" and btype not in END_TYPES:
                continue
            row = _normalize_boundary(item, clip)
            if boundary_evidence_ok(row):
                kept.append(row)
        write_json(
            out_path,
            {
                "clip": clip.__dict__,
                "phase": phase,
                "round_boundary_candidates": kept,
                "raw_response": raw,
            },
        )
        return kept
    except Exception as exc:  # noqa: BLE001
        write_json(
            error_dir / f"{clip.clip_id}.json",
            {"clip": clip.__dict__, "phase": phase, **validation_error_payload(exc, "")},
        )
        write_json(
            out_path,
            {
                "clip": clip.__dict__,
                "phase": phase,
                "round_boundary_candidates": [],
                "error": str(exc),
            },
        )
        return []


def _prepare_window_clip(
    *,
    source_video: Path,
    game_id: str,
    player_id: str,
    start_sec: float,
    end_sec: float,
    clips_dir: Path,
) -> WindowClip:
    start_i = int(round(float(start_sec)))
    end_i = int(round(float(end_sec)))
    clip_id = f"{game_id}_{player_id}_{start_i}_{end_i}"
    clip_path = clips_dir / game_id / player_id / f"{clip_id}.mp4"
    if not clip_path.is_file():
        cut_interval(
            source=source_video,
            output=clip_path,
            start_sec=float(start_i),
            end_sec=float(end_i),
            overwrite=True,
            reencode=False,
        )
    return WindowClip(
        game_id=game_id,
        player_id=player_id,
        clip_id=clip_id,
        clip_path=str(clip_path),
        start_sec=float(start_i),
        end_sec=float(end_i),
    )


def scan_player_continuous(
    *,
    source_video: Path,
    player_id: str,
    game_id: str = "g001",
    backend: ModelBackend,
    run_dir: Path,
    export_dir: Path,
    window_sec: float = WINDOW_SEC,
    overlap_sec: float = WINDOW_OVERLAP_SEC,
    end_skip_sec: float = END_SEARCH_SKIP_SEC,
    max_episodes: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Scan one long POV VOD into episode_XXX_*.mp4 under export_dir/player_id."""
    duration = probe_duration(source_video)
    cand_dir = run_dir / "boundary_candidates"
    err_dir = run_dir / "boundary_errors"
    clips_dir = run_dir / "review_clips"
    lock_path = run_dir / "locked_boundaries.json"
    progress_path = run_dir / "episode_progress.json"

    completed: list[dict[str, Any]] = []
    open_start: float | None = None
    search_after = -1.0
    if resume and lock_path.is_file():
        prior = json.loads(lock_path.read_text(encoding="utf-8"))
        row = (prior.get("players") or {}).get(player_id) or {}
        completed = list(row.get("episodes") or [])
        if row.get("open_start_sec") is not None:
            open_start = float(row["open_start_sec"])
        if row.get("search_after_sec") is not None:
            search_after = float(row["search_after_sec"])
        elif completed:
            search_after = float(completed[-1]["abs_end_sec"])

    def persist() -> None:
        payload = {
            "players": {
                player_id: {
                    "search_after_sec": search_after,
                    "open_start_sec": open_start,
                    "episodes": completed,
                }
            }
        }
        write_json(lock_path, payload)
        write_json(
            progress_path,
            {
                "players": {
                    player_id: {
                        "episode_count": len(completed),
                        "open_start_sec": open_start,
                        "search_after_sec": search_after,
                        "episodes": completed,
                    }
                }
            },
        )

    step = max(1.0, float(window_sec) - float(overlap_sec))
    if open_start is not None:
        cursor = float(open_start) + float(end_skip_sec)
        phase = "end"
    elif search_after >= 0:
        cursor = float(search_after) + float(POST_END_GAP_SEC)
        phase = "start"
    else:
        cursor = 0.0
        phase = "start"

    export_player = export_dir / player_id
    export_player.mkdir(parents=True, exist_ok=True)

    while cursor < duration - 1.0:
        if max_episodes is not None and len(completed) >= int(max_episodes):
            break
        win_end = min(duration, cursor + float(window_sec))
        clip = _prepare_window_clip(
            source_video=source_video,
            game_id=game_id,
            player_id=player_id,
            start_sec=cursor,
            end_sec=win_end,
            clips_dir=clips_dir,
        )
        rows = _annotate_window(clip, backend, cand_dir, err_dir, phase=phase)

        if phase == "start":
            starts = sorted(
                (r for r in rows if r.get("boundary_type") in START_TYPES),
                key=lambda r: float(r["aligned_start_sec"]),
            )
            if starts:
                open_start = float(starts[0]["aligned_start_sec"])
                cursor = open_start + float(end_skip_sec)
                phase = "end"
                persist()
                continue
            cursor += step
            continue

        # phase == end
        ends = sorted(
            (r for r in rows if r.get("boundary_type") in END_TYPES),
            key=lambda r: float(r["aligned_start_sec"]),
        )
        chosen = None
        for row in ends:
            end_sec = float(row["aligned_start_sec"])
            if open_start is None:
                continue
            if end_sec - float(open_start) < float(MIN_EPISODE_SEC):
                continue
            chosen = row
            break
        if chosen is None:
            cursor += step
            if cursor >= duration - 1.0:
                break
            continue

        abs_start = float(open_start)
        abs_end = float(chosen["aligned_start_sec"])
        ep_index = len(completed)
        end_label = end_result_label(
            chosen.get("evidence") or "",
            end_type=str(chosen.get("boundary_type") or ""),
        )
        episode = {
            "episode_id": f"{game_id}_episode_{ep_index:03d}",
            "episode_index": ep_index,
            "game_id": game_id,
            "player_id": player_id,
            "abs_start_sec": abs_start,
            "abs_end_sec": abs_end,
            "end_type": chosen.get("boundary_type") or "game_end",
            "end_evidence": chosen.get("evidence") or "",
            "end_label": end_label,
        }
        out_name = export_episode_filename(ep_index, end_label=end_label)
        out_path = export_player / out_name
        cut_interval(
            source=source_video,
            output=out_path,
            start_sec=abs_start,
            end_sec=abs_end,
            overwrite=True,
            reencode=False,
        )
        episode["output"] = str(out_path)
        completed.append(episode)
        search_after = abs_end
        open_start = None
        cursor = abs_end + float(POST_END_GAP_SEC)
        phase = "start"
        persist()

    persist()
    return {
        "player_id": player_id,
        "episode_count": len(completed),
        "episodes": completed,
        "export_dir": str(export_player),
        "lock_path": str(lock_path),
    }
