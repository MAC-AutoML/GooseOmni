from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_preprocess.align.detect import detect_transition
from data_preprocess.align.materialize import materialize
from data_preprocess.common.backend import ModelBackend
from data_preprocess.common.json_utils import write_json
from data_preprocess.common.media import probe_duration
from data_preprocess.pack.gate import assert_ready_to_align


@dataclass(frozen=True)
class AlignConfig:
    session: str
    source_root: Path
    output_root: Path
    start_template: Path
    backend: ModelBackend
    role_search_sec: float = 30.0
    enter_search_sec: float = 10.0
    coarse_fps: float = 4.0
    refine: bool = True
    max_refine_fps: float = 30.0
    resume: bool = True
    overwrite: bool = False
    game_id: str | None = None
    player_id: str | None = None
    limit: int | None = None
    min_povs_per_game: int = 4
    workspace_root: Path | None = None
    enforce_gate: bool = True


def _game_pov_count(game_dir: Path) -> int:
    return sum(
        1
        for video in game_dir.glob("*.mp4")
        if video.is_file() and not video.name.startswith("._")
    )


def iter_session_videos(
    source_root: Path,
    session: str,
    *,
    game_id: str | None = None,
    player_id: str | None = None,
    min_povs_per_game: int = 0,
) -> list[Path]:
    session_dir = source_root / session
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session directory missing: {session_dir}")
    videos: list[Path] = []
    for game_dir in sorted(session_dir.iterdir()):
        if not game_dir.is_dir() or not game_dir.name.startswith("game_"):
            continue
        if game_id and game_dir.name != game_id:
            continue
        n_povs = _game_pov_count(game_dir)
        if min_povs_per_game > 0 and n_povs < min_povs_per_game:
            print(
                f"[skip] {session}/{game_dir.name}: {n_povs} povs "
                f"< min_povs_per_game={min_povs_per_game}",
                flush=True,
            )
            continue
        for video in sorted(game_dir.glob("*.mp4")):
            if video.name.startswith("._"):
                continue
            if player_id and video.stem != player_id:
                continue
            videos.append(video)
    return videos


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"session": "", "povs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def run_session_align(config: AlignConfig) -> dict[str, Any]:
    workspace = config.workspace_root or config.source_root.parent
    if config.enforce_gate:
        try:
            assert_ready_to_align(config.session, workspace)
        except RuntimeError as exc:
            print(f"[gate] {exc}", flush=True)
            return {
                "stats": {"ok": 0, "review": 0, "skipped": 0, "error": 1},
                "manifest": None,
                "blocked": str(exc),
            }

    videos = iter_session_videos(
        config.source_root,
        config.session,
        game_id=config.game_id,
        player_id=config.player_id,
        min_povs_per_game=config.min_povs_per_game,
    )
    if not videos:
        return {
            "stats": {"ok": 0, "review": 0, "skipped": 0, "error": 0},
            "manifest": None,
            "skipped_thin": True,
        }
    if config.limit is not None:
        videos = videos[: config.limit]

    session_out = config.output_root / config.session
    session_out.mkdir(parents=True, exist_ok=True)
    manifest_path = session_out / "alignment_manifest.json"
    existing = load_manifest(manifest_path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing.get("povs", []):
        if isinstance(row, dict) and "game" in row and "player" in row:
            by_key[(str(row["game"]), str(row["player"]))] = row

    stats = {"ok": 0, "review": 0, "skipped": 0, "error": 0}
    for video in videos:
        game = video.parent.name
        player = video.stem
        key = (game, player)
        output_video = session_out / game / f"{player}.mp4"
        review_dir = session_out / "alignment_review" / game / player
        prior = by_key.get(key)
        if (
            config.resume
            and prior
            and not prior.get("needs_review")
            and output_video.is_file()
        ):
            stats["skipped"] += 1
            continue

        work_dir = review_dir / "frames"
        try:
            duration = probe_duration(video)
            result = detect_transition(
                video,
                backend=config.backend,
                start_template=config.start_template,
                work_dir=work_dir,
                role_search_sec=config.role_search_sec,
                enter_search_sec=config.enter_search_sec,
                coarse_fps=config.coarse_fps,
                refine=config.refine,
                max_refine_fps=config.max_refine_fps,
            )
            written = materialize(
                source=video,
                output_video=output_video,
                review_dir=review_dir,
                result=result,
                overwrite=config.overwrite,
            )
            row = {
                "game": game,
                "player": player,
                "source": str(video),
                "output": None if written is None else str(written),
                "source_duration_sec": duration,
                **result.as_dict(),
            }
            by_key[key] = row
            if result.needs_review and written is None:
                stats["review"] += 1
            else:
                stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            by_key[key] = {
                "game": game,
                "player": player,
                "source": str(video),
                "output": None,
                "needs_review": True,
                "problem": "error",
                "error": str(exc)[:400],
            }

        write_json(
            manifest_path,
            {
                "session": config.session,
                "source_root": str(config.source_root),
                "output_root": str(config.output_root),
                "start_template": str(config.start_template),
                "povs": [by_key[item] for item in sorted(by_key)],
            },
        )

    return {"stats": stats, "manifest": str(manifest_path)}
