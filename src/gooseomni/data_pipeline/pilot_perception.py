from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .pilot_common import cache_path
from .provenance import sha256_tree
from .trajectory_review import recoverable_visual_errors


def _json_file_count(path: Path) -> int:
    return (
        sum(
            1
            for item in path.rglob("*.json")
            if not item.name.startswith(("._", ".__"))
        )
        if path.is_dir()
        else 0
    )


def _active_error_count(error_root: Path, output_root: Path) -> int:
    if not error_root.is_dir():
        return 0
    return sum(
        1
        for item in error_root.rglob("*.json")
        if not item.name.startswith(("._", ".__"))
        and not (output_root / item.relative_to(error_root)).is_file()
    )


def _active_oom_count(error_root: Path, output_root: Path) -> int:
    if not error_root.is_dir():
        return 0
    count = 0
    for item in error_root.rglob("*.json"):
        if item.name.startswith(("._", ".__")):
            continue
        if (output_root / item.relative_to(error_root)).is_file():
            continue
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "CUDA out of memory" in str(payload.get("error", "")):
            count += 1
    return count


def _perception_manifest(context: Any) -> Path:
    manifest = context.run_root / "artifacts/perception_clip_manifest.jsonl"
    if manifest.is_file():
        return manifest
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("phase-aware perception clips must be materialized in Slurm")
    from .perception_windows import materialize_perception_clips

    alignment = json.loads(
        (context.run_root / "artifacts/alignment.json").read_text(encoding="utf-8")
    )
    episodes = json.loads(
        (context.run_root / "artifacts/episodes.json").read_text(encoding="utf-8")
    )
    materialize_perception_clips(
        context.config.games,
        alignment,
        episodes,
        context.run_root / "artifacts/clip_manifest.jsonl",
        context.run_root / "artifacts/perception_clips",
        manifest,
        min(
            context.config.max_visual_chunk_sec,
            context.config.max_audio_chunk_sec,
        ),
    )
    return manifest


def _perception_stage(context: Any, pass_kind: str) -> dict[str, Any]:
    cache_name = f"perception_{pass_kind}"
    target = context.run_root / f"annotations/perception_{pass_kind}"
    cached = cache_path(context, cache_name)
    if cached is not None:
        shutil.copytree(cached, target, ignore=shutil.ignore_patterns("._*", ".__*"))
        return {
            "mode": "cache_replay",
            "output": str(target),
            "input_hash": sha256_tree(cached),
        }
    server_url = os.getenv("QWEN3_OMNI_SERVER_URL")
    if not server_url or not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("fresh dual-pass perception must run in Slurm")
    from .stages import run_command

    manifest_path = _perception_manifest(context)
    command = [
        sys.executable,
        "tools/annotation/run_qwen_annotation.py",
        "--manifest-path",
        str(manifest_path),
        "--output-dir",
        str(target),
        "--error-dir",
        str(context.run_root / f"errors/perception_{pass_kind}"),
        "--backend",
        "local",
        "--server-url",
        server_url,
        "--model",
        context.config.perception_model,
        "--pass-kind",
        pass_kind,
        "--resume",
        "--canonical-players",
        ",".join(
            player.player_id
            for game in context.config.games
            for player in game.players
        ),
        "--speaker-confidence-min",
        str(context.config.speaker_confidence_min),
    ]
    limit = os.getenv(f"GOOSEOMNI_PILOT_LIMIT_{pass_kind.upper()}")
    if limit is None:
        limit = os.getenv("GOOSEOMNI_PILOT_LIMIT_CLIPS")
    if limit:
        command.extend(["--limit", limit])
    executed = run_command(command)
    error_root = context.run_root / f"errors/perception_{pass_kind}"
    historical_error_count = _json_file_count(error_root)
    active_error_count = _active_error_count(error_root, target)
    active_oom_count = _active_oom_count(error_root, target)
    output_count = _json_file_count(target)
    if active_oom_count:
        raise RuntimeError(
            f"{pass_kind} perception has {active_oom_count} active CUDA OOM errors; "
            "restart the model service and resume"
        )
    if active_error_count and output_count == 0 and not recoverable_visual_errors(
        pass_kind, error_root, target
    ):
        raise RuntimeError(
            f"{pass_kind} perception failed closed: {active_error_count} "
            f"quality/schema errors; successful outputs={output_count}"
        )
    return {
        "mode": "qwen_live_partial" if active_error_count else "qwen_live",
        "output": str(target),
        "outputs": output_count,
        "active_errors": active_error_count,
        "active_oom_errors": active_oom_count,
        "historical_errors": historical_error_count,
        "manifest": str(manifest_path),
        "command": executed,
    }


def perceive_visual_stage(context: Any) -> dict[str, Any]:
    return _perception_stage(context, "visual")


def perceive_audio_stage(context: Any) -> dict[str, Any]:
    return _perception_stage(context, "audio")
