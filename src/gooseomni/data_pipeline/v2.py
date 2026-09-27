from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from gooseomni.config import PATHS

from .provenance import sha256_file, write_json

COUNT_PATHS = {
    "public/leaderboard_core/trials.jsonl": 889,
    "private/leaderboard_core/hidden_gold.jsonl": 889,
    "public/raw_video_smoke/raw_video_smoke.jsonl": 48,
    "public/leaderboard_core/probe_groups_public.jsonl": 276,
    "public/agentic_midgame_prediction/trials.jsonl": 160,
    "private/agentic_midgame_prediction/hidden_gold.jsonl": 160,
}
PRIVATE_MARKERS = ('"hidden_gold":', '"forbidden_event_ids":')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _model_verified(value: Any) -> Any:
    if isinstance(value, dict):
        result = {
            key.replace("human_verified", "model_verified"): _model_verified(item)
            for key, item in value.items()
        }
        if "gold_source" in result:
            result["gold_source"] = "model_verified"
        if "recommended_gold_source" in result:
            result["recommended_gold_source"] = "model_verified"
        if result.get("human_reviewer") == "codex_human_reviewer":
            result["review_model"] = "legacy_codex_qwen_review"
            result.pop("human_reviewer", None)
        return result
    if isinstance(value, list):
        return [_model_verified(item) for item in value]
    if isinstance(value, str):
        return value.replace("human_verified", "model_verified")
    return value


def _rewrite_model_provenance(root: Path) -> int:
    rewritten = 0
    for path in sorted(root.rglob("*.jsonl")):
        rows = read_jsonl(path)
        converted = [_model_verified(row) for row in rows]
        if converted != rows:
            write_jsonl(path, converted)
            rewritten += 1
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        converted = _model_verified(value)
        if converted != value:
            write_json(path, converted)
            rewritten += 1
    return rewritten


def _distance(segment: dict[str, Any], cutoff: float) -> float:
    start = float(segment["aligned_start_sec"])
    end = float(segment["aligned_end_sec"])
    if start <= cutoff <= end:
        return 0.0
    return min(abs(cutoff - start), abs(cutoff - end))


def _resolve_pov(
    segments: list[dict[str, Any]], game_id: str, player_id: str, cutoff: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [row for row in segments if row.get("game_id") == game_id]
    if not candidates:
        raise ValueError(f"no segments for game: {game_id}")
    segment = min(candidates, key=lambda row: _distance(row, cutoff))
    pov = next(
        (row for row in segment.get("povs", []) if row.get("player_id") == player_id),
        None,
    )
    if pov is None:
        raise ValueError(f"no POV for {game_id}/{player_id}/{segment['segment_id']}")
    return segment, pov


def _repair_raw_video(
    target: Path, dataset_root: Path, segments_path: Path
) -> dict[str, Any]:
    raw_path = target / "public/raw_video_smoke/raw_video_smoke.jsonl"
    rows = read_jsonl(raw_path)
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    linked: set[str] = set()
    for row in rows:
        player = str(row.get("target_player") or row.get("ego_player"))
        segment, pov = _resolve_pov(
            segments,
            str(row["game_id"]),
            player,
            float(row.get("cutoff_abs_sec", 0.0)),
        )
        source = dataset_root / str(pov["video_file"])
        if not source.is_file():
            raise FileNotFoundError(f"authoritative POV video is missing: {source}")
        relative = Path("assets/videos") / str(segment["segment_id"]) / source.name
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(os.path.relpath(source, destination.parent))
        row["video_file"] = relative.as_posix()
        row["video_resolution_basis"] = {
            "dataset": "data/gooseomni",
            "segment_id": segment["segment_id"],
            "player_id": player,
            "selection": "contains_cutoff_or_nearest_aligned_segment",
        }
        linked.add(relative.as_posix())
    write_jsonl(raw_path, rows)
    return {"rows": len(rows), "unique_video_assets": len(linked)}


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _public_leaks(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted((root / "public").rglob("*")):
        if path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(root)}:{marker}")
    return hits


def build_manifest(root: Path, source_root: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "symlink": path.is_symlink(),
        }
        for path in files
    ]
    return {
        "name": "gooseomni_v2",
        "source": str(source_root),
        "gold_source": "model_verified",
        "review_models": ["qwen3_omni", "legacy_codex_qwen_review"],
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "validation": {
            "line_counts": {path: _line_count(root / path) for path in COUNT_PATHS},
            "public_file_leak_hits": _public_leaks(root),
        },
        "files": records,
    }


def build_v2(
    target: Path,
    source: Path | None = None,
    dataset_root: Path | None = None,
    segments_path: Path | None = None,
) -> dict[str, Any]:
    source = (source or PATHS.root / "benchmark/gooseomni_v1").resolve()
    dataset_root = (dataset_root or PATHS.root / "data/gooseomni").resolve()
    segments_path = (segments_path or dataset_root / "segments.json").resolve()
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)
    try:
        rewritten = _rewrite_model_provenance(target)
        raw = _repair_raw_video(target, dataset_root, segments_path)
        readme = target / "README.md"
        readme.write_text(
            "# GooseOmni v2\n\n"
            "This release freezes GooseOmni v1 trial semantics, repairs raw-video assets, "
            "and labels Qwen3-Omni plus Codex-produced gold as `model_verified`. It is not "
            "a human-verified benchmark.\n",
            encoding="utf-8",
        )
        manifest = build_manifest(target, source)
        write_json(target / "manifest.json", manifest)
        return {"target": str(target), "rewritten_files": rewritten, **raw}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _probe_media(path: Path) -> tuple[bool, bool]:
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
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode:
        return False, False
    streams = json.loads(completed.stdout).get("streams", [])
    types = {row.get("codec_type") for row in streams}
    return "video" in types, "audio" in types


def validate_v2(root: Path, probe_media: bool = True) -> dict[str, Any]:
    issues: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "issues": ["missing manifest.json"]}
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_manifest(root, Path(str(recorded.get("source", "unknown"))))
    expected_counts = recorded.get("validation", {}).get("line_counts", COUNT_PATHS)
    if actual["validation"]["line_counts"] != expected_counts:
        issues.append("line counts do not match the recorded v2 contract")
    if actual["validation"]["public_file_leak_hits"]:
        issues.append("private markers found in public files")
    if recorded.get("files") != actual.get("files"):
        issues.append("manifest inventory or SHA-256 mismatch")
    for path in sorted(root.rglob("*.json*")):
        if path.name == "manifest.json":
            continue
        if "human_verified" in path.read_text(encoding="utf-8"):
            issues.append(
                f"automatic v2 contains human_verified: {path.relative_to(root)}"
            )
    raw_rows = read_jsonl(root / "public/raw_video_smoke/raw_video_smoke.jsonl")
    media_results = []
    for value in sorted({str(row["video_file"]) for row in raw_rows}):
        path = root / value
        if not path.is_file():
            issues.append(f"missing raw-video asset: {value}")
            continue
        if probe_media:
            has_video, has_audio = _probe_media(path)
            media_results.append(
                {"path": value, "video": has_video, "audio": has_audio}
            )
            if not has_video or not has_audio:
                issues.append(f"invalid AV streams: {value}")
    return {
        "ok": not issues,
        "issues": issues,
        "line_counts": actual["validation"]["line_counts"],
        "raw_video_rows": len(raw_rows),
        "unique_media": len({str(row["video_file"]) for row in raw_rows}),
        "media": media_results,
        "public_file_leak_hits": actual["validation"]["public_file_leak_hits"],
    }
