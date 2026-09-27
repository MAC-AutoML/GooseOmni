from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from gooseomni.annotation.eval_export import cut_video
from gooseomni.annotation.json_utils import write_json
from gooseomni.annotation.splitting import load_sync_offsets

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SYNC_CORRECTIONS: dict[str, float] = {
    "Gemini": 0.0,
    "baile": -10.0,
    "beigang": 3.0,
    "mojiang": -13.0,
    "saoyi": -2.0,
    "xiaolu": -16.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export standalone gooseomni videos aligned on absolute game time."
    )
    parser.add_argument(
        "--manifest-path",
        default="data/processed/clip_manifest.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--sync-offsets",
        default="data/processed/sync_offsets.json",
        type=Path,
    )
    parser.add_argument(
        "--round-boundaries-dir",
        default="annotations/round_boundaries",
        type=Path,
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/gooseomni_aligned_v2", type=Path)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--limit-segments", default=None, type=int)
    parser.add_argument(
        "--sync-corrections-json",
        default=None,
        type=Path,
        help="Optional JSON object mapping player_id to correction seconds.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stream-copy",
        action="store_true",
        help="Use ffmpeg stream copy. Faster, but less accurate near non-keyframe cuts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync_corrections = load_sync_corrections(args.sync_corrections_json)
    stats = export_gooseomni_aligned_videos(
        manifest_path=args.manifest_path,
        sync_offsets_path=args.sync_offsets,
        round_boundaries_dir=args.round_boundaries_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        sync_corrections=sync_corrections,
        game_id=args.game_id,
        limit_segments=args.limit_segments,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        reencode=not args.stream_copy,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def load_sync_corrections(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_SYNC_CORRECTIONS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return {str(key): float(value) for key, value in payload.items()}


def export_gooseomni_aligned_videos(
    *,
    manifest_path: Path,
    sync_offsets_path: Path,
    round_boundaries_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    sync_corrections: dict[str, float],
    game_id: str | None,
    limit_segments: int | None,
    overwrite: bool,
    dry_run: bool,
    reencode: bool,
) -> dict[str, int]:
    clips = load_clip_manifest(manifest_path)
    if game_id:
        clips = [clip for clip in clips if clip["game_id"] == game_id]
    segments = build_segments(clips)
    if limit_segments is not None:
        segments = segments[:limit_segments]

    base_offsets = load_sync_offsets(sync_offsets_path)
    output_video_root = output_dir / "videos"
    exported_segments: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    video_count = 0

    for segment in segments:
        segment_record = segment_metadata(segment)
        output_video_root / segment["game_id"] / segment["segment_id"]
        for clip in segment["clips"]:
            base_offset = base_offsets.get((clip["game_id"], clip["player_id"]))
            if base_offset is None:
                skipped.append(skip_record(clip, "missing_sync_offset"))
                continue
            correction = sync_corrections.get(clip["player_id"], 0.0)
            raw_start = base_offset + correction + clip["start_sec"]
            raw_end = base_offset + correction + clip["end_sec"]
            if raw_start < 0:
                skipped.append(
                    skip_record(clip, "negative_raw_start", raw_start_sec=raw_start)
                )
                continue
            if raw_end <= raw_start:
                skipped.append(
                    skip_record(clip, "non_positive_duration", raw_start_sec=raw_start)
                )
                continue

            source_video = raw_dir / clip["game_id"] / f"{clip['player_id']}.mp4"
            if not source_video.exists():
                skipped.append(
                    skip_record(
                        clip, "missing_raw_video", source_video=str(source_video)
                    )
                )
                continue

            rel_video = (
                Path("videos")
                / clip["game_id"]
                / segment["segment_id"]
                / f"{clip['player_id']}.mp4"
            )
            output_video = output_dir / rel_video
            if dry_run:
                print(
                    f"{source_video} -> {output_video} "
                    f"raw=[{raw_start:.3f},{raw_end:.3f}] "
                    f"aligned=[{clip['start_sec']:.3f},{clip['end_sec']:.3f}]"
                )
            else:
                cut_video(
                    source_video=source_video,
                    output_video=output_video,
                    raw_start_sec=raw_start,
                    duration_sec=raw_end - raw_start,
                    overwrite=overwrite,
                    reencode=reencode,
                )

            segment_record["povs"].append(
                {
                    "player_id": clip["player_id"],
                    "video_file": rel_video.as_posix(),
                    "source_raw_video": source_video.as_posix(),
                    "base_sync_raw_start_sec": base_offset,
                    "sync_correction_sec": correction,
                    "corrected_sync_raw_start_sec": base_offset + correction,
                    "raw_start_sec": raw_start,
                    "raw_end_sec": raw_end,
                    "aligned_start_sec": clip["start_sec"],
                    "aligned_end_sec": clip["end_sec"],
                    "qwen_round_boundary_candidates": load_round_candidates(
                        round_boundaries_dir, clip["clip_id"]
                    ),
                }
            )
            video_count += 1

        segment_record["pov_count"] = len(segment_record["povs"])
        exported_segments.append(segment_record)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "segments.json", exported_segments)
        write_json(output_dir / "skipped.json", skipped)
        write_json(
            output_dir / "manifest.json",
            {
                "dataset": "gooseomni",
                "format": "gooseomni_aligned_video_dataset_v2",
                "description": (
                    "Multi-POV Goose Goose Duck video clips aligned on absolute game "
                    "time with phase-transition correction."
                ),
                "source": "raw videos re-cut with corrected sync offsets",
                "source_manifest": manifest_path.as_posix(),
                "base_sync_offsets": sync_offsets_path.as_posix(),
                "qwen_round_boundaries": round_boundaries_dir.as_posix(),
                "sync_corrections_sec": sync_corrections,
                "video_root": "videos",
                "segment_count": len(exported_segments),
                "video_count": video_count,
                "skipped_count": len(skipped),
                "segments_file": "segments.json",
                "jsonl_file": "segments.jsonl",
                "notes": [
                    "Each segment shares aligned_start_sec/aligned_end_sec across POVs.",
                    (
                        "raw_start_sec = base_sync_raw_start_sec + "
                        "sync_correction_sec + aligned_start_sec."
                    ),
                    (
                        "sync_corrections_sec are semantic alignment corrections "
                        "estimated from public phase transitions, not fixed-duration "
                        "round labels."
                    ),
                    (
                        "Qwen3-Omni boundary candidates are preserved as evidence "
                        "metadata, not treated as fixed-duration labels."
                    ),
                ],
            },
        )
        write_jsonl(output_dir / "segments.jsonl", exported_segments)

    return {
        "segments": len(exported_segments),
        "videos": video_count,
        "skipped": len(skipped),
    }


def load_clip_manifest(path: Path) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            clips.append(
                {
                    "game_id": str(item["game_id"]),
                    "player_id": str(item["player_id"]),
                    "clip_id": str(item["clip_id"]),
                    "clip_path": str(item["clip_path"]),
                    "start_sec": float(item["start_sec"]),
                    "end_sec": float(item["end_sec"]),
                }
            )
    return clips


def build_segments(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[(clip["game_id"], clip["start_sec"], clip["end_sec"])].append(clip)

    segments: list[dict[str, Any]] = []
    for index, ((game_id, start, end), group) in enumerate(
        sorted(grouped.items()), start=1
    ):
        segments.append(
            {
                "segment_id": (
                    f"{game_id}_seg_{index:04d}_{int(round(start)):06d}_{int(round(end)):06d}"
                ),
                "game_id": game_id,
                "aligned_start_sec": start,
                "aligned_end_sec": end,
                "duration_sec": end - start,
                "clips": sorted(group, key=lambda item: item["player_id"]),
            }
        )
    return segments


def segment_metadata(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "game_id": segment["game_id"],
        "aligned_start_sec": segment["aligned_start_sec"],
        "aligned_end_sec": segment["aligned_end_sec"],
        "duration_sec": segment["duration_sec"],
        "pov_count": 0,
        "povs": [],
    }


def load_round_candidates(
    round_boundaries_dir: Path, clip_id: str
) -> list[dict[str, Any]]:
    path = round_boundaries_dir / f"{clip_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("round_boundary_candidates", [])
    if not isinstance(candidates, list):
        return []
    return candidates


def skip_record(clip: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    record = {
        "clip_id": clip["clip_id"],
        "game_id": clip["game_id"],
        "player_id": clip["player_id"],
        "aligned_start_sec": clip["start_sec"],
        "aligned_end_sec": clip["end_sec"],
        "reason": reason,
    }
    record.update(extra)
    return record


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


if __name__ == "__main__":
    main()
