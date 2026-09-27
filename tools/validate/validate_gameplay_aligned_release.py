from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import collections
import json
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
REQUIRED_TOP = {"README.md", "videos", "annotations"}
REQUIRED_FIELDS = {
    "dataset",
    "format",
    "game_id",
    "phase_id",
    "phase_type",
    "player_id",
    "video_file",
    "annotation_file",
    "aligned_start_sec",
    "aligned_end_sec",
    "duration_sec",
    "evidence",
    "confidence",
    "time",
    "player_status",
    "role_and_goal",
    "gameplay_trace",
    "utterances",
    "private_memory",
    "belief_state",
    "tom_questions",
    "needs_human_review",
}
SOURCE_TYPES = {
    "direct_visual_observation",
    "speech_claim",
    "public_result",
    "inferred_belief",
    "hidden_or_not_visible_information",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gameplay-aware Omni Goose release layout and per-video annotations.")
    parser.add_argument("--release-root", default=Path("runs/gooseomni_gameplay_pass1/release_gameplay_aligned_v1"), type=Path)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--max-duration-delta-sec", default=1.0, type=float)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, path: Path | None, message: str) -> None:
    issues.append({"severity": severity, "code": code, "path": path.as_posix() if path else None, "message": message})


def read_json(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, "error", "invalid_json", path, repr(exc))
        return None


def ffprobe_duration(path: Path, issues: list[dict[str, Any]]) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, "error", "ffprobe_failed", path, repr(exc))
        return None


def validate_annotation(root: Path, ann_path: Path, video_path: Path, issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    obj = read_json(ann_path, issues)
    if obj is None:
        return None
    missing = sorted(REQUIRED_FIELDS - set(obj))
    if missing:
        add_issue(issues, "error", "missing_required_annotation_fields", ann_path, ",".join(missing))
    if obj.get("format") != "gooseomni_gameplay_aligned_per_video_v1":
        add_issue(issues, "error", "format_mismatch", ann_path, str(obj.get("format")))
    if obj.get("annotation_file") != ann_path.relative_to(root).as_posix():
        add_issue(issues, "error", "annotation_file_mismatch", ann_path, str(obj.get("annotation_file")))
    video_file = obj.get("video_file")
    if not isinstance(video_file, str) or root / video_file != video_path:
        add_issue(issues, "error", "video_file_mismatch", ann_path, str(video_file))
    if obj.get("player_id") != video_path.stem:
        add_issue(issues, "error", "player_id_mismatch", ann_path, str(obj.get("player_id")))
    if obj.get("phase_id") != video_path.parent.name:
        add_issue(issues, "error", "phase_id_mismatch", ann_path, str(obj.get("phase_id")))
    time_obj = obj.get("time") if isinstance(obj.get("time"), dict) else {}
    for key in ["aligned_start_sec", "aligned_end_sec", "duration_sec", "abs_sec_formula"]:
        if key not in time_obj:
            add_issue(issues, "error", "missing_time_field", ann_path, key)
    try:
        start = float(time_obj.get("aligned_start_sec"))
        end = float(time_obj.get("aligned_end_sec"))
        duration = float(time_obj.get("duration_sec"))
        if end <= start or abs((end - start) - duration) > 0.05:
            add_issue(issues, "error", "invalid_time_range", ann_path, f"start={start} end={end} duration={duration}")
        for top_key, expected in [("aligned_start_sec", start), ("aligned_end_sec", end), ("duration_sec", duration)]:
            if abs(float(obj.get(top_key, -10**9)) - expected) > 0.05:
                add_issue(issues, "error", "top_level_time_mismatch", ann_path, f"{top_key}={obj.get(top_key)} expected={expected}")
    except Exception as exc:  # noqa: BLE001
        add_issue(issues, "error", "invalid_time_values", ann_path, repr(exc))
    phase_type = obj.get("phase_type")
    if phase_type == "gameplay" and "gameplay_trace" not in obj:
        add_issue(issues, "error", "gameplay_missing_trace", ann_path, "gameplay phase lacks gameplay_trace field")
    if phase_type == "meeting" and "utterances" not in obj:
        add_issue(issues, "error", "meeting_missing_utterances", ann_path, "meeting phase lacks utterances field")
    if phase_type == "meeting" and isinstance(obj.get("utterances"), list) and not obj.get("utterances"):
        add_issue(issues, "warning", "meeting_empty_utterances", ann_path, "meeting annotation has no utterances")
    if phase_type == "gameplay" and isinstance(obj.get("gameplay_trace"), list) and not obj.get("gameplay_trace"):
        add_issue(issues, "warning", "gameplay_empty_trace", ann_path, "gameplay annotation has no gameplay_trace items")
    for collection in ["gameplay_trace", "utterances", "private_memory", "tom_questions"]:
        value = obj.get(collection)
        if value is not None and not isinstance(value, list):
            add_issue(issues, "error", "annotation_collection_not_list", ann_path, collection)
    if not isinstance(obj.get("belief_state"), dict):
        add_issue(issues, "error", "belief_state_not_object", ann_path, str(type(obj.get("belief_state"))))
    for event in obj.get("gameplay_trace", []) if isinstance(obj.get("gameplay_trace"), list) else []:
        if isinstance(event, dict):
            missing_source = sorted(SOURCE_TYPES - set(event))
            if missing_source:
                add_issue(issues, "warning", "gameplay_event_missing_source_type_fields", ann_path, ",".join(missing_source))
    for q in obj.get("tom_questions", []) if isinstance(obj.get("tom_questions"), list) else []:
        if isinstance(q, dict) and q.get("risk_of_perspective_leakage") in {"medium", "high"} and not q.get("needs_human_review", False):
            add_issue(issues, "error", "leakage_risk_not_reviewed", ann_path, str(q.get("risk_of_perspective_leakage")))
    return obj


def main() -> None:
    args = parse_args()
    root = args.release_root
    issues: list[dict[str, Any]] = []
    if not root.exists():
        add_issue(issues, "error", "missing_release_root", root, "release root does not exist")
    else:
        top = {p.name for p in root.iterdir()}
        extra = sorted(top - REQUIRED_TOP)
        missing_top = sorted(REQUIRED_TOP - top)
        if extra:
            add_issue(issues, "error", "unexpected_top_level_entries", root, ",".join(extra))
        if missing_top:
            add_issue(issues, "error", "missing_top_level_entries", root, ",".join(missing_top))

    videos_root = root / "videos" / args.game_id
    anns_root = root / "annotations" / args.game_id
    videos = sorted(videos_root.glob("*/*.mp4")) if videos_root.exists() else []
    anns = sorted(anns_root.glob("*/*.json")) if anns_root.exists() else []
    if not videos:
        add_issue(issues, "error", "no_videos", videos_root, "no mp4 files found")
    if not anns:
        add_issue(issues, "error", "no_annotations", anns_root, "no json files found")

    video_rel = {v.relative_to(root / "videos").with_suffix(".json"): v for v in videos}
    ann_rel = {a.relative_to(root / "annotations").with_suffix(".mp4"): a for a in anns}
    for rel, v in video_rel.items():
        ann_path = root / "annotations" / rel
        if not ann_path.exists():
            add_issue(issues, "error", "missing_matching_annotation", v, rel.as_posix())
    for rel, a in ann_rel.items():
        video_path = root / "videos" / rel
        if not video_path.exists():
            add_issue(issues, "error", "missing_matching_video", a, rel.as_posix())

    phase_to_videos: dict[str, list[Path]] = collections.defaultdict(list)
    phase_to_annotations: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for v in videos:
        phase_to_videos[v.parent.name].append(v)
        ann_path = root / "annotations" / v.relative_to(root / "videos").with_suffix(".json")
        if ann_path.exists():
            obj = validate_annotation(root, ann_path, v, issues)
            if obj is not None:
                phase_to_annotations[v.parent.name].append(obj)

    for phase_id, phase_videos in sorted(phase_to_videos.items()):
        if "seg_" in phase_id or not phase_id.startswith(f"{args.game_id}_phase_"):
            add_issue(issues, "error", "legacy_or_invalid_phase_id", Path(phase_id), "release phase_id must be semantic g001_phase_* rather than old segment id")
        players = sorted(v.stem for v in phase_videos)
        if players != sorted(PLAYERS):
            add_issue(issues, "error", "phase_not_strict_6pov", Path(phase_id), f"players={players}")
        durations = {v.stem: ffprobe_duration(v, issues) for v in phase_videos}
        numeric = [d for d in durations.values() if d is not None]
        if numeric and max(numeric) - min(numeric) > args.max_duration_delta_sec:
            add_issue(issues, "error", "phase_duration_delta_too_large", Path(phase_id), json.dumps(durations, ensure_ascii=False))
        phase_types = {a.get("phase_type") for a in phase_to_annotations.get(phase_id, [])}
        if len(phase_types) > 1:
            add_issue(issues, "error", "phase_type_not_consistent_across_pov", Path(phase_id), str(sorted(phase_types)))

    phase_times = []
    seen_phase = set()
    for objs in phase_to_annotations.values():
        if not objs:
            continue
        obj = objs[0]
        if obj.get("phase_id") in seen_phase:
            continue
        seen_phase.add(obj.get("phase_id"))
        t = obj.get("time") if isinstance(obj.get("time"), dict) else {}
        try:
            phase_times.append((float(t["aligned_start_sec"]), float(t["aligned_end_sec"]), obj.get("phase_id"), obj.get("phase_type")))
        except Exception:
            pass
    phase_times.sort()
    if phase_times:
        if abs(phase_times[0][0]) > 0.05:
            add_issue(issues, "error", "first_phase_not_game_start", None, f"first phase starts at {phase_times[0][0]:.3f}s")
        durations = [round(end - start, 3) for start, end, _, _ in phase_times]
        ninety_like = sum(1 for d in durations if abs(d - 90.0) <= 1.0)
        if len(durations) >= 5 and ninety_like / len(durations) > 0.8:
            add_issue(issues, "error", "release_looks_like_fixed_90s_windows", None, f"{ninety_like}/{len(durations)} phases are approximately 90s")
    for prev, cur in zip(phase_times, phase_times[1:]):
        gap = cur[0] - prev[1]
        if gap > 1.0:
            if cur[3] != "unknown_gap" and prev[3] != "unknown_gap":
                add_issue(issues, "error", "unexplained_phase_gap", None, f"{prev[2]} -> {cur[2]} gap={gap:.3f}s")
        if gap < -1.0:
            add_issue(issues, "error", "phase_overlap", None, f"{prev[2]} -> {cur[2]} overlap={-gap:.3f}s")

    counts = collections.Counter(issue["severity"] for issue in issues)
    summary = {
        "release_root": root.as_posix(),
        "video_count": len(videos),
        "annotation_count": len(anns),
        "phase_count": len(phase_to_videos),
        "issue_counts": dict(counts),
        "ok": counts.get("error", 0) == 0 and (not args.fail_on_warning or counts.get("warning", 0) == 0),
        "issues": issues,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "issues"}, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
