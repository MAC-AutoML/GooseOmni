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
import re
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
TOP_LEVEL = {"README.md", "inputs", "gold_annotations"}
METADATA_FORBIDDEN = {"answer", "role_and_goal", "belief_state", "tom_questions", "private_memory"}
GOLD_TOP_FORBIDDEN = {"evidence", "confidence", "time", "normalization", "raw_response"}
WORKFLOW_PATTERNS = [
    "raw_response",
    "normalization",
    "risk_of_perspective_leakage_original",
    "Qwen",
    "qwen",
    "phase_boundary_evidence",
]
RISK_VALUES = {"low", "medium", "high", "unknown"}
SOURCE_TYPES = {
    "direct_visual_observation",
    "speech_claim",
    "public_result",
    "inferred_belief",
    "hidden_or_not_visible_information",
}
VISIBILITY_VALUES = {"pov_visible", "public", "heard_speech", "inferred", "hidden"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Omni Goose benchmark release.")
    parser.add_argument("--release-root", default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"), type=Path)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def add(issues: list[dict[str, Any]], severity: str, code: str, path: Path | None, message: str) -> None:
    issues.append({"severity": severity, "code": code, "path": path.as_posix() if path else None, "message": message})


def load(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        add(issues, "error", "invalid_json", path, repr(exc))
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
    except Exception as exc:
        add(issues, "error", "ffprobe_failed", path, repr(exc))
        return None


def check_phase_fields(obj: dict[str, Any], path: Path, issues: list[dict[str, Any]]) -> None:
    required = {
        "episode_id",
        "episode_index",
        "phase_index_global",
        "phase_index_in_episode",
        "phase_order_label_zh",
        "gameplay_round_index",
        "meeting_round_index",
        "previous_phase_id",
        "next_phase_id",
    }
    missing = sorted(required - set(obj))
    if missing:
        add(issues, "error", "missing_phase_order_fields", path, ",".join(missing))
    phase_type = obj.get("phase_type")
    if phase_type == "gameplay" and obj.get("gameplay_round_index") is None:
        add(issues, "error", "gameplay_missing_round_index", path, str(obj.get("gameplay_round_index")))
    if phase_type == "meeting" and obj.get("meeting_round_index") is None:
        add(issues, "error", "meeting_missing_round_index", path, str(obj.get("meeting_round_index")))
    if phase_type == "final" and (obj.get("gameplay_round_index") is not None or obj.get("meeting_round_index") is not None):
        add(issues, "error", "final_has_round_index", path, f"{obj.get('gameplay_round_index')},{obj.get('meeting_round_index')}")


def check_times(gold: dict[str, Any], path: Path, issues: list[dict[str, Any]]) -> None:
    start = float(gold.get("aligned_start_sec", 0.0))
    duration = float(gold.get("duration_sec", 0.0))
    for collection in ["observations", "utterances"]:
        for idx, item in enumerate(gold.get(collection, []) if isinstance(gold.get(collection), list) else []):
            if not isinstance(item, dict):
                add(issues, "error", "non_object_event", path, f"{collection}[{idx}]")
                continue
            for key in ["local_start_sec", "local_end_sec", "abs_start_sec", "abs_end_sec"]:
                if key not in item:
                    add(issues, "error", "missing_event_time", path, f"{collection}[{idx}].{key}")
                    continue
            try:
                ls = float(item["local_start_sec"])
                le = float(item["local_end_sec"])
                abs_s = float(item["abs_start_sec"])
                abs_e = float(item["abs_end_sec"])
            except Exception as exc:
                add(issues, "error", "bad_event_time", path, f"{collection}[{idx}] {exc!r}")
                continue
            if ls < -0.05 or le < ls or le > duration + 0.05:
                add(issues, "error", "event_local_time_out_of_range", path, f"{collection}[{idx}] {ls}-{le} duration={duration}")
            if abs(start + ls - abs_s) > 0.06 or abs(start + le - abs_e) > 0.06:
                add(issues, "error", "event_abs_formula_mismatch", path, f"{collection}[{idx}]")


def check_gold(path: Path, gold: dict[str, Any], root: Path, issues: list[dict[str, Any]]) -> None:
    forbidden = sorted(GOLD_TOP_FORBIDDEN & set(gold))
    if forbidden:
        add(issues, "error", "gold_top_level_workflow_fields", path, ",".join(forbidden))
    check_phase_fields(gold, path, issues)
    if gold.get("annotation_file") != path.relative_to(root).as_posix():
        add(issues, "error", "annotation_file_mismatch", path, str(gold.get("annotation_file")))
    video = root / str(gold.get("video_file", ""))
    metadata = root / str(gold.get("input_metadata_file", ""))
    if not video.exists():
        add(issues, "error", "gold_video_missing", path, str(gold.get("video_file")))
    if not metadata.exists():
        add(issues, "error", "gold_metadata_missing", path, str(gold.get("input_metadata_file")))
    if gold.get("phase_type") == "gameplay" and not gold.get("observations"):
        add(issues, "error", "gameplay_empty_observations", path, "observations empty")
    if gold.get("phase_type") == "meeting" and not gold.get("utterances"):
        add(issues, "error", "meeting_empty_utterances", path, "utterances empty")
    if not gold.get("tom_questions"):
        add(issues, "error", "empty_tom_questions", path, "tom_questions empty")
    check_times(gold, path, issues)
    for idx, obs in enumerate(gold.get("observations", []) if isinstance(gold.get("observations"), list) else []):
        if not isinstance(obs, dict):
            continue
        types = obs.get("source_types")
        if not isinstance(types, list) or not types or any(item not in SOURCE_TYPES for item in types):
            add(issues, "error", "bad_observation_source_types", path, f"observations[{idx}] {types}")
        if obs.get("visibility") not in VISIBILITY_VALUES:
            add(issues, "error", "bad_observation_visibility", path, f"observations[{idx}] {obs.get('visibility')}")
        if any(key in obs for key in SOURCE_TYPES):
            add(issues, "error", "old_source_flag_field_in_observation", path, f"observations[{idx}]")
    for idx, q in enumerate(gold.get("tom_questions", []) if isinstance(gold.get("tom_questions"), list) else []):
        if not isinstance(q, dict):
            add(issues, "error", "non_object_tom_question", path, f"tom_questions[{idx}]")
            continue
        risk = q.get("risk_of_perspective_leakage")
        if risk not in RISK_VALUES:
            add(issues, "error", "bad_leakage_risk", path, f"tom_questions[{idx}] {risk}")
        if risk in {"medium", "high"} and q.get("needs_human_review") is not True:
            add(issues, "error", "leakage_not_reviewed", path, f"tom_questions[{idx}] {risk}")
        for key in ["requires_video_evidence", "required_source_types", "target_player_perspective", "answerable_from_input"]:
            if key not in q:
                add(issues, "error", "missing_tom_v2_field", path, f"tom_questions[{idx}].{key}")


def check_metadata(path: Path, metadata: dict[str, Any], root: Path, issues: list[dict[str, Any]]) -> None:
    check_phase_fields(metadata, path, issues)
    text = json.dumps(metadata, ensure_ascii=False)
    for word in METADATA_FORBIDDEN:
        if word in text:
            add(issues, "error", "metadata_contains_gold_field", path, word)
    if not (root / str(metadata.get("video_file", ""))).exists():
        add(issues, "error", "metadata_video_missing", path, str(metadata.get("video_file")))
    if sorted(metadata.get("roster", [])) != sorted(PLAYERS):
        add(issues, "error", "metadata_bad_roster", path, str(metadata.get("roster")))


def scan_workflow_terms(root: Path, issues: list[dict[str, Any]]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in WORKFLOW_PATTERNS:
            if re.search(re.escape(pattern), text):
                add(issues, "error", "workflow_term_present", path, pattern)


def main() -> None:
    args = parse_args()
    root = args.release_root
    issues: list[dict[str, Any]] = []
    if not root.exists():
        add(issues, "error", "missing_release_root", root, "release root missing")
    else:
        top = {path.name for path in root.iterdir()}
        if top != TOP_LEVEL:
            add(issues, "error", "bad_top_level", root, f"actual={sorted(top)} expected={sorted(TOP_LEVEL)}")

    videos = sorted((root / "inputs" / "videos" / args.game_id).glob("*/*.mp4"))
    metadata_paths = sorted((root / "inputs" / "metadata" / args.game_id).glob("*/*.json"))
    gold_paths = sorted((root / "gold_annotations" / args.game_id).glob("*/*.json"))
    if len(videos) != 168:
        add(issues, "error", "video_count_mismatch", root / "inputs" / "videos", str(len(videos)))
    if len(metadata_paths) != 168:
        add(issues, "error", "metadata_count_mismatch", root / "inputs" / "metadata", str(len(metadata_paths)))
    if len(gold_paths) != 168:
        add(issues, "error", "gold_count_mismatch", root / "gold_annotations", str(len(gold_paths)))

    video_rels = {p.relative_to(root / "inputs" / "videos").with_suffix(".json") for p in videos}
    metadata_rels = {p.relative_to(root / "inputs" / "metadata") for p in metadata_paths}
    gold_rels = {p.relative_to(root / "gold_annotations") for p in gold_paths}
    for rel in video_rels - metadata_rels:
        add(issues, "error", "missing_metadata_for_video", root / "inputs" / "videos" / rel.with_suffix(".mp4"), rel.as_posix())
    for rel in video_rels - gold_rels:
        add(issues, "error", "missing_gold_for_video", root / "inputs" / "videos" / rel.with_suffix(".mp4"), rel.as_posix())

    phase_to_videos: dict[str, list[Path]] = collections.defaultdict(list)
    for video in videos:
        phase_to_videos[video.parent.name].append(video)
    for phase_id, phase_videos in phase_to_videos.items():
        players = sorted(path.stem for path in phase_videos)
        if players != sorted(PLAYERS):
            add(issues, "error", "phase_not_6pov", Path(phase_id), str(players))
        durations = [ffprobe_duration(path, issues) for path in phase_videos]
        values = [value for value in durations if value is not None]
        if values and max(values) - min(values) > 1.0:
            add(issues, "error", "phase_duration_delta_too_large", Path(phase_id), str(values))

    for path in metadata_paths:
        obj = load(path, issues)
        if obj is not None:
            check_metadata(path, obj, root, issues)
    for path in gold_paths:
        obj = load(path, issues)
        if obj is not None:
            check_gold(path, obj, root, issues)

    labels = {}
    for path in metadata_paths:
        obj = load(path, issues)
        if obj and obj["player_id"] == "Gemini":
            labels[obj["phase_id"]] = obj.get("phase_order_label_zh")
    expected_labels = {
        "g001_phase_000_gameplay_000000_000080": "第1局第1次跑动过程",
        "g001_phase_001_meeting_000080_000390": "第1局第1次会议",
        "g001_phase_015_final_003530_003540": "第1局最终结果",
        "g001_phase_016_gameplay_003540_003610": "第2局第1次跑动过程",
    }
    for phase_id, expected in expected_labels.items():
        if labels.get(phase_id) != expected:
            add(issues, "error", "phase_label_mismatch", Path(phase_id), f"{labels.get(phase_id)} != {expected}")

    scan_workflow_terms(root, issues)
    counts = collections.Counter(issue["severity"] for issue in issues)
    summary = {
        "release_root": root.as_posix(),
        "video_count": len(videos),
        "metadata_count": len(metadata_paths),
        "gold_annotation_count": len(gold_paths),
        "phase_count": len(phase_to_videos),
        "issue_counts": dict(counts),
        "ok": counts.get("error", 0) == 0,
        "issues": issues,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "issues"}, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
