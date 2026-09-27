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
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.benchmark.schema import (
    BeliefState,
    CandidateTrial,
    GlobalEventAnnotation,
    InformationState,
    MemoryState,
    POVEventAnnotation,
    PhaseEventAnnotation,
    Segment,
    UtteranceAnnotation,
    VALID_PLAYERS,
)


PLAYER_STAGE_SCHEMAS = {
    "pov_events": POVEventAnnotation,
    "utterances": UtteranceAnnotation,
    "information_states": InformationState,
    "memory_states": MemoryState,
    "belief_states": BeliefState,
}

SINGLE_STAGE_SCHEMAS = {
    "phase_events": PhaseEventAnnotation,
    "global_events": GlobalEventAnnotation,
}

CONDITIONS = {
    "single_pov_video",
    "single_pov_events",
    "multi_pov_global",
    "multi_pov_perspective",
    "text_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate complete Omni Goose GooseOmni benchmark outputs.")
    parser.add_argument("--dataset-root", default=Path("data/gooseomni"), type=Path)
    parser.add_argument("--annotation-root", default=Path("runs/gooseomni_oracle_pass1/annotations"), type=Path)
    parser.add_argument("--benchmark-root", default=Path("runs/gooseomni_oracle_pass1/benchmark"), type=Path)
    parser.add_argument("--output", default=Path("runs/gooseomni_oracle_pass1/completion_validation.json"), type=Path)
    parser.add_argument("--expected-segments", default=82, type=int)
    parser.add_argument("--expected-game-id", default="g001")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, path: Path | None, message: str) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "path": path.as_posix() if path else None,
            "message": message,
        }
    )


def validate_json_file(
    path: Path,
    schema: type,
    issues: list[dict[str, Any]],
    *,
    expected_segment_id: str | None = None,
    expected_player: str | None = None,
    expected_game_id: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        add_issue(issues, "error", "missing_file", path, "required file is missing")
        return None
    try:
        data = read_json(path)
    except Exception as exc:
        add_issue(issues, "error", "invalid_json", path, str(exc))
        return None
    try:
        schema.model_validate(data)
    except Exception as exc:
        add_issue(issues, "error", "schema_validation_failed", path, str(exc))
        return data
    if expected_segment_id is not None and data.get("segment_id") != expected_segment_id:
        add_issue(issues, "error", "segment_id_mismatch", path, f"expected {expected_segment_id}, got {data.get('segment_id')}")
    if expected_player is not None:
        player_key = "player_id" if "player_id" in data else "target_player"
        if data.get(player_key) != expected_player:
            add_issue(issues, "error", "player_mismatch", path, f"expected {expected_player}, got {data.get(player_key)}")
    if expected_game_id is not None and data.get("game_id") != expected_game_id:
        add_issue(issues, "error", "game_id_mismatch", path, f"expected {expected_game_id}, got {data.get('game_id')}")
    return data


def validate_segments(args: argparse.Namespace, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = args.dataset_root / "segments.jsonl"
    rows = read_jsonl(path)
    if len(rows) != args.expected_segments:
        add_issue(issues, "error", "segment_count_mismatch", path, f"expected {args.expected_segments}, got {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        try:
            Segment.model_validate(row)
        except Exception as exc:
            add_issue(issues, "error", "segment_schema_validation_failed", path, f"{row.get('segment_id')}: {exc}")
        segment_id = row.get("segment_id")
        if segment_id in seen:
            add_issue(issues, "error", "duplicate_segment_id", path, str(segment_id))
        seen.add(str(segment_id))
        if row.get("game_id") != args.expected_game_id:
            add_issue(issues, "error", "segment_game_id_mismatch", path, f"{segment_id}: {row.get('game_id')}")
        for pov in row.get("povs", []):
            video_path = args.dataset_root / pov.get("video_file", "")
            if not video_path.exists():
                add_issue(issues, "error", "missing_video", video_path, f"{segment_id} {pov.get('player_id')}")
    return rows


def validate_annotation_stage_outputs(
    args: argparse.Namespace,
    segments: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_counts: dict[str, int] = collections.Counter()
    segment_summaries = []
    for segment in segments:
        segment_id = segment["segment_id"]
        summary: dict[str, Any] = {"segment_id": segment_id}
        for stage, schema in PLAYER_STAGE_SCHEMAS.items():
            ok_players = []
            for player in VALID_PLAYERS:
                path = args.annotation_root / stage / args.expected_game_id / segment_id / f"{player}.json"
                data = validate_json_file(
                    path,
                    schema,
                    issues,
                    expected_segment_id=segment_id,
                    expected_player=player,
                    expected_game_id=args.expected_game_id,
                )
                if data is not None:
                    ok_players.append(player)
                    stage_counts[stage] += 1
            summary[stage] = ok_players
        for stage, schema in SINGLE_STAGE_SCHEMAS.items():
            path = args.annotation_root / stage / args.expected_game_id / f"{segment_id}.json"
            data = validate_json_file(
                path,
                schema,
                issues,
                expected_segment_id=segment_id,
                expected_game_id=args.expected_game_id,
            )
            summary[stage] = data is not None
            if data is not None:
                stage_counts[stage] += 1
        segment_summaries.append(summary)
    return {"stage_counts": dict(stage_counts), "segments": segment_summaries}


def validate_candidate_trials(args: argparse.Namespace, segments: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = args.annotation_root / "candidate_trials" / f"{args.expected_game_id}_candidate_trials.jsonl"
    rows = read_jsonl(path)
    segment_ids = {row["segment_id"] for row in segments}
    counts = collections.Counter()
    seen_trial_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        row_path = Path(f"{path.as_posix()}:{index}")
        try:
            CandidateTrial.model_validate(row)
        except Exception as exc:
            add_issue(issues, "error", "candidate_trial_schema_validation_failed", row_path, str(exc))
        segment_id = row.get("segment_id")
        counts[segment_id] += 1
        if segment_id not in segment_ids:
            add_issue(issues, "error", "candidate_trial_unknown_segment", row_path, str(segment_id))
        trial_key = (str(segment_id), str(row.get("trial_id")))
        if trial_key in seen_trial_keys:
            add_issue(
                issues,
                "error",
                "duplicate_candidate_trial_id",
                row_path,
                f"{trial_key[0]}::{trial_key[1]}",
            )
        seen_trial_keys.add(trial_key)
    for segment_id in segment_ids:
        if counts[segment_id] <= 0:
            add_issue(issues, "error", "missing_candidate_trials_for_segment", path, segment_id)
    return rows


def validate_benchmark(args: argparse.Namespace, candidate_rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    trials_path = args.benchmark_root / "weak" / "trials.jsonl"
    gold_path = args.benchmark_root / "weak" / "gold_weak.jsonl"
    metadata_path = args.benchmark_root / "weak" / "metadata.json"
    review_path = args.benchmark_root / "human_review_queue.jsonl"
    trials = read_jsonl(trials_path)
    gold = read_jsonl(gold_path)
    review = read_jsonl(review_path)
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    expected_trial_count = len(candidate_rows) * len(CONDITIONS)
    if len(trials) != expected_trial_count:
        add_issue(issues, "error", "benchmark_trial_count_mismatch", trials_path, f"expected {expected_trial_count}, got {len(trials)}")
    if len(gold) != len(trials):
        add_issue(issues, "error", "gold_count_mismatch", gold_path, f"expected {len(trials)}, got {len(gold)}")
    if metadata.get("candidate_count") != len(candidate_rows):
        add_issue(issues, "error", "metadata_candidate_count_mismatch", metadata_path, f"expected {len(candidate_rows)}, got {metadata.get('candidate_count')}")
    condition_counts = collections.Counter()
    trial_ids: set[str] = set()
    for index, row in enumerate(trials, start=1):
        row_path = Path(f"{trials_path.as_posix()}:{index}")
        trial_id = str(row.get("trial_id"))
        if trial_id in trial_ids:
            add_issue(issues, "error", "duplicate_benchmark_trial_id", row_path, trial_id)
        trial_ids.add(trial_id)
        condition = row.get("input_condition")
        condition_counts[condition] += 1
        if condition not in CONDITIONS:
            add_issue(issues, "error", "invalid_input_condition", row_path, str(condition))
        if condition == "single_pov_video":
            for video_file in row.get("input_video_files", []):
                video_path = args.dataset_root / video_file
                if not video_path.exists():
                    add_issue(issues, "error", "benchmark_input_video_missing", video_path, trial_id)
    for condition in CONDITIONS:
        if condition_counts[condition] != len(candidate_rows):
            add_issue(issues, "error", "condition_count_mismatch", trials_path, f"{condition}: expected {len(candidate_rows)}, got {condition_counts[condition]}")
    for report_name in ("dataset_card.md", "annotation_quality.md", "tom_benchmark_card.md"):
        report_path = args.benchmark_root / "reports" / report_name
        if not report_path.exists() or not report_path.read_text(encoding="utf-8").strip():
            add_issue(issues, "error", "missing_benchmark_report", report_path, "required benchmark report is missing or empty")
    return {
        "trial_count": len(trials),
        "gold_count": len(gold),
        "review_count": len(review),
        "condition_counts": dict(condition_counts),
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    issues: list[dict[str, Any]] = []
    segments = validate_segments(args, issues)
    annotation_summary = validate_annotation_stage_outputs(args, segments, issues)
    candidate_rows = validate_candidate_trials(args, segments, issues)
    benchmark_summary = validate_benchmark(args, candidate_rows, issues)
    severity_counts = collections.Counter(issue["severity"] for issue in issues)
    issue_code_counts = collections.Counter(issue["code"] for issue in issues)
    payload = {
        "complete": not any(issue["severity"] == "error" for issue in issues),
        "dataset_root": args.dataset_root.as_posix(),
        "annotation_root": args.annotation_root.as_posix(),
        "benchmark_root": args.benchmark_root.as_posix(),
        "expected_segments": args.expected_segments,
        "players": list(VALID_PLAYERS),
        "annotation_summary": annotation_summary,
        "candidate_trial_count": len(candidate_rows),
        "benchmark_summary": benchmark_summary,
        "issue_counts": dict(severity_counts),
        "issue_code_counts": dict(issue_code_counts),
        "issues": issues[:500],
        "truncated_issues": max(0, len(issues) - 500),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    lines = [
        "# Omni Goose Completion Validation",
        "",
        f"- complete: {payload['complete']}",
        f"- expected_segments: {args.expected_segments}",
        f"- candidate_trial_count: {len(candidate_rows)}",
        f"- benchmark_trial_count: {benchmark_summary['trial_count']}",
        f"- issue_counts: {json.dumps(dict(severity_counts), ensure_ascii=False, sort_keys=True)}",
        f"- issue_code_counts: {json.dumps(dict(issue_code_counts), ensure_ascii=False, sort_keys=True)}",
        "",
        "## First Issues",
        "",
    ]
    for issue in issues[:80]:
        lines.append(f"- [{issue['severity']}] {issue['code']}: {issue['path']} - {issue['message']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": payload["complete"], "issues": len(issues), "output": args.output.as_posix()}, ensure_ascii=False))
    if args.fail_on_incomplete and not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
