# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent
    for _parent in _Path(__file__).resolve().parents
    if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.benchmark.io import load_segments_jsonl, write_json, write_jsonl
from gooseomni.benchmark.pipeline import (
    annotation_path,
    normalize_candidate_trial_payload,
    normalize_global_event_payload,
    normalize_phase_event_payload,
    normalize_utterance_payload,
    parse_json_array,
    parse_json_object,
    parse_partial_json_array_objects,
    parse_partial_json_object,
)
from gooseomni.benchmark.schema import (
    BeliefState,
    CandidateTrial,
    GlobalEvent,
    GlobalEventAnnotation,
    InformationState,
    MemoryState,
    PhaseEvent,
    PhaseEventAnnotation,
    Utterance,
    UtteranceAnnotation,
)
from tools.annotation.run_belief_state_annotation import _normalize_belief_payload
from tools.annotation.run_information_state import _normalize_information_payload
from tools.annotation.run_memory_state_update import _normalize_memory_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover usable annotations from saved Omni Goose error raw responses."
    )
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument(
        "--annotation-root",
        default=Path("runs/gooseomni_oracle_pass1/annotations"),
        type=Path,
    )
    parser.add_argument("--stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _segments_by_id(path: Path) -> dict[str, Any]:
    return {segment.segment_id: segment for segment in load_segments_jsonl(path)}


def _parse_array(raw: str, max_items: int = 5) -> list[dict[str, Any]]:
    try:
        return parse_json_array(raw)
    except Exception:
        return parse_partial_json_array_objects(raw, max_items=max_items)


def _parse_object(raw: str) -> dict[str, Any] | None:
    try:
        return parse_json_object(raw)
    except Exception:
        return parse_partial_json_object(raw)


def _append_candidate_trials(
    path: Path,
    trials: list[CandidateTrial],
    segment_id: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and not overwrite:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("segment_id") != segment_id:
                existing_rows.append(row)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [*existing_rows, *trials], append=False)


def _candidate_trials_exist(path: Path, segment_id: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("segment_id") == segment_id:
                return True
        except Exception:
            continue
    return False


def recover_one(
    error_path: Path, args: argparse.Namespace, segments: dict[str, Any]
) -> tuple[bool, str]:
    record = json.loads(error_path.read_text(encoding="utf-8"))
    stage = record.get("stage")
    segment_id = record.get("segment_id")
    raw = record.get("raw_response") or ""
    if not raw.strip():
        return False, "empty_raw_response"
    if args.stage and stage != args.stage:
        return False, "stage_filtered"
    segment = segments.get(segment_id)
    if segment is None:
        return False, "missing_segment"

    if stage == "phase_events":
        items = _parse_array(raw, max_items=5)
        events = [
            PhaseEvent.model_validate(
                normalize_phase_event_payload(item, segment, index)
            )
            for index, item in enumerate(items, start=1)
        ]
        if not events:
            return False, "no_recovered_items"
        output_path = annotation_path(
            args.dataset_root,
            "phase_events",
            segment,
            annotation_root=args.annotation_root,
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        annotation = PhaseEventAnnotation(
            game_id=segment.game_id,
            segment_id=segment.segment_id,
            aligned_start_sec=segment.aligned_start_sec,
            aligned_end_sec=segment.aligned_end_sec,
            phase_events=events,
            raw_response=raw,
        )
        if not args.dry_run:
            write_json(output_path, annotation)
        return True, output_path.as_posix()

    if stage == "utterances":
        player_id = record.get("player_id")
        pov = next((pov for pov in segment.povs if pov.player_id == player_id), None)
        if pov is None:
            return False, "missing_pov"
        items = _parse_array(raw, max_items=5)
        utterances = [
            Utterance.model_validate(
                normalize_utterance_payload(item, segment, pov, index)
            )
            for index, item in enumerate(items, start=1)
        ]
        if not utterances:
            return False, "no_recovered_items"
        output_path = annotation_path(
            args.dataset_root, "utterances", segment, player_id, args.annotation_root
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        annotation = UtteranceAnnotation(
            game_id=segment.game_id,
            segment_id=segment.segment_id,
            player_id=player_id,
            video_file=pov.video_file,
            aligned_start_sec=segment.aligned_start_sec,
            aligned_end_sec=segment.aligned_end_sec,
            utterances=utterances,
            raw_response=raw,
        )
        if not args.dry_run:
            write_json(output_path, annotation)
        return True, output_path.as_posix()

    if stage == "global_events":
        items = _parse_array(raw, max_items=3)
        events = [
            GlobalEvent.model_validate(
                normalize_global_event_payload(item, segment, index)
            )
            for index, item in enumerate(items, start=1)
        ]
        if not events:
            return False, "no_recovered_items"
        output_path = annotation_path(
            args.dataset_root,
            "global_events",
            segment,
            annotation_root=args.annotation_root,
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        annotation = GlobalEventAnnotation(
            game_id=segment.game_id,
            segment_id=segment.segment_id,
            aligned_start_sec=segment.aligned_start_sec,
            aligned_end_sec=segment.aligned_end_sec,
            global_events=events,
            raw_response=raw,
        )
        if not args.dry_run:
            write_json(output_path, annotation)
        return True, output_path.as_posix()

    if stage == "memory_states":
        target = record.get("target_player")
        payload = _parse_object(raw)
        if payload is None:
            return False, "no_recovered_object"
        state = MemoryState.model_validate(
            _normalize_memory_payload(payload, segment, target)
        )
        output_path = annotation_path(
            args.dataset_root, "memory_states", segment, target, args.annotation_root
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        if not args.dry_run:
            write_json(output_path, state)
        return True, output_path.as_posix()

    if stage == "information_states":
        target = record.get("target_player")
        payload = _parse_object(raw)
        if payload is None:
            return False, "no_recovered_object"
        state = InformationState.model_validate(
            _normalize_information_payload(payload, segment, target)
        )
        output_path = annotation_path(
            args.dataset_root,
            "information_states",
            segment,
            target,
            args.annotation_root,
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        if not args.dry_run:
            write_json(output_path, state)
        return True, output_path.as_posix()

    if stage == "belief_states":
        target = record.get("target_player")
        payload = _parse_object(raw)
        if payload is None:
            return False, "no_recovered_object"
        global_path = annotation_path(
            args.dataset_root,
            "global_events",
            segment,
            annotation_root=args.annotation_root,
        )
        global_payload = (
            json.loads(global_path.read_text(encoding="utf-8"))
            if global_path.exists()
            else {}
        )
        state = BeliefState.model_validate(
            _normalize_belief_payload(
                payload, segment, target, global_payload.get("global_events", [])
            )
        )
        output_path = annotation_path(
            args.dataset_root, "belief_states", segment, target, args.annotation_root
        )
        if output_path.exists() and not args.overwrite:
            return False, "output_exists"
        if not args.dry_run:
            write_json(output_path, state)
        return True, output_path.as_posix()

    if stage == "candidate_trials":
        output_path = (
            args.annotation_root / "candidate_trials" / "g001_candidate_trials.jsonl"
        )
        if (
            output_path.exists()
            and not args.overwrite
            and _candidate_trials_exist(output_path, segment.segment_id)
        ):
            return False, "output_exists"
        items = _parse_array(raw, max_items=5)
        trials = [
            CandidateTrial.model_validate(
                normalize_candidate_trial_payload(item, segment, index)
            )
            for index, item in enumerate(items, start=1)
        ]
        if not trials:
            return False, "no_recovered_items"
        _append_candidate_trials(
            output_path, trials, segment.segment_id, args.overwrite, args.dry_run
        )
        return True, output_path.as_posix()

    return False, f"unsupported_stage:{stage}"


def main() -> None:
    args = parse_args()
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    segments = _segments_by_id(segments_path)
    stats: dict[str, int] = {"recovered": 0, "skipped": 0, "error": 0}
    for error_path in sorted((args.annotation_root / "errors").rglob("*.json")):
        try:
            ok, reason = recover_one(error_path, args, segments)
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"error {error_path}: {exc}")
            continue
        if ok:
            stats["recovered"] += 1
            print(f"recovered {error_path} -> {reason}")
        else:
            stats["skipped"] += 1
            print(f"skip {error_path}: {reason}")
    print(stats)


if __name__ == "__main__":
    main()
