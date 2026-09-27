from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gooseomni.benchmark.checkers import run_perspective_leakage_checker
from gooseomni.benchmark.backends import MockBackend
from gooseomni.benchmark.io import load_segments_jsonl, resolve_video_path
from gooseomni.benchmark.pipeline import (
    abs_time,
    normalize_candidate_trial_payload,
    normalize_cutoff_payload,
    normalize_global_event_payload,
    normalize_utterance_payload,
    parse_partial_json_array_objects,
    save_error,
)
from gooseomni.benchmark.schema import CandidateTrial, GlobalEvent, MemoryState, POVEvent, Segment, Utterance


def _segment_row() -> dict:
    players = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
    return {
        "segment_id": "g001_seg_0001_000080_000170",
        "game_id": "g001",
        "aligned_start_sec": 80.0,
        "aligned_end_sec": 170.0,
        "duration_sec": 90.0,
        "pov_count": 6,
        "povs": [
            {
                "player_id": player,
                "video_file": f"videos/g001/g001_seg_0001_000080_000170/{player}.mp4",
                "source_raw_video": f"data/raw/g001/{player}.mp4",
                "aligned_start_sec": 80.0,
                "aligned_end_sec": 170.0,
            }
            for player in players
        ],
    }


def _write_segments_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "gooseomni" / "segments.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_segment_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_load_segments_jsonl_reads_strict_segment(tmp_path: Path) -> None:
    path = _write_segments_jsonl(tmp_path)

    segments = load_segments_jsonl(path)

    assert len(segments) == 1
    assert segments[0].segment_id == "g001_seg_0001_000080_000170"
    assert len(segments[0].povs) == 6


def test_resolve_video_path_uses_dataset_root(tmp_path: Path) -> None:
    path = resolve_video_path(tmp_path / "data" / "gooseomni", "videos/g001/s/Gemini.mp4")

    assert path == tmp_path / "data" / "gooseomni" / "videos" / "g001" / "s" / "Gemini.mp4"


def test_local_to_abs_conversion() -> None:
    segment = Segment.model_validate(_segment_row())

    assert abs_time(segment, 30.0) == 110.0


def test_mock_backend_payload_can_validate_schema() -> None:
    segment = Segment.model_validate(_segment_row())
    raw = MockBackend().annotate_video(Path("Gemini.mp4"), "TASK: pov_event")
    item = json.loads(raw)[0]
    item.update(
        {
            "game_id": segment.game_id,
            "segment_id": segment.segment_id,
            "player_id": "Gemini",
            "abs_start_sec": abs_time(segment, item["local_start_sec"]),
            "abs_end_sec": abs_time(segment, item["local_end_sec"]),
            "source_pov": ["Gemini"],
        }
    )

    event = POVEvent.model_validate(item)

    assert event.dataset == "gooseomni"
    assert event.player_id == "Gemini"


def test_error_output_can_be_saved(tmp_path: Path) -> None:
    segment = Segment.model_validate(_segment_row())
    path = save_error(
        dataset_root=tmp_path / "data" / "gooseomni",
        annotation_root=tmp_path / "annotations",
        stage="pov_events",
        segment=segment,
        prompt="prompt",
        raw_response="{bad",
        error=ValueError("bad json"),
        pov=segment.povs[0],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "pov_events"
    assert payload["segment_id"] == segment.segment_id
    assert payload["player_id"] == "Gemini"


def test_resume_mode_does_not_reannotate(tmp_path: Path) -> None:
    segments = _write_segments_jsonl(tmp_path)
    output_root = tmp_path / "annotations"
    script = Path(__file__).resolve().parents[1] / "tools" / "annotation" / "run_pov_event_annotation.py"
    base_cmd = [
        ".venv/bin/python",
        str(script),
        "--dataset-root",
        str(segments.parent),
        "--segments-jsonl",
        str(segments),
        "--output-root",
        str(output_root),
        "--backend",
        "mock",
        "--segment-id",
        "g001_seg_0001_000080_000170",
        "--player-id",
        "Gemini",
    ]

    subprocess.run(base_cmd, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    result = subprocess.run(
        base_cmd + ["--resume"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "'skipped': 1" in result.stdout


def test_player_id_must_be_valid() -> None:
    row = _segment_row()
    row["povs"][0]["player_id"] = "bad_player"

    with pytest.raises(ValueError, match="6 valid players|player_id"):
        Segment.model_validate(row)



def test_utterance_normalization_recovers_list_time_referred() -> None:
    segment = Segment.model_validate(_segment_row())
    payload = normalize_utterance_payload(
        {
            "local_start_sec": 1.0,
            "local_end_sec": 2.0,
            "speaker": "unknown",
            "text": "为什么不说话？",
            "claims": [
                {
                    "content": "有人不说话。",
                    "certainty": 0.7,
                    "time_referred": [],
                }
            ],
            "certainty": 0.7,
            "evidence": "audio",
        },
        segment,
        segment.povs[0],
        1,
    )

    utterance = Utterance.model_validate(payload)

    assert utterance.claims[0].time_referred == "unknown"


def test_memory_state_marks_items_without_source_for_review() -> None:
    segment = Segment.model_validate(_segment_row())
    payload = normalize_cutoff_payload(
        {
            "memory_items": [
                {
                    "memory_id": "mem_001",
                    "first_observed_abs_sec": 80.0,
                    "last_referenced_abs_sec": 170.0,
                    "memory_type": "inferred",
                    "content": "unsupported memory",
                    "source_event_ids": [],
                    "source_claim_ids": [],
                    "confidence": 0.4,
                    "decay_status": "active",
                    "visibility": "private",
                }
            ],
            "memory_delta": [],
        },
        segment,
        "Gemini",
    )

    state = MemoryState.model_validate(payload)

    assert state.memory_items[0].needs_human_review is True


def test_perspective_leakage_checker_flags_forbidden_id_usage() -> None:
    findings = run_perspective_leakage_checker(
        {
            "target_player": "saoyi",
            "evidence": "uses ge_051",
            "forbidden_information": [{"hidden_event_id": "ge_051", "reason": "hidden"}],
        }
    )

    assert findings[0].leakage is True
    assert findings[0].verdict == "fail"


def test_benchmark_export_and_scoring(tmp_path: Path) -> None:
    ann_root = tmp_path / "annotations"
    candidate_dir = ann_root / "candidate_trials"
    candidate_dir.mkdir(parents=True)
    candidate_dir.joinpath("g001_candidate_trials.jsonl").write_text(
        json.dumps(
            {
                "dataset": "gooseomni",
                "game_id": "g001",
                "segment_id": "g001_seg_0001_000080_000170",
                "trial_id": "trial_001",
                "question_type": "first_order_belief",
                "trial_type": "belief_state",
                "target_player": "Gemini",
                "cutoff_abs_sec": 170.0,
                "question": "Q?",
                "answer": "A",
                "available_information": ["visible"],
                "hidden_information": ["hidden"],
                "expected_answer_basis": "visible only",
                "risk_of_perspective_leakage": "medium",
                "certainty": 0.8,
                "evidence": "e",
                "source_pov": ["Gemini"],
                "needs_human_review": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bench = tmp_path / "benchmark"
    repo = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            ".venv/bin/python",
            "tools/package/export_tom_benchmark.py",
            "--dataset-root",
            "data/gooseomni",
            "--annotation-root",
            str(ann_root),
            "--benchmark-root",
            str(bench),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            ".venv/bin/python",
            "tools/eval/run_tom_eval.py",
            "--trials",
            str(bench / "weak" / "trials.jsonl"),
            "--output",
            str(bench / "weak" / "predictions_mock.jsonl"),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            ".venv/bin/python",
            "tools/eval/score_tom_eval.py",
            "--trials",
            str(bench / "weak" / "trials.jsonl"),
            "--predictions",
            str(bench / "weak" / "predictions_mock.jsonl"),
            "--output",
            str(bench / "reports" / "eval_scores.json"),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    trials = (bench / "weak" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    scores = json.loads((bench / "reports" / "eval_scores.json").read_text(encoding="utf-8"))
    assert len(trials) == 5
    assert scores["json_parse_success"] == 1.0
    assert scores["perspective_leakage_rate"] == 0.0


def test_candidate_trial_normalizer_fills_required_fields() -> None:
    segment = Segment.model_validate(_segment_row())

    payload = normalize_candidate_trial_payload(
        {
            "question_type": "belief_state",
            "question": "Gemini 知道什么？",
            "answer": "只知道自己看到的信息",
        },
        segment,
        1,
    )
    trial = CandidateTrial.model_validate(payload)

    assert trial.certainty == 0.5
    assert trial.evidence
    assert trial.source_pov
    assert trial.needs_human_review is True


def test_global_event_normalizer_limits_supporting_ids() -> None:
    segment = Segment.model_validate(_segment_row())

    payload = normalize_global_event_payload(
        {
            "local_start_sec": 0,
            "local_end_sec": 90,
            "event_type": "discussion",
            "description": "x" * 300,
            "actors": ["Gemini"],
            "visible_to": ["Gemini"],
            "heard_by": ["Gemini"],
            "supporting_pov_event_ids": [f"event_{idx:03d}" for idx in range(100)],
            "supporting_utterance_ids": [f"utt_{idx:03d}" for idx in range(100)],
            "certainty": "high",
            "evidence": "e" * 300,
            "source_pov": ["Gemini"],
        },
        segment,
        1,
    )
    event = GlobalEvent.model_validate(payload)

    assert len(event.supporting_pov_event_ids) == 6
    assert len(event.supporting_utterance_ids) == 6
    assert event.certainty == 0.5
    assert len(event.description) == 160


def test_partial_json_recovery_handles_bad_certainty_string() -> None:
    raw = '[{"local_start_sec":0,"local_end_sec":1,"certainty":"high","evidence":"e"}'

    rows = parse_partial_json_array_objects(raw)

    assert rows[0]["certainty"] == 0.5
    assert rows[0]["needs_human_review"] is True
