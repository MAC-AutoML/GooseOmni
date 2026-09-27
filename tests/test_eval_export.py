from __future__ import annotations

import json
from pathlib import Path

from gooseomni.annotation.eval_export import (
    EventAlignedExportConfig,
    _ffmpeg_cut_command,
    _aligned_window,
    export_event_aligned_omni_eval,
)


def test_ffmpeg_cut_command_reencodes_by_default(tmp_path: Path) -> None:
    command = _ffmpeg_cut_command(
        source_video=tmp_path / "input.mp4",
        output_video=tmp_path / "output.mp4",
        raw_start_sec=12.3456,
        duration_sec=3.2,
        overwrite=True,
        reencode=True,
    )

    assert command[:7] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "12.346",
    ]
    assert "-c:v" in command
    assert "mpeg4" in command
    assert "-q:v" in command
    assert command[-1].endswith("output.mp4")


def test_aligned_window_clamps_long_events_to_max_duration() -> None:
    start, end = _aligned_window(
        [{"start_sec": 0, "end_sec": 300}],
        pre_context_sec=8,
        post_context_sec=4,
        max_duration_sec=120,
    )

    assert start == 180
    assert end == 300


def test_export_event_aligned_omni_eval_writes_level2_shape(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_video = raw_dir / "g001" / "P1.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"fake")

    trials = tmp_path / "trials.json"
    trials.write_text(
        json.dumps(
            [
                {
                    "game_id": "g001",
                    "trial_id": "trial_000001",
                    "question_type": "first_order_belief",
                    "target_player_id": "P1",
                    "cutoff_time": 12,
                    "question": "P1 看到了什么？",
                    "answer": "P1 看到了 P2。",
                    "supporting_global_event_ids": ["ge_000001"],
                    "confidence": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    global_events = tmp_path / "global_events.json"
    global_events.write_text(
        json.dumps(
            [
                {
                    "global_event_id": "ge_000001",
                    "game_id": "g001",
                    "start_sec": 10,
                    "end_sec": 12,
                    "event_type": "encounter",
                    "description": "P1 看到了 P2。",
                    "source_player_ids": ["P1"],
                    "source_clip_ids": ["g001_P1_0_90"],
                }
            ]
        ),
        encoding="utf-8",
    )
    offsets = tmp_path / "sync_offsets.json"
    offsets.write_text(
        json.dumps(
            {
                "offsets": [
                    {
                        "game_id": "g001",
                        "player_id": "P1",
                        "raw_start_sec": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("gooseomni.annotation.eval_export.cut_video", lambda **kwargs: None)

    stats = export_event_aligned_omni_eval(
        EventAlignedExportConfig(
            trials_path=trials,
            global_events_path=global_events,
            sync_offsets_path=offsets,
            raw_dir=raw_dir,
            output_dir=tmp_path / "level2",
            pre_context_sec=2,
            post_context_sec=3,
            dry_run=False,
        )
    )

    annotations = json.loads((tmp_path / "level2" / "annotations.json").read_text())
    samples = (tmp_path / "level2" / "samples.jsonl").read_text(encoding="utf-8").splitlines()

    assert stats == {"samples": 1, "skipped": 0}
    assert len(samples) == 1
    assert annotations[0]["video_file"] == (
        "g001/P1/g001_trial_000001_first_order_belief_P1_ge_000001.mp4"
    )
    assert annotations[0]["question_1"]["timestamp"] == "4.000"
    assert annotations[0]["question_2"]["answer"] == "P1 看到了 P2。"
    metadata = annotations[0]["metadata"]
    assert metadata["crop_primary_global_event_id"] == "ge_000001"
    assert metadata["sync_raw_start_sec"] == 100.0
    assert metadata["aligned_window_start_sec"] == 8.0
    assert metadata["raw_window_start_sec"] == 108.0
    assert metadata["supporting_events"][0]["clip_start_sec"] == 2.0
    assert metadata["supporting_events"][0]["included_in_clip"] is True
