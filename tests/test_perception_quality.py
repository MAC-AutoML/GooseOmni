from __future__ import annotations

from gooseomni.annotation.perception_quality import (
    normalize_pass_event,
    validate_pass_events,
)


def test_visual_gate_rejects_audio_and_uniform_templates() -> None:
    rows = [
        {
            "event_type": "audio_cue" if index == 0 else "movement",
            "description": "same template",
            "start_sec": index * 5,
            "end_sec": index * 5 + 5,
        }
        for index in range(4)
    ]
    issues = validate_pass_events(rows, "visual")
    assert any("forbidden event" in issue for issue in issues)
    assert "perception output repeats templated descriptions" in issues
    assert "perception output uses an artificial uniform time grid" in issues


def test_visual_gate_rejects_three_event_uniform_grid() -> None:
    rows = [
        {
            "event_type": event_type,
            "description": event_type,
            "start_sec": index * 5,
            "end_sec": index * 5 + 5,
        }
        for index, event_type in enumerate(("meeting", "vote", "action_outcome"))
    ]
    assert "perception output uses an artificial uniform time grid" in (
        validate_pass_events(rows, "visual")
    )


def test_visual_task_ui_is_not_movement_or_plain_task() -> None:
    row = normalize_pass_event(
        {
            "event_type": "task",
            "description": "点击任务界面进度条",
            "movement_transition": "street -> park",
        },
        "visual",
    )
    assert row["event_type"] == "task_ui"
    assert row["movement_transition"] is None


def test_visual_gate_rejects_passive_task_sidebar() -> None:
    issues = validate_pass_events(
        [
            {
                "event_type": "task_ui",
                "description": "左侧常驻任务列表和任务进度可见",
                "start_sec": 10,
                "end_sec": 18,
            }
        ],
        "visual",
    )
    assert "passive task list/sidebar is not interactive task_ui" in issues


def test_visual_gate_rejects_long_or_overlapping_task_ui() -> None:
    rows = [
        {
            "event_type": "task_ui",
            "description": "玩家操作占据主要画面的连线小游戏",
            "start_sec": 0,
            "end_sec": 25,
        },
        {
            "event_type": "movement",
            "description": "角色从广场移动到街道",
            "start_sec": 5,
            "end_sec": 24,
        },
    ]
    issues = validate_pass_events(rows, "visual")
    assert "task_ui duration exceeds the interactive UI limit" in issues
    assert "task_ui overlaps movement for most of its duration" in issues


def test_plain_progress_bar_does_not_promote_task_to_task_ui() -> None:
    row = normalize_pass_event(
        {
            "event_type": "task",
            "description": "左侧任务进度条常驻",
        },
        "visual",
    )
    assert row["event_type"] == "task"


def test_result_screen_is_normalized_to_action_outcome() -> None:
    row = normalize_pass_event(
        {
            "event_type": "task_ui",
            "description": "游戏结算界面显示基础经验和获胜者奖励",
        },
        "visual",
    )
    assert row["event_type"] == "action_outcome"


def test_visual_gate_requires_matching_transcribed_abs_timestamps() -> None:
    valid = {
        "event_type": "vote",
        "description": "投票",
        "start_sec": 10.0,
        "end_sec": 12.0,
        "timestamp_evidence": "first=ABS 10.00s; last=ABS 12.00s",
    }
    assert validate_pass_events([valid], "visual") == []
    missing = validate_pass_events([{**valid, "timestamp_evidence": ""}], "visual")
    assert "visual event lacks two transcribed ABS timestamps" in missing
    mismatch = validate_pass_events(
        [{**valid, "timestamp_evidence": "first=ABS 8.00s; last=ABS 9.00s"}],
        "visual",
    )
    assert "visual event times disagree with transcribed ABS timestamps" in mismatch


def test_audio_gate_drops_uncertain_speaker_and_rejects_broadcast() -> None:
    row = normalize_pass_event(
        {
            "event_type": "claim",
            "speaker_id": "a",
            "speaker_confidence": 0.7,
            "utterance": "text",
        },
        "audio",
    )
    assert row["speaker_id"] is None
    assert row["utterance"] is None
    issues = validate_pass_events(
        [
            {
                "event_type": "claim",
                "description": "claim",
                "heard_by": ["a", "b", "c", "d", "e", "f"],
                "meeting_public": False,
            }
        ],
        "audio",
    )
    assert "non-public audio event broadcasts heard_by to all players" in issues


def test_audio_gate_enforces_phase_and_canonical_players() -> None:
    issues = validate_pass_events(
        [
            {
                "event_type": "claim",
                "phase_type": "gameplay",
                "meeting_public": True,
                "speaker_id": "invented",
                "heard_by": ["a", "invented"],
            }
        ],
        "audio",
        {"a", "b"},
    )
    assert "gameplay audio event cannot be meeting_public" in issues
