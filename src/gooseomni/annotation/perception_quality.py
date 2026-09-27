from __future__ import annotations

import re
from collections import Counter
from typing import Any

VISUAL_EVENT_TYPES = {
    "movement",
    "encounter",
    "task",
    "task_ui",
    "interaction",
    "meeting",
    "vote",
    "kill",
    "body_report",
    "action_outcome",
}
AUDIO_EVENT_TYPES = {
    "utterance",
    "claim",
    "reaction",
    "accusation",
    "defense",
    "audio_cue",
}
INTERACTIVE_UI_MARKERS = {"点击", "小游戏", "连线", "交互界面", "操作界面"}
PASSIVE_TASK_MARKERS = {
    "任务列表",
    "任务栏",
    "任务面板",
    "任务目标",
    "左侧任务",
    "角落任务",
    "常驻",
    "sidebar",
    "task list",
    "objective list",
}
RESULT_SCREEN_MARKERS = {"结算界面", "经验奖励", "获胜者奖励", "总经验", "胜利界面"}
MAX_TASK_UI_DURATION_SEC = 20.0
ABS_TIMESTAMP_PATTERN = re.compile(r"ABS\s+(-?\d+(?:\.\d+)?)s", re.IGNORECASE)


def normalize_pass_event(
    row: dict[str, Any], pass_kind: str, speaker_confidence_min: float = 0.85
) -> dict[str, Any]:
    normalized = dict(row)
    searchable = " ".join(
        str(normalized.get(key, "")) for key in ("description", "evidence", "location")
    ).lower()
    if pass_kind == "visual" and normalized.get("event_type") == "task":
        if any(marker in searchable for marker in INTERACTIVE_UI_MARKERS):
            normalized["event_type"] = "task_ui"
            normalized["movement_transition"] = None
    if pass_kind == "visual" and normalized.get("event_type") == "task_ui":
        if any(marker in searchable for marker in RESULT_SCREEN_MARKERS):
            normalized["event_type"] = "action_outcome"
            normalized["movement_transition"] = None
    if pass_kind == "audio":
        confidence = float(
            normalized.get("speaker_confidence", normalized.get("confidence", 0.0))
            or 0.0
        )
        if confidence < speaker_confidence_min:
            normalized["speaker_id"] = None
            normalized["utterance"] = None
    return normalized


def validate_pass_events(
    rows: list[dict[str, Any]],
    pass_kind: str,
    canonical_players: set[str] | None = None,
) -> list[str]:
    if pass_kind == "combined":
        return []
    allowed = VISUAL_EVENT_TYPES if pass_kind == "visual" else AUDIO_EVENT_TYPES
    issues: list[str] = []
    forbidden = sorted({str(row.get("event_type")) for row in rows} - allowed)
    if forbidden:
        issues.append(f"{pass_kind} pass contains forbidden event types: {forbidden}")
    descriptions = [
        " ".join(str(row.get("description", "")).lower().split()) for row in rows
    ]
    if len(rows) >= 4 and len(set(descriptions)) / len(rows) < 0.6:
        issues.append("perception output repeats templated descriptions")
    if descriptions and max(Counter(descriptions).values()) > 2:
        issues.append("one perception description is repeated more than twice")
    if len(rows) >= 3:
        starts = [round(float(row.get("start_sec", 0.0)), 3) for row in rows]
        durations = [
            round(float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0)), 3)
            for row in rows
        ]
        gaps = [
            round(right - left, 3)
            for left, right in zip(starts, starts[1:], strict=False)
        ]
        if len(set(durations)) == 1 and gaps and len(set(gaps)) == 1:
            issues.append("perception output uses an artificial uniform time grid")
    if pass_kind == "visual":
        for row in rows:
            values = [
                float(value)
                for value in ABS_TIMESTAMP_PATTERN.findall(
                    str(row.get("timestamp_evidence", ""))
                )
            ]
            if len(values) < 2:
                issues.append("visual event lacks two transcribed ABS timestamps")
                break
            if (
                abs(values[0] - float(row.get("start_sec", 0.0))) > 0.75
                or abs(values[-1] - float(row.get("end_sec", 0.0))) > 0.75
            ):
                issues.append(
                    "visual event times disagree with transcribed ABS timestamps"
                )
                break
        task_ui_rows = [row for row in rows if row.get("event_type") == "task_ui"]
        movement_rows = [row for row in rows if row.get("event_type") == "movement"]
        for row in task_ui_rows:
            searchable = " ".join(
                str(row.get(key, "")) for key in ("description", "evidence", "location")
            ).lower()
            duration = float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0))
            if any(marker in searchable for marker in PASSIVE_TASK_MARKERS):
                issues.append("passive task list/sidebar is not interactive task_ui")
                break
            if duration > MAX_TASK_UI_DURATION_SEC:
                issues.append("task_ui duration exceeds the interactive UI limit")
                break
        for task_ui in task_ui_rows:
            ui_start = float(task_ui.get("start_sec", 0.0))
            ui_end = float(task_ui.get("end_sec", 0.0))
            ui_duration = max(ui_end - ui_start, 0.0)
            if ui_duration <= 0:
                continue
            for movement in movement_rows:
                overlap = max(
                    0.0,
                    min(ui_end, float(movement.get("end_sec", 0.0)))
                    - max(ui_start, float(movement.get("start_sec", 0.0))),
                )
                if overlap / ui_duration > 0.5:
                    issues.append("task_ui overlaps movement for most of its duration")
                    break
            if issues and issues[-1].startswith("task_ui overlaps"):
                break
    if pass_kind == "audio":
        for row in rows:
            if row.get("phase_type") == "gameplay" and row.get("meeting_public"):
                issues.append("gameplay audio event cannot be meeting_public")
                break
            heard_by = row.get("heard_by", [])
            if heard_by and not row.get("meeting_public") and len(set(heard_by)) >= 6:
                issues.append(
                    "non-public audio event broadcasts heard_by to all players"
                )
                break
            if canonical_players is not None:
                speaker = row.get("speaker_id")
                if speaker is not None and str(speaker) not in canonical_players:
                    issues.append("audio event contains non-canonical speaker")
                    break
                if {str(player) for player in heard_by} - canonical_players:
                    issues.append("audio event contains non-canonical heard_by player")
                    break
    return issues
