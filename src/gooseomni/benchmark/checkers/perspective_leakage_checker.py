from __future__ import annotations

from typing import Any

from ..schema import CheckerFinding


def run_perspective_leakage_checker(belief_or_trial: dict[str, Any]) -> list[CheckerFinding]:
    forbidden = belief_or_trial.get("forbidden_information") or belief_or_trial.get("hidden_information") or []
    text = " ".join(
        str(belief_or_trial.get(field, ""))
        for field in ("question", "answer", "evidence", "expected_answer_basis")
    )
    leaked = []
    for item in forbidden:
        source_id = item.get("hidden_event_id") or item.get("source_id")
        if source_id and source_id in text:
            leaked.append({"hidden_event_id": source_id, "used_in_field": "text_fields"})
    return [
        CheckerFinding(
            checker_name="perspective_leakage",
            annotation_id=belief_or_trial.get("trial_id") or belief_or_trial.get("target_player"),
            verdict="fail" if leaked else "pass",
            reason="forbidden information appears in answer/evidence" if leaked else "no direct forbidden id usage detected",
            suggested_fix="remove forbidden information from target-player reasoning" if leaked else None,
            confidence=0.7,
            leakage=bool(leaked),
            leaked_items=leaked,
        )
    ]

