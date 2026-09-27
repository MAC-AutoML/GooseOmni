from __future__ import annotations

from typing import Any

from ..schema import CheckerFinding


def run_claim_fact_checker(annotation: dict[str, Any]) -> list[CheckerFinding]:
    findings: list[CheckerFinding] = []
    for item in annotation.get("events", []):
        confused = bool(item.get("is_speech_claim")) and bool(item.get("is_direct_observation"))
        findings.append(
            CheckerFinding(
                checker_name="claim_fact",
                annotation_id=item.get("event_id"),
                verdict="fail" if confused else "pass",
                reason="speech claim is also marked as direct observation" if confused else "claim/fact flags are consistent",
                suggested_fix="set only one of is_speech_claim/is_direct_observation unless explicitly justified" if confused else None,
                confidence=0.8,
            )
        )
    return findings

