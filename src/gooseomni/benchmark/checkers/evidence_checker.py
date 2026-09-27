from __future__ import annotations

from typing import Any

from ..schema import CheckerFinding


def run_evidence_checker(annotation: dict[str, Any]) -> list[CheckerFinding]:
    findings: list[CheckerFinding] = []
    items = annotation.get("events") or annotation.get("utterances") or annotation.get("global_events") or []
    for item in items:
        evidence = str(item.get("evidence", "")).strip()
        findings.append(
            CheckerFinding(
                checker_name="evidence",
                annotation_id=item.get("event_id") or item.get("utterance_id") or item.get("global_event_id"),
                verdict="supported" if evidence else "unsupported",
                reason="evidence present" if evidence else "missing evidence",
                suggested_fix=None if evidence else "add concrete visual/audio evidence",
                confidence=0.7 if evidence else 0.3,
            )
        )
    return findings

