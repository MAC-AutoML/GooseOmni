from __future__ import annotations

from typing import Any

from ..schema import CheckerFinding, VALID_PLAYERS


def run_consistency_checker(global_annotation: dict[str, Any]) -> list[CheckerFinding]:
    findings: list[CheckerFinding] = []
    for item in global_annotation.get("global_events", []):
        source_pov = set(item.get("source_pov", []))
        invalid = sorted(source_pov - set(VALID_PLAYERS))
        findings.append(
            CheckerFinding(
                checker_name="consistency",
                annotation_id=item.get("global_event_id"),
                verdict="fail" if invalid else "pass",
                reason=f"invalid source_pov values: {invalid}" if invalid else "source_pov values are valid",
                suggested_fix="remove invalid source_pov values" if invalid else None,
                confidence=0.8,
            )
        )
    return findings

