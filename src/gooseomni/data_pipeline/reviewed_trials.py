from __future__ import annotations

from typing import Any

IMMUTABLE_TRIAL_FIELDS = {
    "trial_id",
    "probe_group_id",
    "variant",
    "episode_id",
    "cutoff_abs_sec",
    "subject_player",
    "target_player",
    "tom_layer",
    "split",
    "evidence_ids",
}


def accepted_trials(
    candidates: dict[str, dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply Codex repairs while preserving trial identity and evidence scope."""
    accepted: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("decision") not in {"accept", "repair"}:
            continue
        if decision.get("probe_group_id") is not None:
            selected = [
                row
                for row in candidates.values()
                if row.get("probe_group_id") == decision["probe_group_id"]
            ]
        else:
            selected = [candidates[str(decision["trial_id"])]]
        for candidate in selected:
            repaired = decision.get("final_value")
            row = (
                {**candidate, **repaired}
                if isinstance(repaired, dict)
                else dict(candidate)
            )
            for field in IMMUTABLE_TRIAL_FIELDS:
                if field in candidate:
                    row[field] = candidate[field]
            row["gold_source"] = "model_verified"
            accepted.append(row)
    return accepted
