from __future__ import annotations

from pathlib import Path
from typing import Any

from .pilot_common import cache_path
from .v2 import read_jsonl, write_jsonl


def _frame_packs(run_root: Path, rows: list[dict[str, Any]]) -> list[str]:
    evidence = {
        *rows[0].get("evidence_ids", []),
        *rows[0].get("review_only_hidden_evidence", []),
    }
    paths = []
    for evidence_id in evidence:
        clip_id = str(evidence_id).split(":visual", 1)[0]
        path = run_root / "local_codex/frame_packs" / f"{clip_id}.jpg"
        if path.is_file():
            paths.append(str(path))
    return sorted(set(paths))


def _review_queue(run_root: Path, groups: dict[str, list[dict[str, Any]]]) -> Path:
    queue = run_root / "local_codex/review_queue.jsonl"
    write_jsonl(
        queue,
        [
            {
                "probe_group_id": group_id,
                "candidate": rows[0],
                "variants": sorted(row["variant"] for row in rows),
                "available_evidence_ids": sorted(
                    {
                        *rows[0].get("evidence_ids", []),
                        *rows[0].get("review_only_hidden_evidence", []),
                    }
                ),
                "review_only_hidden_evidence": rows[0].get(
                    "review_only_hidden_evidence", []
                ),
                "frame_pack_paths": _frame_packs(run_root, rows),
                "required_schema": {
                    "decision": "accept|repair|reject",
                    "reason": "string",
                    "evidence_ids": "list[string]",
                    "final_value": "object|null",
                },
            }
            for group_id, rows in sorted(groups.items())
        ],
    )
    return queue


def codex_review_stage(context: Any) -> dict[str, Any]:
    candidates = read_jsonl(context.run_root / "candidates/tom_trials.jsonl")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row["probe_group_id"]), []).append(row)
    if not groups:
        write_jsonl(context.run_root / "review_decisions.jsonl", [])
        write_jsonl(context.run_root / "quarantine/rejected_trials.jsonl", [])
        return {"mode": "no_candidates", "accepted": 0, "quarantined": 0}
    cached = cache_path(context, "codex_decisions")
    if cached is not None:
        decisions = read_jsonl(cached)
        mode = "audited_cache"
    else:
        local_decisions = context.run_root / "local_codex/decisions.jsonl"
        if not local_decisions.is_file():
            queue = _review_queue(context.run_root, groups)
            raise RuntimeError(
                f"local Codex review is pending: {queue}; "
                "write strict decisions and resume"
            )
        decisions = read_jsonl(local_decisions)
        mode = "local_codex"
    decision_ids = [str(row.get("probe_group_id")) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Codex decisions contain duplicate probe group IDs")
    if set(decision_ids) != set(groups):
        raise ValueError(
            "Codex decisions must cover exactly the candidate probe groups"
        )
    from .stages import validate_judgement

    validated = []
    for row in decisions:
        group_id = str(row["probe_group_id"])
        representative = groups[group_id][0]
        candidate = {
            **representative,
            "available_evidence_ids": sorted(
                {
                    *representative.get("evidence_ids", []),
                    *representative.get("review_only_hidden_evidence", []),
                }
            ),
        }
        validated.append(
            {
                "probe_group_id": group_id,
                **validate_judgement(
                    candidate,
                    {
                        "decision": row.get("decision"),
                        "reason": row.get("reason"),
                        "evidence_ids": row.get("evidence_ids", []),
                        "final_value": row.get("final_value"),
                    },
                    context.config.judge_model,
                    str(row.get("raw_response", "")),
                ),
            }
        )
    accepted = [row for row in validated if row.get("decision") in {"accept", "repair"}]
    rejected = [row for row in validated if row not in accepted]
    write_jsonl(context.run_root / "review_decisions.jsonl", validated)
    write_jsonl(context.run_root / "quarantine/rejected_trials.jsonl", rejected)
    return {"mode": mode, "accepted": len(accepted), "quarantined": len(rejected)}
