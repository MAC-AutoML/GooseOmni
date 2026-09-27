from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def group_semantic_key(
    row: dict[str, Any], include_template: bool = True
) -> tuple[Any, ...]:
    query_variable = row.get("query_variable") or {}
    parts: list[Any] = [
        row.get("target_player"),
        tuple(row.get("anchor_event_ids", [])),
        tuple(row.get("related_claim_ids", [])),
        query_variable.get("type"),
    ]
    if include_template:
        parts.append(row.get("template"))
    return tuple(parts)


def review_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("review_record_id")), str(record.get("probe_group_id")))


def load_accept_records(work_root: Path) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(work_root.glob("codex_human_review*.jsonl")):
        for row in read_jsonl(path):
            if row.get("gate_decision") != "accept_human_verified":
                continue
            row = dict(row)
            row["_source_review_file"] = path.as_posix()
            records.setdefault(review_key(row), row)
    return list(records.values())


def load_groups_from_runs(runs_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(runs_root.glob("*/annotations/diagnostics/probe_groups.jsonl")):
        pass_name = path.parts[-4] if len(path.parts) >= 4 else path.as_posix()
        for row in read_jsonl(path):
            row = dict(row)
            row["_source_pass"] = pass_name
            by_id.setdefault(str(row.get("probe_group_id")), []).append(row)
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit accepted Codex-human review records against the current combined pass."
    )
    parser.add_argument("--main-pass-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main_groups = read_jsonl(
        args.main_pass_root / "annotations/diagnostics/probe_groups.jsonl"
    )
    main_ids = {str(row.get("probe_group_id")) for row in main_groups}
    main_exact_sem = {
        group_semantic_key(row, include_template=True) for row in main_groups
    }
    main_loose_sem = {
        group_semantic_key(row, include_template=False) for row in main_groups
    }
    accepted = load_accept_records(args.work_root)
    groups_by_id = load_groups_from_runs(args.runs_root)

    rows = []
    for record in accepted:
        group_id = str(record.get("probe_group_id"))
        candidates = groups_by_id.get(group_id, [])
        exact_match = False
        loose_match = False
        candidate_summary = []
        for candidate in candidates:
            exact_key = group_semantic_key(candidate, include_template=True)
            loose_key = group_semantic_key(candidate, include_template=False)
            exact_match = exact_match or exact_key in main_exact_sem
            loose_match = loose_match or loose_key in main_loose_sem
            candidate_summary.append(
                {
                    "source_pass": candidate.get("_source_pass"),
                    "probe_group_id": candidate.get("probe_group_id"),
                    "target_player": candidate.get("target_player"),
                    "anchor_event_ids": candidate.get("anchor_event_ids", []),
                    "related_claim_ids": candidate.get("related_claim_ids", []),
                    "template": candidate.get("template"),
                    "query_type": (candidate.get("query_variable") or {}).get("type"),
                    "gold_source": candidate.get("gold_source"),
                }
            )
        if group_id in main_ids:
            status = "direct_id_in_main"
        elif exact_match:
            status = "semantic_exact_in_main"
        elif loose_match:
            status = "semantic_loose_in_main"
        else:
            status = "not_in_main"
        rows.append(
            {
                "status": status,
                "review_record_id": record.get("review_record_id"),
                "probe_group_id": group_id,
                "source_review_file": record.get("_source_review_file"),
                "review_summary": record.get("review_summary"),
                "candidate_groups": candidate_summary,
            }
        )

    summary = {
        "main_pass_root": args.main_pass_root.as_posix(),
        "accepted_review_records_unique": len(accepted),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "not_in_main": [row for row in rows if row["status"] == "not_in_main"],
        "semantic_loose_in_main": [
            row for row in rows if row["status"] == "semantic_loose_in_main"
        ],
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "main_pass_root",
                    "accepted_review_records_unique",
                    "status_counts",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
