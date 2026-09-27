from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(results_root: Path) -> dict[str, Any]:
    rows = []
    errors = []
    for path in sorted(results_root.glob("*.json")):
        if path.name.endswith(".error.json"):
            errors.append(read_json(path))
            continue
        payload = read_json(path)
        parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
        source = payload.get("source_task") if isinstance(payload.get("source_task"), dict) else {}
        candidate = source.get("candidate") if isinstance(source.get("candidate"), dict) else {}
        rows.append(
            {
                "result_file": path.as_posix(),
                "review_task_id": payload.get("review_task_id"),
                "task_type": payload.get("task_type"),
                "parse_ok": bool(payload.get("parse_ok")),
                "decision": parsed.get("decision", "missing"),
                "gold_source_after_review": parsed.get("gold_source_after_review", "missing"),
                "needs_human_review": parsed.get("needs_human_review"),
                "acceptance_scope": parsed.get("acceptance_scope"),
                "reject_reasons": parsed.get("reject_reasons", []),
                "remaining_uncertainties": parsed.get("remaining_uncertainties", []),
                "candidate_id": candidate.get("candidate_id"),
                "claim_id": candidate.get("claim_id"),
                "event_id": candidate.get("event_id") or candidate.get("vote_event_id"),
                "target": candidate.get("target") or candidate.get("listener"),
            }
        )
    return {
        "results_root": results_root.as_posix(),
        "files": len(rows),
        "errors": len(errors),
        "parse_ok": sum(1 for row in rows if row["parse_ok"]),
        "decisions": dict(Counter(row["decision"] for row in rows)),
        "gold_source_after_review": dict(Counter(row["gold_source_after_review"] for row in rows)),
        "needs_human_review": dict(Counter(str(row["needs_human_review"]) for row in rows)),
        "accepted_task_ids": [row["review_task_id"] for row in rows if row["decision"] == "accept"],
        "rows": rows,
        "error_rows": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3-Omni clip review results for Codex human-gold gating.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.results_root)
    write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "error_rows"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
