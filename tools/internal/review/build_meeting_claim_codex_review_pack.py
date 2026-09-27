from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_contact_sheet(video: Path, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video.as_posix(),
            "-vf",
            "fps=1,scale=480:-1,tile=5x4",
            "-frames:v",
            "1",
            output.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0 and output.exists() and output.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict Codex-human review pack for Qwen meeting-claim candidates.")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--priority", action="append", default=["high"])
    parser.add_argument("--allow-row-issues", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.audit_root / "qwen_meeting_claim_audit_rows.jsonl")
    row_by_task = {row["review_task_id"]: row for row in rows}
    candidates = read_jsonl(args.audit_root / "candidate_claims_for_codex_human_review.jsonl")
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("review_priority") not in set(args.priority):
            continue
        if candidate.get("audit_issues"):
            continue
        source_id = candidate["source_result_id"]
        source_row = row_by_task.get(source_id, {})
        if source_row.get("audit_issues") and not args.allow_row_issues:
            continue
        result_path = args.results_root / f"{source_id}.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        video = Path(result["video_file"])
        contact = args.output_root / "contact_sheets" / f"{source_id}_contact.jpg"
        if not contact.exists():
            make_contact_sheet(video, contact)
        selected.append(
            {
                "review_item_id": candidate["candidate_id"],
                "review_type": "codex_human_meeting_claim_grounding",
                "source_result_id": source_id,
                "source_result_file": result_path.as_posix(),
                "video_file": video.as_posix(),
                "contact_sheet": contact.as_posix(),
                "review_priority": candidate["review_priority"],
                "row_audit_issues": source_row.get("audit_issues", []),
                "candidate_audit_issues": candidate.get("audit_issues", []),
                "promotion_allowed_without_codex_human_review": False,
                "acceptance_criteria": [
                    "speaker canonical ID is visually or audibly grounded",
                    "utterance text is correct enough for benchmark gold",
                    "claim_type and strategic_function match the utterance",
                    "claim_target_players contains only verified six-POV canonical players",
                    "non-POV display names remain display-only",
                    "no vote/reaction claim is accepted without direct visual evidence",
                ],
                "candidate": candidate,
            }
        )
        if args.limit is not None and len(selected) >= args.limit:
            break

    write_jsonl(args.output_root / "codex_human_review_queue.jsonl", selected)
    summary = {
        "ok": True,
        "audit_root": args.audit_root.as_posix(),
        "results_root": args.results_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "selected_review_items": len(selected),
        "priorities": args.priority,
        "allow_row_issues": args.allow_row_issues,
        "promotion_allowed_without_codex_human_review": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
