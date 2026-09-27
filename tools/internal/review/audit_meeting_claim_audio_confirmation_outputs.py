from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANONICAL_PLAYERS = {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}
ACCEPTABLE_DECISIONS = {
    "confirm_candidate",
    "correct_transcript",
    "correct_speaker_or_alias",
    "reject",
    "uncertain",
}
PROMOTABLE_DECISIONS = {
    "confirm_candidate",
    "correct_transcript",
    "correct_speaker_or_alias",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def audit_result(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = read_json(path)
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    issues: list[str] = []
    if not result.get("parse_ok"):
        issues.append("parse_not_ok")

    decision = parsed.get("audio_confirmation_decision")
    if decision not in ACCEPTABLE_DECISIONS:
        issues.append("invalid_audio_confirmation_decision")

    gold_policy = (
        parsed.get("gold_policy") if isinstance(parsed.get("gold_policy"), dict) else {}
    )
    if gold_policy.get("eligible_for_human_gold_merge") is True:
        issues.append("qwen_claims_direct_human_gold_eligibility")
    if gold_policy.get("needs_human_review") is not True:
        issues.append("needs_human_review_not_true")

    speaker = (
        parsed.get("confirmed_speaker")
        if isinstance(parsed.get("confirmed_speaker"), dict)
        else {}
    )
    canonical_speaker = speaker.get("canonical_speaker")
    if canonical_speaker not in CANONICAL_PLAYERS and canonical_speaker != "unknown":
        issues.append("invalid_canonical_speaker")

    claim = (
        parsed.get("claim_grounding")
        if isinstance(parsed.get("claim_grounding"), dict)
        else {}
    )
    invalid_targets = [
        target
        for target in (claim.get("claim_target_players") or [])
        if target not in CANONICAL_PLAYERS
    ]
    if invalid_targets:
        issues.append("invalid_claim_target_players")

    transcript = (
        parsed.get("transcript") if isinstance(parsed.get("transcript"), dict) else {}
    )
    evidence_quality = parsed.get("evidence_quality")
    transcript_match = transcript.get("transcript_match")
    can_enter_merge_gate = (
        not issues
        and decision in PROMOTABLE_DECISIONS
        and evidence_quality in {"high", "medium"}
        and transcript_match in {"exact", "minor_correction"}
        and speaker.get("canonical_speaker") in CANONICAL_PLAYERS
        and speaker.get("speaker_confidence") in {"high", "medium"}
    )

    row = {
        "review_task_id": result.get("review_task_id"),
        "result_file": path.as_posix(),
        "source_review_item_id": (result.get("source_task") or {}).get(
            "source_review_item_id"
        ),
        "parse_ok": bool(result.get("parse_ok")),
        "audio_confirmation_decision": decision,
        "evidence_quality": evidence_quality,
        "transcript_match": transcript_match,
        "canonical_speaker": canonical_speaker,
        "claim_target_players": claim.get("claim_target_players") or [],
        "issues": issues,
        "can_enter_codex_human_gold_merge_gate": can_enter_merge_gate,
        "promotion_to_human_verified_gold": False,
    }

    merge_candidate = None
    if can_enter_merge_gate:
        merge_candidate = {
            "merge_candidate_id": result.get("review_task_id"),
            "source_review_item_id": row["source_review_item_id"],
            "result_file": path.as_posix(),
            "video_file": result.get("video_file"),
            "confirmed_speaker": speaker,
            "transcript": transcript,
            "claim_grounding": claim,
            "evidence_quality": evidence_quality,
            "remaining_uncertainties": parsed.get("remaining_uncertainties") or [],
            "gold_policy": {
                "qwen_only_is_not_human_gold": True,
                "requires_codex_human_gold_merge_gate": True,
                "promotion_to_human_verified_gold": False,
            },
        }
    return row, merge_candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Qwen audio-confirmation outputs for meeting claims."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    merge_candidates: list[dict[str, Any]] = []
    for path in sorted(args.results_root.glob("*.json")):
        if path.name.endswith(".error.json"):
            rows.append(
                {
                    "review_task_id": path.name.removesuffix(".error.json"),
                    "result_file": path.as_posix(),
                    "parse_ok": False,
                    "issues": ["error_result"],
                    "can_enter_codex_human_gold_merge_gate": False,
                    "promotion_to_human_verified_gold": False,
                }
            )
            continue
        row, merge_candidate = audit_result(path)
        rows.append(row)
        if merge_candidate is not None:
            merge_candidates.append(merge_candidate)

    issue_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision = row.get("audio_confirmation_decision") or "unknown"
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        for issue in row.get("issues") or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    summary = {
        "ok": True,
        "results": len(rows),
        "merge_gate_candidates": len(merge_candidates),
        "decision_counts": dict(sorted(decision_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "promotion_to_human_verified_gold": False,
    }
    write_jsonl(args.output_root / "audio_confirmation_audit_rows.jsonl", rows)
    write_jsonl(
        args.output_root / "codex_human_gold_merge_candidates.jsonl", merge_candidates
    )
    write_json(args.output_root / "audio_confirmation_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
