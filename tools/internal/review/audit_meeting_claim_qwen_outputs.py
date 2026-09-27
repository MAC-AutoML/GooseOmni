from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PLAYERS = {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}
NON_CANONICAL_NAMES = {"紫林", "猫手", "牛六", "温天", "雪豹", "海螺", "小杨鸭", "革命你"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def has_noncanonical_display(display_names: list[Any]) -> bool:
    text = " ".join(str(item) for item in display_names)
    return any(name in text for name in NON_CANONICAL_NAMES)


def audit_utterance(result_id: str, utterance: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    target_players = utterance.get("claim_target_players") or []
    target_display_names = utterance.get("claim_target_display_names") or []
    if not isinstance(target_players, list):
        issues.append("claim_target_players_not_list")
        target_players = []
    invalid_targets = [player for player in target_players if player not in PLAYERS]
    if invalid_targets:
        issues.append("invalid_claim_target_players")
    if target_players and has_noncanonical_display(target_display_names):
        issues.append("noncanonical_display_mapped_to_canonical_target")
    if utterance.get("needs_human_review") is not True:
        issues.append("needs_human_review_not_true")
    if utterance.get("canonical_speaker") not in PLAYERS and utterance.get("canonical_speaker") != "unknown":
        issues.append("invalid_canonical_speaker")
    if utterance.get("tom_relevance") == "high" and utterance.get("text_confidence") == "high":
        review_priority = "high"
    elif utterance.get("tom_relevance") in {"high", "medium"}:
        review_priority = "medium"
    else:
        review_priority = "low"
    candidate = {
        "candidate_id": f"{result_id}_{utterance.get('utterance_id', 'utt')}",
        "source_result_id": result_id,
        "candidate_type": "meeting_claim_utterance",
        "review_priority": review_priority,
        "audit_issues": issues,
        "promotion_allowed_without_codex_human_review": False,
        "utterance": utterance,
    }
    return issues, candidate


def audit_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = read_json(path)
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    result_id = str(result.get("review_task_id") or path.stem)
    issues: list[str] = []
    candidates: list[dict[str, Any]] = []
    if not result.get("parse_ok"):
        issues.append("parse_not_ok")
    if parsed.get("vote_events"):
        issues.append("contains_vote_events_requires_visual_gate")
    if parsed.get("reaction_links"):
        issues.append("contains_reaction_links_requires_visual_gate")
    for utterance in parsed.get("utterances") or []:
        if not isinstance(utterance, dict):
            issues.append("utterance_not_object")
            continue
        utterance_issues, candidate = audit_utterance(result_id, utterance)
        issues.extend(utterance_issues)
        candidates.append(candidate)
    row = {
        "result_file": path.as_posix(),
        "review_task_id": result_id,
        "parse_ok": bool(result.get("parse_ok")),
        "clip_evidence_quality": parsed.get("clip_evidence_quality"),
        "utterance_count": len(parsed.get("utterances") or []),
        "vote_event_count": len(parsed.get("vote_events") or []),
        "reaction_link_count": len(parsed.get("reaction_links") or []),
        "audit_issues": sorted(set(issues)),
        "promotion_allowed_without_codex_human_review": False,
    }
    return row, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Qwen meeting-claim grounding outputs before Codex-human review.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for path in sorted(args.results_root.glob("*.json")):
        row, result_candidates = audit_result(path)
        rows.append(row)
        candidates.extend(result_candidates)

    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row["audit_issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    summary = {
        "ok": True,
        "results": len(rows),
        "candidate_utterances": len(candidates),
        "high_priority_candidates": sum(1 for row in candidates if row["review_priority"] == "high"),
        "issue_counts": dict(sorted(issue_counts.items())),
        "promotion_allowed_without_codex_human_review": False,
    }
    write_json(args.output_root / "qwen_meeting_claim_audit_summary.json", summary)
    write_jsonl(args.output_root / "qwen_meeting_claim_audit_rows.jsonl", rows)
    write_jsonl(args.output_root / "candidate_claims_for_codex_human_review.jsonl", candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
