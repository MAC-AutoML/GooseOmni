from __future__ import annotations

import argparse
import json
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


def classify_probe_use(candidate: dict[str, Any]) -> list[str]:
    claim = (
        candidate.get("claim_grounding")
        if isinstance(candidate.get("claim_grounding"), dict)
        else {}
    )
    claim_type = claim.get("claim_type")
    strategic = claim.get("strategic_function")
    uses: list[str] = []
    if claim_type in {
        "location",
        "route",
        "sighting",
        "defense",
        "accusation",
        "role_claim",
    }:
        uses.append("claim_truth_vs_claim_awareness")
    if claim_type in {"route", "location", "sighting"}:
        uses.append("route_belief")
    if strategic in {
        "accuse",
        "defend_self",
        "defend_other",
        "coordinate_vote",
        "ask_for_information",
    }:
        uses.append("D_perspective_taking_prediction")
    if strategic in {"coordinate_vote", "accuse"}:
        uses.append("vote_influence")
    return list(dict.fromkeys(uses))


def build_merge_item(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    transcript = (
        candidate.get("transcript")
        if isinstance(candidate.get("transcript"), dict)
        else {}
    )
    claim = (
        candidate.get("claim_grounding")
        if isinstance(candidate.get("claim_grounding"), dict)
        else {}
    )
    speaker = (
        candidate.get("confirmed_speaker")
        if isinstance(candidate.get("confirmed_speaker"), dict)
        else {}
    )
    return {
        "merge_review_item_id": f"mcg_merge_{index:05d}_{candidate.get('merge_candidate_id')}",
        "source_merge_candidate_id": candidate.get("merge_candidate_id"),
        "source_review_item_id": candidate.get("source_review_item_id"),
        "source_result_file": candidate.get("result_file"),
        "video_file": candidate.get("video_file"),
        "review_type": "codex_human_final_meeting_claim_gold_merge",
        "priority": "high"
        if "D_perspective_taking_prediction" in classify_probe_use(candidate)
        else "medium",
        "suggested_probe_uses": classify_probe_use(candidate),
        "confirmed_speaker": speaker,
        "confirmed_transcript": {
            "text": transcript.get("corrected_text")
            or transcript.get("candidate_text"),
            "transcript_match": transcript.get("transcript_match"),
            "text_confidence": transcript.get("text_confidence"),
        },
        "claim_grounding": claim,
        "evidence_quality": candidate.get("evidence_quality"),
        "remaining_uncertainties": candidate.get("remaining_uncertainties") or [],
        "acceptance_criteria": [
            "The transcript is sufficiently exact for benchmark gold.",
            "The canonical speaker is one of Gemini, baile, beigang, mojiang, saoyi, xiaolu and is grounded in the clip.",
            "Every canonical claim target is independently safe; otherwise target must stay display-only.",
            "The claim has clear ToM use for claim-awareness, route-belief, vote-influence, or D perspective-taking.",
            "No Qwen-only field is treated as human_verified without this final Codex-human merge review.",
        ],
        "merge_policy": {
            "promotion_to_human_verified_gold": False,
            "requires_explicit_accept_record": True,
            "reject_if_any_acceptance_criterion_fails": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final Codex-human merge queue for audio-confirmed meeting claims."
    )
    parser.add_argument("--merge-candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    candidates = read_jsonl(args.merge_candidates)
    items = [
        build_merge_item(candidate, index + 1)
        for index, candidate in enumerate(candidates)
    ]
    use_counts: dict[str, int] = {}
    for item in items:
        for use in item["suggested_probe_uses"]:
            use_counts[use] = use_counts.get(use, 0) + 1

    write_jsonl(args.output_root / "codex_human_gold_merge_queue.jsonl", items)
    summary = {
        "ok": True,
        "source_merge_candidates": args.merge_candidates.as_posix(),
        "merge_review_items": len(items),
        "suggested_probe_use_counts": dict(sorted(use_counts.items())),
        "promotion_to_human_verified_gold": False,
        "requires_explicit_accept_record": True,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
