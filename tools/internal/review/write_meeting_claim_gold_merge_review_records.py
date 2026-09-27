from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def review_item(item: dict[str, Any]) -> dict[str, Any]:
    uncertainties = item.get("remaining_uncertainties") or []
    confirmed_transcript = item.get("confirmed_transcript") or {}
    confirmed_speaker = item.get("confirmed_speaker") or {}
    claim = item.get("claim_grounding") or {}
    transcript_ok = (
        confirmed_transcript.get("transcript_match") == "exact"
        and confirmed_transcript.get("text_confidence") == "high"
    )
    speaker_ok = (
        confirmed_speaker.get("canonical_speaker")
        in {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}
        and confirmed_speaker.get("speaker_confidence") == "high"
    )
    no_unsafe_targets = not claim.get("claim_target_players") or all(
        target in {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}
        for target in claim.get("claim_target_players", [])
    )
    has_tom_use = bool(item.get("suggested_probe_uses"))

    if transcript_ok and speaker_ok and no_unsafe_targets and has_tom_use:
        decision = "accept_for_qwen_checked_merge_candidate"
        next_gate = "codex_human_audio_spotcheck_or_independent_transcript_before_human_verified"
        notes = [
            "Audio-confirmation output is internally consistent and suitable for building candidate claim-awareness/D-probe drafts.",
            "Display-only aliases remain unmapped when not proved in this clip.",
            "Do not mark human_verified until an explicit final audio spot-check or independent transcript review is recorded.",
        ]
    else:
        decision = "reject_for_gold_merge"
        next_gate = "none_rejected"
        notes = [
            "At least one final merge acceptance criterion failed.",
            "Reject for gold merge unless the source confirmation is repaired and re-reviewed.",
        ]

    return {
        "merge_review_item_id": item["merge_review_item_id"],
        "source_merge_candidate_id": item["source_merge_candidate_id"],
        "source_review_item_id": item["source_review_item_id"],
        "source_result_file": item["source_result_file"],
        "video_file": item["video_file"],
        "confirmed_speaker": confirmed_speaker,
        "confirmed_transcript": confirmed_transcript,
        "claim_grounding": claim,
        "suggested_probe_uses": item.get("suggested_probe_uses") or [],
        "remaining_uncertainties": uncertainties,
        "codex_human_merge_decision": decision,
        "review_notes": notes,
        "promotion_to_human_verified_gold": False,
        "safe_for_probe_draft_generation": decision
        == "accept_for_qwen_checked_merge_candidate",
        "next_required_gate": next_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write conservative final merge review records for meeting-claim candidates."
    )
    parser.add_argument("--merge-queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    items = read_jsonl(args.merge_queue)
    records = [review_item(item) for item in items]
    decision_counts: dict[str, int] = {}
    for record in records:
        decision = record["codex_human_merge_decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    summary = {
        "ok": True,
        "merge_queue": args.merge_queue.as_posix(),
        "records": len(records),
        "decision_counts": dict(sorted(decision_counts.items())),
        "safe_for_probe_draft_generation": sum(
            1 for record in records if record["safe_for_probe_draft_generation"]
        ),
        "promotion_to_human_verified_gold": False,
        "note": "These records authorize qwen_checked probe drafting only; human_verified gold still requires explicit final audio spot-check or independent transcript review.",
    }
    write_jsonl(
        args.output_root / "codex_human_gold_merge_review_records.jsonl", records
    )
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
