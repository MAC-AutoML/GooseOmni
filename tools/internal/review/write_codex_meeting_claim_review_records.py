from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REVIEW_DECISIONS: dict[str, dict[str, Any]] = {
    "mcgw_00001_g001_phase_001_meeting_000080_000390_w000_Gemini_u001": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Contact sheet supports meeting/vote UI context and mojiang/末将 as visible discussion participant.",
            "Qwen transcript is plausible and ToM-relevant, but exact speech cannot be human-verified from frames alone.",
            "Keep as claim-grounding candidate; do not promote to human_verified gold yet.",
        ],
    },
    "mcgw_00002_g001_phase_001_meeting_000080_000390_w001_Gemini_u001": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Speaker and public meeting context are visually plausible.",
            "The utterance contains non-POV display names only; canonical target list correctly remains empty.",
            "Speech content needs audio confirmation before use as gold.",
        ],
    },
    "mcgw_00003_g001_phase_001_meeting_000080_000390_w002_Gemini_u001": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Meeting/vote context is visible; non-POV display name 海螺 is not mapped to a canonical target.",
            "The claim is strategically useful for public claim awareness, but transcript still needs audio confirmation.",
        ],
    },
    "mcgw_00004_g001_phase_001_meeting_000080_000390_w003_Gemini_u001": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Gemini POV and public meeting context are visible.",
            "Candidate is useful as a sighting/question claim about 海螺, but exact wording requires audio confirmation.",
        ],
    },
    "mcgw_00005_g001_phase_001_meeting_000080_000390_w004_Gemini_u001": {
        "decision": "reject_for_gold",
        "review_confidence": "high",
        "review_notes": [
            "Qwen mapped several display names to canonical players without sufficient visual proof.",
            "The candidate mentions 心怡/埃迪/明锅/车车; canonical target mapping to saoyi/mojiang/beigang is not safe.",
            "Reject for human gold unless a separate alias-mapping review proves each mapping in this exact clip.",
        ],
    },
    "mcgw_00006_g001_phase_001_meeting_000080_000390_w005_Gemini_u002": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Display names remain non-canonical and are not force-mapped.",
            "The utterance is useful as a self-defense claim, but exact speech needs audio confirmation.",
        ],
    },
    "mcgw_00009_g001_phase_001_meeting_000080_000390_w008_Gemini_u003": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Role-claim/accusation content is ToM-relevant.",
            "Frame evidence cannot verify exact spoken role names; keep candidate only.",
        ],
    },
    "mcgw_00010_g001_phase_001_meeting_000080_000390_w009_Gemini_u001": {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Public vote/meeting context and chat panel are visible.",
            "Question about deaths is useful for public-history tracking; exact transcript needs audio confirmation.",
        ],
    },
}


CANONICAL_PLAYERS = {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}


def infer_conservative_decision(item: dict[str, Any]) -> dict[str, Any]:
    utterance = item["candidate"]["utterance"]
    speaker = utterance.get("canonical_speaker")
    speaker_display = utterance.get("speaker_display_name")
    target_players = utterance.get("claim_target_players") or []
    text = utterance.get("utterance_text") or ""
    row_issues = item.get("row_audit_issues") or []
    candidate_issues = item.get("candidate_audit_issues") or []

    if row_issues or candidate_issues:
        return {
            "decision": "reject_for_gold",
            "review_confidence": "high",
            "review_notes": [
                "Audit issues remain on this candidate or its source row.",
                "Reject for human gold unless the source extraction is repaired and re-reviewed.",
            ],
        }

    if not text.strip():
        return {
            "decision": "reject_for_gold",
            "review_confidence": "high",
            "review_notes": [
                "No usable utterance text is available.",
                "A meeting claim probe cannot be grounded without a claim transcript.",
            ],
        }

    invalid_targets = [target for target in target_players if target not in CANONICAL_PLAYERS]
    if invalid_targets:
        return {
            "decision": "reject_for_gold",
            "review_confidence": "high",
            "review_notes": [
                f"Invalid canonical target IDs are present: {invalid_targets}.",
                "Reject for gold to avoid leaking unsafe alias mapping into benchmark labels.",
            ],
        }

    if speaker not in CANONICAL_PLAYERS:
        return {
            "decision": "needs_speaker_and_audio_confirmation",
            "review_confidence": "medium",
            "review_notes": [
                f"Speaker is not resolved to a six-POV canonical player: {speaker_display!r} / {speaker!r}.",
                "Keep as a claim-grounding candidate only; do not promote before speaker identity and transcript are independently confirmed.",
            ],
        }

    if target_players:
        return {
            "decision": "needs_alias_and_audio_confirmation",
            "review_confidence": "medium",
            "review_notes": [
                "Candidate contains canonical target players and may be useful for claim-awareness or perspective-taking probes.",
                "Canonical target mapping and exact transcript still need independent confirmation before gold promotion.",
            ],
        }

    return {
        "decision": "needs_audio_confirmation",
        "review_confidence": "medium",
        "review_notes": [
            "Canonical speaker and public meeting context are plausible from the structured candidate.",
            "Exact spoken content still requires audio or transcript confirmation before use as human-verified gold.",
        ],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write conservative Codex-human review records for meeting claim candidates.")
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    queue = read_jsonl(args.review_queue)
    queue_by_id = {row["review_item_id"]: row for row in queue}
    records: list[dict[str, Any]] = []
    for item in queue:
        item_id = item["review_item_id"]
        decision = REVIEW_DECISIONS.get(item_id) or infer_conservative_decision(item)
        next_gate = "audio_or_transcript_confirmation"
        if decision["decision"] == "needs_speaker_and_audio_confirmation":
            next_gate = "speaker_identity_and_audio_confirmation"
        elif decision["decision"] == "needs_alias_and_audio_confirmation":
            next_gate = "alias_mapping_and_audio_confirmation"
        elif decision["decision"] == "reject_for_gold":
            next_gate = "none_rejected"
        record = {
            "review_item_id": item_id,
            "source_result_id": item["source_result_id"],
            "video_file": item["video_file"],
            "contact_sheet": item["contact_sheet"],
            "candidate_utterance": item["candidate"]["utterance"],
            "codex_human_decision": decision["decision"],
            "review_confidence": decision["review_confidence"],
            "review_notes": decision["review_notes"],
            "promotion_to_human_verified_gold": False,
            "safe_for_next_stage": decision["decision"].startswith("needs_"),
            "next_required_gate": next_gate,
        }
        records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        decision = record["codex_human_decision"]
        counts[decision] = counts.get(decision, 0) + 1
    summary = {
        "ok": True,
        "review_queue": args.review_queue.as_posix(),
        "records": len(records),
        "decision_counts": dict(sorted(counts.items())),
        "promotion_to_human_verified_gold": False,
        "note": "These are conservative Codex-human visual/context review records. Audio-confirmed records are still required before gold promotion.",
    }
    write_jsonl(args.output_root / "codex_human_review_records.jsonl", records)
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
