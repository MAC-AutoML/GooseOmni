from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """TASK: gooseomni_meeting_claim_audio_confirmation_v1

You are doing a narrow second-pass audio/video confirmation for a paper-grade GooseOmni Theory-of-Mind benchmark.
Canonical player IDs: Gemini, baile, beigang, mojiang, saoyi, xiaolu.

Goal:
- Confirm or correct one candidate meeting utterance from a short Goose Goose Duck meeting clip.
- Focus only on exact spoken text, speaker grounding, and alias/canonical target safety.
- Do not generate new unrelated claims.
- Prefer "unknown" over guessing.

Important policy:
- TASK_JSON contains a Qwen candidate, not gold.
- The video/audio is the evidence.
- Qwen-only confirmation is still qwen_checked_audio_confirmation, not human_verified.
- A later Codex-human merge step may use this output as evidence, but you must keep needs_human_review=true.
- If the utterance is not clearly audible, set transcript_match="uncertain" or "reject".

Canonical / alias safety:
- The six canonical player IDs are benchmark POV players, not every display name in the match.
- claim_target_players may contain only Gemini, baile, beigang, mojiang, saoyi, xiaolu.
- Do not map non-POV display names such as 紫林, 猫手, 牛六, 温天, 雪豹, 海螺, 小杨鸭, 小艺, 新一酱, 明哥, 牛牛, EDDIE to a canonical player unless this exact clip proves it.
- If alias mapping is not directly proved, keep the raw name in claim_target_display_names and leave claim_target_players empty.
- Never put "unknown" in claim_target_players.

Check the candidate utterance:
1. Is the candidate speaker audible/visible enough?
2. Is the candidate transcript exact enough for benchmark use?
3. If not exact, provide a corrected transcript only for the utterance in question.
4. Are canonical targets safe? If not, remove unsafe canonical targets and preserve display names.
5. Is the utterance strategically relevant for claim-awareness, trust-update, vote influence, or perspective-taking probes?

Return strict JSON only:
{
  "review_task_id": "string",
  "audio_confirmation_decision": "confirm_candidate|correct_transcript|correct_speaker_or_alias|reject|uncertain",
  "evidence_quality": "high|medium|low|unusable",
  "confirmed_speaker": {
    "speaker_display_name": "string|unknown",
    "canonical_speaker": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
    "speaker_confidence": "high|medium|low|unknown",
    "speaker_evidence_types": ["voice", "nameplate", "visible_chat_row", "prior_alias", "unclear"]
  },
  "transcript": {
    "candidate_text": "string",
    "transcript_match": "exact|minor_correction|major_correction|uncertain|reject",
    "corrected_text": "string",
    "text_confidence": "high|medium|low|unknown"
  },
  "claim_grounding": {
    "claim_type": "accusation|defense|location|route|sighting|role_claim|vote_suggestion|question|agreement|contradiction|other|none",
    "strategic_function": "accuse|defend_self|defend_other|coordinate_vote|ask_for_information|answer_question|social_chatter|unknown",
    "claim_target_players": ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"],
    "claim_target_display_names": ["string"],
    "unsafe_alias_mappings_removed": ["string"],
    "tom_relevance": "high|medium|low"
  },
  "gold_policy": {
    "gold_source_after_this_pass": "qwen_checked_audio_confirmation",
    "eligible_for_human_gold_merge": false,
    "needs_human_review": true,
    "remaining_required_gate": "codex_human_or_independent_transcript_review"
  },
  "evidence_notes": ["string"],
  "remaining_uncertainties": ["string"]
}
"""


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


def build_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_task_id": "audio_" + row["audio_confirmation_item_id"],
        "task_type": "meeting_claim_audio_confirmation",
        "priority": "high",
        "primary_video_file": row["video_file"],
        "context_video_files": [],
        "source_result_id": row["source_result_id"],
        "source_review_item_id": row["audio_confirmation_item_id"],
        "next_required_gate": row["next_required_gate"],
        "candidate_utterance": {
            "canonical_speaker": row.get("canonical_speaker"),
            "speaker_display_name": row.get("speaker_display_name"),
            "abs_start_sec": row.get("abs_start_sec"),
            "abs_end_sec": row.get("abs_end_sec"),
            "claim_type": row.get("claim_type"),
            "strategic_function": row.get("strategic_function"),
            "claim_target_players": row.get("claim_target_players") or [],
            "claim_target_display_names": row.get("claim_target_display_names") or [],
            "utterance_text": row.get("qwen_transcript_to_confirm"),
        },
        "acceptance_policy": {
            "qwen_only_is_not_human_gold": True,
            "eligible_outputs": [
                "confirm_candidate",
                "correct_transcript",
                "correct_speaker_or_alias",
                "reject",
                "uncertain",
            ],
            "human_gold_requires_later_merge_gate": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qwen audio-confirmation pack for strict meeting-claim candidates."
    )
    parser.add_argument("--audio-confirmation-queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.audio_confirmation_queue)
    if args.limit is not None:
        rows = rows[: args.limit]
    tasks = [build_task(row) for row in rows]
    write_jsonl(
        args.output_root / "meeting_claim_audio_confirmation_queue.jsonl", tasks
    )
    (args.output_root / "meeting_claim_audio_confirmation_prompt.md").write_text(
        PROMPT_TEMPLATE, encoding="utf-8"
    )
    summary = {
        "ok": True,
        "source_audio_confirmation_queue": args.audio_confirmation_queue.as_posix(),
        "output_root": args.output_root.as_posix(),
        "tasks": len(tasks),
        "prompt": (
            args.output_root / "meeting_claim_audio_confirmation_prompt.md"
        ).as_posix(),
        "queue": (
            args.output_root / "meeting_claim_audio_confirmation_queue.jsonl"
        ).as_posix(),
        "promotion_to_human_verified_gold": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
