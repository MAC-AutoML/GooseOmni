from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PLAYERS = {"Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"}


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


def compact_text(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def infer_cutoff_abs_sec(source_review_item_id: str) -> float:
    match = re.search(r"_meeting_(\d+)_(\d+)_", source_review_item_id or "")
    if not match:
        return 0.0
    return float(match.group(2))


def infer_source_segment_ids(source_review_item_id: str) -> list[str]:
    match = re.search(r"(g001_phase_\d+_meeting_\d+_\d+)", source_review_item_id or "")
    return [match.group(1)] if match else []


def target_player_for_group(speaker_id: str, canonical_targets: list[str]) -> str:
    for player in canonical_targets:
        if player in PLAYERS:
            return player
    return speaker_id if speaker_id in PLAYERS else "Gemini"


def build_group(record: dict[str, Any], index: int) -> dict[str, Any]:
    transcript = record.get("confirmed_transcript") or {}
    speaker = record.get("confirmed_speaker") or {}
    claim = record.get("claim_grounding") or {}
    speaker_id = speaker.get("canonical_speaker") or "unknown"
    claim_text = transcript.get("text") or ""
    claim_type = claim.get("claim_type") or "other"
    strategic = claim.get("strategic_function") or "unknown"
    display_targets = claim.get("claim_target_display_names") or []
    canonical_targets = claim.get("claim_target_players") or []
    target_summary = (
        ", ".join(canonical_targets or display_targets) or "the discussed player/event"
    )
    source_id = record["source_review_item_id"]
    group_id = f"mcg_pg_{index:05d}_{source_id}"
    query_type = (
        "claim_truth_vs_claim_awareness"
        if "claim_truth_vs_claim_awareness"
        in (record.get("suggested_probe_uses") or [])
        else "trust_update"
    )
    return {
        "probe_group_id": group_id,
        "game_id": "g001",
        "template": "meeting_claim_audio_confirmed",
        "source_segment_ids": infer_source_segment_ids(source_id),
        "cutoff_abs_sec": infer_cutoff_abs_sec(source_id),
        "target_player": target_player_for_group(speaker_id, canonical_targets),
        "anchor_event_ids": [source_id],
        "related_claim_ids": [source_id],
        "hidden_event_ids_for_target": [],
        "available_evidence_ids_for_target": [source_id],
        "selection_reason": "Audio-confirmed meeting claim with exact/high-confidence transcript and speaker gate.",
        "source_merge_review_item_id": record["merge_review_item_id"],
        "source_review_item_id": source_id,
        "source_result_file": record["source_result_file"],
        "video_file": record["video_file"],
        "query_variable": {
            "type": query_type,
            "description": f"How should listeners interpret {speaker_id}'s meeting claim about {target_summary}?",
        },
        "speaker": speaker_id,
        "claim_text": claim_text,
        "claim_type": claim_type,
        "strategic_function": strategic,
        "claim_target_players": canonical_targets,
        "claim_target_display_names": display_targets,
        "diagnostic_families": [
            family
            for family in [
                "claim_verification",
                "false_belief",
                "representational_change",
                "perspective_taking",
                "strategy_communication"
                if strategic
                in {"accuse", "defend_self", "defend_other", "coordinate_vote"}
                else "",
            ]
            if family
        ],
        "quality": {
            "audio_confirmation": "qwen_checked",
            "merge_review_decision": record["codex_human_merge_decision"],
            "needs_human_review": True,
            "promotion_to_human_verified_gold": False,
        },
        "needs_human_review": True,
        "review_reasons": [
            "Probe group is generated from audio-confirmed qwen_checked meeting claim, not final human_verified gold.",
            "Final benchmark inclusion requires explicit Codex-human accept record and claim-truth/visibility linkage.",
        ],
    }


def build_probes(group: dict[str, Any]) -> list[dict[str, Any]]:
    group_id = group["probe_group_id"]
    speaker = group["speaker"]
    claim_text = group["claim_text"]
    claim_short = compact_text(claim_text)
    display_targets = group.get("claim_target_display_names") or []
    canonical_targets = group.get("claim_target_players") or []
    target_text = (
        ", ".join(canonical_targets or display_targets) or "the referenced player/event"
    )
    target_player = group["target_player"]
    cutoff = group["cutoff_abs_sec"]
    public_form = {
        "speaker": speaker,
        "claim_text": claim_text,
        "claim_type": group.get("claim_type"),
        "target_text": target_text,
        "source_review_item_id": group["source_review_item_id"],
    }
    expected_schema = {
        "belief_or_interpretation": "string",
        "knows_global_truth": "true|false|unknown",
        "evidence_used": ["string"],
        "forbidden_evidence_used": ["string"],
        "confidence": "low|medium|high",
    }
    common = {
        "probe_group_id": group_id,
        "game_id": "g001",
        "target_player": target_player,
        "cutoff_abs_sec": cutoff,
        "gold_source": "qwen_checked",
        "needs_human_review": True,
        "promotion_to_human_verified_gold": False,
        "acceptable_evidence_ids": [group["source_review_item_id"]],
        "forbidden_event_ids": [],
        "expected_output_schema": expected_schema,
    }
    return [
        {
            **common,
            "probe_id": f"{group_id}_A",
            "probe_type": "A_pre_reveal_belief",
            "input_condition": "target_available_events",
            "prompt": (
                "QUERY_VARIABLE_PUBLIC_FORM_JSON="
                + json.dumps(public_form, ensure_ascii=False)
                + "\n"
                f"在会议中，{speaker} 说：『{claim_short}』。只根据当时公开可听到的发言，"
                f"目标玩家 {target_player} 应如何理解这条关于 {target_text} 的 claim？不要使用其他 POV 的隐藏事实。"
            ),
        },
        {
            **common,
            "probe_id": f"{group_id}_B",
            "probe_type": "B_post_reveal_reconstruct_previous_belief",
            "input_condition": "oracle_truth_revealed",
            "prompt": (
                "ORACLE_TRUTH_JSON="
                + json.dumps(
                    {
                        "claim_text": claim_text,
                        "source": "meeting_claim_audio_confirmed",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                f"现在可以看到后验全局信息，但请重建 claim 刚说完时 {target_player} 的主观状态。"
                f"当时听到 {speaker} 的发言『{claim_short}』后，{target_player} 应如何理解这条 claim？"
                "不要把后验真相倒灌成当时已经知道的证据。"
            ),
        },
        {
            **common,
            "probe_id": f"{group_id}_C",
            "probe_type": "C_other_agent_false_belief",
            "input_condition": "public_history_only",
            "prompt": (
                "QUERY_VARIABLE_PUBLIC_FORM_JSON="
                + json.dumps(public_form, ensure_ascii=False)
                + "\n"
                f"另一个只听到同一段公开会议发言、但没有额外私人 POV 证据的玩家，"
                f"在 {cutoff:.1f}s 时应如何理解 {speaker} 的发言『{claim_short}』？"
            ),
        },
        {
            **common,
            "probe_id": f"{group_id}_D",
            "probe_type": "D_perspective_taking_prediction",
            "input_condition": "speaker_perspective",
            "prompt": (
                "SPEAKER_AVAILABLE_CONTEXT_JSON="
                + json.dumps(
                    {"speaker": speaker, "claim_text": claim_text}, ensure_ascii=False
                )
                + "\n"
                "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON="
                + json.dumps(public_form, ensure_ascii=False)
                + "\n"
                f"从发言者 {speaker} 的视角看，说完『{claim_short}』后，"
                f"{speaker} 应该预期其他玩家会如何理解、相信、质疑或跟随这条关于 {target_text} 的 claim？"
            ),
        },
    ]


def build_hidden_gold(
    group: dict[str, Any], probes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    d_reference = {
        "human_verified_scope": "Audio-confirmed meeting claim and public listener interpretation; does not certify hidden global truth beyond the claim text.",
        "speaker": group["speaker"],
        "claim_text": group["claim_text"],
        "strategic_function": group["strategic_function"],
        "expected_listener_basis": "public meeting speech only",
    }
    return [
        {
            "probe_id": probe["probe_id"],
            "probe_group_id": group["probe_group_id"],
            "gold_source": "qwen_checked",
            "claim_text": group["claim_text"],
            "speaker": group["speaker"],
            "claim_type": group["claim_type"],
            "strategic_function": group["strategic_function"],
            "claim_target_players": group.get("claim_target_players") or [],
            "claim_target_display_names": group.get("claim_target_display_names") or [],
            "allowed_source_review_item_id": group["source_review_item_id"],
            "D_PT_reference_enriched_by": "meeting_claim_audio_confirmed_public_speech_gate",
            "D_PT_reference": d_reference,
            "promotion_to_human_verified_gold": False,
            "needs_human_review": True,
        }
        for probe in probes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build qwen_checked Decrypto-style probe drafts from meeting-claim merge records."
    )
    parser.add_argument("--merge-review-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    records = [
        row
        for row in read_jsonl(args.merge_review_records)
        if row.get("safe_for_probe_draft_generation") is True
        and row.get("codex_human_merge_decision")
        == "accept_for_qwen_checked_merge_candidate"
    ]
    groups: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    hidden_gold: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        group = build_group(record, index)
        group_probes = build_probes(group)
        groups.append(group)
        probes.extend(group_probes)
        hidden_gold.extend(build_hidden_gold(group, group_probes))

    write_jsonl(args.output_root / "probe_groups.qwen_checked_draft.jsonl", groups)
    write_jsonl(args.output_root / "probes.qwen_checked_draft.jsonl", probes)
    write_jsonl(args.output_root / "hidden_gold.qwen_checked_draft.jsonl", hidden_gold)
    summary = {
        "ok": True,
        "source_merge_review_records": args.merge_review_records.as_posix(),
        "probe_groups": len(groups),
        "probes": len(probes),
        "hidden_gold": len(hidden_gold),
        "gold_source": "qwen_checked",
        "promotion_to_human_verified_gold": False,
        "needs_human_review": True,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
