from __future__ import annotations

from tests._decrypto_diagnostics_support import (
    Path,
    audio_confirmation_audit,
    audio_confirmation_pack,
    claim_truth_precheck,
    clip_review,
    gold_merge_review_records,
    json,
    meeting_claim_gold_merge,
    meeting_claim_probe_drafts,
    meeting_claim_sync,
    read_jsonl,
    write_jsonl,
)


def test_claim_truth_supported_precheck_requires_speaker_direct_witness(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "ledger"
    write_jsonl(
        ledger / "world_events.jsonl",
        [
            {
                "world_event_id": "ge_supported",
                "game_id": "g001",
                "phase_type": "gameplay",
                "event_type": "movement",
                "description": "Gemini 在街道上移动。",
                "actors": ["Gemini"],
                "source_povs": ["Gemini"],
                "source_segment_ids": ["g001_phase_001_gameplay_000080_000170"],
                "abs_start_sec": 100.0,
                "abs_end_sec": 105.0,
                "certainty": 0.9,
            }
        ],
    )
    write_jsonl(
        ledger / "claims.jsonl",
        [
            {
                "claim_id": "claim_supported",
                "speaker": "baile",
                "heard_by": ["baile", "saoyi"],
                "claim_type": "location",
                "content": "我看到 Gemini 刚才在街道那边移动。",
                "source_segment_ids": ["g001_phase_002_meeting_000170_000260"],
                "abs_start_sec": 180.0,
                "abs_end_sec": 190.0,
                "certainty": 0.9,
            }
        ],
    )
    write_jsonl(
        ledger / "claim_truth_links.jsonl",
        [
            {
                "claim_id": "claim_supported",
                "world_event_ids": ["ge_supported"],
                "truth_status_global": "supported",
                "confidence": 0.9,
            }
        ],
    )
    write_jsonl(
        ledger / "visibility_edges.jsonl",
        [
            {
                "event_id": "ge_supported",
                "player_id": "Gemini",
                "visibility": "direct_visual",
            },
            {
                "event_id": "ge_supported",
                "player_id": "saoyi",
                "visibility": "not_visible",
            },
        ],
    )

    rows = claim_truth_precheck.build_rows(
        ledger_root=ledger,
        video_root=tmp_path / "videos",
        output_dir=tmp_path / "out",
        limit=10,
        skipped=set(),
        max_targets_per_cluster=1,
        allow_speaker_mismatch=False,
        truth_status="supported",
    )
    assert rows == []

    def fake_extract_frame(video_root, phase_id, player, abs_sec, output):
        return {
            "ok": True,
            "phase_id": phase_id,
            "player": player,
            "abs_sec": abs_sec,
            "output": output.as_posix(),
        }

    monkeypatch.setattr(claim_truth_precheck, "extract_frame", fake_extract_frame)
    monkeypatch.setattr(claim_truth_precheck, "compose_quad", lambda row, output: True)
    monkeypatch.setattr(
        claim_truth_precheck, "make_contact", lambda quad_paths, output: True
    )

    claim_rows = read_jsonl(ledger / "claims.jsonl")
    claim_rows[0]["speaker"] = "Gemini"
    write_jsonl(ledger / "claims.jsonl", claim_rows)

    rows = claim_truth_precheck.build_rows(
        ledger_root=ledger,
        video_root=tmp_path / "videos",
        output_dir=tmp_path / "out",
        limit=10,
        skipped=set(),
        max_targets_per_cluster=1,
        allow_speaker_mismatch=False,
        truth_status="supported",
    )

    assert len(rows) == 1
    assert rows[0]["event_id"] == "ge_supported"
    assert rows[0]["claim_id"] == "claim_supported"
    assert rows[0]["target"] == "saoyi"
    assert rows[0]["claim_speaker_frame"]["player"] == "Gemini"


def test_clip_review_prompt_requires_visual_identity_checks() -> None:
    prompt = clip_review.PROMPT_TEMPLATE

    assert "candidate as a hypothesis" in prompt
    assert "animation labels" in prompt
    assert "nameplates" in prompt
    assert "vote UI" in prompt
    assert "short or vague claims" in prompt
    assert "exact visual/audio labels" in prompt


def test_meeting_claim_audio_confirmation_pack_keeps_qwen_out_of_human_gold() -> None:
    task = audio_confirmation_pack.build_task(
        {
            "audio_confirmation_item_id": "mcgw_00001_u001",
            "source_result_id": "mcgw_00001",
            "video_file": "clips/mcgw_00001.mp4",
            "next_required_gate": "audio_or_transcript_confirmation",
            "canonical_speaker": "Gemini",
            "speaker_display_name": "Gemini",
            "abs_start_sec": 80.0,
            "abs_end_sec": 84.0,
            "claim_type": "accusation",
            "strategic_function": "accuse",
            "claim_target_players": ["baile"],
            "claim_target_display_names": ["白乐"],
            "qwen_transcript_to_confirm": "白乐刚才在右边。",
        }
    )

    assert task["review_task_id"] == "audio_mcgw_00001_u001"
    assert task["primary_video_file"] == "clips/mcgw_00001.mp4"
    assert task["acceptance_policy"]["qwen_only_is_not_human_gold"] is True
    assert task["acceptance_policy"]["human_gold_requires_later_merge_gate"] is True


def test_audio_confirmation_audit_blocks_direct_human_gold_claim(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "audio_result.json"
    result_path.write_text(
        json.dumps(
            {
                "review_task_id": "audio_mcgw_00001_u001",
                "video_file": "clips/mcgw_00001.mp4",
                "parse_ok": True,
                "source_task": {"source_review_item_id": "mcgw_00001_u001"},
                "parsed": {
                    "audio_confirmation_decision": "confirm_candidate",
                    "evidence_quality": "high",
                    "confirmed_speaker": {
                        "canonical_speaker": "Gemini",
                        "speaker_confidence": "high",
                    },
                    "transcript": {
                        "candidate_text": "白乐刚才在右边。",
                        "transcript_match": "exact",
                        "text_confidence": "high",
                    },
                    "claim_grounding": {
                        "claim_type": "accusation",
                        "strategic_function": "accuse",
                        "claim_target_players": ["baile"],
                    },
                    "gold_policy": {
                        "eligible_for_human_gold_merge": True,
                        "needs_human_review": False,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    row, merge_candidate = audio_confirmation_audit.audit_result(result_path)

    assert "qwen_claims_direct_human_gold_eligibility" in row["issues"]
    assert "needs_human_review_not_true" in row["issues"]
    assert row["can_enter_codex_human_gold_merge_gate"] is False
    assert row["promotion_to_human_verified_gold"] is False
    assert merge_candidate is None


def test_audio_confirmation_audit_allows_only_merge_gate_candidate(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "audio_result.json"
    result_path.write_text(
        json.dumps(
            {
                "review_task_id": "audio_mcgw_00002_u001",
                "video_file": "clips/mcgw_00002.mp4",
                "parse_ok": True,
                "source_task": {"source_review_item_id": "mcgw_00002_u001"},
                "parsed": {
                    "audio_confirmation_decision": "correct_transcript",
                    "evidence_quality": "medium",
                    "confirmed_speaker": {
                        "speaker_display_name": "末将",
                        "canonical_speaker": "mojiang",
                        "speaker_confidence": "medium",
                    },
                    "transcript": {
                        "candidate_text": "我不在现场。",
                        "transcript_match": "minor_correction",
                        "corrected_text": "我当时不在现场。",
                        "text_confidence": "medium",
                    },
                    "claim_grounding": {
                        "claim_type": "defense",
                        "strategic_function": "defend_self",
                        "claim_target_players": [],
                        "claim_target_display_names": [],
                    },
                    "gold_policy": {
                        "eligible_for_human_gold_merge": False,
                        "needs_human_review": True,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    row, merge_candidate = audio_confirmation_audit.audit_result(result_path)

    assert row["issues"] == []
    assert row["can_enter_codex_human_gold_merge_gate"] is True
    assert row["promotion_to_human_verified_gold"] is False
    assert merge_candidate is not None
    assert (
        merge_candidate["gold_policy"]["requires_codex_human_gold_merge_gate"] is True
    )
    assert merge_candidate["gold_policy"]["promotion_to_human_verified_gold"] is False


def test_meeting_claim_gold_merge_pack_requires_explicit_accept() -> None:
    item = meeting_claim_gold_merge.build_merge_item(
        {
            "merge_candidate_id": "audio_mcgw_00002_u001",
            "source_review_item_id": "mcgw_00002_u001",
            "result_file": "results/audio_mcgw_00002_u001.json",
            "video_file": "clips/mcgw_00002.mp4",
            "confirmed_speaker": {"canonical_speaker": "mojiang"},
            "transcript": {
                "corrected_text": "我当时不在现场。",
                "transcript_match": "minor_correction",
                "text_confidence": "medium",
            },
            "claim_grounding": {
                "claim_type": "vote_suggestion",
                "strategic_function": "coordinate_vote",
                "claim_target_players": ["baile"],
            },
            "evidence_quality": "medium",
        },
        1,
    )

    assert item["merge_policy"]["promotion_to_human_verified_gold"] is False
    assert item["merge_policy"]["requires_explicit_accept_record"] is True
    assert "D_perspective_taking_prediction" in item["suggested_probe_uses"]
    assert "vote_influence" in item["suggested_probe_uses"]


def test_sync_audio_confirmation_queue_filters_rejected_and_unknown_speaker(
    tmp_path: Path,
) -> None:
    records = [
        {
            "review_item_id": "keep_audio",
            "source_result_id": "mcgw_00001",
            "video_file": "clips/1.mp4",
            "contact_sheet": "contacts/1.jpg",
            "next_required_gate": "audio_or_transcript_confirmation",
            "codex_human_decision": "needs_audio_confirmation",
            "candidate_utterance": {
                "canonical_speaker": "Gemini",
                "speaker_display_name": "Gemini",
                "utterance_text": "我看到了白乐。",
            },
        },
        {
            "review_item_id": "keep_alias",
            "source_result_id": "mcgw_00002",
            "video_file": "clips/2.mp4",
            "contact_sheet": "contacts/2.jpg",
            "next_required_gate": "alias_mapping_and_audio_confirmation",
            "codex_human_decision": "needs_alias_and_audio_confirmation",
            "candidate_utterance": {
                "canonical_speaker": "mojiang",
                "speaker_display_name": "末将",
                "utterance_text": "白乐可以出。",
                "claim_target_players": ["baile"],
            },
        },
        {
            "review_item_id": "drop_unknown",
            "source_result_id": "mcgw_00003",
            "video_file": "clips/3.mp4",
            "contact_sheet": "contacts/3.jpg",
            "next_required_gate": "speaker_identity_and_audio_confirmation",
            "codex_human_decision": "needs_speaker_and_audio_confirmation",
            "candidate_utterance": {"canonical_speaker": "unknown"},
        },
        {
            "review_item_id": "drop_reject",
            "source_result_id": "mcgw_00004",
            "video_file": "clips/4.mp4",
            "contact_sheet": "contacts/4.jpg",
            "next_required_gate": "none_rejected",
            "codex_human_decision": "reject_for_gold",
            "candidate_utterance": {},
        },
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )

    summary = meeting_claim_sync.refresh_audio_confirmation_queue(
        records_path, tmp_path / "out"
    )
    queue = read_jsonl(tmp_path / "out" / "audio_confirmation_queue.jsonl")

    assert summary["audio_confirmation_items"] == 2
    assert {row["audio_confirmation_item_id"] for row in queue} == {
        "keep_audio",
        "keep_alias",
    }
    assert summary["promotion_to_human_verified_gold"] is False


def test_gold_merge_review_record_never_promotes_qwen_checked_candidate() -> None:
    record = gold_merge_review_records.review_item(
        {
            "merge_review_item_id": "mcg_merge_00001",
            "source_merge_candidate_id": "audio_mcgw_00001_u001",
            "source_review_item_id": "mcgw_00001_u001",
            "source_result_file": "results/audio_mcgw_00001_u001.json",
            "video_file": "clips/mcgw_00001.mp4",
            "confirmed_speaker": {
                "canonical_speaker": "mojiang",
                "speaker_confidence": "high",
            },
            "confirmed_transcript": {
                "text": "我当时不在现场。",
                "transcript_match": "exact",
                "text_confidence": "high",
            },
            "claim_grounding": {
                "claim_type": "defense",
                "strategic_function": "defend_self",
                "claim_target_players": [],
            },
            "suggested_probe_uses": [
                "claim_truth_vs_claim_awareness",
                "D_perspective_taking_prediction",
            ],
            "remaining_uncertainties": [],
        }
    )

    assert (
        record["codex_human_merge_decision"]
        == "accept_for_qwen_checked_merge_candidate"
    )
    assert record["safe_for_probe_draft_generation"] is True
    assert record["promotion_to_human_verified_gold"] is False
    assert (
        record["next_required_gate"]
        == "codex_human_audio_spotcheck_or_independent_transcript_before_human_verified"
    )


def test_meeting_claim_probe_drafts_are_qwen_checked_only() -> None:
    record = {
        "merge_review_item_id": "mcg_merge_00001",
        "source_review_item_id": "mcgw_00001_u001",
        "source_result_file": "results/audio_mcgw_00001_u001.json",
        "video_file": "clips/mcgw_00001.mp4",
        "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
        "safe_for_probe_draft_generation": True,
        "confirmed_speaker": {"canonical_speaker": "mojiang"},
        "confirmed_transcript": {
            "text": "我当时不在现场。",
            "transcript_match": "exact",
            "text_confidence": "high",
        },
        "claim_grounding": {
            "claim_type": "defense",
            "strategic_function": "defend_self",
            "claim_target_players": [],
            "claim_target_display_names": [],
        },
        "suggested_probe_uses": [
            "claim_truth_vs_claim_awareness",
            "D_perspective_taking_prediction",
        ],
    }

    group = meeting_claim_probe_drafts.build_group(record, 1)
    probes = meeting_claim_probe_drafts.build_probes(group)
    hidden_gold = meeting_claim_probe_drafts.build_hidden_gold(group, probes)

    assert group["quality"]["audio_confirmation"] == "qwen_checked"
    assert group["quality"]["promotion_to_human_verified_gold"] is False
    assert group["needs_human_review"] is True
    assert {probe["probe_type"] for probe in probes} == {
        "A_pre_reveal_belief",
        "B_post_reveal_reconstruct_previous_belief",
        "C_other_agent_false_belief",
        "D_perspective_taking_prediction",
    }
    assert group["template"] == "meeting_claim_audio_confirmed"
    assert group["target_player"] == "mojiang"
    assert all(probe["target_player"] == "mojiang" for probe in probes)
    assert all(probe["cutoff_abs_sec"] == 0.0 for probe in probes)
    assert (
        "QUERY_VARIABLE_PUBLIC_FORM_JSON"
        in next(
            probe for probe in probes if probe["probe_type"] == "A_pre_reveal_belief"
        )["prompt"]
    )
    d_prompt = next(
        probe
        for probe in probes
        if probe["probe_type"] == "D_perspective_taking_prediction"
    )["prompt"]
    assert "SPEAKER_AVAILABLE_CONTEXT_JSON" in d_prompt
    assert "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON" in d_prompt
    assert all(probe["gold_source"] == "qwen_checked" for probe in probes)
    assert all(probe["promotion_to_human_verified_gold"] is False for probe in probes)
    assert all(row["promotion_to_human_verified_gold"] is False for row in hidden_gold)
    assert all(row["D_PT_reference_enriched_by"] for row in hidden_gold)
