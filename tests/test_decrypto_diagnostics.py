from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gooseomni.benchmark.decrypto_diagnostics import (
    PLAYERS,
    build_claim_truth_links,
    build_decrypto_diagnostics,
    build_oracle_ledger,
    build_visibility_edges,
    canonicalize_events,
    export_gooseomni_benchmark,
    read_json,
    read_jsonl,
    score_decrypto_diagnostics,
    select_probe_groups,
    validate_decrypto_outputs,
    write_jsonl,
)
from tools.build.build_decrypto_human_verified_subset import merge_human_verified_passes
from tools.internal.review.build_claim_truth_extra_targets import edge_lookup, validate_claim_truth_spec
import tools.internal.review.build_behavior_outcome_d_candidates as behavior_outcome_d
import tools.internal.review.build_decrypto_clip_review_pack as clip_review
import tools.internal.review.build_claim_truth_visual_precheck_pack as claim_truth_precheck
import tools.internal.review.audit_meeting_claim_audio_confirmation_outputs as audio_confirmation_audit
import tools.internal.review.build_combined_audio_confirmation_results as combined_audio_results
import tools.internal.review.build_meeting_claim_audio_confirmation_pack as audio_confirmation_pack
import tools.internal.review.build_meeting_claim_gold_merge_pack as meeting_claim_gold_merge
import tools.internal.review.build_meeting_claim_human_gold_accept_candidates as human_gold_accept_candidates
import tools.internal.review.build_meeting_claim_scope_limited_accept_candidates as scope_limited_accept
import tools.internal.review.build_meeting_claim_human_verified_promotion_pass as meeting_claim_promotion
import tools.internal.review.build_meeting_claim_probe_drafts as meeting_claim_probe_drafts
import tools.internal.review.report_meeting_claim_benchmark_progress as meeting_claim_progress_report
import tools.internal.review.report_meeting_claim_total_progress as meeting_claim_total_progress
import tools.internal.review.report_scope_limited_gold_release as scope_release_report
import tools.internal.review.sync_meeting_claim_extension_state as meeting_claim_sync
import tools.internal.review.write_meeting_claim_gold_merge_review_records as gold_merge_review_records
from tools.annotation import run_decrypto_high_quality_review as high_quality_review
from tools.annotation import submit_gooseomni_oracle_jobs as submit_oracle_jobs
from tools.internal.review.build_delayed_reveal_from_verified_anchors import validate_delayed_reveal_spec
from tools.internal.review.build_delayed_reveal_visual_precheck_pack import compose_quad, load_skipped_clusters


def _write_gold_annotation(
    root: Path,
    phase_id: str,
    player_id: str,
    phase_type: str,
    observations: list[dict],
    utterances: list[dict],
) -> None:
    path = root / "gold_annotations" / "g001" / phase_id / f"{player_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "gooseomni",
        "version": "release_benchmark_v2",
        "game_id": "g001",
        "episode_id": "g001_episode_000",
        "episode_index": 0,
        "phase_id": phase_id,
        "phase_type": phase_type,
        "phase_index_global": 0 if phase_type == "gameplay" else 1,
        "phase_index_in_episode": 0 if phase_type == "gameplay" else 1,
        "phase_order_label_zh": "第1局第1次跑动过程" if phase_type == "gameplay" else "第1局第1次会议",
        "gameplay_round_index": 1 if phase_type == "gameplay" else None,
        "meeting_round_index": 1 if phase_type == "meeting" else None,
        "previous_phase_id": None,
        "next_phase_id": None,
        "player_id": player_id,
        "video_file": f"inputs/videos/g001/{phase_id}/{player_id}.mp4",
        "annotation_file": f"gold_annotations/g001/{phase_id}/{player_id}.json",
        "aligned_start_sec": 400.0 if phase_type == "gameplay" else 418.0,
        "aligned_end_sec": 490.0 if phase_type == "gameplay" else 508.0,
        "duration_sec": 90.0,
        "observations": observations,
        "utterances": utterances,
        "player_status": {},
        "role_and_goal": {},
        "private_memory": [],
        "belief_state": [],
        "tom_questions": [],
        "needs_human_review": False,
        "review_reasons": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _release_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "release_benchmark_v2"
    gameplay = "g001_phase_000_gameplay_000400_000490"
    meeting = "g001_phase_001_meeting_000490_000580"
    obs = {
        "abs_start_sec": 410.0,
        "abs_end_sec": 414.0,
        "event_type": "route_near_body",
        "actor": "baile",
        "location": "right hallway",
        "description": "mojiang sees baile pass near the right-side body area.",
        "source_types": ["direct_visual_observation"],
        "evidence": "baile is visible near body area in mojiang POV",
        "certainty": 0.9,
        "needs_human_review": False,
    }
    duplicate_obs = dict(obs, abs_start_sec=411.0, abs_end_sec=414.5)
    for player in PLAYERS:
        observations = [obs, duplicate_obs] if player == "mojiang" else []
        _write_gold_annotation(root, gameplay, player, "gameplay", observations, [])

    utterance = {
        "abs_start_sec": 418.2,
        "abs_end_sec": 423.5,
        "speaker": "baile",
        "transcript": "我刚才一直在下面，没有去过右边。",
        "claim_text": "我刚才一直在下面，没有去过右边。",
        "certainty": 0.85,
        "needs_human_review": False,
    }
    for player in PLAYERS:
        _write_gold_annotation(root, meeting, player, "meeting", [], [utterance if player == "Gemini" else dict(utterance)])
    return root


def test_overlap_event_dedup() -> None:
    candidates = [
        {
            "local_event_id": "seg4_mojiang_e1",
            "game_id": "g001",
            "source_segment_ids": ["seg4"],
            "source_povs": ["mojiang"],
            "abs_start_sec": 410.0,
            "abs_end_sec": 414.0,
            "phase_type": "gameplay",
            "event_type": "route_near_body",
            "actors": ["baile"],
            "patients": [],
            "location": "right hallway",
            "description": "baile near body",
            "direct_visual_evidence": ["seen"],
            "direct_audio_evidence": [],
            "public_evidence": [],
            "inferred_fields": [],
            "certainty": 0.8,
            "needs_human_review": False,
        },
        {
            "local_event_id": "seg5_mojiang_e2",
            "game_id": "g001",
            "source_segment_ids": ["seg5"],
            "source_povs": ["mojiang"],
            "abs_start_sec": 411.0,
            "abs_end_sec": 414.5,
            "phase_type": "gameplay",
            "event_type": "route_near_body",
            "actors": ["baile"],
            "patients": [],
            "location": "right hallway",
            "description": "baile near body duplicate",
            "direct_visual_evidence": ["seen again"],
            "direct_audio_evidence": [],
            "public_evidence": [],
            "inferred_fields": [],
            "certainty": 0.9,
            "needs_human_review": False,
        },
    ]

    events, maps, _ = canonicalize_events(candidates)

    assert len(events) == 1
    assert maps[0]["duplicate_local_event_ids"] == ["seg4_mojiang_e1", "seg5_mojiang_e2"]


def test_visibility_edges_cover_six_players() -> None:
    event = {
        "world_event_id": "ge_000001",
        "game_id": "g001",
        "source_povs": ["mojiang"],
        "phase_type": "gameplay",
        "public_evidence": [],
        "abs_end_sec": 414.0,
        "certainty": 0.9,
    }

    edges = build_visibility_edges([event], [])

    assert len(edges) == 6
    assert {edge["player_id"] for edge in edges} == set(PLAYERS)
    assert next(edge for edge in edges if edge["player_id"] == "mojiang")["visibility"] == "direct_visual"
    assert next(edge for edge in edges if edge["player_id"] == "saoyi")["visibility"] == "not_visible"


def test_claim_truth_linking_detects_contradicted_alibi() -> None:
    event = {
        "world_event_id": "ge_000001",
        "game_id": "g001",
        "source_povs": ["mojiang"],
        "phase_type": "gameplay",
        "event_type": "route_near_body",
        "actors": ["baile"],
        "location": "右边",
        "description": "baile 经过右边尸体附近",
        "abs_start_sec": 410.0,
        "abs_end_sec": 414.0,
        "certainty": 0.9,
        "public_evidence": [],
    }
    claim = {
        "claim_id": "claim_000001",
        "game_id": "g001",
        "speaker": "baile",
        "heard_by": PLAYERS[:],
        "abs_start_sec": 418.0,
        "abs_end_sec": 423.0,
        "claim_type": "location",
        "content": "我刚才一直在下面，没有去过右边。",
        "normalized_content": "我刚才一直在下面，没有去过右边。",
        "target_entities": [],
        "certainty": 0.8,
    }
    edges = build_visibility_edges([event], [claim])

    links = build_claim_truth_links([claim], [event], edges)

    assert links
    assert links[0]["truth_status_global"] == "contradicted"
    assert links[0]["local_awareness_by_player"]["saoyi"] == "not_enough_information"
    assert links[0]["local_awareness_by_player"]["mojiang"] == "has_contradictory_visual_evidence"


def test_manual_claim_truth_spec_rejects_weak_or_visible_anchor() -> None:
    task_event = {
        "world_event_id": "ge_task",
        "game_id": "g001",
        "source_povs": ["Gemini"],
        "phase_type": "gameplay",
        "event_type": "task",
        "actors": ["Gemini"],
        "location": "游戏厅",
        "description": "Gemini在游戏厅内执行任务。",
        "abs_start_sec": 1700.0,
        "abs_end_sec": 1710.0,
        "certainty": 0.85,
        "needs_human_review": False,
        "public_evidence": [],
    }
    clean_event = {
        **task_event,
        "world_event_id": "ge_move",
        "event_type": "movement",
        "description": "Gemini 在下水道移动。",
        "location": "下水道",
    }
    residual_ui_event = {
        **clean_event,
        "world_event_id": "ge_residual_ui",
        "source_segment_ids": ["g001_phase_026_gameplay_006400_006490"],
        "abs_start_sec": 6400.0,
        "abs_end_sec": 6410.0,
    }
    claim = {
        "claim_id": "claim_route",
        "game_id": "g001",
        "speaker": "Gemini",
        "heard_by": PLAYERS[:],
        "abs_start_sec": 1890.0,
        "abs_end_sec": 1920.0,
        "claim_type": "location",
        "content": "我这一轮一直在学校这块，没有去过下水道。",
        "certainty": 1.0,
    }
    edges = edge_lookup(build_visibility_edges([task_event, clean_event, residual_ui_event], [claim]))

    task_errors = validate_claim_truth_spec(task_event, claim, "baile", "contradicted", edges)
    visible_errors = validate_claim_truth_spec(clean_event, claim, "Gemini", "contradicted", edges)
    clean_errors = validate_claim_truth_spec(clean_event, claim, "baile", "contradicted", edges)
    residual_errors = validate_claim_truth_spec(residual_ui_event, claim, "baile", "contradicted", edges)

    assert "contradicted_anchor_event_type_too_weak:task" in task_errors
    assert "anchor_description_has_ui_or_weak_semantics" in task_errors
    assert "target_anchor_visibility_not_hidden:direct_visual" in visible_errors
    assert clean_errors == []
    assert "anchor_event_too_close_to_gameplay_phase_start:0.0s" in residual_errors


def test_delayed_reveal_spec_requires_delay_and_semantic_match() -> None:
    event = {
        "world_event_id": "ge_reveal",
        "source_povs": ["baile"],
        "actors": ["baile"],
        "patients": ["Gemini"],
        "description": "Gemini 被 baile 的角色杀死。",
        "location": "街道",
        "abs_end_sec": 6170.0,
    }
    too_close_claim = {
        "claim_id": "claim_close",
        "speaker": "baile",
        "heard_by": PLAYERS[:],
        "abs_start_sec": 6180.0,
        "content": "我杀了一个人。",
        "normalized_content": "baile 说自己杀了一个人。",
    }
    mismatch_claim = {
        "claim_id": "claim_mismatch",
        "speaker": "saoyi",
        "heard_by": PLAYERS[:],
        "claim_type": "accusation",
        "abs_start_sec": 6300.0,
        "content": "牛牛死亡，所以不是路姐刀的。",
        "normalized_content": "讨论牛牛死亡。",
    }
    good_claim = {
        "claim_id": "claim_good",
        "speaker": "baile",
        "heard_by": PLAYERS[:],
        "claim_type": "accusation",
        "abs_start_sec": 6300.0,
        "content": "我杀了 Gemini。",
        "normalized_content": "baile 说自己杀了 Gemini。",
    }
    role_claim = {
        **good_claim,
        "claim_id": "claim_role",
        "claim_type": "role",
    }
    victim_claim = {
        **good_claim,
        "claim_id": "claim_victim",
        "content": "一刀下去，六将军人头落地。",
        "normalized_content": "baile声称自己一刀杀死了六将军。",
    }
    victim_event = {
        **event,
        "description": "baile 的角色在街道上被击杀。",
        "actors": [],
        "patients": [],
        "source_povs": ["saoyi"],
    }
    residual_anchor_event = {
        **event,
        "source_segment_ids": ["g001_phase_026_gameplay_006400_006490"],
        "phase_type": "gameplay",
        "abs_start_sec": 6400.0,
        "abs_end_sec": 6410.0,
    }
    residual_claim = {
        **good_claim,
        "claim_id": "claim_residual",
        "source_segment_ids": ["g001_phase_027_meeting_006490_006650"],
        "abs_start_sec": 6490.0,
        "abs_end_sec": 6500.0,
    }

    assert "reveal_too_close_to_event:10.0s" in validate_delayed_reveal_spec(event, "xiaolu", too_close_claim)
    assert "claim_does_not_semantically_match_event" in validate_delayed_reveal_spec(event, "xiaolu", mismatch_claim)
    assert "claim_type_not_reveal:role" in validate_delayed_reveal_spec(event, "xiaolu", role_claim)
    assert "claim_does_not_semantically_match_event" in validate_delayed_reveal_spec(victim_event, "Gemini", victim_claim)
    assert "anchor_event_too_close_to_gameplay_phase_start:0.0s" in validate_delayed_reveal_spec(residual_anchor_event, "xiaolu", good_claim)
    assert "reveal_claim_too_close_to_phase_start:0.0s" in validate_delayed_reveal_spec(event, "xiaolu", residual_claim)
    assert validate_delayed_reveal_spec(event, "xiaolu", good_claim) == []


def test_clip_review_window_respects_phase_bounds() -> None:
    phase_id = "g001_phase_026_gameplay_006400_006490"

    local_start, duration = clip_review.clip_window(phase_id, 6401.0, pre_sec=5.0, post_sec=7.0)
    late_start, late_duration = clip_review.clip_window(phase_id, 6488.0, pre_sec=5.0, post_sec=7.0)

    assert local_start == 0.0
    assert duration == 8.0
    assert late_start == 83.0
    assert late_duration == 7.0


def test_high_quality_review_uses_explicit_primary_and_context_video(tmp_path: Path) -> None:
    primary = tmp_path / "primary.mp4"
    primary.write_bytes(b"fake")
    task = {
        "review_task_id": "clip_review_001",
        "primary_video_file": primary.as_posix(),
        "context_video_files": ["context_a.mp4", "context_b.mp4"],
    }

    resolved = high_quality_review.resolve_video(task, tmp_path / "release")
    prompt = high_quality_review.build_prompt("TEMPLATE", task, resolved)

    assert resolved == primary
    assert "context_a.mp4" in prompt
    assert "context_b.mp4" in prompt
    assert "TASK_JSON" in prompt


def test_end_to_end_probe_generation_export_validation_and_scoring(tmp_path: Path) -> None:
    release_root = _release_fixture(tmp_path)
    annotation_root = tmp_path / "annotations"
    benchmark_root = tmp_path / "benchmark" / "gooseomni_v1"

    ledger_counts = build_oracle_ledger(release_root, annotation_root)
    diag_counts = build_decrypto_diagnostics(annotation_root, annotation_root, limit=12)
    export_counts = export_gooseomni_benchmark(annotation_root, benchmark_root)
    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert ledger_counts["world_events"] == 1
    assert ledger_counts["claims"] == 1
    assert ledger_counts["claim_truth_links"] == 1
    assert diag_counts["probe_groups"] >= 1
    assert diag_counts["A"] == diag_counts["probe_groups"]
    assert diag_counts["B"] == diag_counts["probe_groups"]
    assert diag_counts["C"] == diag_counts["probe_groups"]
    assert diag_counts["D"] >= 1
    assert export_counts["static_trials"] >= 4
    assert validation["ok"], validation
    assert all(row["gold_source"] == row["hidden_gold"]["gold_source"] for row in read_jsonl(benchmark_root / "static_trials" / "hidden_gold.jsonl"))

    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    templates = {group["template"] for group in groups}
    assert any(group["template"] == "contradicted_alibi" for group in groups)
    assert "private_witness" in templates
    assert "vote_influence" in templates
    assert "delayed_public_reveal" in templates
    assert all(group["target_player"] != "mojiang" for group in groups if group["template"] == "contradicted_alibi")
    assert all(group["hidden_event_ids_for_target"] for group in groups if group["template"] == "contradicted_alibi")

    trials_text = (benchmark_root / "static_trials" / "trials.jsonl").read_text(encoding="utf-8")
    assert "hidden_gold" not in trials_text
    assert "forbidden_event_ids" not in trials_text

    prompts = read_jsonl(benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
    for prompt in prompts:
        if prompt["probe_type"] == "A_pre_reveal_belief":
            assert "QUERY_VARIABLE_PUBLIC_FORM_JSON" in prompt["prompt"]
            assert "ORACLE_TRUTH_JSON" not in prompt["prompt"]
            assert not any(event_id in prompt["prompt"] for event_id in prompt.get("forbidden_event_ids", []))
        if prompt["probe_type"] == "B_post_reveal_reconstruct_previous_belief":
            assert "answer to probe A" not in prompt["prompt"]
        if prompt["probe_type"] == "D_perspective_taking_prediction":
            assert "SPEAKER_AVAILABLE_CONTEXT_JSON" in prompt["prompt"]
            assert "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON" in prompt["prompt"]
            assert "TARGET_LISTENER_CONTEXT_JSON" not in prompt["prompt"]

    group_id = prompts[0]["probe_group_id"]
    response_rows = [
        {
            "probe_group_id": group_id,
            "probe_type": "A_pre_reveal_belief",
            "parsed": {
                "knows_truth": False,
                "belief_label": "does_not_know",
                "likely_belief": "saoyi only heard the claim.",
            },
        },
        {
            "probe_group_id": group_id,
            "probe_type": "B_post_reveal_reconstruct_previous_belief",
            "parsed": {
                "target_knew_truth_at_cutoff": False,
                "reconstructed_prior_belief": "does_not_know: saoyi only heard the claim.",
            },
        },
        {
            "probe_group_id": group_id,
            "probe_type": "C_other_agent_false_belief",
            "parsed": {
                "other_player_knew_truth_at_cutoff": False,
                "other_player_likely_belief": "uncertain from their own perspective",
            },
        },
    ]
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, response_rows)
    score_out = tmp_path / "scores.json"

    aggregate = score_decrypto_diagnostics(responses, annotation_root / "diagnostics" / "hidden_gold.jsonl", score_out)
    scores = read_json(score_out)["scores"]

    assert aggregate["groups_scored"] == 1
    assert scores[0]["RC_weak"] is True
    assert scores[0]["FB_weak"] is True


def test_hidden_event_selector_selects_non_visible_target() -> None:
    event = {
        "world_event_id": "ge_000001",
        "game_id": "g001",
        "source_segment_ids": ["seg"],
        "source_povs": ["mojiang"],
        "abs_start_sec": 410.0,
        "abs_end_sec": 414.0,
        "phase_type": "gameplay",
        "event_type": "route_near_body",
        "actors": ["baile"],
        "location": "right",
        "description": "baile near body",
        "certainty": 0.9,
        "needs_human_review": False,
    }
    ledger = {
        "world_events": [event],
        "claims": [],
        "claim_truth_links": [],
        "visibility_edges": build_visibility_edges([event], []),
    }

    groups = select_probe_groups(ledger, limit=3)
    hidden_groups = [group for group in groups if group["template"] == "hidden_event_awareness"]

    assert hidden_groups
    assert all("ge_000001" in group["hidden_event_ids_for_target"] for group in hidden_groups)
    assert all(group["target_player"] != "mojiang" for group in hidden_groups)


def test_scoring_claim_verification_and_forbidden_evidence(tmp_path: Path) -> None:
    hidden_gold = tmp_path / "hidden_gold.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "scores.json"
    write_jsonl(
        hidden_gold,
        [
            {
                "probe_group_id": "pg_ok",
                "forbidden_event_ids_for_target": ["ge_hidden"],
                "acceptable_evidence_ids_for_target": ["claim_public"],
                "claim_truth_global": "contradicted",
                "claim_awareness_local_target": "not_enough_information",
            },
            {
                "probe_group_id": "pg_leak",
                "forbidden_event_ids_for_target": ["ge_secret"],
                "acceptable_evidence_ids_for_target": ["claim_allowed"],
                "claim_truth_global": "supported",
                "claim_awareness_local_target": "has_evidence",
            },
        ],
    )
    write_jsonl(
        responses,
        [
            {
                "probe_group_id": "pg_ok",
                "probe_type": "A_pre_reveal_belief",
                "parsed": {
                    "knows_truth": False,
                    "belief_label": "does_not_know",
                    "claim_truth_global": "contradicted",
                    "claim_awareness_local_target": "not_enough_information",
                    "evidence_ids": ["claim_public"],
                },
            },
            {
                "probe_group_id": "pg_ok",
                "probe_type": "B_post_reveal_reconstruct_previous_belief",
                "parsed": {
                    "target_knew_truth_at_cutoff": False,
                    "reconstructed_prior_belief": "does_not_know",
                },
            },
            {
                "probe_group_id": "pg_ok",
                "probe_type": "C_other_agent_false_belief",
                "parsed": {
                    "other_player_knew_truth_at_cutoff": False,
                    "other_player_likely_belief": "uncertain",
                },
            },
            {
                "probe_group_id": "pg_leak",
                "probe_type": "A_pre_reveal_belief",
                "parsed": {
                    "knows_truth": False,
                    "belief_label": "uncertain",
                    "claim_truth_global": "contradicted",
                    "target_has_evidence_for_claim_truth": False,
                    "evidence_ids": ["ge_secret", "claim_not_allowed"],
                },
            },
        ],
    )

    aggregate = score_decrypto_diagnostics(responses, hidden_gold, output)
    rows = {row["probe_group_id"]: row for row in read_json(output)["scores"]}

    assert rows["pg_ok"]["claim_verification_global"] is True
    assert rows["pg_ok"]["claim_verification_local"] is True
    assert rows["pg_ok"]["evidence_support"] is True
    assert rows["pg_leak"]["claim_verification_global"] is False
    assert rows["pg_leak"]["claim_verification_local"] is False
    assert rows["pg_leak"]["perspective_leakage"] is True
    assert rows["pg_leak"]["forbidden_evidence_usage"] is True
    assert rows["pg_leak"]["evidence_support"] is False
    assert aggregate["claim_verification_global"] == 0.5
    assert aggregate["claim_verification_local"] == 0.5
    assert aggregate["forbidden_evidence_usage_rate"] == 0.5
    assert aggregate["evidence_support_rate"] == 0.5


def test_scoring_parse_and_schema_validation_failures(tmp_path: Path) -> None:
    hidden_gold = tmp_path / "hidden_gold.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "scores.json"
    write_jsonl(
        hidden_gold,
        [
            {
                "probe_group_id": "pg_bad",
                "forbidden_event_ids_for_target": [],
                "claim_truth_global": "unverified",
                "claim_awareness_local_target": "unknown",
            }
        ],
    )
    write_jsonl(
        responses,
        [
            {
                "probe_group_id": "pg_bad",
                "probe_type": "A_pre_reveal_belief",
                "raw_response": "not json",
            }
        ],
    )

    aggregate = score_decrypto_diagnostics(responses, hidden_gold, output)
    row = read_json(output)["scores"][0]

    assert aggregate["json_parse_success"] == 0.0
    assert aggregate["schema_validation_success"] == 0.0
    assert row["json_parse_success"] is False
    assert row["schema_validation_success"] is False


def test_scoring_d_probe_uses_enriched_speaker_listener_gold(tmp_path: Path) -> None:
    hidden_gold = tmp_path / "hidden_gold.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "scores.json"
    write_jsonl(
        hidden_gold,
        [
            {
                "probe_group_id": "pg_d",
                "forbidden_event_ids_for_target": [],
                "D_PT_reference": {
                    "speaker": "baile",
                    "listener": "Gemini",
                    "expected_information_state_difference": True,
                },
            }
        ],
    )
    write_jsonl(
        responses,
        [
            {
                "probe_group_id": "pg_d",
                "probe_type": "D_perspective_taking_prediction",
                "parsed": {
                    "speaker": "baile",
                    "listener": "Gemini",
                    "predicted_listener_trust_update": "unchanged",
                    "predicted_listener_next_action": "ignore",
                },
            }
        ],
    )

    aggregate = score_decrypto_diagnostics(responses, hidden_gold, output)
    row = read_json(output)["scores"][0]

    assert row["PT_weak"] is True
    assert row["PT_strong"] is True
    assert aggregate["groups_scored"] == 1

    write_jsonl(
        responses,
        [
            {
                "probe_group_id": "pg_d",
                "probe_type": "D_perspective_taking_prediction",
                "parsed": {
                    "speaker": "baile",
                    "listener": "saoyi",
                    "predicted_listener_trust_update": "unchanged",
                    "predicted_listener_next_action": "ignore",
                },
            }
        ],
    )
    score_decrypto_diagnostics(responses, hidden_gold, output)
    row = read_json(output)["scores"][0]

    assert row["PT_weak"] is False
    assert row["PT_strong"] is False


def test_validator_rejects_unsafe_d_prompt(tmp_path: Path) -> None:
    release_root = _release_fixture(tmp_path)
    annotation_root = tmp_path / "annotations"
    benchmark_root = tmp_path / "benchmark" / "gooseomni_v1"

    build_oracle_ledger(release_root, annotation_root)
    build_decrypto_diagnostics(annotation_root, annotation_root, limit=12)
    export_gooseomni_benchmark(annotation_root, benchmark_root)

    prompts_path = benchmark_root / "interactive_diagnostics" / "prompts.jsonl"
    prompts = read_jsonl(prompts_path)
    for prompt in prompts:
        if prompt["probe_type"] == "D_perspective_taking_prediction":
            prompt["prompt"] += "\nTARGET_LISTENER_CONTEXT_JSON:\n{}"
            break
    write_jsonl(prompts_path, prompts)

    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert not validation["ok"]
    assert any(issue["code"] == "D_prompt_contains_listener_private_context" for issue in validation["issues"])


def test_validator_requires_enriched_gold_for_human_verified_d(tmp_path: Path) -> None:
    release_root = _release_fixture(tmp_path)
    annotation_root = tmp_path / "annotations"
    benchmark_root = tmp_path / "benchmark" / "gooseomni_v1"

    build_oracle_ledger(release_root, annotation_root)
    build_decrypto_diagnostics(annotation_root, annotation_root, limit=12)
    export_gooseomni_benchmark(annotation_root, benchmark_root)

    d_group_id = read_jsonl(annotation_root / "diagnostics" / "probes_D_perspective_taking.jsonl")[0]["probe_group_id"]
    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    for group in groups:
        if group["probe_group_id"] == d_group_id:
            group["gold_source"] = "human_verified"
    write_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl", groups)

    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert not validation["ok"]
    assert any(issue["code"] == "human_verified_D_missing_enriched_gold" for issue in validation["issues"])

    hidden_rows = read_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl")
    for row in hidden_rows:
        if row["probe_group_id"] == d_group_id:
            row["D_PT_reference_enriched_by"] = "codex_human_reviewer"
            row["D_PT_reference"]["human_verified_scope"] = "information_state_difference_and_prompt_safety"
    write_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl", hidden_rows)

    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert validation["ok"], validation


def _write_minimal_verified_pass(pass_root: Path, group_id: str, anchor_event_id: str) -> None:
    diag = pass_root / "annotations" / "diagnostics"
    group = {
        "probe_group_id": group_id,
        "game_id": "g001",
        "source_segment_ids": ["seg"],
        "cutoff_abs_sec": 100.0,
        "target_player": "baile",
        "query_variable": {"type": "private_witness", "description": anchor_event_id},
        "anchor_event_ids": [anchor_event_id],
        "related_claim_ids": [],
        "hidden_event_ids_for_target": [anchor_event_id],
        "available_evidence_ids_for_target": [],
        "selection_reason": "fixture",
        "diagnostic_families": ["false_belief", "representational_change"],
        "template": "private_witness",
        "quality": {"needs_human_review": False},
        "needs_human_review": False,
        "gold_source": "human_verified",
    }
    probe_base = {
        "probe_group_id": group_id,
        "target_player": "baile",
        "cutoff_abs_sec": 100.0,
        "input_condition": "target_available_events",
        "prompt": "fixture",
        "expected_output_schema": {},
        "forbidden_event_ids": [anchor_event_id],
        "acceptable_evidence_ids": [],
        "gold_source": "human_verified",
    }
    write_jsonl(diag / "probe_groups.jsonl", [group])
    write_jsonl(diag / "hidden_gold.jsonl", [{"probe_group_id": group_id, "gold_source": "human_verified", "forbidden_event_ids_for_target": [anchor_event_id]}])
    write_jsonl(diag / "diagnostic_quality.jsonl", [{"probe_group_id": group_id, "needs_human_review": False, "diagnostic_score": 1.0}])
    write_jsonl(diag / "probes_A_pre_reveal.jsonl", [{**probe_base, "probe_id": f"{group_id}_A", "probe_type": "A_pre_reveal_belief"}])
    write_jsonl(diag / "probes_B_reconstruct.jsonl", [{**probe_base, "probe_id": f"{group_id}_B", "probe_type": "B_post_reveal_reconstruct_previous_belief"}])
    write_jsonl(diag / "probes_C_false_belief.jsonl", [{**probe_base, "probe_id": f"{group_id}_C", "probe_type": "C_other_agent_false_belief"}])
    write_jsonl(diag / "probes_D_perspective_taking.jsonl", [])


def test_human_verified_merge_preserves_distinct_groups_with_colliding_ids(tmp_path: Path) -> None:
    base = tmp_path / "base"
    oracle = base / "annotations" / "oracle_ledger"
    oracle.mkdir(parents=True)
    (oracle / "world_events.jsonl").write_text("", encoding="utf-8")
    pass_a = tmp_path / "pass_a"
    pass_b = tmp_path / "pass_b"
    output = tmp_path / "combined"
    _write_minimal_verified_pass(pass_a, "g001_pwit_000002_baile", "ge_000061")
    _write_minimal_verified_pass(pass_b, "g001_pwit_000002_baile", "ge_000585")

    summary = merge_human_verified_passes(base, [pass_a, pass_b], output)
    groups = read_jsonl(output / "annotations" / "diagnostics" / "probe_groups.jsonl")
    hidden = read_jsonl(output / "annotations" / "diagnostics" / "hidden_gold.jsonl")
    probes = read_jsonl(output / "annotations" / "diagnostics" / "probes_A_pre_reveal.jsonl")

    assert summary["counts"]["probe_groups"] == 2
    assert {tuple(row["anchor_event_ids"]) for row in groups} == {("ge_000061",), ("ge_000585",)}
    assert len({row["probe_group_id"] for row in groups}) == 2
    assert {row["probe_group_id"] for row in hidden} == {row["probe_group_id"] for row in groups}
    assert {row["probe_group_id"] for row in probes} == {row["probe_group_id"] for row in groups}
    assert len({row["probe_id"] for row in probes}) == 2


def test_delayed_reveal_visual_precheck_quad_generation(tmp_path: Path) -> None:
    image_paths = []
    for name, color in [
        ("anchor_source", "red"),
        ("anchor_target", "green"),
        ("reveal_speaker", "blue"),
        ("reveal_listener", "purple"),
    ]:
        path = tmp_path / f"{name}.jpg"
        Image.new("RGB", (64, 48), color).save(path)
        image_paths.append(path)

    row = {
        "idx": 1,
        "event_id": "ge_fixture",
        "claim_id": "claim_fixture",
        "anchor_source_frame": image_paths[0].as_posix(),
        "anchor_target_frame": image_paths[1].as_posix(),
        "reveal_speaker_frame": image_paths[2].as_posix(),
        "reveal_listener_frame": image_paths[3].as_posix(),
        "anchor_source_label": "source sees hidden event",
        "anchor_target_label": "target does not see event",
        "reveal_speaker_label": "speaker reveals claim",
        "reveal_listener_label": "listener hears claim",
        "event_description": "Gemini sees a body.",
        "claim_text": "Gemini says somebody died.",
    }
    output = tmp_path / "quad.jpg"

    assert compose_quad(row, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_delayed_reveal_visual_precheck_loads_skipped_clusters(tmp_path: Path) -> None:
    review_path = tmp_path / "review.jsonl"
    write_jsonl(
        review_path,
        [
            {"event_id": "ge_a", "claim_id": "claim_a"},
            {"cluster_key": "ge_b:claim_b"},
        ],
    )

    skipped = load_skipped_clusters([review_path, tmp_path / "missing.jsonl"])

    assert skipped == {("ge_a", "claim_a"), ("ge_b", "claim_b")}


def test_behavior_outcome_candidates_prefer_aligned_release_frames(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "ledger"
    write_jsonl(
        ledger / "claims.jsonl",
        [
            {
                "claim_id": "claim_fixture",
                "speaker": "baile",
                "heard_by": ["saoyi"],
                "claim_type": "accusation",
                "content": "我认为应该投 Gemini，因为他刚才行为很可疑。",
                "abs_start_sec": 90.0,
                "abs_end_sec": 100.0,
                "source_segment_ids": ["g001_phase_001_meeting_000080_000170"],
            }
        ],
    )
    write_jsonl(
        ledger / "world_events.jsonl",
        [
            {
                "world_event_id": "ge_vote_fixture",
                "game_id": "g001",
                "description": "saoyi 在投票中选择了 Gemini",
                "abs_start_sec": 120.0,
                "abs_end_sec": 121.0,
                "source_povs": ["saoyi"],
                "source_segment_ids": ["g001_phase_001_meeting_000080_000170"],
            }
        ],
    )

    calls = []

    def fake_extract_release_frame(release_video_root, phase_id, player, abs_sec, output):
        calls.append((release_video_root, phase_id, player, abs_sec, output))
        return {
            "ok": True,
            "video": f"{release_video_root}/{phase_id}/{player}.mp4",
            "phase_id": phase_id,
            "player": player,
            "abs_sec": abs_sec,
            "local_sec": abs_sec - behavior_outcome_d.phase_start(phase_id),
            "output": output.as_posix(),
        }

    monkeypatch.setattr(behavior_outcome_d, "extract_release_frame", fake_extract_release_frame)

    rows = behavior_outcome_d.build_candidates(
        ledger_root=ledger,
        clip_root=None,
        release_video_root=tmp_path / "release" / "inputs" / "videos" / "g001",
        output_root=tmp_path / "out",
        limit=10,
    )

    assert len(rows) == 1
    assert rows[0]["frame_source"] == "release_aligned_phase_video"
    assert rows[0]["claim_phase_id"] == "g001_phase_001_meeting_000080_000170"
    assert rows[0]["vote_phase_id"] == "g001_phase_001_meeting_000080_000170"
    assert rows[0]["claim_frame"]["local_sec"] == 20.0
    assert rows[0]["vote_frame"]["local_sec"] == 41.0
    assert [call[2] for call in calls] == ["saoyi", "saoyi"]


def test_claim_truth_supported_precheck_requires_speaker_direct_witness(tmp_path: Path, monkeypatch) -> None:
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
            {"event_id": "ge_supported", "player_id": "Gemini", "visibility": "direct_visual"},
            {"event_id": "ge_supported", "player_id": "saoyi", "visibility": "not_visible"},
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
        return {"ok": True, "phase_id": phase_id, "player": player, "abs_sec": abs_sec, "output": output.as_posix()}

    monkeypatch.setattr(claim_truth_precheck, "extract_frame", fake_extract_frame)
    monkeypatch.setattr(claim_truth_precheck, "compose_quad", lambda row, output: True)
    monkeypatch.setattr(claim_truth_precheck, "make_contact", lambda quad_paths, output: True)

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


def test_audio_confirmation_audit_blocks_direct_human_gold_claim(tmp_path: Path) -> None:
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


def test_audio_confirmation_audit_allows_only_merge_gate_candidate(tmp_path: Path) -> None:
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
    assert merge_candidate["gold_policy"]["requires_codex_human_gold_merge_gate"] is True
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


def test_sync_audio_confirmation_queue_filters_rejected_and_unknown_speaker(tmp_path: Path) -> None:
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
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    summary = meeting_claim_sync.refresh_audio_confirmation_queue(records_path, tmp_path / "out")
    queue = read_jsonl(tmp_path / "out" / "audio_confirmation_queue.jsonl")

    assert summary["audio_confirmation_items"] == 2
    assert {row["audio_confirmation_item_id"] for row in queue} == {"keep_audio", "keep_alias"}
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
            "suggested_probe_uses": ["claim_truth_vs_claim_awareness", "D_perspective_taking_prediction"],
            "remaining_uncertainties": [],
        }
    )

    assert record["codex_human_merge_decision"] == "accept_for_qwen_checked_merge_candidate"
    assert record["safe_for_probe_draft_generation"] is True
    assert record["promotion_to_human_verified_gold"] is False
    assert record["next_required_gate"] == "codex_human_audio_spotcheck_or_independent_transcript_before_human_verified"


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
        "suggested_probe_uses": ["claim_truth_vs_claim_awareness", "D_perspective_taking_prediction"],
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
    assert "QUERY_VARIABLE_PUBLIC_FORM_JSON" in next(probe for probe in probes if probe["probe_type"] == "A_pre_reveal_belief")["prompt"]
    d_prompt = next(probe for probe in probes if probe["probe_type"] == "D_perspective_taking_prediction")["prompt"]
    assert "SPEAKER_AVAILABLE_CONTEXT_JSON" in d_prompt
    assert "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON" in d_prompt
    assert all(probe["gold_source"] == "qwen_checked" for probe in probes)
    assert all(probe["promotion_to_human_verified_gold"] is False for probe in probes)
    assert all(row["promotion_to_human_verified_gold"] is False for row in hidden_gold)
    assert all(row["D_PT_reference_enriched_by"] for row in hidden_gold)


def test_human_gold_accept_candidates_filter_uncertain_aliases(tmp_path: Path) -> None:
    groups = [
        {
            "probe_group_id": "pg_keep",
            "source_review_item_id": "keep",
            "quality": {},
            "needs_human_review": True,
        },
        {
            "probe_group_id": "pg_drop",
            "source_review_item_id": "drop",
            "quality": {},
            "needs_human_review": True,
        },
    ]
    probes = [
        {"probe_id": "p_keep", "probe_group_id": "pg_keep", "probe_type": "D_perspective_taking_prediction"},
        {"probe_id": "p_drop", "probe_group_id": "pg_drop", "probe_type": "D_perspective_taking_prediction"},
    ]
    hidden = [
        {"probe_id": "p_keep", "probe_group_id": "pg_keep"},
        {"probe_id": "p_drop", "probe_group_id": "pg_drop"},
    ]
    merge = [
        {
            "source_review_item_id": "keep",
            "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
            "confirmed_transcript": {"transcript_match": "exact", "text_confidence": "high"},
            "confirmed_speaker": {"speaker_confidence": "high"},
            "remaining_uncertainties": [],
        },
        {
            "source_review_item_id": "drop",
            "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
            "confirmed_transcript": {"transcript_match": "exact", "text_confidence": "high"},
            "confirmed_speaker": {"speaker_confidence": "high"},
            "remaining_uncertainties": ["alias not resolved"],
        },
    ]
    for name, rows in {
        "groups.jsonl": groups,
        "probes.jsonl": probes,
        "hidden.jsonl": hidden,
        "merge.jsonl": merge,
    }.items():
        (tmp_path / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    accepted_groups, accepted_probes, accepted_hidden = human_gold_accept_candidates.build_accept_candidates(
        tmp_path / "groups.jsonl",
        tmp_path / "probes.jsonl",
        tmp_path / "hidden.jsonl",
        tmp_path / "merge.jsonl",
    )

    assert [row["probe_group_id"] for row in accepted_groups] == ["pg_keep"]
    assert [row["probe_id"] for row in accepted_probes] == ["p_keep"]
    assert [row["probe_id"] for row in accepted_hidden] == ["p_keep"]
    assert accepted_groups[0]["quality"]["human_gold_accept_candidate"] is True
    assert accepted_groups[0]["quality"]["promotion_to_human_verified_gold"] is False


def test_combined_audio_confirmation_results_preserves_queue_order_and_dedups(tmp_path: Path) -> None:
    queue = tmp_path / "queue.jsonl"
    write_jsonl(
        queue,
        [
            {"review_task_id": "audio_001"},
            {"review_task_id": "audio_002"},
            {"review_task_id": "audio_003"},
        ],
    )
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "audio_001.json").write_text(
        json.dumps({"review_task_id": "audio_001", "source": "a"}), encoding="utf-8"
    )
    (root_a / "audio_002.json").write_text(
        json.dumps({"review_task_id": "audio_002", "source": "a"}), encoding="utf-8"
    )
    (root_b / "audio_002.json").write_text(
        json.dumps({"review_task_id": "audio_002", "source": "b"}), encoding="utf-8"
    )
    (root_b / "unrelated.json").write_text(
        json.dumps({"review_task_id": "unrelated"}), encoding="utf-8"
    )
    output_root = tmp_path / "combined"
    output_root.mkdir()
    (output_root / "stale.json").write_text(json.dumps({"review_task_id": "stale"}), encoding="utf-8")

    summary = combined_audio_results.build_combined_results(
        queue=queue,
        result_roots=[root_a, root_b],
        output_root=output_root,
    )

    assert summary["expected_tasks"] == 3
    assert summary["stale_results_removed"] == 1
    assert summary["combined_results"] == 2
    assert summary["missing_review_task_ids"] == ["audio_003"]
    assert len(summary["duplicate_results"]) == 1
    assert (tmp_path / "combined" / "audio_001.json").exists()
    assert read_json(tmp_path / "combined" / "audio_002.json")["source"] == "a"
    assert not (tmp_path / "combined" / "combined_results_summary.json").exists()
    assert (tmp_path / "combined_results_summary.json").exists()






def test_scope_limited_release_report_requires_scope_metadata(tmp_path: Path) -> None:
    root = tmp_path / "pass"
    reports = root / "benchmark/gooseomni_v1/reports"
    reports.mkdir(parents=True)
    (reports / "validation.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    write_jsonl(
        root / "annotations/diagnostics/probe_groups.jsonl",
        [
            {
                "probe_group_id": "pg_scope",
                "quality": {"scope_limited_public_speech_gold": True},
                "scope_limitations": {"scope": "public_meeting_speech_interpretation_only"},
                "query_variable": {"type": "claim_truth_vs_claim_awareness"},
            }
        ],
    )
    write_jsonl(
        root / "annotations/diagnostics/diagnostic_quality.jsonl",
        [{"probe_group_id": "pg_scope", "scope_limited_public_speech_gold": True}],
    )
    write_jsonl(
        root / "annotations/diagnostics/hidden_gold.jsonl",
        [{"probe_group_id": "pg_scope", "scope_limited_public_speech_gold": True}],
    )
    write_jsonl(
        root / "benchmark/gooseomni_v1/static_trials/trials.jsonl",
        [{"trial_id": "t1", "probe_type": "A_pre_reveal_belief"}],
    )

    report = scope_release_report.build_scope_report(root)
    scope_release_report.write_benchmark_card(root, report)

    assert report["ok"] is True
    assert report["counts"]["scope_limited_groups"] == 1
    assert "scope_limited_public_speech_groups: 1" in (reports / "benchmark_card.md").read_text(encoding="utf-8")


def test_scope_limited_accept_preserves_alias_uncertainty(tmp_path: Path) -> None:
    group = {
        "probe_group_id": "pg_scope",
        "source_review_item_id": "src_scope",
        "quality": {},
        "needs_human_review": True,
    }
    probe = {"probe_id": "p_scope", "probe_group_id": "pg_scope", "probe_type": "A_pre_reveal_belief"}
    hidden = {"probe_id": "p_scope", "probe_group_id": "pg_scope"}
    merge = {
        "source_review_item_id": "src_scope",
        "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
        "safe_for_probe_draft_generation": True,
        "confirmed_transcript": {"transcript_match": "exact", "text_confidence": "high"},
        "confirmed_speaker": {"speaker_confidence": "high"},
        "remaining_uncertainties": ["The canonical identity of display name 海螺 is not confirmed."],
    }
    write_jsonl(tmp_path / "groups.jsonl", [group])
    write_jsonl(tmp_path / "probes.jsonl", [probe])
    write_jsonl(tmp_path / "hidden.jsonl", [hidden])
    write_jsonl(tmp_path / "merge.jsonl", [merge])

    groups, probes, hidden_rows, records = scope_limited_accept.build_accept_candidates(
        tmp_path / "groups.jsonl",
        tmp_path / "probes.jsonl",
        tmp_path / "hidden.jsonl",
        tmp_path / "merge.jsonl",
    )

    assert len(groups) == 1
    assert groups[0]["quality"]["scope_limited_public_speech_gold"] is True
    assert groups[0]["scope_limitations"]["remaining_uncertainties_preserved"] == merge["remaining_uncertainties"]
    assert probes[0]["scope_limited_public_speech_gold"] is True
    assert hidden_rows[0]["scope_limited_public_speech_gold"] is True
    assert records[0]["scope_limited_accept"] is True


def test_meeting_claim_promotion_pass_only_promotes_accept_candidates(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "annotations/oracle_ledger").mkdir(parents=True)
    write_jsonl(base / "annotations/oracle_ledger/world_events.jsonl", [])
    accept = tmp_path / "accept"
    group = {
        "probe_group_id": "pg_keep",
        "quality": {"human_gold_accept_candidate": True},
        "needs_human_review": True,
        "source_result_file": "result.json",
        "video_file": "clip.mp4",
    }
    write_jsonl(accept / "probe_groups.human_gold_accept_candidates.jsonl", [group])
    write_jsonl(
        accept / "probes.human_gold_accept_candidates.jsonl",
        [
            {
                "probe_group_id": "pg_keep",
                "probe_id": "pg_keep_A",
                "probe_type": "A_pre_reveal_belief",
                "gold_source": "human_gold_accept_candidate",
            }
        ],
    )
    write_jsonl(
        accept / "hidden_gold.human_gold_accept_candidates.jsonl",
        [{"probe_group_id": "pg_keep", "probe_id": "pg_keep_A", "gold_source": "human_gold_accept_candidate"}],
    )

    summary = meeting_claim_promotion.build_promotion_pass(base, [accept], tmp_path / "out")
    groups = read_jsonl(tmp_path / "out/annotations/diagnostics/probe_groups.jsonl")
    probes = read_jsonl(tmp_path / "out/annotations/diagnostics/probes_A_pre_reveal.jsonl")
    records = read_jsonl(tmp_path / "out/review/codex_human_review_records.jsonl")

    assert summary["counts"]["probe_groups"] == 1
    assert groups[0]["gold_source"] == "human_verified"
    assert groups[0]["needs_human_review"] is False
    assert probes[0]["gold_source"] == "human_verified"
    assert records[0]["gate_decision"] == "accept_human_verified"
    assert records[0]["remaining_uncertainties"] == []


def test_meeting_claim_total_progress_sums_layers(tmp_path: Path) -> None:
    stable = tmp_path / "stable"
    (stable / "benchmark/gooseomni_v1/reports").mkdir(parents=True)
    (stable / "benchmark/gooseomni_v1/reports/validation.json").write_text(
        json.dumps({"ok": True, "issue_count": 0, "counts": {"probe_groups": 1}}), encoding="utf-8"
    )
    write_jsonl(stable / "benchmark/gooseomni_v1/static_trials/trials.jsonl", [{"probe_type": "A"}])
    write_jsonl(stable / "annotations/diagnostics/probe_groups.jsonl", [{"query_variable": {"type": "route_belief"}}])

    roots = []
    for idx, probes in enumerate([3, 6], start=1):
        root = tmp_path / f"audio_{idx}"
        (root / "audit").mkdir(parents=True)
        (root / "audit/audio_confirmation_audit_summary.json").write_text(
            json.dumps({"results": idx, "merge_gate_candidates": idx, "decision_counts": {}, "issue_counts": {}}),
            encoding="utf-8",
        )
        (root / "codex_human_gold_merge_review_records").mkdir(parents=True)
        (root / "codex_human_gold_merge_review_records/summary.json").write_text(
            json.dumps({"records": idx, "safe_for_probe_draft_generation": idx}), encoding="utf-8"
        )
        (root / "probe_drafts_qwen_checked").mkdir(parents=True)
        (root / "probe_drafts_qwen_checked/summary.json").write_text(
            json.dumps({"probe_groups": idx, "probes": probes}), encoding="utf-8"
        )
        write_jsonl(
            root / "probe_drafts_qwen_checked/probes.qwen_checked_draft.jsonl",
            [{"probe_type": "D_perspective_taking_prediction"}] * probes,
        )
        (root / "human_gold_accept_candidates").mkdir(parents=True)
        (root / "human_gold_accept_candidates/summary.json").write_text(
            json.dumps({"probe_groups": 1, "probes": 3}), encoding="utf-8"
        )
        write_jsonl(
            root / "human_gold_accept_candidates/probes.human_gold_accept_candidates.jsonl",
            [{"probe_type": "D_perspective_taking_prediction"}] * 3,
        )
        roots.append(root)

    report = meeting_claim_total_progress.build_total_report(stable, roots)

    assert report["stable_human_verified_core"]["validation_ok"] is True
    assert report["extension_totals"]["qwen_checked_probe_groups"] == 3
    assert report["extension_totals"]["qwen_checked_probes"] == 9
    assert report["extension_totals"]["human_gold_accept_candidate_groups"] == 2
    assert report["extension_totals"]["human_gold_accept_candidate_probes"] == 6
    assert report["promotion_to_human_verified_gold"] is False


def test_meeting_claim_progress_report_reads_missing_files_as_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    missing_jsonl = tmp_path / "missing.jsonl"

    assert meeting_claim_progress_report.read_json(missing) == {}
    assert meeting_claim_progress_report.read_jsonl(missing_jsonl) == []


def test_oracle_submit_defaults_use_high_quality_qwen_settings(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["submit_gooseomni_oracle_jobs.py"])

    args = submit_oracle_jobs.parse_args()

    assert args.qwen_max_tokens == "16384"
    assert args.qwen_text_merge_max_tokens == "32768"
    assert args.qwen_video_fps == "1.0"
    assert args.qwen_video_max_frames == "48"
    assert args.qwen_video_max_pixels == "401408"
    assert args.omni_http_timeout_sec == "900"


def test_oracle_slurm_defaults_use_high_quality_qwen_settings() -> None:
    for rel in [
        "configs/slurm/qwen3_omni_oracle_2gpu_stage.slurm",
        "configs/slurm/qwen3_omni_oracle_4x2_local.slurm",
        "configs/slurm/qwen3_omni_oracle_local_array.slurm",
    ]:
        text = Path(rel).read_text(encoding="utf-8")

        assert 'QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"' in text
        assert 'QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"' in text
        assert 'QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"' in text
        assert 'QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"' in text
        assert 'QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"' in text
        assert 'OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-900}"' in text
