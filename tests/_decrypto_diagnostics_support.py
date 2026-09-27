# ruff: noqa: F401
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import tools.internal.review.audit_meeting_claim_audio_confirmation_outputs as audio_confirmation_audit
import tools.internal.review.build_behavior_outcome_d_candidates as behavior_outcome_d
import tools.internal.review.build_claim_truth_visual_precheck_pack as claim_truth_precheck
import tools.internal.review.build_combined_audio_confirmation_results as combined_audio_results
import tools.internal.review.build_decrypto_clip_review_pack as clip_review
import tools.internal.review.build_meeting_claim_audio_confirmation_pack as audio_confirmation_pack
import tools.internal.review.build_meeting_claim_gold_merge_pack as meeting_claim_gold_merge
import tools.internal.review.build_meeting_claim_human_gold_accept_candidates as human_gold_accept_candidates
import tools.internal.review.build_meeting_claim_human_verified_promotion_pass as meeting_claim_promotion
import tools.internal.review.build_meeting_claim_probe_drafts as meeting_claim_probe_drafts
import tools.internal.review.build_meeting_claim_scope_limited_accept_candidates as scope_limited_accept
import tools.internal.review.report_meeting_claim_benchmark_progress as meeting_claim_progress_report
import tools.internal.review.report_meeting_claim_total_progress as meeting_claim_total_progress
import tools.internal.review.report_scope_limited_gold_release as scope_release_report
import tools.internal.review.sync_meeting_claim_extension_state as meeting_claim_sync
import tools.internal.review.write_meeting_claim_gold_merge_review_records as gold_merge_review_records
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
from tools.annotation import run_decrypto_high_quality_review as high_quality_review
from tools.annotation import submit_gooseomni_oracle_jobs as submit_oracle_jobs
from tools.build.build_decrypto_human_verified_subset import merge_human_verified_passes
from tools.internal.review.build_claim_truth_extra_targets import (
    edge_lookup,
    validate_claim_truth_spec,
)
from tools.internal.review.build_delayed_reveal_from_verified_anchors import (
    validate_delayed_reveal_spec,
)
from tools.internal.review.build_delayed_reveal_visual_precheck_pack import (
    compose_quad,
    load_skipped_clusters,
)


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
        "phase_order_label_zh": "第1局第1次跑动过程"
        if phase_type == "gameplay"
        else "第1局第1次会议",
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
        _write_gold_annotation(
            root,
            meeting,
            player,
            "meeting",
            [],
            [utterance if player == "Gemini" else dict(utterance)],
        )
    return root


def _write_minimal_verified_pass(
    pass_root: Path, group_id: str, anchor_event_id: str
) -> None:
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
    write_jsonl(
        diag / "hidden_gold.jsonl",
        [
            {
                "probe_group_id": group_id,
                "gold_source": "human_verified",
                "forbidden_event_ids_for_target": [anchor_event_id],
            }
        ],
    )
    write_jsonl(
        diag / "diagnostic_quality.jsonl",
        [
            {
                "probe_group_id": group_id,
                "needs_human_review": False,
                "diagnostic_score": 1.0,
            }
        ],
    )
    write_jsonl(
        diag / "probes_A_pre_reveal.jsonl",
        [
            {
                **probe_base,
                "probe_id": f"{group_id}_A",
                "probe_type": "A_pre_reveal_belief",
            }
        ],
    )
    write_jsonl(
        diag / "probes_B_reconstruct.jsonl",
        [
            {
                **probe_base,
                "probe_id": f"{group_id}_B",
                "probe_type": "B_post_reveal_reconstruct_previous_belief",
            }
        ],
    )
    write_jsonl(
        diag / "probes_C_false_belief.jsonl",
        [
            {
                **probe_base,
                "probe_id": f"{group_id}_C",
                "probe_type": "C_other_agent_false_belief",
            }
        ],
    )
    write_jsonl(diag / "probes_D_perspective_taking.jsonl", [])


__all__ = [name for name in globals() if not name.startswith("__")]
