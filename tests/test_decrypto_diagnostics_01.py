from __future__ import annotations

from tests._decrypto_diagnostics_support import (
    PLAYERS,
    Path,
    _release_fixture,
    build_claim_truth_links,
    build_decrypto_diagnostics,
    build_oracle_ledger,
    build_visibility_edges,
    canonicalize_events,
    clip_review,
    edge_lookup,
    export_gooseomni_benchmark,
    high_quality_review,
    read_json,
    read_jsonl,
    score_decrypto_diagnostics,
    select_probe_groups,
    validate_claim_truth_spec,
    validate_decrypto_outputs,
    validate_delayed_reveal_spec,
    write_jsonl,
)


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
    assert maps[0]["duplicate_local_event_ids"] == [
        "seg4_mojiang_e1",
        "seg5_mojiang_e2",
    ]


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
    assert (
        next(edge for edge in edges if edge["player_id"] == "mojiang")["visibility"]
        == "direct_visual"
    )
    assert (
        next(edge for edge in edges if edge["player_id"] == "saoyi")["visibility"]
        == "not_visible"
    )


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
    assert (
        links[0]["local_awareness_by_player"]["mojiang"]
        == "has_contradictory_visual_evidence"
    )


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
    edges = edge_lookup(
        build_visibility_edges([task_event, clean_event, residual_ui_event], [claim])
    )

    task_errors = validate_claim_truth_spec(
        task_event, claim, "baile", "contradicted", edges
    )
    visible_errors = validate_claim_truth_spec(
        clean_event, claim, "Gemini", "contradicted", edges
    )
    clean_errors = validate_claim_truth_spec(
        clean_event, claim, "baile", "contradicted", edges
    )
    residual_errors = validate_claim_truth_spec(
        residual_ui_event, claim, "baile", "contradicted", edges
    )

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

    assert "reveal_too_close_to_event:10.0s" in validate_delayed_reveal_spec(
        event, "xiaolu", too_close_claim
    )
    assert "claim_does_not_semantically_match_event" in validate_delayed_reveal_spec(
        event, "xiaolu", mismatch_claim
    )
    assert "claim_type_not_reveal:role" in validate_delayed_reveal_spec(
        event, "xiaolu", role_claim
    )
    assert "claim_does_not_semantically_match_event" in validate_delayed_reveal_spec(
        victim_event, "Gemini", victim_claim
    )
    assert (
        "anchor_event_too_close_to_gameplay_phase_start:0.0s"
        in validate_delayed_reveal_spec(residual_anchor_event, "xiaolu", good_claim)
    )
    assert "reveal_claim_too_close_to_phase_start:0.0s" in validate_delayed_reveal_spec(
        event, "xiaolu", residual_claim
    )
    assert validate_delayed_reveal_spec(event, "xiaolu", good_claim) == []


def test_clip_review_window_respects_phase_bounds() -> None:
    phase_id = "g001_phase_026_gameplay_006400_006490"

    local_start, duration = clip_review.clip_window(
        phase_id, 6401.0, pre_sec=5.0, post_sec=7.0
    )
    late_start, late_duration = clip_review.clip_window(
        phase_id, 6488.0, pre_sec=5.0, post_sec=7.0
    )

    assert local_start == 0.0
    assert duration == 8.0
    assert late_start == 83.0
    assert late_duration == 7.0


def test_high_quality_review_uses_explicit_primary_and_context_video(
    tmp_path: Path,
) -> None:
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


def test_end_to_end_probe_generation_export_validation_and_scoring(
    tmp_path: Path,
) -> None:
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
    assert all(
        row["gold_source"] == row["hidden_gold"]["gold_source"]
        for row in read_jsonl(benchmark_root / "static_trials" / "hidden_gold.jsonl")
    )

    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    templates = {group["template"] for group in groups}
    assert any(group["template"] == "contradicted_alibi" for group in groups)
    assert "private_witness" in templates
    assert "vote_influence" in templates
    assert "delayed_public_reveal" in templates
    assert all(
        group["target_player"] != "mojiang"
        for group in groups
        if group["template"] == "contradicted_alibi"
    )
    assert all(
        group["hidden_event_ids_for_target"]
        for group in groups
        if group["template"] == "contradicted_alibi"
    )

    trials_text = (benchmark_root / "static_trials" / "trials.jsonl").read_text(
        encoding="utf-8"
    )
    assert "hidden_gold" not in trials_text
    assert "forbidden_event_ids" not in trials_text

    prompts = read_jsonl(benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
    for prompt in prompts:
        if prompt["probe_type"] == "A_pre_reveal_belief":
            assert "QUERY_VARIABLE_PUBLIC_FORM_JSON" in prompt["prompt"]
            assert "ORACLE_TRUTH_JSON" not in prompt["prompt"]
            assert not any(
                event_id in prompt["prompt"]
                for event_id in prompt.get("forbidden_event_ids", [])
            )
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

    aggregate = score_decrypto_diagnostics(
        responses, annotation_root / "diagnostics" / "hidden_gold.jsonl", score_out
    )
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
    hidden_groups = [
        group for group in groups if group["template"] == "hidden_event_awareness"
    ]

    assert hidden_groups
    assert all(
        "ge_000001" in group["hidden_event_ids_for_target"] for group in hidden_groups
    )
    assert all(group["target_player"] != "mojiang" for group in hidden_groups)
