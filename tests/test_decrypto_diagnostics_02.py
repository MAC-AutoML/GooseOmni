from __future__ import annotations

from tests._decrypto_diagnostics_support import (
    Image,
    Path,
    _release_fixture,
    _write_minimal_verified_pass,
    behavior_outcome_d,
    build_decrypto_diagnostics,
    build_oracle_ledger,
    compose_quad,
    export_gooseomni_benchmark,
    load_skipped_clusters,
    merge_human_verified_passes,
    read_json,
    read_jsonl,
    score_decrypto_diagnostics,
    validate_decrypto_outputs,
    write_jsonl,
)


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
    assert any(
        issue["code"] == "D_prompt_contains_listener_private_context"
        for issue in validation["issues"]
    )


def test_validator_requires_enriched_gold_for_human_verified_d(tmp_path: Path) -> None:
    release_root = _release_fixture(tmp_path)
    annotation_root = tmp_path / "annotations"
    benchmark_root = tmp_path / "benchmark" / "gooseomni_v1"

    build_oracle_ledger(release_root, annotation_root)
    build_decrypto_diagnostics(annotation_root, annotation_root, limit=12)
    export_gooseomni_benchmark(annotation_root, benchmark_root)

    d_group_id = read_jsonl(
        annotation_root / "diagnostics" / "probes_D_perspective_taking.jsonl"
    )[0]["probe_group_id"]
    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    for group in groups:
        if group["probe_group_id"] == d_group_id:
            group["gold_source"] = "human_verified"
    write_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl", groups)

    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert not validation["ok"]
    assert any(
        issue["code"] == "human_verified_D_missing_enriched_gold"
        for issue in validation["issues"]
    )

    hidden_rows = read_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl")
    for row in hidden_rows:
        if row["probe_group_id"] == d_group_id:
            row["D_PT_reference_enriched_by"] = "codex_human_reviewer"
            row["D_PT_reference"]["human_verified_scope"] = (
                "information_state_difference_and_prompt_safety"
            )
    write_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl", hidden_rows)

    validation = validate_decrypto_outputs(annotation_root, benchmark_root)

    assert validation["ok"], validation


def test_human_verified_merge_preserves_distinct_groups_with_colliding_ids(
    tmp_path: Path,
) -> None:
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
    probes = read_jsonl(
        output / "annotations" / "diagnostics" / "probes_A_pre_reveal.jsonl"
    )

    assert summary["counts"]["probe_groups"] == 2
    assert {tuple(row["anchor_event_ids"]) for row in groups} == {
        ("ge_000061",),
        ("ge_000585",),
    }
    assert len({row["probe_group_id"] for row in groups}) == 2
    assert {row["probe_group_id"] for row in hidden} == {
        row["probe_group_id"] for row in groups
    }
    assert {row["probe_group_id"] for row in probes} == {
        row["probe_group_id"] for row in groups
    }
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


def test_behavior_outcome_candidates_prefer_aligned_release_frames(
    tmp_path: Path, monkeypatch
) -> None:
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

    def fake_extract_release_frame(
        release_video_root, phase_id, player, abs_sec, output
    ):
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

    monkeypatch.setattr(
        behavior_outcome_d, "extract_release_frame", fake_extract_release_frame
    )

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
