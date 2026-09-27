from __future__ import annotations

from tests._strict_tom_pipeline_support import (
    TOM_LAYERS,
    Anchor,
    _read_records,
    build_episodes,
    build_information_state,
    build_trial_group,
    consensus_boundaries,
    evidence_for_layer,
    fit_affine_ransac,
    fuse_trajectory,
    normalize_audio_event,
    normalize_visual_event,
    observed_cutoffs,
    raw_interval,
    structured_gold,
    subject_only_evidence,
    valid_clip,
    validate_alignment,
    validate_audited_episode_payload,
    validate_claim_hearing,
    validate_episode_anchors,
    validate_trials,
)


def test_affine_alignment_rejects_outlier_and_round_trips() -> None:
    anchors = [
        Anchor(0, 12, "game_start", "a0"),
        Anchor(100, 112.1, "meeting_start", "a1"),
        Anchor(200, 212.2, "return_to_game", "a2"),
        Anchor(300, 999, "vote_result", "outlier"),
    ]
    fitted = fit_affine_ransac(anchors, residual_threshold_sec=1.0)
    assert fitted["inlier_count"] == 3
    assert abs(fitted["scale"] - 1.001) < 0.001
    assert raw_interval(fitted, 10, 20, 1000) == (
        fitted["scale"] * 10 + fitted["offset"],
        fitted["scale"] * 20 + fitted["offset"],
    )


def test_zero_byte_clip_is_never_resumed(tmp_path) -> None:
    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"")
    assert valid_clip(clip, 90) is False


def test_missing_perception_directory_is_an_empty_record_set(tmp_path) -> None:
    assert _read_records(tmp_path / "missing", "events") == []


def test_alignment_requires_auditable_mapping_for_every_player() -> None:
    payload = {
        "reference_player": "Gemini",
        "common_coverage_start_sec": 0,
        "mappings": {
            "Gemini": {
                "scale": 1,
                "offset": 0,
                "inlier_count": 3,
                "median_residual_sec": 0.2,
                "p95_residual_sec": 0.5,
            }
        },
    }
    assert validate_alignment(payload, ["Gemini"]) == []
    assert "missing alignment mapping: xiaolu" in validate_alignment(
        payload, ["Gemini", "xiaolu"]
    )


def test_alignment_common_coverage_prevents_negative_raw_time() -> None:
    payload = {
        "reference_player": "Gemini",
        "common_coverage_start_sec": 15,
        "mappings": {
            "Gemini": {
                "scale": 1,
                "offset": 110,
                "inlier_count": 3,
                "median_residual_sec": 0,
                "p95_residual_sec": 0,
            },
            "xiaolu": {
                "scale": 1,
                "offset": -14.5,
                "inlier_count": 3,
                "median_residual_sec": 0.2,
                "p95_residual_sec": 0.5,
            },
        },
    }
    assert validate_alignment(payload, ["Gemini", "xiaolu"]) == []
    payload["common_coverage_start_sec"] = 0
    assert (
        "common coverage start does not prevent negative raw time"
        in validate_alignment(payload, ["Gemini", "xiaolu"])
    )


def test_episode_consensus_ignores_single_pov_boundary() -> None:
    rows = [
        {
            "player_id": player,
            "clip_id": player,
            "boundary_type": "game_start",
            "aligned_start_sec": second,
        }
        for player, second in (("Gemini", 10), ("xiaolu", 11))
    ]
    rows.append(
        {
            "player_id": "Gemini",
            "clip_id": "private",
            "boundary_type": "game_end",
            "aligned_start_sec": 50,
        }
    )
    consensus = consensus_boundaries(rows)
    episodes = build_episodes("g001", consensus, 100)
    assert len(consensus) == 1
    assert episodes[0]["abs_end_sec"] == 100


def test_episode_alignment_requires_three_anchors_per_pov() -> None:
    rows = []
    for second, boundary_type in (
        (0, "game_start"),
        (20, "meeting_start"),
        (40, "return_to_game"),
    ):
        for player, drift in (("Gemini", 0.0), ("xiaolu", 0.4)):
            rows.append(
                {
                    "player_id": player,
                    "clip_id": f"{player}-{second}",
                    "boundary_type": boundary_type,
                    "aligned_start_sec": second + drift,
                }
            )
    episodes = [{"episode_id": "e1", "abs_start_sec": 0, "abs_end_sec": 60}]
    report = validate_episode_anchors(rows, episodes, ["Gemini", "xiaolu"])
    assert report["ok"] is True
    assert all(row["anchor_count"] == 3 for row in report["reports"])


def test_audited_episode_cache_requires_existing_evidence_and_contiguous_phases(
    tmp_path,
) -> None:
    frame = tmp_path / "anchors.jpg"
    frame.write_bytes(b"frame")
    evidence = [
        {
            "evidence_id": f"a{i}",
            "player_id": "Gemini",
            "frame_pack_path": str(frame),
        }
        for i in range(3)
    ]
    payload = {
        "episodes": [
            {
                "episode_id": "e1",
                "abs_start_sec": 1,
                "abs_end_sec": 3,
                "phases": [
                    {"abs_start_sec": 1, "abs_end_sec": 2},
                    {"abs_start_sec": 2, "abs_end_sec": 3},
                ],
            }
        ],
        "evidence_assets": evidence,
        "alignment_report": {
            "reports": [
                {
                    "episode_id": "e1",
                    "player_id": "Gemini",
                    "median_residual_sec": 0,
                    "p95_residual_sec": 0,
                    "evidence_ids": ["a0", "a1", "a2"],
                }
            ]
        },
    }
    assert validate_audited_episode_payload(payload, ["Gemini"], 3, 1, 2) == []
    payload["episodes"][0]["phases"][1]["abs_start_sec"] = 2.5
    assert "audited episode phases are not contiguous: e1" in (
        validate_audited_episode_payload(payload, ["Gemini"], 3, 1, 2)
    )


def test_task_ui_never_becomes_movement() -> None:
    row = normalize_visual_event(
        {
            "player_id": "Gemini",
            "event_type": "task_ui",
            "movement_transition": "street -> park",
            "visible_players": [],
        }
    )
    assert row["event_type"] == "task_ui"
    assert row["movement_transition"] is None


def test_private_event_does_not_propagate_and_future_is_forbidden() -> None:
    nodes = [
        {
            "trajectory_node_id": "n1",
            "episode_id": "e1",
            "abs_start_sec": 5,
            "source_povs": ["Gemini"],
            "public_ui": False,
            "global_fact": False,
            "modality": "visual",
            "evidence_asset_ids": ["frame1"],
        },
        {
            "trajectory_node_id": "n2",
            "episode_id": "e1",
            "abs_start_sec": 15,
            "source_povs": ["xiaolu"],
            "public_ui": False,
            "global_fact": False,
            "modality": "visual",
            "evidence_asset_ids": ["frame2"],
        },
    ]
    state = build_information_state("e1", "Gemini", 10, nodes)
    assert state["known_facts"] == ["n1"]
    assert set(state["forbidden_events"]) == {"n2"}


def test_information_state_cutoffs_only_use_observed_trajectory_boundaries() -> None:
    nodes = [{"abs_end_sec": 12}, {"abs_end_sec": 8}, {"abs_end_sec": 12}]
    assert observed_cutoffs(nodes) == [8.0, 12.0]


def test_claim_hearing_fails_closed() -> None:
    issues = validate_claim_hearing(
        {"heard_by": ["a", "b"], "audio_admissible": False}, ["a", "b"]
    )
    assert "non-public claim cannot default to all players" in issues
    assert "claim lacks admissible audio or visual text evidence" in issues


def test_audio_discards_visual_fields_and_rejects_noncanonical_identities() -> None:
    normalized = normalize_audio_event(
        {
            "player_id": "a",
            "event_type": "claim",
            "visible_players": ["invented"],
            "location": "hall",
            "speaker_id": "Player1",
            "speaker_confidence": 0.95,
            "utterance": "b is suspicious",
        }
    )
    assert normalized["visible_players"] == []
    assert normalized["location"] is None
    issues = validate_claim_hearing(
        {
            **normalized,
            "heard_by": ["a", "Player2"],
            "audio_admissible": True,
        },
        ["a", "b"],
    )
    assert "claim heard_by contains non-canonical players" in issues
    assert "claim speaker is not a canonical player" in issues


def test_audio_null_speaker_confidence_fails_closed() -> None:
    normalized = normalize_audio_event(
        {
            "player_id": "a",
            "event_type": "claim",
            "speaker_id": "a",
            "speaker_confidence": None,
            "utterance": "b is suspicious",
        }
    )
    assert normalized["audio_candidate_ok"] is False
    assert normalized["speaker_id"] is None
    assert normalized["utterance"] is None


def test_audio_gold_requires_cross_pov_agreement() -> None:
    episodes = [{"episode_id": "e1", "abs_start_sec": 0, "abs_end_sec": 20}]
    base = {
        "start_sec": 5,
        "end_sec": 6,
        "event_type": "claim",
        "speaker_id": "a",
        "speaker_confidence": 0.9,
        "utterance": "我看见 b",
        "confidence": 0.9,
        "evidence_id": "audio-a",
        "heard_by": ["a"],
    }
    single = fuse_trajectory([], [{**base, "player_id": "a"}], episodes)
    assert single[0]["audio_admissible"] is False
    agreed = fuse_trajectory(
        [],
        [
            {**base, "player_id": "a"},
            {**base, "player_id": "b", "evidence_id": "audio-b"},
        ],
        episodes,
    )
    assert agreed[0]["audio_admissible"] is True
    assert agreed[0]["heard_by"] == ["a", "b"]


def test_trajectory_does_not_merge_private_visual_or_different_audio() -> None:
    episodes = [{"episode_id": "e1", "abs_start_sec": 0, "abs_end_sec": 20}]
    visual = {
        "episode_id": "e1",
        "phase_index": 0,
        "phase_type": "gameplay",
        "start_sec": 5,
        "end_sec": 6,
        "event_type": "movement",
        "confidence": 0.9,
        "visible_players": [],
        "evidence_id": "v1",
    }
    audio = {
        "episode_id": "e1",
        "phase_index": 0,
        "phase_type": "gameplay",
        "start_sec": 7,
        "end_sec": 8,
        "event_type": "claim",
        "speaker_confidence": 0.9,
        "confidence": 0.9,
        "heard_by": [],
    }
    rows = fuse_trajectory(
        [
            {**visual, "player_id": "a"},
            {**visual, "player_id": "b", "evidence_id": "v2"},
        ],
        [
            {
                **audio,
                "player_id": "a",
                "speaker_id": "a",
                "utterance": "x",
                "evidence_id": "a1",
            },
            {
                **audio,
                "player_id": "b",
                "speaker_id": "b",
                "utterance": "x",
                "evidence_id": "a2",
            },
        ],
        episodes,
    )
    assert len(rows) == 4
    assert all(len(row["source_povs"]) == 1 for row in rows)


def test_five_tom_layers_keep_group_in_one_split_and_intent_distribution() -> None:
    state = {
        "episode_id": "e1",
        "player_id": "a",
        "cutoff_abs_sec": 10,
        "available_evidence_ids": ["e1"],
    }
    splits = set()
    for layer in TOM_LAYERS:
        gold = (
            {
                "certainty": "inferred_distribution",
                "distribution": [{"value": "x", "p": 0.5}],
            }
            if layer == "intent_and_action_prediction"
            else {"certainty": "explicit", "answer": "x"}
        )
        rows = build_trial_group(state, "b", "q", "gap", layer, ["e1"], gold)
        assert {row["variant"] for row in rows} == {"A", "B", "C", "D"}
        assert len({row["split"] for row in rows}) == 1
        splits.add(rows[0]["split"])
    assert splits


def test_tom_layer_evidence_is_target_related_accessible_and_cutoff_safe() -> None:
    state = {
        "episode_id": "e1",
        "player_id": "a",
        "cutoff_abs_sec": 10,
        "available_evidence_ids": ["visible-b", "future-b", "private-c"],
    }
    nodes = [
        {
            "abs_start_sec": 5,
            "event_type": "encounter",
            "player_id": "a",
            "visible_players": ["b"],
            "evidence_asset_ids": ["visible-b"],
        },
        {
            "abs_start_sec": 11,
            "event_type": "encounter",
            "player_id": "a",
            "visible_players": ["b"],
            "evidence_asset_ids": ["future-b"],
        },
        {
            "abs_start_sec": 5,
            "event_type": "encounter",
            "player_id": "a",
            "visible_players": ["c"],
            "evidence_asset_ids": ["private-c"],
        },
    ]
    assert evidence_for_layer(state, nodes, "b", "perception_access") == ["visible-b"]
    assert evidence_for_layer(state, nodes, "b", "stance_trust_and_agreement") == []


def test_public_outcome_source_pov_is_not_a_target_and_shared_evidence_has_no_gap() -> (
    None
):
    state = {
        "episode_id": "e1",
        "player_id": "a",
        "cutoff_abs_sec": 10,
        "available_evidence_ids": ["public-result"],
    }
    node = {
        "abs_start_sec": 5,
        "event_type": "action_outcome",
        "player_id": "b",
        "speaker_id": None,
        "visible_players": [],
        "mentioned_players": [],
        "evidence_asset_ids": ["public-result"],
    }
    assert evidence_for_layer(state, [node], "b", "knowledge_state") == []
    assert (
        subject_only_evidence(
            ["public-result"], {"available_evidence_ids": ["public-result"]}
        )
        == []
    )


def test_final_trial_validation_rejects_unknown_and_empty_distribution() -> None:
    base = {
        "trial_id": "t1",
        "probe_group_id": "g1",
        "split": "train",
        "cutoff_abs_sec": 1,
        "tom_layer": "intent_and_action_prediction",
        "gold_source": "model_verified",
    }
    issues = validate_trials(
        [
            {**base, "gold": {"certainty": "unknown"}},
            {
                **base,
                "trial_id": "t2",
                "probe_group_id": "g2",
                "gold": {"certainty": "inferred_distribution", "distribution": []},
            },
        ]
    )
    assert "unresolved ToM gold: t1" in issues
    assert "empty ToM gold distribution: t2" in issues


def test_structured_gold_is_explicit_and_future_behavior_stays_private() -> None:
    state = {"player_id": "a", "cutoff_abs_sec": 10}
    nodes = [
        {
            "abs_start_sec": 5,
            "event_type": "encounter",
            "visible_players": ["b"],
            "evidence_asset_ids": ["visible-b"],
        },
        {
            "abs_start_sec": 12,
            "event_type": "movement",
            "player_id": "b",
            "location": "park",
            "evidence_asset_ids": ["future-b"],
        },
    ]
    perception = structured_gold("perception_access", state, "b", nodes, ["visible-b"])
    intent = structured_gold(
        "intent_and_action_prediction", state, "b", nodes, ["visible-b"]
    )
    assert perception["answer"]["access"] == "direct_visual"
    assert intent["answer"]["next_observed_event_type"] == "movement"
    assert intent["future_evidence_ids"] == ["future-b"]
