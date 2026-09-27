from __future__ import annotations

from tests._strict_tom_pipeline_support import (
    _apply_decisions,
    accepted_trials,
    json,
    load_visual_error_candidates,
    pytest,
    recoverable_visual_errors,
    review_and_write_trajectory,
    validate_pilot,
)


def test_smoke_coverage_gaps_do_not_bypass_formal_release_gates() -> None:
    alignment = {
        "reference_player": "Gemini",
        "common_coverage_start_sec": 0,
        "mappings": {
            player: {
                "scale": 1,
                "offset": 0,
                "inlier_count": 3,
                "median_residual_sec": 0,
                "p95_residual_sec": 0,
            }
            for player in ("Gemini", "b")
        },
    }
    smoke = validate_pilot(
        alignment,
        [{"episode_id": "e1"}],
        [],
        [],
        [],
        ["Gemini", "b"],
        minimum_episodes=1,
        minimum_trials_per_layer=1,
    )
    formal = validate_pilot(alignment, [], [], [], [], ["Gemini", "b"])
    assert smoke["ok"] is True
    assert smoke["coverage_gaps"]
    assert smoke["release_eligible"] is False
    assert formal["ok"] is False


def test_codex_trial_repair_is_applied_without_changing_identity_or_evidence() -> None:
    candidate = {
        "trial_id": "t1",
        "probe_group_id": "g1",
        "variant": "A",
        "episode_id": "e1",
        "cutoff_abs_sec": 10,
        "subject_player": "a",
        "target_player": "b",
        "tom_layer": "knowledge_state",
        "split": "test",
        "evidence_ids": ["allowed"],
        "gold": {"certainty": "unknown"},
    }
    rows = accepted_trials(
        {"t1": candidate},
        [
            {
                "trial_id": "t1",
                "decision": "repair",
                "final_value": {
                    "trial_id": "invented",
                    "evidence_ids": ["invented"],
                    "gold": {"certainty": "explicit", "answer": "x"},
                },
            }
        ],
    )
    assert rows[0]["trial_id"] == "t1"
    assert rows[0]["evidence_ids"] == ["allowed"]
    assert rows[0]["gold"] == {"certainty": "explicit", "answer": "x"}
    assert rows[0]["gold_source"] == "model_verified"


def test_codex_group_repair_applies_atomically_to_all_variants() -> None:
    candidates = {
        f"t-{variant}": {
            "trial_id": f"t-{variant}",
            "probe_group_id": "g1",
            "variant": variant,
            "evidence_ids": ["allowed"],
            "gold": {"certainty": "unknown"},
        }
        for variant in "ABCD"
    }
    rows = accepted_trials(
        candidates,
        [
            {
                "probe_group_id": "g1",
                "decision": "repair",
                "final_value": {"gold": {"certainty": "explicit", "answer": "x"}},
            }
        ],
    )
    assert {row["variant"] for row in rows} == set("ABCD")
    assert all(row["gold"]["answer"] == "x" for row in rows)


def test_codex_trajectory_repair_keeps_identity_and_evidence() -> None:
    candidate = {
        "trajectory_node_id": "n1",
        "episode_id": "e1",
        "abs_start_sec": 1.0,
        "abs_end_sec": 2.0,
        "source_povs": ["a"],
        "evidence_asset_ids": ["frame-a"],
        "event_type": "vote",
    }
    decisions = [
        {
            "trajectory_node_id": "n1",
            "decision": "repair",
            "reason": "keyframe timestamp",
            "evidence_ids": ["frame-a"],
            "final_value": {
                "trajectory_node_id": "invented",
                "abs_start_sec": 3.0,
                "abs_end_sec": 4.0,
                "event_type": "action_outcome",
            },
        }
    ]
    rows = _apply_decisions(
        [candidate],
        decisions,
        [{"episode_id": "e1", "abs_start_sec": 0.0, "abs_end_sec": 10.0}],
    )
    assert rows[0]["trajectory_node_id"] == "n1"
    assert rows[0]["evidence_asset_ids"] == ["frame-a"]
    assert rows[0]["abs_start_sec"] == 3.0
    assert rows[0]["gold_source"] == "model_verified"


def test_codex_trajectory_accept_canonicalizes_unknown_player_names() -> None:
    candidate = {
        "trajectory_node_id": "n1",
        "episode_id": "e1",
        "abs_start_sec": 1.0,
        "abs_end_sec": 2.0,
        "source_povs": ["a"],
        "evidence_asset_ids": ["frame-a"],
        "event_type": "interaction",
        "visible_players": ["screen-name"],
        "mentioned_players": ["a", "screen-name"],
    }
    rows = _apply_decisions(
        [candidate],
        [
            {
                "trajectory_node_id": "n1",
                "decision": "accept",
                "reason": "interaction is visible",
                "evidence_ids": ["frame-a"],
                "final_value": None,
            }
        ],
        [{"episode_id": "e1", "abs_start_sec": 0.0, "abs_end_sec": 10.0}],
        ["a", "b"],
    )
    assert rows[0]["visible_players"] == ["unknown"]
    assert rows[0]["mentioned_players"] == ["a", "unknown"]
    assert rows[0]["canonicalization_original"]["visible_players"] == ["screen-name"]


def test_visual_error_candidate_is_quarantined_until_codex_review(
    tmp_path, monkeypatch
) -> None:
    error_root = tmp_path / "errors/perception_visual"
    error_root.mkdir(parents=True)
    clip = {
        "game_id": "g001",
        "player_id": "Gemini",
        "clip_id": "clip-1",
        "clip_path": "/unused/clip.mp4",
        "start_sec": 10.0,
        "end_sec": 20.0,
    }
    raw_event = {
        "start_sec": 11.0,
        "end_sec": 12.0,
        "event_type": "vote",
        "description": "投票界面",
        "visible_players": ["xiaolu"],
        "confidence": 0.9,
        "timestamp_evidence": "first=ABS 11.00s; last=ABS 12.00s",
    }
    (error_root / "clip-1.json").write_text(
        json.dumps({"clip": clip, "raw_response": json.dumps([raw_event])}),
        encoding="utf-8",
    )

    rows = load_visual_error_candidates(error_root)

    assert len(rows) == 1
    assert rows[0]["evidence_id"] == "clip-1:visual_quarantine:00"
    assert rows[0]["quality_gate_failed"] is True
    assert recoverable_visual_errors("visual", error_root) is True
    assert recoverable_visual_errors("audio", error_root) is False

    def fake_contact_sheet(_video_path, output_path, _start_sec):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"frame-pack")
        return output_path

    monkeypatch.setattr(
        "gooseomni.data_pipeline.trajectory_review.create_timestamp_contact_sheet",
        fake_contact_sheet,
    )
    with pytest.raises(RuntimeError, match="trajectory review is pending"):
        review_and_write_trajectory(
            tmp_path,
            rows,
            [],
            [{"episode_id": "e1", "abs_start_sec": 10.0, "abs_end_sec": 20.0}],
            ["Gemini", "xiaolu"],
        )
    queue = tmp_path / "local_codex/trajectory_review_queue.jsonl"
    assert queue.is_file()
    queued = json.loads(queue.read_text(encoding="utf-8"))
    assert "visual_quarantine" in queued["available_evidence_ids"][0]
    assert queued["frame_pack_path"].endswith("clip-1.jpg")
