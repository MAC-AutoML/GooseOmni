from __future__ import annotations

from tests._decrypto_diagnostics_support import (
    Path,
    combined_audio_results,
    human_gold_accept_candidates,
    json,
    meeting_claim_progress_report,
    meeting_claim_promotion,
    meeting_claim_total_progress,
    read_json,
    read_jsonl,
    scope_limited_accept,
    scope_release_report,
    submit_oracle_jobs,
    write_jsonl,
)


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
        {
            "probe_id": "p_keep",
            "probe_group_id": "pg_keep",
            "probe_type": "D_perspective_taking_prediction",
        },
        {
            "probe_id": "p_drop",
            "probe_group_id": "pg_drop",
            "probe_type": "D_perspective_taking_prediction",
        },
    ]
    hidden = [
        {"probe_id": "p_keep", "probe_group_id": "pg_keep"},
        {"probe_id": "p_drop", "probe_group_id": "pg_drop"},
    ]
    merge = [
        {
            "source_review_item_id": "keep",
            "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
            "confirmed_transcript": {
                "transcript_match": "exact",
                "text_confidence": "high",
            },
            "confirmed_speaker": {"speaker_confidence": "high"},
            "remaining_uncertainties": [],
        },
        {
            "source_review_item_id": "drop",
            "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
            "confirmed_transcript": {
                "transcript_match": "exact",
                "text_confidence": "high",
            },
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

    accepted_groups, accepted_probes, accepted_hidden = (
        human_gold_accept_candidates.build_accept_candidates(
            tmp_path / "groups.jsonl",
            tmp_path / "probes.jsonl",
            tmp_path / "hidden.jsonl",
            tmp_path / "merge.jsonl",
        )
    )

    assert [row["probe_group_id"] for row in accepted_groups] == ["pg_keep"]
    assert [row["probe_id"] for row in accepted_probes] == ["p_keep"]
    assert [row["probe_id"] for row in accepted_hidden] == ["p_keep"]
    assert accepted_groups[0]["quality"]["human_gold_accept_candidate"] is True
    assert accepted_groups[0]["quality"]["promotion_to_human_verified_gold"] is False


def test_combined_audio_confirmation_results_preserves_queue_order_and_dedups(
    tmp_path: Path,
) -> None:
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
    (output_root / "stale.json").write_text(
        json.dumps({"review_task_id": "stale"}), encoding="utf-8"
    )

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
                "scope_limitations": {
                    "scope": "public_meeting_speech_interpretation_only"
                },
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
    assert "scope_limited_public_speech_groups: 1" in (
        reports / "benchmark_card.md"
    ).read_text(encoding="utf-8")


def test_scope_limited_accept_preserves_alias_uncertainty(tmp_path: Path) -> None:
    group = {
        "probe_group_id": "pg_scope",
        "source_review_item_id": "src_scope",
        "quality": {},
        "needs_human_review": True,
    }
    probe = {
        "probe_id": "p_scope",
        "probe_group_id": "pg_scope",
        "probe_type": "A_pre_reveal_belief",
    }
    hidden = {"probe_id": "p_scope", "probe_group_id": "pg_scope"}
    merge = {
        "source_review_item_id": "src_scope",
        "codex_human_merge_decision": "accept_for_qwen_checked_merge_candidate",
        "safe_for_probe_draft_generation": True,
        "confirmed_transcript": {
            "transcript_match": "exact",
            "text_confidence": "high",
        },
        "confirmed_speaker": {"speaker_confidence": "high"},
        "remaining_uncertainties": [
            "The canonical identity of display name 海螺 is not confirmed."
        ],
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
    assert (
        groups[0]["scope_limitations"]["remaining_uncertainties_preserved"]
        == merge["remaining_uncertainties"]
    )
    assert probes[0]["scope_limited_public_speech_gold"] is True
    assert hidden_rows[0]["scope_limited_public_speech_gold"] is True
    assert records[0]["scope_limited_accept"] is True


def test_meeting_claim_promotion_pass_only_promotes_accept_candidates(
    tmp_path: Path,
) -> None:
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
        [
            {
                "probe_group_id": "pg_keep",
                "probe_id": "pg_keep_A",
                "gold_source": "human_gold_accept_candidate",
            }
        ],
    )

    summary = meeting_claim_promotion.build_promotion_pass(
        base, [accept], tmp_path / "out"
    )
    groups = read_jsonl(tmp_path / "out/annotations/diagnostics/probe_groups.jsonl")
    probes = read_jsonl(
        tmp_path / "out/annotations/diagnostics/probes_A_pre_reveal.jsonl"
    )
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
        json.dumps({"ok": True, "issue_count": 0, "counts": {"probe_groups": 1}}),
        encoding="utf-8",
    )
    write_jsonl(
        stable / "benchmark/gooseomni_v1/static_trials/trials.jsonl",
        [{"probe_type": "A"}],
    )
    write_jsonl(
        stable / "annotations/diagnostics/probe_groups.jsonl",
        [{"query_variable": {"type": "route_belief"}}],
    )

    roots = []
    for idx, probes in enumerate([3, 6], start=1):
        root = tmp_path / f"audio_{idx}"
        (root / "audit").mkdir(parents=True)
        (root / "audit/audio_confirmation_audit_summary.json").write_text(
            json.dumps(
                {
                    "results": idx,
                    "merge_gate_candidates": idx,
                    "decision_counts": {},
                    "issue_counts": {},
                }
            ),
            encoding="utf-8",
        )
        (root / "codex_human_gold_merge_review_records").mkdir(parents=True)
        (root / "codex_human_gold_merge_review_records/summary.json").write_text(
            json.dumps({"records": idx, "safe_for_probe_draft_generation": idx}),
            encoding="utf-8",
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
            root
            / "human_gold_accept_candidates/probes.human_gold_accept_candidates.jsonl",
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


def test_meeting_claim_progress_report_reads_missing_files_as_empty(
    tmp_path: Path,
) -> None:
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
        assert (
            'QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"'
            in text
        )
        assert 'QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"' in text
        assert (
            'QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-48}"' in text
        )
        assert (
            'QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"'
            in text
        )
        assert 'OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-900}"' in text
