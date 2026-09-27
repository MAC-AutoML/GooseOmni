from __future__ import annotations

from types import SimpleNamespace

import pytest

from gooseomni.data_pipeline.pilot_review import codex_review_stage
from gooseomni.data_pipeline.reviewed_trials import accepted_trials
from gooseomni.data_pipeline.tom_trials import public_trial
from gooseomni.data_pipeline.v2 import read_jsonl, write_jsonl


def test_codex_review_is_group_level_and_keeps_hidden_evidence_private(
    tmp_path,
) -> None:
    candidates = [
        {
            "trial_id": f"t-{variant}",
            "probe_group_id": "g1",
            "variant": variant,
            "evidence_ids": ["clip:visual:00"],
            "review_only_hidden_evidence": ["future:audio:00"],
            "gold": {"certainty": "unknown"},
        }
        for variant in "ABCD"
    ]
    write_jsonl(tmp_path / "candidates/tom_trials.jsonl", candidates)
    frame = tmp_path / "local_codex/frame_packs/clip.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    context = SimpleNamespace(
        run_root=tmp_path,
        config=SimpleNamespace(
            stage_cache={},
            judge_model="gpt-5.6-sol",
        ),
    )
    with pytest.raises(RuntimeError, match="review is pending"):
        codex_review_stage(context)
    queue = read_jsonl(tmp_path / "local_codex/review_queue.jsonl")
    assert len(queue) == 1
    assert queue[0]["variants"] == list("ABCD")
    assert queue[0]["frame_pack_paths"] == [str(frame)]
    write_jsonl(
        tmp_path / "local_codex/decisions.jsonl",
        [
            {
                "probe_group_id": "g1",
                "decision": "repair",
                "reason": "future behavior verifies the answer",
                "evidence_ids": ["future:audio:00"],
                "final_value": {
                    "gold": {"certainty": "behaviorally_supported", "answer": "x"}
                },
            }
        ],
    )
    result = codex_review_stage(context)
    decisions = read_jsonl(tmp_path / "review_decisions.jsonl")
    rows = accepted_trials({row["trial_id"]: row for row in candidates}, decisions)
    assert result["accepted"] == 1
    assert len(rows) == 4
    assert "review_only_hidden_evidence" not in public_trial(rows[0])


def test_codex_review_allows_empty_diagnostic_run(tmp_path) -> None:
    write_jsonl(tmp_path / "candidates/tom_trials.jsonl", [])
    context = SimpleNamespace(
        run_root=tmp_path,
        config=SimpleNamespace(stage_cache={}, judge_model="gpt-5.6-sol"),
    )
    result = codex_review_stage(context)
    assert result == {"mode": "no_candidates", "accepted": 0, "quarantined": 0}
    assert read_jsonl(tmp_path / "review_decisions.jsonl") == []
    assert read_jsonl(tmp_path / "quarantine/rejected_trials.jsonl") == []
