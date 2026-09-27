from __future__ import annotations

import json

import pytest

from gooseomni.data_pipeline.local_review import (
    apply_local_review,
    prepare_local_review,
)
from gooseomni.data_pipeline.v2 import read_jsonl, write_jsonl


def _trajectory_queue(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    queue = tmp_path / "local_codex/trajectory_review_queue.jsonl"
    write_jsonl(
        queue,
        [
            {
                "trajectory_node_id": "n1",
                "candidate": {"event_type": "meeting"},
                "available_evidence_ids": ["frame:visual:00"],
                "frame_pack_path": str(frame),
            }
        ],
    )
    return queue


def test_prepare_and_apply_local_codex_review(tmp_path) -> None:
    queue = _trajectory_queue(tmp_path)
    prepared = prepare_local_review(tmp_path, "trajectory")
    assert prepared["queue_items"] == 1
    assert prepared["image_count"] == 1
    assert "每个 trajectory_node_id" in (
        tmp_path / "local_codex/automation/trajectory/prompt.txt"
    ).read_text(encoding="utf-8")
    schema = json.loads(
        (tmp_path / "local_codex/automation/trajectory/response_schema.json").read_text(
            encoding="utf-8"
        )
    )
    final_value_schema = schema["properties"]["decisions"]["items"]["properties"][
        "final_value"
    ]
    assert final_value_schema == {"type": ["string", "null"]}

    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "id": "n1",
                        "decision": "accept",
                        "reason": "frame supports meeting UI",
                        "evidence_ids": ["frame:visual:00"],
                        "final_value": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = apply_local_review(tmp_path, "trajectory", response)
    decisions = read_jsonl(tmp_path / "local_codex/trajectory_decisions.jsonl")
    assert result["decision_counts"]["accept"] == 1
    assert decisions[0]["trajectory_node_id"] == "n1"
    assert result["queue_sha256"]
    assert queue.is_file()


def test_apply_local_codex_review_decodes_repair_json(tmp_path) -> None:
    _trajectory_queue(tmp_path)
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "id": "n1",
                        "decision": "repair",
                        "reason": "use the visible public UI",
                        "evidence_ids": ["frame:visual:00"],
                        "final_value": '{"event_type":"public_ui"}',
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    apply_local_review(tmp_path, "trajectory", response)
    decisions = read_jsonl(tmp_path / "local_codex/trajectory_decisions.jsonl")
    assert decisions[0]["final_value"] == {"event_type": "public_ui"}


def test_apply_local_codex_review_fails_closed(tmp_path) -> None:
    _trajectory_queue(tmp_path)
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "id": "n1",
                        "decision": "accept",
                        "reason": "bad citation",
                        "evidence_ids": ["invented"],
                        "final_value": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside allow-list"):
        apply_local_review(tmp_path, "trajectory", response)
