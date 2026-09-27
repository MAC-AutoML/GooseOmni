from __future__ import annotations

import json
from pathlib import Path

from gooseomni.benchmark.agentic_midgame import (
    PLAYERS,
    build_agentic_midgame_prediction,
    contains_death_skill_overclaim,
    public_leak_hits,
    score_agentic_midgame_prediction,
    write_jsonl,
)


def _event(
    event_id: str, source: str, start: float, event_type: str, actor: str | None = None
) -> dict:
    return {
        "world_event_id": event_id,
        "game_id": "g001",
        "source_povs": [source],
        "source_segment_ids": ["seg001"],
        "phase_type": "gameplay",
        "event_type": event_type,
        "actors": [actor or source],
        "location": "right hallway",
        "description": f"{actor or source} does {event_type} near right hallway.",
        "abs_start_sec": start,
        "abs_end_sec": start + 4,
        "certainty": 0.9,
    }


def _edges(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        source = set(event["source_povs"])
        for player in PLAYERS:
            visible = "direct_visual" if player in source else "not_visible"
            rows.append(
                {
                    "edge_id": f"vis_{event['world_event_id']}_{player}",
                    "game_id": "g001",
                    "event_id": event["world_event_id"],
                    "player_id": player,
                    "cutoff_abs_sec": event["abs_end_sec"],
                    "visibility": visible,
                    "evidence_ids": [event["world_event_id"]]
                    if visible == "direct_visual"
                    else [],
                    "confidence": 0.9,
                }
            )
    return rows


def _snapshot(player: str, cutoff: float) -> dict:
    return {
        "snapshot_id": f"mem_{player}_{int(cutoff)}",
        "game_id": "g001",
        "target_player": player,
        "cutoff_abs_sec": cutoff,
        "public_history": [
            {"evidence_id": "claim_000001", "content": "Gemini said an alibi."}
        ],
        "private_observations": [
            {
                "evidence_id": f"obs_{player}",
                "world_event_id": "ge_visible",
                "content": "local visible context",
            }
        ],
        "heard_claims": [
            {
                "evidence_id": "claim_000001",
                "claim_id": "claim_000001",
                "speaker": "Gemini",
                "content": "我没去右边。",
            }
        ],
        "inferred_beliefs": [],
        "available_evidence_ids": ["claim_000001", f"obs_{player}"],
        "forbidden_event_ids": [],
    }


def _ledger(root: Path) -> Path:
    ann = root / "annotations"
    ledger = ann / "oracle_ledger"
    events = [
        _event("ge_000001", "Gemini", 10, "movement", "Gemini"),
        _event("ge_000002", "baile", 80, "task", "baile"),
    ]
    claim = {
        "claim_id": "claim_000001",
        "game_id": "g001",
        "speaker": "Gemini",
        "heard_by": PLAYERS[:],
        "abs_start_sec": 30.0,
        "abs_end_sec": 35.0,
        "claim_type": "location",
        "content": "我没去右边。",
    }
    link = {
        "claim_truth_link_id": "ctl_1",
        "claim_id": "claim_000001",
        "world_event_ids": ["ge_000001"],
        "truth_status_global": "contradicted",
        "local_awareness_by_player": {
            player: (
                "has_contradictory_visual_evidence"
                if player == "Gemini"
                else "not_enough_information"
            )
            for player in PLAYERS
        },
    }
    write_jsonl(ledger / "world_events.jsonl", events)
    write_jsonl(ledger / "claims.jsonl", [claim])
    write_jsonl(ledger / "claim_truth_links.jsonl", [link])
    write_jsonl(ledger / "visibility_edges.jsonl", _edges(events))
    write_jsonl(
        ledger / "belief_memory_snapshots.jsonl",
        [
            _snapshot(player, cutoff)
            for player in PLAYERS
            for cutoff in [2.0, 23.0, 36.0]
        ],
    )
    return ann


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_agentic_midgame_builds_four_public_safe_families(tmp_path: Path) -> None:
    ann = _ledger(tmp_path)
    bench = tmp_path / "benchmark" / "gooseomni_v1"

    counts = build_agentic_midgame_prediction(ann, bench, limit=4)
    trials = _read_jsonl(
        bench / "public" / "agentic_midgame_prediction" / "trials.jsonl"
    )
    hidden = _read_jsonl(
        bench / "private" / "agentic_midgame_prediction" / "hidden_gold.jsonl"
    )

    assert counts["trials"] == 4
    assert {row["probe_family"] for row in trials} == {
        "E_hidden_world_state_estimation",
        "F_other_player_current_action_prediction",
        "G_next_behavior_prediction",
        "H_deception_state_inference",
    }
    assert len(hidden) == 4
    assert all(
        {"ego_player", "cutoff_abs_sec", "prediction_target", "prediction_window"}
        <= set(row)
        for row in trials
    )
    assert public_leak_hits(trials) == []


def test_agentic_scoring_detects_accuracy_and_overclaim(tmp_path: Path) -> None:
    ann = _ledger(tmp_path)
    bench = tmp_path / "benchmark" / "gooseomni_v1"
    build_agentic_midgame_prediction(ann, bench, limit=4)
    trials = _read_jsonl(
        bench / "public" / "agentic_midgame_prediction" / "trials.jsonl"
    )
    responses = []
    for row in trials:
        if row["probe_family"] == "E_hidden_world_state_estimation":
            parsed = {
                "oracle_guess": {"event_type": "location"},
                "evidence_ids": ["claim_000001"],
                "must_not_claim_direct_observation": True,
            }
        elif row["probe_family"] == "F_other_player_current_action_prediction":
            parsed = {"predicted_action": "location", "evidence_ids": ["claim_000001"]}
        elif row["probe_family"] == "G_next_behavior_prediction":
            parsed = {
                "top1_next_behavior": "task",
                "topk_next_behaviors": ["task", "move"],
                "evidence_ids": ["claim_000001"],
            }
        else:
            parsed = {
                "global_truth_guess": "contradicted",
                "local_knowability": "cannot_know",
                "evidence_ids": ["claim_000001"],
            }
        responses.append(
            {
                "trial_id": row["trial_id"],
                "probe_family": row["probe_family"],
                "parsed": parsed,
                "raw_response": json.dumps(parsed),
            }
        )
    response_path = tmp_path / "responses.jsonl"
    write_jsonl(response_path, responses)

    aggregate = score_agentic_midgame_prediction(
        response_path,
        bench / "private" / "agentic_midgame_prediction" / "hidden_gold.jsonl",
        tmp_path / "scores.json",
    )

    assert aggregate["hidden_world_state_accuracy"] == 1.0
    assert aggregate["next_behavior_top1_accuracy"] == 1.0
    assert aggregate["deception_detection_global"] == 1.0
    assert aggregate["death_skill_overclaim_rate"] == 0.0


def test_death_skill_overclaim_detector() -> None:
    assert contains_death_skill_overclaim("从尸体和血迹可知 X 用鸭子技能杀了 Y")
    assert not contains_death_skill_overclaim("画面只能说明这里有尸体，不能确认凶手。")
