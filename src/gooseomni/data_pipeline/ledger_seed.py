from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .v2 import write_jsonl


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"fusion artifact must be a JSON array: {path}")
    return value


def _world_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for index, row in enumerate(rows, start=1):
        sources = [str(value) for value in row.get("source_player_ids", [])]
        actors = [
            str(value)
            for value in row.get("involved_players", [])
            if str(value) in sources
        ] or sources
        events.append(
            {
                "world_event_id": f"seed_ge_{index:06d}",
                "game_id": str(row.get("game_id", "unknown")),
                "source_segment_ids": list(row.get("source_clip_ids", [])),
                "source_povs": sources,
                "abs_start_sec": float(row.get("start_sec", 0.0)),
                "abs_end_sec": float(row.get("end_sec", row.get("start_sec", 0.0))),
                "phase_type": "meeting"
                if "meeting" in str(row.get("event_type", "")).lower()
                else "gameplay",
                "event_type": str(row.get("event_type", "unknown")),
                "actors": actors,
                "patients": [],
                "location": str(row.get("location", "unknown")),
                "description": str(row.get("description", "")),
                "direct_visual_evidence": list(row.get("source_clip_ids", [])),
                "direct_audio_evidence": [],
                "public_evidence": [],
                "inferred_fields": [],
                "certainty": float(row.get("confidence", 0.5)),
                "needs_human_review": False,
                "gold_source": "qwen_seed",
            }
        )
    return events


def _claims(rows: list[dict[str, Any]], players: list[str]) -> list[dict[str, Any]]:
    claims = []
    for index, row in enumerate(rows, start=1):
        speaker = str(row.get("speaker_id") or row.get("player_id") or "unknown")
        claims.append(
            {
                "claim_id": f"seed_claim_{index:06d}",
                "game_id": str(row.get("game_id", "unknown")),
                "source_segment_ids": [str(row.get("clip_id", ""))],
                "speaker": speaker,
                "heard_by": players,
                "abs_start_sec": float(row.get("start_sec", 0.0)),
                "abs_end_sec": float(row.get("end_sec", row.get("start_sec", 0.0))),
                "claim_type": str(row.get("speech_act", "other")),
                "content": str(row.get("text", "")),
                "normalized_content": str(row.get("text", "")),
                "time_referred": {"type": "unknown"},
                "target_entities": list(row.get("mentioned_players", [])),
                "related_event_ids": [],
                "strategic_role": "information_sharing",
                "certainty": float(row.get("confidence", 0.5)),
                "needs_human_review": False,
                "gold_source": "qwen_seed",
            }
        )
    return claims


def _visibility(
    events: list[dict[str, Any]], players: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        sources = set(event["source_povs"])
        for player in players:
            visible = player in sources
            rows.append(
                {
                    "edge_id": f"vis_{event['world_event_id']}_{player}",
                    "game_id": event["game_id"],
                    "event_id": event["world_event_id"],
                    "player_id": player,
                    "cutoff_abs_sec": event["abs_end_sec"],
                    "visibility": "direct_visual" if visible else "not_visible",
                    "evidence_ids": [event["world_event_id"]] if visible else [],
                    "explanation": "source POV observed event"
                    if visible
                    else "event absent from player source POV",
                    "confidence": event["certainty"],
                }
            )
    return rows


def _snapshots(
    events: list[dict[str, Any]], claims: list[dict[str, Any]], players: list[str]
) -> list[dict[str, Any]]:
    end = max((float(row["abs_end_sec"]) for row in events), default=0.0)
    cutoffs = [
        float(value) for value in range(0, int(math.ceil(end / 10.0)) * 10 + 1, 10)
    ]
    if end not in cutoffs:
        cutoffs.append(end)
    rows = []
    for player in players:
        for index, cutoff in enumerate(sorted(set(cutoffs)), start=1):
            visible = [
                row
                for row in events
                if player in row["source_povs"] and float(row["abs_end_sec"]) <= cutoff
            ][-12:]
            heard = [row for row in claims if float(row["abs_end_sec"]) <= cutoff][-12:]
            available = [row["world_event_id"] for row in visible] + [
                row["claim_id"] for row in heard
            ]
            hidden = [
                row
                for row in events
                if player not in row["source_povs"]
                and float(row["abs_end_sec"]) <= cutoff
            ][-24:]
            rows.append(
                {
                    "snapshot_id": f"seed_mem_{player}_{index:06d}",
                    "game_id": events[0]["game_id"] if events else "unknown",
                    "target_player": player,
                    "cutoff_abs_sec": cutoff,
                    "public_history": [
                        {
                            "evidence_id": row["claim_id"],
                            "abs_sec": row["abs_end_sec"],
                            "type": "meeting_statement",
                            "content": row["content"],
                        }
                        for row in heard
                    ],
                    "private_observations": [
                        {
                            "evidence_id": row["world_event_id"],
                            "world_event_id": row["world_event_id"],
                            "visibility": "direct_visual",
                            "content": row["description"],
                            "confidence": row["certainty"],
                        }
                        for row in visible
                    ],
                    "heard_claims": [
                        {
                            "evidence_id": row["claim_id"],
                            "claim_id": row["claim_id"],
                            "speaker": row["speaker"],
                            "content": row["content"],
                            "local_truth_awareness": "not_enough_information",
                        }
                        for row in heard
                    ],
                    "inferred_beliefs": [],
                    "hidden_events_for_target": [
                        {
                            "world_event_id": row["world_event_id"],
                            "reason_hidden": "not_visible",
                            "visible_to": row["source_povs"],
                        }
                        for row in hidden
                    ],
                    "forbidden_event_ids": [row["world_event_id"] for row in hidden],
                    "available_evidence_ids": available,
                    "gold_source": "qwen_seed",
                }
            )
    return rows


def build_seed_ledger(
    fusion_root: Path, output_root: Path, players: list[str]
) -> dict[str, int]:
    events = _world_events(_load(fusion_root / "global_events.json"))
    claims = _claims(_load(fusion_root / "meeting_utterances.json"), players)
    visibility = _visibility(events, players)
    snapshots = _snapshots(events, claims, players)
    game_id = events[0]["game_id"] if events else "unknown"
    end = max((float(row["abs_end_sec"]) for row in events), default=0.0)
    phase_events = [
        {
            "phase_event_id": "seed_phase_000001",
            "game_id": game_id,
            "episode_id": f"{game_id}_episode_000",
            "phase_id": f"{game_id}_seed_gameplay",
            "phase_type": "gameplay",
            "phase_order_label_zh": "Qwen seed phase",
            "abs_start_sec": 0.0,
            "abs_end_sec": end,
            "previous_phase_id": None,
            "next_phase_id": None,
        }
    ]
    canonical = [
        {
            "canonical_event_id": row["world_event_id"],
            "duplicate_local_event_ids": row["source_segment_ids"],
            "abs_time_cluster": [row["abs_start_sec"], row["abs_end_sec"]],
            "canonical_source": row["source_segment_ids"][0]
            if row["source_segment_ids"]
            else row["world_event_id"],
            "merge_reason": "Qwen POV deterministic seed fusion",
        }
        for row in events
    ]
    links = [
        {
            "claim_truth_link_id": f"seed_ctl_{row['claim_id']}",
            "claim_id": row["claim_id"],
            "world_event_ids": [],
            "truth_status_global": "unverified",
            "local_awareness_by_player": dict.fromkeys(players, "unknown"),
            "explanation": "Seed claim requires Codex adjudication",
            "confidence": row["certainty"],
            "needs_human_review": False,
        }
        for row in claims
    ]
    payloads = {
        "world_events.jsonl": events,
        "claims.jsonl": claims,
        "phase_events.jsonl": phase_events,
        "visibility_edges.jsonl": visibility,
        "belief_memory_snapshots.jsonl": snapshots,
        "claim_truth_links.jsonl": links,
        "canonical_event_map.jsonl": canonical,
    }
    for name, values in payloads.items():
        write_jsonl(output_root / name, values)
    return {
        name.removesuffix(".jsonl"): len(values) for name, values in payloads.items()
    }
