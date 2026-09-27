from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gooseomni.annotation.json_utils import parse_json
from gooseomni.annotation.perception_quality import normalize_pass_event
from gooseomni.annotation.runner import normalize_pov_event_payload
from gooseomni.annotation.schemas import Clip
from gooseomni.models.model_server.local_common.timestamp_overlay import (
    create_timestamp_contact_sheet,
)

from .information_state import validate_claim_hearing
from .provenance import write_json
from .trajectory import fuse_trajectory
from .v2 import read_jsonl, write_jsonl

PUBLIC_VISUAL_TYPES = {"meeting", "vote", "body_report", "action_outcome"}


def _visual_payloads(root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(root.glob("*.json")):
        if not path.name.startswith("._"):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def load_visual_error_candidates(
    error_root: Path, output_root: Path | None = None
) -> list[dict[str, Any]]:
    """Recover failed visual rows only as quarantined Codex-review candidates."""
    rows: list[dict[str, Any]] = []
    if not error_root.is_dir():
        return rows
    for path in sorted(error_root.glob("*.json")):
        if path.name.startswith("._"):
            continue
        if output_root is not None and (output_root / path.name).is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        clip = Clip.model_validate(record["clip"])
        try:
            payload = parse_json(str(record.get("raw_response", "")))
        except (TypeError, ValueError):
            continue
        for index, item in enumerate(payload):
            normalized = normalize_pass_event(
                normalize_pov_event_payload(item, clip), "visual"
            )
            normalized["evidence_id"] = (
                f"{clip.clip_id}:visual_quarantine:{index:02d}"
            )
            normalized["quality_gate_failed"] = True
            rows.append(normalized)
    return rows


def recoverable_visual_errors(
    pass_kind: str, error_root: Path, output_root: Path | None = None
) -> bool:
    """Allow visual-only failures to proceed solely through Codex quarantine."""
    return pass_kind == "visual" and bool(
        load_visual_error_candidates(error_root, output_root)
    )


def _frame_packs(run_root: Path) -> dict[str, str]:
    packs: dict[str, str] = {}
    target = run_root / "local_codex/frame_packs"
    payloads = _visual_payloads(run_root / "annotations/perception_visual")
    payloads.extend(_visual_payloads(run_root / "errors/perception_visual"))
    clips = {
        str(payload["clip"]["clip_id"]): payload["clip"]
        for payload in payloads
        if isinstance(payload.get("clip"), dict)
    }
    for clip_id, clip in sorted(clips.items()):
        path = create_timestamp_contact_sheet(
            clip["clip_path"],
            target / f"{clip_id}.jpg",
            float(clip["start_sec"]),
        )
        packs[clip_id] = str(path)
    return packs


def _pack_for_node(node: dict[str, Any], packs: dict[str, str]) -> str | None:
    for evidence_id in node.get("evidence_asset_ids", []):
        clip_id = str(evidence_id).split(":visual", 1)[0]
        if clip_id in packs:
            return packs[clip_id]
    return None


def _prepare_queue(
    run_root: Path,
    visual_nodes: list[dict[str, Any]],
) -> Path:
    packs = _frame_packs(run_root)
    queue = run_root / "local_codex/trajectory_review_queue.jsonl"
    write_jsonl(
        queue,
        [
            {
                "trajectory_node_id": node["trajectory_node_id"],
                "candidate": node,
                "frame_pack_path": _pack_for_node(node, packs),
                "available_evidence_ids": node.get("evidence_asset_ids", []),
                "required_schema": {
                    "decision": "accept|repair|reject",
                    "reason": "string",
                    "evidence_ids": "list[string]",
                    "final_value": "object|null",
                },
            }
            for node in visual_nodes
        ],
    )
    return queue


def _apply_decisions(
    visual_nodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    players: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = {str(row["trajectory_node_id"]): row for row in visual_nodes}
    decision_ids = [str(row.get("trajectory_node_id")) for row in decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(candidates):
        raise ValueError("trajectory decisions must cover each visual node exactly once")
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    accepted = []
    for decision in decisions:
        node_id = str(decision["trajectory_node_id"])
        candidate = candidates[node_id]
        action = decision.get("decision")
        if action not in {"accept", "repair", "reject"}:
            raise ValueError(f"invalid trajectory decision: {node_id}")
        cited = {str(row) for row in decision.get("evidence_ids", [])}
        allowed = {str(row) for row in candidate.get("evidence_asset_ids", [])}
        if cited - allowed:
            raise ValueError(f"trajectory decision cites unknown evidence: {node_id}")
        if action == "reject":
            continue
        if action == "repair":
            repaired = decision.get("final_value")
            if not isinstance(repaired, dict):
                raise ValueError(f"trajectory repair requires final_value: {node_id}")
            node = {**candidate, **repaired}
            node["review_original_value"] = candidate
            node["review_diff_fields"] = sorted(
                key for key, value in repaired.items() if candidate.get(key) != value
            )
        else:
            node = dict(candidate)
        node["trajectory_node_id"] = node_id
        node["episode_id"] = candidate["episode_id"]
        node["source_povs"] = candidate["source_povs"]
        node["evidence_asset_ids"] = candidate["evidence_asset_ids"]
        episode = episode_by_id[str(node["episode_id"])]
        start = float(node["abs_start_sec"])
        end = float(node["abs_end_sec"])
        if not (episode["abs_start_sec"] <= start < end <= episode["abs_end_sec"]):
            raise ValueError(f"trajectory repair has invalid time range: {node_id}")
        if node.get("event_type") in {"task_ui", "interaction"}:
            node["movement_transition"] = None
        if node.get("global_fact") and not node.get("public_ui"):
            raise ValueError(f"private trajectory cannot become a global fact: {node_id}")
        if node.get("public_ui") and node.get("event_type") not in PUBLIC_VISUAL_TYPES:
            raise ValueError(f"invalid public UI event type: {node_id}")
        if players is not None:
            canonical = set(players)
            normalized_fields = {}
            for key in ("visible_players", "mentioned_players"):
                original = [str(value) for value in node.get(key, [])]
                normalized = sorted(
                    {value if value in canonical else "unknown" for value in original}
                )
                if normalized != original:
                    normalized_fields[key] = original
                    node[key] = normalized
            if normalized_fields:
                node["canonicalization_original"] = normalized_fields
                node["canonicalization_fields"] = sorted(normalized_fields)
        node["review_decision"] = action
        node["review_reason"] = str(decision.get("reason", ""))
        node["gold_source"] = "model_verified"
        accepted.append(node)
    return accepted


def review_and_write_trajectory(
    run_root: Path,
    visual: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    players: list[str],
) -> dict[str, Any]:
    nodes = fuse_trajectory(visual, audio, episodes)
    visual_nodes = [row for row in nodes if row.get("modality") == "visual"]
    audio_nodes = [row for row in nodes if row.get("modality") == "audio"]
    decisions_path = run_root / "local_codex/trajectory_decisions.jsonl"
    if not decisions_path.is_file():
        queue = _prepare_queue(run_root, visual_nodes)
        raise RuntimeError(
            f"local Codex trajectory review is pending: {queue}; "
            "write trajectory_decisions.jsonl and resume"
        )
    reviewed_visual = _apply_decisions(
        visual_nodes, read_jsonl(decisions_path), episodes, players
    )
    audio_issues = [
        {"trajectory_node_id": row["trajectory_node_id"], "issues": issues}
        for row in audio_nodes
        if (issues := validate_claim_hearing(row, players))
    ]
    if audio_issues:
        write_json(run_root / "quarantine/audio_conflicts.json", audio_issues)
    rejected_audio_ids = {row["trajectory_node_id"] for row in audio_issues}
    accepted_audio = [
        row for row in audio_nodes
        if row["trajectory_node_id"] not in rejected_audio_ids
    ]
    accepted = reviewed_visual + accepted_audio
    target = run_root / "artifacts/trajectory_nodes.jsonl"
    write_jsonl(target, accepted)
    return {
        "nodes": len(accepted),
        "visual_reviewed": len(reviewed_visual),
        "audio_quarantined": len(audio_issues),
        "output": str(target),
    }
