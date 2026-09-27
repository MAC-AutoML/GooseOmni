from __future__ import annotations

from typing import Any

EDGE_TYPES = {
    "direct_visual",
    "direct_audio",
    "public_ui",
    "heard_claim",
    "inferred_only",
    "not_visible",
    "post_cutoff",
    "unknown",
}


def observed_cutoffs(nodes: list[dict[str, Any]]) -> list[float]:
    """Use only reviewed event boundaries represented in the current run."""
    return sorted({float(row["abs_end_sec"]) for row in nodes})


def access_edge(node: dict[str, Any], player: str, cutoff: float) -> str:
    if float(node["abs_start_sec"]) > cutoff:
        return "post_cutoff"
    if node.get("public_ui"):
        return "public_ui"
    if player in node.get("source_povs", []):
        if node.get("modality") == "audio" or node.get("utterance"):
            return "direct_audio"
        return "direct_visual"
    if player in node.get("heard_by", []) and node.get("audio_admissible"):
        return "heard_claim"
    if node.get("global_fact"):
        return "not_visible"
    return "unknown"


def build_information_state(
    episode_id: str,
    player: str,
    cutoff: float,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    edges = [
        {
            "trajectory_node_id": node["trajectory_node_id"],
            "access": access_edge(node, player, cutoff),
        }
        for node in nodes
    ]
    accessible_ids = {
        row["trajectory_node_id"]
        for row in edges
        if row["access"] in {"direct_visual", "direct_audio", "public_ui", "heard_claim"}
    }
    public_ids = {
        node["trajectory_node_id"]
        for node in nodes
        if node.get("public_ui") and float(node["abs_start_sec"]) <= cutoff
    }
    forbidden = {
        node["trajectory_node_id"]
        for node in nodes
        if float(node["abs_start_sec"]) > cutoff
        or node["trajectory_node_id"] not in accessible_ids | public_ids
    }
    return {
        "information_state_id": f"{episode_id}:{player}:{cutoff:.3f}",
        "episode_id": episode_id,
        "player_id": player,
        "cutoff_abs_sec": cutoff,
        "private_observations": sorted(accessible_ids - public_ids),
        "public_history": sorted(public_ids),
        "heard_claims": sorted(
            row["trajectory_node_id"]
            for row in edges
            if row["access"] == "heard_claim"
        ),
        "known_facts": sorted(accessible_ids | public_ids),
        "belief_distribution": [],
        "intent_hypotheses": [],
        "stance_and_trust": [],
        "forbidden_events": sorted(forbidden),
        "available_evidence_ids": sorted(
            {
                evidence
                for node in nodes
                if node["trajectory_node_id"] in accessible_ids | public_ids
                for evidence in node.get("evidence_asset_ids", [])
            }
        ),
        "visibility_edges": edges,
    }


def validate_claim_hearing(row: dict[str, Any], players: list[str]) -> list[str]:
    issues = []
    canonical_players = set(players)
    heard_by = row.get("heard_by")
    if heard_by is None:
        issues.append("claim has no evidence-derived heard_by")
    else:
        heard_by_players = set(heard_by)
        if heard_by_players - canonical_players:
            issues.append("claim heard_by contains non-canonical players")
        if heard_by_players == canonical_players and not row.get("meeting_public"):
            issues.append("non-public claim cannot default to all players")
    speaker_id = row.get("speaker_id")
    if speaker_id is not None and speaker_id not in canonical_players:
        issues.append("claim speaker is not a canonical player")
    if not row.get("audio_admissible") and not row.get("visual_text_evidence_id"):
        issues.append("claim lacks admissible audio or visual text evidence")
    return issues
