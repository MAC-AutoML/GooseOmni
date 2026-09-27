from __future__ import annotations

from collections import defaultdict
from typing import Any

VISUAL_TYPES = {
    "movement",
    "encounter",
    "task",
    "task_ui",
    "interaction",
    "meeting",
    "vote",
    "kill",
    "body_report",
    "action_outcome",
}
AUDIO_TYPES = {"utterance", "claim", "reaction", "accusation", "defense", "audio_cue"}
PUBLIC_TYPES = {"meeting", "vote", "vote_result", "ejection", "action_outcome"}


def normalize_visual_event(row: dict[str, Any]) -> dict[str, Any]:
    event_type = str(row.get("event_type", "unknown"))
    if event_type not in VISUAL_TYPES:
        event_type = "unknown"
    movement = row.get("movement_transition")
    if event_type in {"task_ui", "interaction"}:
        movement = None
    return {
        **row,
        "event_type": event_type,
        "movement_transition": movement,
        "visible_players": sorted({str(item) for item in row.get("visible_players", [])}),
        "source_pov": str(row["player_id"]),
        "modality": "visual",
    }


def normalize_audio_event(row: dict[str, Any]) -> dict[str, Any]:
    event_type = str(row.get("event_type", "unknown"))
    if event_type not in AUDIO_TYPES:
        event_type = "unknown"
    speaker = row.get("speaker_id")
    confidence_value = row.get("speaker_confidence", row.get("confidence", 0.0))
    try:
        confidence = float(confidence_value or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    verified_text = bool(row.get("visual_text_evidence_id"))
    candidate_ok = speaker is not None and confidence >= 0.85 and bool(row.get("utterance"))
    return {
        **row,
        "event_type": event_type,
        "visible_players": [],
        "location": None,
        "speaker_id": speaker if candidate_ok else None,
        "utterance": row.get("utterance") if candidate_ok else None,
        "audio_candidate_ok": candidate_ok,
        "visual_text_verified": verified_text,
        "audio_admissible": False,
        "source_pov": str(row["player_id"]),
        "modality": "audio",
    }


def _audio_signature(row: dict[str, Any]) -> tuple[str, str] | None:
    if not row.get("audio_candidate_ok"):
        return None
    return (
        str(row["speaker_id"]),
        " ".join(str(row["utterance"]).lower().split()),
    )


def _same_phase(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("episode_id") == right.get("episode_id")
        and left.get("phase_index") == right.get("phase_index")
        and left.get("phase_type") == right.get("phase_type")
    )


def _can_fuse(
    group: list[dict[str, Any]], row: dict[str, Any], tolerance_sec: float
) -> bool:
    first = group[0]
    if (
        first["modality"] != row["modality"]
        or first["event_type"] != row["event_type"]
        or not _same_phase(first, row)
        or abs(float(first["start_sec"]) - float(row["start_sec"])) > tolerance_sec
    ):
        return False
    if row["modality"] == "audio":
        signature = _audio_signature(row)
        return signature is not None and signature == _audio_signature(first)
    return bool(first.get("public_ui")) and bool(row.get("public_ui")) and (
        first["event_type"] in PUBLIC_TYPES
    )


def fuse_trajectory(
    visual_rows: list[dict[str, Any]],
    audio_rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    tolerance_sec: float = 1.5,
) -> list[dict[str, Any]]:
    rows = [normalize_visual_event(row) for row in visual_rows]
    rows.extend(normalize_audio_event(row) for row in audio_rows)
    grouped: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: float(item["start_sec"])):
        match = next(
            (
                group
                for group in reversed(grouped)
                if _can_fuse(group, row, tolerance_sec)
            ),
            None,
        )
        if match is None:
            match = []
            grouped.append(match)
        match.append(row)
    nodes = []
    for index, group in enumerate(grouped):
        first = group[0]
        episode = next(
            (
                item
                for item in episodes
                if item["abs_start_sec"] <= float(first["start_sec"])
                < item["abs_end_sec"]
            ),
            None,
        )
        if episode is None:
            continue
        source_povs = sorted({str(row["source_pov"]) for row in group})
        public_support = first["event_type"] in PUBLIC_TYPES and all(
            bool(row.get("public_ui")) for row in group
        )
        grouped_audio = [row for row in group if row["modality"] == "audio"]
        agreed_audio = len({str(row["source_pov"]) for row in grouped_audio}) >= 2
        visual_text_verified = any(
            row.get("visual_text_verified") for row in grouped_audio
        )
        audio_admissible = bool(grouped_audio) and (
            agreed_audio or visual_text_verified
        )
        meeting_public = (
            first.get("phase_type") == "meeting"
            and any(bool(row.get("meeting_public")) for row in grouped_audio)
        )
        heard_by = sorted(
            {str(row["source_pov"]) for row in grouped_audio}
            | {
                str(player)
                for row in grouped_audio
                if meeting_public
                for player in row.get("heard_by", [])
            }
        )
        nodes.append(
            {
                "trajectory_node_id": f"{episode['episode_id']}_node_{index:05d}",
                "episode_id": episode["episode_id"],
                "phase_index": first.get("phase_index"),
                "phase_type": first.get("phase_type"),
                "abs_start_sec": min(float(row["start_sec"]) for row in group),
                "abs_end_sec": max(float(row["end_sec"]) for row in group),
                "player_id": first.get("player_id"),
                "event_type": first["event_type"],
                "modality": first["modality"],
                "location": first.get("location"),
                "description": first.get("description"),
                "movement_transition": first.get("movement_transition"),
                "visible_players": sorted(
                    {
                        str(item)
                        for item_row in group
                        for item in item_row.get("visible_players", [])
                    }
                ),
                "mentioned_players": sorted(
                    {
                        str(item)
                        for item_row in group
                        for item in item_row.get("mentioned_players", [])
                    }
                ),
                "source_povs": source_povs,
                "evidence_asset_ids": sorted(
                    {
                        str(item_row["evidence_id"])
                        for item_row in group
                        if item_row.get("evidence_id")
                    }
                ),
                "confidence": min(float(row.get("confidence", 0.0)) for row in group),
                "quality_gate_failed": any(
                    bool(row.get("quality_gate_failed")) for row in group
                ),
                "global_fact": public_support,
                "public_ui": public_support,
                "utterance": first.get("utterance") if audio_admissible else None,
                "speaker_id": first.get("speaker_id") if audio_admissible else None,
                "audio_admissible": audio_admissible if grouped_audio else None,
                "heard_by": heard_by if audio_admissible else [],
                "meeting_public": meeting_public,
            }
        )
    return nodes


def by_episode(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        result[str(node["episode_id"])].append(node)
    return dict(result)
