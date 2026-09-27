from __future__ import annotations

import traceback
import os
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .io import resolve_video_path, safe_json_loads, write_json, write_jsonl
from .schema import AnnotationError, POVRef, Segment, VALID_PLAYER_SET

PLAYER_ALIASES = {
    "末将": "mojiang",
    "墨将": "mojiang",
    "魔将": "mojiang",
    "北港": "beigang",
    "北刚": "beigang",
    "小鹿": "xiaolu",
    "扫一": "saoyi",
    "扫姨": "saoyi",
    "骚艺": "saoyi",
}


def _clamp_certainty(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _short_text(value: Any, limit: int = 120) -> str:
    if value is None:
        return "unknown"
    text = str(value).replace("\n", " ").strip()
    return text[:limit] if text else "unknown"


def _short_list(values: Any, limit: int = 6) -> list[str]:
    if not isinstance(values, list):
        return []
    rows: list[str] = []
    for value in values:
        text = _short_text(value, 80)
        if text != "unknown":
            rows.append(text)
        if len(rows) >= limit:
            break
    return list(dict.fromkeys(rows))


def _short_optional_text(value: Any, limit: int = 120) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return "unknown" if not value else _short_text(", ".join(str(item) for item in value), limit)
    if isinstance(value, dict):
        return _short_text(json.dumps(value, ensure_ascii=False), limit)
    return _short_text(value, limit)


def output_root(dataset_root: Path) -> Path:
    configured = os.getenv("GOOSEOMNI_ANNOTATION_ROOT")
    if configured:
        return Path(configured)
    resolved = dataset_root.resolve()
    if resolved.name == "gooseomni" and resolved.parent.name == "data":
        return resolved.parents[1] / "annotations"
    return Path.cwd() / "annotations"


def abs_time(segment: Segment, local_sec: float) -> float:
    return segment.aligned_start_sec + local_sec


def context_fields(segment: Segment, pov: POVRef | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "dataset": segment.dataset,
        "game_id": segment.game_id,
        "segment_id": segment.segment_id,
    }
    if pov is not None:
        fields["player_id"] = pov.player_id
        fields["source_pov"] = [pov.player_id]
    return fields


def prepare_timed_payload(
    item: dict[str, Any],
    segment: Segment,
    *,
    pov: POVRef | None = None,
    target_player: str | None = None,
    item_id: str | None = None,
    id_field: str | None = None,
) -> dict[str, Any]:
    payload = dict(item)
    payload.update(context_fields(segment, pov))
    if "source_povs" in payload and "source_pov" not in payload:
        payload["source_pov"] = payload.pop("source_povs")
    if id_field and item_id and not payload.get(id_field):
        payload[id_field] = item_id
    if target_player is not None:
        payload["target_player"] = target_player
    local_start, local_end = normalize_model_time_window(
        segment,
        float(payload["local_start_sec"]),
        float(payload["local_end_sec"]),
    )
    validate_local_window(segment, local_start, local_end)
    payload["local_start_sec"] = local_start
    payload["local_end_sec"] = local_end
    payload["abs_start_sec"] = abs_time(segment, local_start)
    payload["abs_end_sec"] = abs_time(segment, local_end)
    if pov is not None:
        payload["player_id"] = pov.player_id
        payload.setdefault("source_pov", [pov.player_id])
    return payload


def normalize_pov_event_payload(item: dict[str, Any], segment: Segment, pov: POVRef, index: int) -> dict[str, Any]:
    payload = prepare_timed_payload(
        item,
        segment,
        pov=pov,
        item_id=f"{segment.segment_id}_{pov.player_id}_event_{index:03d}",
        id_field="event_id",
    )
    payload["actor"] = normalize_optional_player(payload.get("actor"), default=pov.player_id)
    payload["speaker"] = normalize_optional_player(payload.get("speaker"), default=None)
    payload.setdefault("utterance", None)
    payload.setdefault("claim_type", None)
    if not payload.get("location"):
        payload["location"] = "unknown"
    payload["visible_players"] = normalize_player_list(payload.get("visible_players", []))
    payload["mentioned_players"] = normalize_player_list(payload.get("mentioned_players", []))
    return payload


def normalize_optional_player(value: Any, default: str | None = None) -> str | None:
    if value in {None, "", "unknown"}:
        return default if default in VALID_PLAYER_SET else None
    text = str(value)
    mapped = PLAYER_ALIASES.get(text, text)
    if mapped in VALID_PLAYER_SET:
        return mapped
    return default if default in VALID_PLAYER_SET else "unknown" if default == "unknown" else None


def normalize_player_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        mapped = normalize_optional_player(value, default=None)
        if mapped in VALID_PLAYER_SET:
            normalized.append(mapped)
    return list(dict.fromkeys(normalized))


def normalize_utterance_payload(item: dict[str, Any], segment: Segment, pov: POVRef, index: int) -> dict[str, Any]:
    payload = dict(item)
    if "transcript" in payload and "text" not in payload:
        payload["text"] = payload["transcript"]
    if "text" in payload and "transcript" not in payload:
        payload["transcript"] = payload["text"]
    payload["speaker"] = normalize_optional_player(payload.get("speaker"), default="unknown") or "unknown"
    payload.setdefault("speaker_confidence", payload.get("certainty", 0.0))
    payload["addressee"] = normalize_player_list(payload.get("addressee", []))
    payload["mentioned_players"] = normalize_player_list(payload.get("mentioned_players", []))
    payload.setdefault("claims", [])
    payload.setdefault("possible_intents", [])
    payload.setdefault("is_direct_observation", False)
    payload.setdefault("is_speech_claim", True)
    payload.setdefault("needs_human_review", payload.get("speaker") == "unknown")
    normalized = prepare_timed_payload(
        payload,
        segment,
        pov=pov,
        item_id=f"{segment.segment_id}_{pov.player_id}_utt_{index:03d}",
        id_field="utterance_id",
    )
    for claim_index, claim in enumerate(normalized.get("claims", []), start=1):
        claim.setdefault(
            "claim_id",
            f"{normalized['utterance_id']}_claim_{claim_index:03d}",
        )
        claim["speaker"] = normalize_optional_player(claim.get("speaker"), default=normalized.get("speaker", "unknown")) or "unknown"
        claim.setdefault("evidence", normalized.get("evidence", ""))
        claim["mentioned_players"] = normalize_player_list(claim.get("mentioned_players", []))
        claim["subject_players"] = normalize_player_list(claim.get("subject_players", []))
        claim["object_players"] = normalize_player_list(claim.get("object_players", []))
        claim["locations"] = _short_list(claim.get("locations", []), 6)
        claim["time_referred"] = _short_optional_text(claim.get("time_referred"), 80)
        claim["content"] = _short_text(claim.get("content"), 160)
        claim["claim_type"] = _short_text(claim.get("claim_type"), 80)
        claim["evidence"] = _short_text(claim.get("evidence"), 160)
        claim["certainty"] = _clamp_certainty(claim.get("certainty"), normalized.get("certainty", 0.5))
    return normalized


def normalize_global_event_payload(item: dict[str, Any], segment: Segment, index: int) -> dict[str, Any]:
    payload = dict(item)
    if "source_povs" in payload and "source_pov" not in payload:
        payload["source_pov"] = payload.pop("source_povs")
    if "actors" in payload and "involved_players" not in payload:
        payload["involved_players"] = payload["actors"]
    payload.setdefault("target_player", "global")
    payload.setdefault("actors", payload.get("involved_players", []))
    payload.setdefault("visible_to", [])
    payload.setdefault("heard_by", [])
    if isinstance(payload.get("source_pov"), str):
        payload["source_pov"] = [payload["source_pov"]]
    payload["source_pov"] = normalize_player_list(payload.get("source_pov", []))
    payload["actors"] = normalize_player_list(payload.get("actors", []))
    payload["involved_players"] = normalize_player_list(payload.get("involved_players", []))
    payload["visible_to"] = normalize_player_list(payload.get("visible_to", []))
    payload["heard_by"] = normalize_player_list(payload.get("heard_by", []))
    payload["not_visible_to"] = normalize_player_list(payload.get("not_visible_to", []))
    if not payload["source_pov"]:
        payload["source_pov"] = [segment.povs[0].player_id]
        payload["needs_human_review"] = True
    payload.setdefault("is_direct_observation", True)
    payload.setdefault("is_speech_claim", False)
    payload.setdefault("needs_human_review", False)
    payload["description"] = _short_text(payload.get("description"), 160)
    payload["evidence"] = _short_text(payload.get("evidence"), 160)
    payload["supporting_pov_event_ids"] = _short_list(payload.get("supporting_pov_event_ids", []), 6)
    payload["supporting_utterance_ids"] = _short_list(payload.get("supporting_utterance_ids", []), 6)
    payload["related_claim_ids"] = _short_list(payload.get("related_claim_ids", []), 6)
    payload["certainty"] = _clamp_certainty(payload.get("certainty"), 0.5)
    payload.setdefault("conflict", False)
    return prepare_timed_payload(
        payload,
        segment,
        item_id=f"{segment.segment_id}_ge_{index:03d}",
        id_field="global_event_id",
    )


def normalize_candidate_trial_payload(item: dict[str, Any], segment: Segment, index: int) -> dict[str, Any]:
    payload = dict(item)
    payload.update(
        {
            "dataset": segment.dataset,
            "game_id": segment.game_id,
            "segment_id": segment.segment_id,
        }
    )
    trial_id = str(payload.get("trial_id") or f"trial_{index:03d}")
    if not trial_id.startswith(f"{segment.segment_id}_"):
        trial_id = f"{segment.segment_id}_{trial_id}"
    payload["trial_id"] = trial_id
    if payload.get("question_type") == "belief_state":
        payload["question_type"] = "first_order_belief"
        payload.setdefault("trial_type", "belief_state")
    if payload.get("question_type") not in {
        "first_order_belief",
        "second_order_belief",
        "hidden_information",
        "false_belief",
        "intent_inference",
        "knowledge_access",
        "belief_state",
    }:
        payload["question_type"] = "first_order_belief"
        payload["needs_human_review"] = True
    payload.setdefault("trial_type", payload.get("question_type"))
    target = normalize_optional_player(payload.get("target_player"), default=None)
    if target is None:
        text = " ".join(
            _short_text(payload.get(key), 200)
            for key in ["question", "answer", "expected_answer_basis"]
        )
        target = next((player for player in VALID_PLAYER_SET if player in text), None)
    payload["target_player"] = target or segment.povs[0].player_id
    cutoff = payload.get("cutoff_abs_sec")
    try:
        cutoff_float = float(cutoff)
    except (TypeError, ValueError):
        cutoff_float = segment.aligned_end_sec
    if cutoff_float < segment.aligned_start_sec or cutoff_float > segment.aligned_end_sec:
        cutoff_float = segment.aligned_end_sec
    payload["cutoff_abs_sec"] = cutoff_float
    payload["question"] = _short_text(payload.get("question"), 200)
    payload["answer"] = _short_text(payload.get("answer"), 200)
    payload["distractors"] = _short_list(payload.get("distractors", []), 4)
    payload["available_information"] = _short_list(payload.get("available_information", []), 6)
    payload["hidden_information"] = _short_list(payload.get("hidden_information", []), 6)
    payload["supporting_global_event_ids"] = _short_list(payload.get("supporting_global_event_ids", []), 6)
    payload["supporting_information_state_ids"] = _short_list(payload.get("supporting_information_state_ids", []), 6)
    payload["expected_answer_basis"] = _short_text(payload.get("expected_answer_basis"), 160)
    if isinstance(payload.get("source_pov"), str):
        payload["source_pov"] = [payload["source_pov"]]
    if "source_povs" in payload and "source_pov" not in payload:
        payload["source_pov"] = payload.pop("source_povs")
    payload["source_pov"] = normalize_player_list(payload.get("source_pov", []))
    if not payload["source_pov"]:
        payload["source_pov"] = [pov.player_id for pov in segment.povs]
        payload["needs_human_review"] = True
    if not payload.get("evidence"):
        basis = payload.get("expected_answer_basis") or payload.get("answer") or payload.get("question") or "unknown"
        payload["evidence"] = str(basis)
        payload["needs_human_review"] = True
    payload["evidence"] = _short_text(payload.get("evidence"), 160)
    payload["certainty"] = _clamp_certainty(
        payload.get("certainty"),
        0.5 if payload.get("needs_human_review") else 0.7,
    )
    if payload.get("risk_of_perspective_leakage") not in {"low", "medium", "high"}:
        payload["risk_of_perspective_leakage"] = "medium" if payload.get("needs_human_review") else "low"
    payload.setdefault("needs_human_review", False)
    return payload


def normalize_phase_event_payload(item: dict[str, Any], segment: Segment, index: int) -> dict[str, Any]:
    payload = dict(item)
    payload.setdefault("target_player", "global")
    payload.setdefault("visible_to_all", False)
    if isinstance(payload.get("source_pov"), str):
        payload["source_pov"] = [payload["source_pov"]]
    if isinstance(payload.get("evidence_povs"), str):
        payload["evidence_povs"] = [payload["evidence_povs"]]
    payload.setdefault("evidence_povs", payload.get("source_pov", []))
    payload["evidence_povs"] = normalize_player_list(payload.get("evidence_povs", []))
    valid_phase_types = {"discussion", "voting", "vote_result", "exile", "action", "body_report", "meeting_start", "unknown"}
    if payload.get("phase_type") not in valid_phase_types:
        payload["phase_type"] = "unknown"
        payload["needs_human_review"] = True
    return prepare_timed_payload(
        payload,
        segment,
        item_id=f"{segment.segment_id}_phase_{index:03d}",
        id_field="phase_event_id",
    )


def normalize_cutoff_payload(item: dict[str, Any], segment: Segment, target_player: str) -> dict[str, Any]:
    payload = dict(item)
    payload.update(
        {
            "dataset": segment.dataset,
            "game_id": segment.game_id,
            "segment_id": segment.segment_id,
            "target_player": target_player,
            "cutoff_abs_sec": segment.aligned_end_sec,
        }
    )
    return payload


def normalize_model_time_window(segment: Segment, start_sec: float, end_sec: float) -> tuple[float, float]:
    if 0 <= start_sec <= segment.duration_sec and 0 <= end_sec <= segment.duration_sec:
        local_start, local_end = start_sec, end_sec
    elif segment.aligned_start_sec <= start_sec <= segment.aligned_end_sec and segment.aligned_start_sec <= end_sec <= segment.aligned_end_sec:
        local_start = start_sec - segment.aligned_start_sec
        local_end = end_sec - segment.aligned_start_sec
    else:
        return start_sec, end_sec
    if local_end <= local_start:
        local_end = min(segment.duration_sec, local_start + 1.0)
        if local_end <= local_start and local_start > 0:
            local_start = max(0.0, local_end - 1.0)
    return local_start, local_end


def validate_local_window(segment: Segment, local_start_sec: float, local_end_sec: float) -> None:
    if local_start_sec < 0 or local_end_sec > segment.duration_sec:
        raise ValueError(
            f"local time window [{local_start_sec}, {local_end_sec}] exceeds "
            f"segment duration {segment.duration_sec}"
        )
    if local_end_sec <= local_start_sec:
        raise ValueError("local_end_sec must be greater than local_start_sec")


def validate_player_id(player_id: str) -> None:
    if player_id not in VALID_PLAYER_SET:
        raise ValueError(f"player_id must be one of {sorted(VALID_PLAYER_SET)}")


def save_error(
    *,
    dataset_root: Path,
    annotation_root: Path | None = None,
    stage: str,
    segment: Segment,
    prompt: str,
    error: Exception,
    raw_response: str | None = None,
    pov: POVRef | None = None,
    target_player: str | None = None,
) -> Path:
    root = (annotation_root or output_root(dataset_root)) / "errors" / stage
    player = pov.player_id if pov else target_player
    suffix = player or "global"
    path = root / f"{segment.segment_id}_{suffix}.json"
    write_json(
        path,
        AnnotationError(
            stage=stage,
            segment_id=segment.segment_id,
            player_id=pov.player_id if pov else None,
            target_player=target_player,
            video_file=pov.video_file if pov else None,
            raw_response=raw_response,
            error_message="".join(traceback.format_exception_only(type(error), error)).strip(),
            prompt=prompt,
        ),
    )
    return path


