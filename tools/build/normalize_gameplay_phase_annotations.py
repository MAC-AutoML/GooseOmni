from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_KEYS = [
    "direct_visual_observation",
    "speech_claim",
    "public_result",
    "inferred_belief",
    "hidden_or_not_visible_information",
]

PUBLIC_EVENT_TYPES = {
    "meeting",
    "meeting_start",
    "discussion",
    "vote",
    "voting",
    "vote_result",
    "exile",
    "public_result",
    "phase_transition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize gameplay-aware Omni Goose phase annotations before release.")
    parser.add_argument("--root", type=Path, required=True, help="Annotation root, e.g. phase_annotations/annotations/g001")
    parser.add_argument("--write", action="store_true", help="Write normalized JSON files. Without this, only report.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def rounded(value: float) -> float:
    return round(float(value), 3)


def parse_raw_response(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def normalize_timed_event(event: dict[str, Any], aligned_start: float, aligned_end: float, duration: float) -> tuple[bool, str]:
    local_start = as_float(event.get("local_start_sec"))
    local_end = as_float(event.get("local_end_sec"))
    if local_start is None or local_end is None:
        return False, "missing_time"

    eps = 0.05
    local_ok = -eps <= local_start <= duration + eps and -eps <= local_end <= duration + eps
    abs_ok = aligned_start - eps <= local_start <= aligned_end + eps and aligned_start - eps <= local_end <= aligned_end + eps

    if local_ok:
        abs_start = aligned_start + local_start
        abs_end = aligned_start + local_end
        reason = "already_local"
    elif abs_ok or local_start > duration + eps or local_end > duration + eps:
        abs_start = local_start
        abs_end = local_end
        local_start = abs_start - aligned_start
        local_end = abs_end - aligned_start
        reason = "converted_abs_to_local"
    else:
        return False, "unrecognized_time"

    if abs_end < abs_start:
        abs_start, abs_end = abs_end, abs_start
        reason = "swapped_time_order"
    if abs_end - abs_start < 0.1 and aligned_start - eps <= abs_start <= aligned_end + eps:
        abs_end = min(aligned_end, abs_start + 1.0)
        if abs_end - abs_start < 0.1:
            abs_start = max(aligned_start, abs_end - 1.0)
        reason = "expanded_point_time"

    clipped_abs_start = max(aligned_start, abs_start)
    clipped_abs_end = min(aligned_end, abs_end)
    if clipped_abs_end - clipped_abs_start < 0.1:
        return False, "outside_phase_time"

    if clipped_abs_start != abs_start or clipped_abs_end != abs_end:
        event["needs_human_review"] = True
        event["time_clipped_to_phase"] = True
        reason = "clipped_to_phase"

    new_values = {
        "local_start_sec": rounded(clipped_abs_start - aligned_start),
        "local_end_sec": rounded(clipped_abs_end - aligned_start),
        "abs_start_sec": rounded(clipped_abs_start),
        "abs_end_sec": rounded(clipped_abs_end),
    }
    previous_values = {key: event.get(key) for key in new_values}
    event.update(new_values)
    if reason == "already_local" and previous_values != new_values:
        reason = "already_local_added_abs"
    return True, reason


def ensure_gameplay_source_fields(event: dict[str, Any], phase_type: str) -> bool:
    changed = False
    event_type = str(event.get("event_type", "")).lower()
    if "direct_visual_observation" not in event:
        event["direct_visual_observation"] = True
        changed = True
    if "speech_claim" not in event:
        event["speech_claim"] = False
        changed = True
    if "public_result" not in event:
        event["public_result"] = phase_type == "meeting" or event_type in PUBLIC_EVENT_TYPES
        changed = True
    if "inferred_belief" not in event:
        event["inferred_belief"] = False
        changed = True
    if "hidden_or_not_visible_information" not in event:
        event["hidden_or_not_visible_information"] = False
        changed = True
    return changed


def ensure_utterance_fields(event: dict[str, Any]) -> bool:
    changed = False
    if "speech_claim" not in event:
        event["speech_claim"] = True
        changed = True
    return changed


def add_review_reason(obj: dict[str, Any], reason: str) -> None:
    reasons = obj.setdefault("review_reasons", [])
    if isinstance(reasons, list) and reason not in reasons:
        reasons.append(reason)
    obj["needs_human_review"] = True


def normalize_leakage_risk(question: dict[str, Any]) -> tuple[bool, str]:
    raw = question.get("risk_of_perspective_leakage", "unknown")
    text = str(raw).strip().lower()
    if text in {"high", "高"} or "高" in text:
        normalized = "high"
    elif text in {"medium", "中", "中等"} or "中" in text:
        normalized = "medium"
    elif text in {"low", "低", "无", "none"} or "低" in text:
        normalized = "low"
    elif text in {"unknown", "未知", ""} or "未知" in text:
        normalized = "unknown"
    else:
        normalized = text
    changed = normalized != raw
    if changed:
        question.setdefault("risk_of_perspective_leakage_original", raw)
        question["risk_of_perspective_leakage"] = normalized
    return changed, normalized


def fallback_gameplay_trace(obj: dict[str, Any], duration: float, aligned_start: float) -> dict[str, Any]:
    status = obj.get("player_status") if isinstance(obj.get("player_status"), dict) else {}
    memory = obj.get("private_memory") if isinstance(obj.get("private_memory"), list) else []
    memory0 = memory[0] if memory and isinstance(memory[0], dict) else {}
    evidence = status.get("death_evidence") or memory0.get("evidence") or "Qwen annotation had empty gameplay_trace; fallback status event requires review."
    description = memory0.get("description") or evidence
    local_end = min(duration, 5.0) if duration > 5.0 else duration
    return {
        "local_start_sec": 0.0,
        "local_end_sec": rounded(local_end),
        "abs_start_sec": rounded(aligned_start),
        "abs_end_sec": rounded(aligned_start + local_end),
        "event_type": "spectator_or_status_observation" if status.get("alive_state") == "dead" else "visible_gameplay_status",
        "location": "unknown",
        "visible_players": [],
        "actor": obj.get("player_id", "unknown"),
        "direct_visual_observation": True,
        "speech_claim": False,
        "public_result": False,
        "inferred_belief": False,
        "hidden_or_not_visible_information": False,
        "description": description,
        "evidence": evidence,
        "certainty": 0.45,
        "needs_human_review": True,
        "fallback_from_player_status": True,
    }


def normalize_file(path: Path) -> tuple[dict[str, int], bool]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    stats = {
        "files": 1,
        "events_converted_abs": 0,
        "events_removed_outside": 0,
        "events_already_local": 0,
        "source_fields_added": 0,
        "utterance_fields_added": 0,
        "fallback_gameplay_trace_added": 0,
        "leakage_review_marked": 0,
    }
    changed = False

    aligned_start = float(obj.get("aligned_start_sec", obj.get("time", {}).get("aligned_start_sec", 0.0)))
    aligned_end = float(obj.get("aligned_end_sec", obj.get("time", {}).get("aligned_end_sec", aligned_start)))
    duration = float(obj.get("duration_sec", obj.get("time", {}).get("duration_sec", aligned_end - aligned_start)))
    phase_type = str(obj.get("phase_type", "unknown"))

    if phase_type == "meeting" and not obj.get("utterances"):
        raw_obj = parse_raw_response(obj.get("raw_response"))
        raw_utterances = raw_obj.get("utterances") if isinstance(raw_obj, dict) else None
        if isinstance(raw_utterances, list) and raw_utterances:
            obj["utterances"] = raw_utterances
            add_review_reason(obj, "utterances_restored_from_raw_response")
            changed = True

    for collection in ["gameplay_trace", "utterances"]:
        value = obj.get(collection)
        if not isinstance(value, list):
            obj[collection] = []
            changed = True
            continue
        kept = []
        for event in value:
            if not isinstance(event, dict):
                changed = True
                continue
            ok, reason = normalize_timed_event(event, aligned_start, aligned_end, duration)
            if not ok:
                stats["events_removed_outside"] += 1
                changed = True
                add_review_reason(obj, f"removed_{collection}_{reason}")
                continue
            if reason == "already_local":
                stats["events_already_local"] += 1
            else:
                stats["events_converted_abs"] += 1
                changed = True
            if collection == "gameplay_trace" and ensure_gameplay_source_fields(event, phase_type):
                stats["source_fields_added"] += 1
                changed = True
            if collection == "utterances" and ensure_utterance_fields(event):
                stats["utterance_fields_added"] += 1
                changed = True
            kept.append(event)
        obj[collection] = kept

    if phase_type == "gameplay" and not obj.get("gameplay_trace"):
        obj["gameplay_trace"] = [fallback_gameplay_trace(obj, duration, aligned_start)]
        add_review_reason(obj, "fallback_gameplay_trace_added")
        stats["fallback_gameplay_trace_added"] += 1
        changed = True

    for question in obj.get("tom_questions", []) if isinstance(obj.get("tom_questions"), list) else []:
        if not isinstance(question, dict):
            continue
        risk_changed, risk = normalize_leakage_risk(question)
        if risk_changed:
            changed = True
        if risk in {"medium", "high"} and question.get("needs_human_review") is not True:
            question["needs_human_review"] = True
            add_review_reason(obj, "medium_high_leakage_question")
            stats["leakage_review_marked"] += 1
            changed = True

    if changed:
        obj["normalization"] = {
            "event_times": "local_start_sec/local_end_sec are local to this phase; abs_start_sec/abs_end_sec are aligned game seconds.",
            "source_fields": SOURCE_KEYS,
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats, changed


def main() -> None:
    args = parse_args()
    totals: dict[str, int] = {}
    changed_paths = []
    for path in sorted(args.root.glob("*/*.json")):
        if args.write:
            stats, changed = normalize_file(path)
        else:
            original = path.read_text(encoding="utf-8")
            stats, changed = normalize_file(path)
            path.write_text(original, encoding="utf-8")
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
        if changed:
            changed_paths.append(path.as_posix())
    totals["changed_files"] = len(changed_paths)
    print(json.dumps({"stats": totals, "changed_paths": changed_paths[:50]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
