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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import (
    annotation_path,
    annotate_text_with_segment_context,
    append_review_items,
    filter_segments,
    load_json_if_exists,
    parse_json_object,
    save_error,
    validate_player_id,
)
from gooseomni.benchmark.prompts import information_state_prompt
from gooseomni.benchmark.schema import InformationState, VALID_PLAYERS


EVENT_FIELDS = {
    "event_id",
    "player_id",
    "local_start_sec",
    "local_end_sec",
    "event_type",
    "description",
    "actor",
    "visible_players",
    "mentioned_players",
    "location",
    "speaker",
    "utterance",
    "claim_type",
    "certainty",
    "evidence",
    "source_pov",
    "is_direct_observation",
    "is_speech_claim",
    "needs_human_review",
}
UTTERANCE_FIELDS = {
    "utterance_id",
    "player_id",
    "speaker",
    "speaker_confidence",
    "local_start_sec",
    "local_end_sec",
    "text",
    "mentioned_players",
    "speech_act",
    "claims",
    "certainty",
    "evidence",
    "source_pov",
    "is_direct_observation",
    "is_speech_claim",
    "needs_human_review",
}
GLOBAL_FIELDS = {
    "global_event_id",
    "local_start_sec",
    "local_end_sec",
    "event_type",
    "description",
    "actors",
    "involved_players",
    "visible_to",
    "heard_by",
    "supporting_pov_event_ids",
    "supporting_utterance_ids",
    "conflict",
    "certainty",
    "evidence",
    "source_pov",
    "is_direct_observation",
    "is_speech_claim",
    "needs_human_review",
}
PUBLIC_EVENT_TYPES = {
    "meeting",
    "discussion",
    "vote",
    "voting",
    "vote_result",
    "ejection",
    "exile",
    "body_report",
    "meeting_start",
    "public_ui",
}


def _truncate(value: object, limit: int) -> object:
    if isinstance(value, str):
        return value[:limit]
    return value


def _compact_item(item: dict, allowed: set[str]) -> dict:
    compact = {key: item.get(key) for key in allowed if key in item}
    for key, limit in [("evidence", 90), ("description", 120), ("text", 120), ("utterance", 120)]:
        if key in compact:
            compact[key] = _truncate(compact[key], limit)
    claims = compact.get("claims")
    if isinstance(claims, list):
        compact["claims"] = [
            {
                key: _truncate(
                    claim.get(key),
                    120 if key == "content" else 90 if key == "evidence" else 10_000,
                )
                for key in [
                    "claim_id",
                    "speaker",
                    "claim_type",
                    "content",
                    "mentioned_players",
                    "locations",
                    "certainty",
                    "evidence",
                ]
                if key in claim
            }
            for claim in claims[:2]
            if isinstance(claim, dict)
        ]
    return compact


def _top_items(items: list[dict], limit: int) -> list[dict]:
    def score(item: dict) -> tuple[int, float]:
        review_boost = 1 if item.get("needs_human_review") else 0
        return (review_boost, float(item.get("certainty") or 0.0))

    return sorted(items, key=score, reverse=True)[:limit]


def _target_can_access_global_event(event: dict, target_player: str) -> bool:
    event_type = str(event.get("event_type") or "").lower()
    visible_to = event.get("visible_to") or []
    heard_by = event.get("heard_by") or []
    if target_player in visible_to or target_player in heard_by:
        return True
    return event_type in PUBLIC_EVENT_TYPES


def _hidden_global_ref(event: dict, target_player: str) -> dict:
    return {
        "hidden_event_id": event.get("global_event_id", "unknown"),
        "reason": f"not visible/heard/public for {target_player}",
        "source_pov": event.get("source_pov", []),
    }


def _projection_for_target(
    target_player: str,
    global_events: list[dict],
    pov_events: list[dict],
    utterances: list[dict],
) -> dict:
    own_pov_events = [item for item in pov_events if item.get("player_id") == target_player]
    own_utterances = [item for item in utterances if item.get("player_id") == target_player]
    visible_global_events = []
    hidden_global_events = []
    for event in global_events:
        compact = _compact_item(event, GLOBAL_FIELDS)
        if _target_can_access_global_event(event, target_player):
            visible_global_events.append(compact)
        else:
            hidden_global_events.append(_hidden_global_ref(event, target_player))
    return {
        "visible_or_public_global_events": _top_items(visible_global_events, 4),
        "target_pov_events": _top_items([_compact_item(item, EVENT_FIELDS) for item in own_pov_events], 5),
        "target_heard_utterances": _top_items([_compact_item(item, UTTERANCE_FIELDS) for item in own_utterances], 5),
        "hidden_global_event_refs_for_unknowns_only": hidden_global_events[:4],
        "projection_rules": [
            "known_facts/beliefs can only use visible_or_public_global_events, target_pov_events, or target_heard_utterances",
            "hidden_global_event_refs_for_unknowns_only may only populate unknown_or_unseen_information or unknowns",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-player information states.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--target-player", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-url", default=None)
    return parser.parse_args()


def _load_annotations(
    dataset_root: Path, output_root: Path | None, stage: str, segment: object
) -> list[dict]:
    rows: list[dict] = []
    if stage == "global_events":
        payload = load_json_if_exists(annotation_path(dataset_root, stage, segment, annotation_root=output_root))
        return [] if payload is None else payload.get("global_events", [])
    for pov in segment.povs:
        payload = load_json_if_exists(annotation_path(dataset_root, stage, segment, pov.player_id, output_root))
        if payload is not None:
            key = "events" if stage == "pov_events" else "utterances"
            rows.extend(payload.get(key, []))
    return rows


def _compact_object_retry_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\n重要：上一次 information_state 输出过长或 JSON 不完整。"
        + "请重新输出一个完整 strict JSON 对象，不要 Markdown。"
        + "所有列表最多 5 项；known_facts/beliefs/unknowns/private_observations/public_information/false_or_uncertain_beliefs 只能是短字符串数组。"
        + "available_information 和 unknown_or_unseen_information 最多 5 个短对象，每个对象只包含 type, content, source_ids。"
        + "suspicion_state/trust_state 只保留玩家名到短字符串或 0-1 分数的映射。"
        + "evidence 不超过 80 个中文字符；不确定写 unknown 并降低 certainty。"
    )


def _stringify_short(value: object, limit: int = 120) -> str:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        for key in ["content", "description", "evidence", "reason", "text"]:
            if isinstance(value.get(key), str):
                return value[key][:limit]
        return str({k: value[k] for k in list(value)[:3]})[:limit]
    return str(value)[:limit]


def _as_string_list(value: object, limit: int = 5) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        nested = []
        for nested_key in ["known_facts", "beliefs", "unknowns", "items", "facts"]:
            if nested_key in value:
                nested = value[nested_key]
                break
        value = nested if nested else list(value.values())
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_stringify_short(item) for item in value[:limit]]


def _as_object_list(value: object, limit: int = 5) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if isinstance(item, list):
                for sub in item[:limit - len(rows)]:
                    rows.append({"type": str(key), "content": _stringify_short(sub)})
            else:
                rows.append({"type": str(key), "content": _stringify_short(item)})
            if len(rows) >= limit:
                break
        return rows[:limit]
    if isinstance(value, str):
        return [{"type": "summary", "content": value[:120]}]
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:limit]:
        if isinstance(item, dict):
            rows.append({
                "type": _stringify_short(item.get("type") or item.get("event_type") or "information", 40),
                "content": _stringify_short(item.get("content") or item.get("description") or item.get("evidence") or item),
                "source_ids": item.get("source_ids") or item.get("source_event_ids") or item.get("supporting_global_event_ids") or [],
            })
        else:
            rows.append({"type": "information", "content": _stringify_short(item)})
    return rows


def _as_dict(value: object) -> dict:
    if isinstance(value, dict):
        compact = {}
        for key, item in list(value.items())[:8]:
            if isinstance(item, dict):
                compact[str(key)] = {sub_key: item[sub_key] for sub_key in list(item)[:4]}
            else:
                compact[str(key)] = item
        return compact
    return {}


def _normalize_information_payload(payload: dict, segment: object, target: str) -> dict:
    normalized = dict(payload)
    normalized.update(
        {
            "dataset": segment.dataset,
            "game_id": segment.game_id,
            "segment_id": segment.segment_id,
            "target_player": target,
            "cutoff_abs_sec": segment.aligned_end_sec,
        }
    )
    available = normalized.get("available_information")
    if isinstance(available, dict):
        normalized.setdefault("known_facts", available.get("known_facts"))
        normalized.setdefault("beliefs", available.get("beliefs"))
        normalized.setdefault("unknowns", available.get("unknowns"))
        normalized["available_information"] = _as_object_list(available, 5)
    else:
        normalized["available_information"] = _as_object_list(available, 5)
    normalized["unknown_or_unseen_information"] = _as_object_list(normalized.get("unknown_or_unseen_information"), 5)
    for key in ["known_facts", "beliefs", "unknowns", "private_observations", "public_information", "false_or_uncertain_beliefs"]:
        normalized[key] = _as_string_list(normalized.get(key), 5)
    normalized["suspicion_state"] = _as_dict(normalized.get("suspicion_state"))
    normalized["trust_state"] = _as_dict(normalized.get("trust_state"))
    try:
        normalized["certainty"] = float(normalized.get("certainty", 0.5))
    except (TypeError, ValueError):
        normalized["certainty"] = 0.5
    normalized["certainty"] = max(0.0, min(1.0, normalized["certainty"]))
    normalized["evidence"] = _stringify_short(normalized.get("evidence") or "有限视角投影摘要", 120)
    source_pov = normalized.get("source_pov")
    if isinstance(source_pov, str):
        source_pov = [source_pov]
    if not isinstance(source_pov, list) or target not in source_pov:
        source_pov = [target]
    normalized["source_pov"] = source_pov
    normalized.setdefault("needs_human_review", normalized["certainty"] < 0.5)
    return normalized


def main() -> None:
    args = parse_args()
    if args.target_player is not None:
        validate_player_id(args.target_player)
    backend = create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for segment in filter_segments(
        load_segments_jsonl(segments_path),
        game_id=args.game_id,
        segment_id=args.segment_id,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
    ):
        targets = [args.target_player] if args.target_player else list(VALID_PLAYERS)
        global_events = _load_annotations(args.dataset_root, args.output_root, "global_events", segment)
        pov_events = _load_annotations(args.dataset_root, args.output_root, "pov_events", segment)
        utterances = _load_annotations(args.dataset_root, args.output_root, "utterances", segment)
        for target in targets:
            output_path = annotation_path(
                args.dataset_root, "information_states", segment, target, args.output_root
            )
            if output_path.exists() and args.resume and not args.overwrite:
                stats["skipped"] += 1
                continue
            projection = _projection_for_target(target, global_events, pov_events, utterances)
            prompt = information_state_prompt(
                segment,
                target,
                projection["visible_or_public_global_events"],
                projection["target_pov_events"],
                projection["target_heard_utterances"],
                hidden_global_event_refs=projection["hidden_global_event_refs_for_unknowns_only"],
                projection_rules=projection["projection_rules"],
            )
            raw_response = ""
            try:
                raw_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment, target)
                try:
                    payload = parse_json_object(raw_response)
                except Exception as first_error:  # noqa: BLE001
                    prompt = _compact_object_retry_prompt(prompt)
                    retry_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment, target)
                    try:
                        payload = parse_json_object(retry_response)
                        raw_response = retry_response
                    except Exception as second_error:  # noqa: BLE001
                        raw_response = (
                            "FIRST_ERROR:\n"
                            + repr(first_error)
                            + "\n\nFIRST_RESPONSE:\n"
                            + raw_response
                            + "\n\nRETRY_ERROR:\n"
                            + repr(second_error)
                            + "\n\nRETRY_RESPONSE:\n"
                            + retry_response
                        )
                        raise ValueError(raw_response) from second_error
                state = InformationState.model_validate(_normalize_information_payload(payload, segment, target))
                write_json(output_path, state)
                append_review_items(
                    args.dataset_root,
                    annotation_root=args.output_root,
                    stage="information_states",
                    segment=segment,
                    items=[state],
                )
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(
                    dataset_root=args.dataset_root,
                    annotation_root=args.output_root,
                    stage="information_states",
                    segment=segment,
                    target_player=target,
                    prompt=prompt,
                    raw_response=raw_response,
                    error=exc,
                )
                stats["error"] += 1
    print(stats)


if __name__ == "__main__":
    main()
