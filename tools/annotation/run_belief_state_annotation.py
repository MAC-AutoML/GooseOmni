from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import (
    annotate_text_with_segment_context,
    annotation_path,
    filter_segments,
    load_json_if_exists,
    normalize_cutoff_payload,
    parse_json_object,
    parse_partial_json_object,
    retry_prompt_for_compact_object,
    save_error,
    validate_player_id,
)
from gooseomni.benchmark.prompts import belief_state_prompt
from gooseomni.benchmark.schema import VALID_PLAYERS, BeliefState

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-player belief states.")
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


def _load(
    dataset_root: Path,
    output_root: Path | None,
    stage: str,
    segment: object,
    player_id: str | None = None,
) -> dict:
    return (
        load_json_if_exists(
            annotation_path(dataset_root, stage, segment, player_id, output_root)
        )
        or {}
    )


def _forbidden_from_global(global_events: list[dict], target: str) -> list[dict]:
    forbidden = []
    for event in global_events:
        if target not in event.get("visible_to", []) and target not in event.get(
            "heard_by", []
        ):
            forbidden.append(
                {
                    "hidden_event_id": event.get("global_event_id", "unknown"),
                    "reason": f"not visible/heard by {target}",
                }
            )
    return forbidden


def _stringify_short(value: object, limit: int = 120) -> str:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        for key in ["content", "description", "reason", "evidence", "text"]:
            if isinstance(value.get(key), str):
                return value[key][:limit]
        return str({k: value[k] for k in list(value)[:3]})[:limit]
    return str(value)[:limit]


def _as_list(value: object, limit: int = 8) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if isinstance(item, list):
                for sub in item[: limit - len(rows)]:
                    rows.append({"type": str(key), "content": _stringify_short(sub)})
            else:
                rows.append({"type": str(key), "content": _stringify_short(item)})
            if len(rows) >= limit:
                break
        return rows[:limit]
    if isinstance(value, str):
        return [{"type": "belief", "content": value[:120]}]
    if isinstance(value, list):
        rows = []
        for item in value[:limit]:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"type": "belief", "content": _stringify_short(item)})
        return rows
    return []


def _belief_items(value: object, limit: int = 8) -> list[dict]:
    rows = _as_list(value, limit)
    normalized = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        player = row.get("player")
        if player not in VALID_PLAYERS:
            text = _stringify_short(row.get("content") or row)
            player = next(
                (candidate for candidate in VALID_PLAYERS if candidate in text), None
            )
        normalized.append(
            {
                "player": player,
                "belief_type": str(
                    row.get("belief_type") or row.get("type") or "unknown"
                ),
                "content": _stringify_short(row.get("content") or row),
                "score": row.get("score"),
                "evidence_for": row.get("evidence_for")
                if isinstance(row.get("evidence_for"), list)
                else [],
                "evidence_against": row.get("evidence_against")
                if isinstance(row.get("evidence_against"), list)
                else [],
                "confidence": max(
                    0.0,
                    min(
                        1.0,
                        float(row.get("confidence", row.get("certainty", 0.5)) or 0.5),
                    ),
                ),
            }
        )
    return normalized


def _forbidden_list(
    value: object, global_events: list[dict], target: str
) -> list[dict]:
    if not value:
        return _forbidden_from_global(global_events, target)
    rows = []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        for index, item in enumerate(value[:12], start=1):
            if isinstance(item, dict):
                rows.append(
                    {
                        "hidden_event_id": str(
                            item.get("hidden_event_id")
                            or item.get("event_id")
                            or f"hidden_{index:03d}"
                        ),
                        "reason": _stringify_short(
                            item.get("reason") or item.get("content") or item
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "hidden_event_id": f"hidden_{index:03d}",
                        "reason": _stringify_short(item),
                    }
                )
    return rows or _forbidden_from_global(global_events, target)


def _normalize_belief_payload(
    payload: dict, segment: object, target: str, global_events: list[dict]
) -> dict:
    normalized = normalize_cutoff_payload(payload, segment, target)
    normalized["knows"] = _as_list(normalized.get("knows"), 8)
    normalized["does_not_know"] = _as_list(normalized.get("does_not_know"), 8)
    normalized["believes_or_suspects"] = _belief_items(
        normalized.get("believes_or_suspects"), 8
    )
    trust = normalized.get("trust_state")
    normalized["trust_state"] = trust if isinstance(trust, dict) else {}
    normalized["forbidden_information"] = _forbidden_list(
        normalized.get("forbidden_information"), global_events, target
    )
    try:
        normalized["certainty"] = float(normalized.get("certainty", 0.5))
    except (TypeError, ValueError):
        normalized["certainty"] = 0.5
    normalized["certainty"] = max(0.0, min(1.0, normalized["certainty"]))
    normalized["evidence"] = _stringify_short(
        normalized.get("evidence") or "belief state normalized from finite-view inputs",
        160,
    )
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
    if args.target_player:
        validate_player_id(args.target_player)
    backend = create_backend(
        args.backend,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        server_url=args.server_url,
    )
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
        global_events = _load(
            args.dataset_root, args.output_root, "global_events", segment
        ).get("global_events", [])
        for target in (
            [args.target_player] if args.target_player else list(VALID_PLAYERS)
        ):
            output_path = annotation_path(
                args.dataset_root, "belief_states", segment, target, args.output_root
            )
            if output_path.exists() and args.resume and not args.overwrite:
                stats["skipped"] += 1
                continue
            memory = _load(
                args.dataset_root, args.output_root, "memory_states", segment, target
            )
            pov = _load(
                args.dataset_root, args.output_root, "pov_events", segment, target
            ).get("events", [])
            utt = _load(
                args.dataset_root, args.output_root, "utterances", segment, target
            ).get("utterances", [])
            prompt = belief_state_prompt(
                segment, target, memory, pov, utt, global_events
            )
            raw_response = ""
            try:
                raw_response = annotate_text_with_segment_context(
                    backend, prompt, args.dataset_root, segment, target
                )
                try:
                    parsed_payload = parse_json_object(raw_response)
                except Exception as first_error:  # noqa: BLE001
                    retry_prompt = retry_prompt_for_compact_object(prompt)
                    retry_response = annotate_text_with_segment_context(
                        backend, retry_prompt, args.dataset_root, segment, target
                    )
                    try:
                        parsed_payload = parse_json_object(retry_response)
                        prompt = retry_prompt
                        raw_response = retry_response
                    except Exception as second_error:  # noqa: BLE001
                        parsed_payload = parse_partial_json_object(
                            retry_response
                        ) or parse_partial_json_object(raw_response)
                        if parsed_payload is None:
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
                        prompt = retry_prompt
                        raw_response = (
                            retry_response if "{" in retry_response else raw_response
                        )
                payload = _normalize_belief_payload(
                    parsed_payload, segment, target, global_events
                )
                state = BeliefState.model_validate(payload)
                write_json(output_path, state)
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(
                    dataset_root=args.dataset_root,
                    annotation_root=args.output_root,
                    stage="belief_states",
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
