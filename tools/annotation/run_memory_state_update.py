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
    save_error,
    validate_player_id,
)
from gooseomni.benchmark.prompts import memory_state_prompt
from gooseomni.benchmark.schema import VALID_PLAYERS, MemoryState

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update rolling player memory states.")
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


def _stage(
    dataset_root: Path,
    output_root: Path | None,
    stage: str,
    segment: object,
    player_id: str | None = None,
) -> dict | None:
    return load_json_if_exists(
        annotation_path(dataset_root, stage, segment, player_id, output_root)
    )


MEMORY_TYPE_ALIASES = {
    "gameplay_event": "direct_visual",
    "event": "direct_visual",
    "observation": "direct_visual",
    "direct_observation": "direct_visual",
    "visual": "direct_visual",
    "speech_claim": "heard_claim",
    "claim": "heard_claim",
    "utterance": "heard_claim",
    "public_event": "public_result",
    "vote": "public_result",
    "vote_result": "public_result",
    "meeting_result": "public_result",
    "self": "self_action",
    "self_observation": "self_action",
    "inference": "inferred",
    "suspicion": "inferred",
    "phase_event": "phase",
}
DECAY_STATUS_ALIASES = {
    "stable": "active",
    "current": "active",
    "remembered": "active",
    "new": "active",
    "updated": "active",
    "outdated": "stale",
    "old": "stale",
    "conflicted": "contradicted",
}
VALID_MEMORY_TYPES = {
    "direct_visual",
    "heard_claim",
    "public_result",
    "self_action",
    "inferred",
    "phase",
}
VALID_DECAY_STATUS = {"active", "stale", "contradicted"}
VALID_VISIBILITY = {"private", "public"}


def _previous_memory_before(
    dataset_root: Path,
    output_root: Path | None,
    all_segments: list,
    current_segment: object,
    target: str,
) -> dict | None:
    current_index = next(
        (
            index
            for index, segment in enumerate(all_segments)
            if segment.segment_id == current_segment.segment_id
        ),
        None,
    )
    if current_index is None:
        return None
    for previous_segment in reversed(all_segments[:current_index]):
        payload = _stage(
            dataset_root, output_root, "memory_states", previous_segment, target
        )
        if payload is not None:
            return payload
    return None


def _normalize_memory_payload(payload: dict, segment: object, target: str) -> dict:
    normalized = normalize_cutoff_payload(payload, segment, target)
    items = normalized.get("memory_items")
    if not isinstance(items, list):
        normalized["memory_items"] = []
        items = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        item.setdefault(
            "memory_id", f"{segment.segment_id}_{target}_memory_{index:03d}"
        )
        memory_type = str(item.get("memory_type") or "inferred").strip().lower()
        item["memory_type"] = MEMORY_TYPE_ALIASES.get(memory_type, memory_type)
        if item["memory_type"] not in VALID_MEMORY_TYPES:
            item["memory_type"] = "inferred"
            item["needs_human_review"] = True
        decay_status = str(item.get("decay_status") or "active").strip().lower()
        item["decay_status"] = DECAY_STATUS_ALIASES.get(decay_status, decay_status)
        if item["decay_status"] not in VALID_DECAY_STATUS:
            item["decay_status"] = "active"
            item["needs_human_review"] = True
        visibility = str(item.get("visibility") or "private").strip().lower()
        item["visibility"] = visibility if visibility in VALID_VISIBILITY else "private"
        if not isinstance(item.get("source_event_ids"), list):
            item["source_event_ids"] = []
        if not isinstance(item.get("source_claim_ids"), list):
            item["source_claim_ids"] = []
        try:
            item["first_observed_abs_sec"] = float(
                item.get("first_observed_abs_sec", segment.aligned_start_sec)
            )
        except (TypeError, ValueError):
            item["first_observed_abs_sec"] = segment.aligned_start_sec
        try:
            item["last_referenced_abs_sec"] = float(
                item.get("last_referenced_abs_sec", segment.aligned_end_sec)
            )
        except (TypeError, ValueError):
            item["last_referenced_abs_sec"] = segment.aligned_end_sec
    delta = normalized.get("memory_delta")
    if isinstance(delta, dict):
        converted = []
        for operation, keys in {
            "add": ["add", "added"],
            "update": ["update", "updated"],
            "contradict": ["contradict", "contradicted"],
            "forget_low_confidence": ["forget_low_confidence", "removed", "forgotten"],
        }.items():
            values = []
            for key in keys:
                if key in delta:
                    values = delta.get(key) or []
                    break
            if isinstance(values, int):
                values = []
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for value in values[:12]:
                    converted.append(
                        {
                            "operation": operation,
                            "memory_id": str(value),
                            "reason": "normalized from model memory_delta",
                        }
                    )
        normalized["memory_delta"] = converted
    elif not isinstance(delta, list):
        normalized["memory_delta"] = []
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
    all_segments = load_segments_jsonl(segments_path)
    segments = filter_segments(
        all_segments,
        game_id=args.game_id,
        segment_id=args.segment_id,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
    )
    stats = {"ok": 0, "error": 0, "skipped": 0}
    previous: dict[str, dict | None] = dict.fromkeys(VALID_PLAYERS)
    for segment in segments:
        targets = [args.target_player] if args.target_player else list(VALID_PLAYERS)
        phase = (
            _stage(args.dataset_root, args.output_root, "phase_events", segment) or {}
        )
        for target in targets:
            output_path = annotation_path(
                args.dataset_root, "memory_states", segment, target, args.output_root
            )
            if output_path.exists() and args.resume and not args.overwrite:
                previous[target] = load_json_if_exists(output_path)
                stats["skipped"] += 1
                continue
            pov_ann = (
                _stage(
                    args.dataset_root, args.output_root, "pov_events", segment, target
                )
                or {}
            )
            utt_ann = (
                _stage(
                    args.dataset_root, args.output_root, "utterances", segment, target
                )
                or {}
            )
            previous_state = previous.get(target) or _previous_memory_before(
                args.dataset_root, args.output_root, all_segments, segment, target
            )
            prompt = memory_state_prompt(
                segment,
                target,
                previous_state,
                pov_ann.get("events", []),
                utt_ann.get("utterances", []),
                phase.get("phase_events", []),
            )
            raw_response = ""
            try:
                raw_response = annotate_text_with_segment_context(
                    backend, prompt, args.dataset_root, segment, target
                )
                state = MemoryState.model_validate(
                    _normalize_memory_payload(
                        parse_json_object(raw_response), segment, target
                    )
                )
                write_json(output_path, state)
                previous[target] = state.model_dump()
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(
                    dataset_root=args.dataset_root,
                    annotation_root=args.output_root,
                    stage="memory_states",
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
