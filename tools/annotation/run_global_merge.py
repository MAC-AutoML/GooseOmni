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
    parse_json_array,
    parse_partial_json_array_objects,
    normalize_global_event_payload,
    retry_prompt_for_compact_json,
    save_error,
)
from gooseomni.benchmark.prompts import global_merge_prompt
from gooseomni.benchmark.schema import GlobalEvent, GlobalEventAnnotation


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge 6-POV annotations into global events.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--segment-id", default=None)
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


def _compact_item(item: dict, allowed: set[str]) -> dict:
    compact = {key: item.get(key) for key in allowed if key in item}
    if isinstance(compact.get("evidence"), str):
        compact["evidence"] = compact["evidence"][:90]
    if isinstance(compact.get("description"), str):
        compact["description"] = compact["description"][:120]
    if isinstance(compact.get("text"), str):
        compact["text"] = compact["text"][:120]
    for list_key in ["supporting_pov_event_ids", "supporting_utterance_ids"]:
        if isinstance(compact.get(list_key), list):
            compact[list_key] = compact[list_key][:6]
    claims = compact.get("claims")
    if isinstance(claims, list):
        compact["claims"] = [
            {
                key: claim.get(key)
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
        for claim in compact["claims"]:
            if isinstance(claim.get("content"), str):
                claim["content"] = claim["content"][:120]
            if isinstance(claim.get("evidence"), str):
                claim["evidence"] = claim["evidence"][:90]
    return compact


def _top_items(items: list[dict], limit: int) -> list[dict]:
    def score(item: dict) -> tuple[float, int]:
        review_boost = 1 if item.get("needs_human_review") else 0
        return (float(item.get("certainty") or 0.0), review_boost)

    return sorted(items, key=score, reverse=True)[:limit]


def _load_stage_annotations(
    dataset_root: Path, output_root: Path | None, stage: str, segment: object
) -> list[dict]:
    rows: list[dict] = []
    for pov in segment.povs:
        payload = load_json_if_exists(annotation_path(dataset_root, stage, segment, pov.player_id, output_root))
        if payload is None:
            continue
        if stage == "pov_events":
            items = [_compact_item(item, EVENT_FIELDS) for item in payload.get("events", [])]
            rows.append(
                {
                    "player_id": payload.get("player_id", pov.player_id),
                    "events": _top_items(items, 3),
                }
            )
        elif stage == "utterances":
            items = [_compact_item(item, UTTERANCE_FIELDS) for item in payload.get("utterances", [])]
            rows.append(
                {
                    "player_id": payload.get("player_id", pov.player_id),
                    "utterances": _top_items(items, 3),
                }
            )
        else:
            rows.append(payload)
    return rows


def main() -> None:
    args = parse_args()
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
        output_path = annotation_path(args.dataset_root, "global_events", segment, annotation_root=args.output_root)
        if output_path.exists() and args.resume and not args.overwrite:
            stats["skipped"] += 1
            continue
        pov_events = _load_stage_annotations(args.dataset_root, args.output_root, "pov_events", segment)
        utterances = _load_stage_annotations(args.dataset_root, args.output_root, "utterances", segment)
        prompt = global_merge_prompt(segment, pov_events, utterances)
        raw_response = ""
        try:
            raw_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment)
            try:
                parsed_items = parse_json_array(raw_response)
            except Exception as first_error:  # noqa: BLE001
                prompt = retry_prompt_for_compact_json(prompt, max_items=3)
                retry_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment)
                try:
                    parsed_items = parse_json_array(retry_response)
                    raw_response = retry_response
                except Exception as second_error:  # noqa: BLE001
                    recovered = parse_partial_json_array_objects(retry_response, max_items=3)
                    if not recovered:
                        recovered = parse_partial_json_array_objects(raw_response, max_items=3)
                    if recovered:
                        parsed_items = recovered
                        raw_response = retry_response if "{" in retry_response else raw_response
                    else:
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
            events = [
                GlobalEvent.model_validate(normalize_global_event_payload(item, segment, index))
                for index, item in enumerate(parsed_items, start=1)
            ]
            annotation = GlobalEventAnnotation(
                game_id=segment.game_id,
                segment_id=segment.segment_id,
                aligned_start_sec=segment.aligned_start_sec,
                aligned_end_sec=segment.aligned_end_sec,
                global_events=events,
                raw_response=raw_response,
            )
            write_json(output_path, annotation)
            append_review_items(
                args.dataset_root,
                annotation_root=args.output_root,
                stage="global_events",
                segment=segment,
                items=events,
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            save_error(
                dataset_root=args.dataset_root,
                annotation_root=args.output_root,
                stage="global_events",
                segment=segment,
                prompt=prompt,
                raw_response=raw_response,
                error=exc,
            )
            stats["error"] += 1
    print(stats)


if __name__ == "__main__":
    main()
