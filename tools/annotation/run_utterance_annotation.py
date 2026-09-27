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
    append_review_items,
    filter_povs,
    filter_segments,
    parse_json_array,
    parse_partial_json_array_objects,
    parse_json_array_with_video_retry,
    normalize_utterance_payload,
    retry_prompt_for_compact_json,
    save_error,
    video_path_for,
)
from gooseomni.benchmark.prompts import utterance_prompt
from gooseomni.benchmark.schema import Utterance, UtteranceAnnotation


MEETING_BOUNDARIES = {"meeting_start", "meeting_or_alert", "vote_result", "ejection", "discussion", "vote"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate aligned Omni Goose utterances.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--prefer-meeting", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-url", default=None)
    return parser.parse_args()


def _has_meeting_candidate(pov: object) -> bool:
    return any(
        item.get("boundary_type") in MEETING_BOUNDARIES
        for item in getattr(pov, "qwen_round_boundary_candidates", [])
    )


def main() -> None:
    args = parse_args()
    backend = create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    stats = {"ok": 0, "error": 0, "skipped": 0}
    segments = filter_segments(
        load_segments_jsonl(segments_path),
        game_id=args.game_id,
        segment_id=args.segment_id,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
    )
    for segment in segments:
        for pov in filter_povs(segment, args.player_id):
            if args.prefer_meeting and not _has_meeting_candidate(pov):
                stats["skipped"] += 1
                continue
            output_path = annotation_path(
                args.dataset_root, "utterances", segment, pov.player_id, args.output_root
            )
            if output_path.exists() and args.resume and not args.overwrite:
                stats["skipped"] += 1
                continue
            prompt = utterance_prompt(segment, pov)
            raw_response = ""
            try:
                video_path = video_path_for(args.dataset_root, pov)
                if hasattr(backend, "annotate_audio"):
                    raw_response = backend.annotate_audio(video_path, prompt)
                    try:
                        parsed_items = parse_json_array(raw_response)
                    except Exception as first_error:  # noqa: BLE001
                        prompt = retry_prompt_for_compact_json(prompt, max_items=4)
                        retry_response = backend.annotate_audio(video_path, prompt)
                        try:
                            parsed_items = parse_json_array(retry_response)
                            raw_response = retry_response
                        except Exception as second_error:  # noqa: BLE001
                            recovered = parse_partial_json_array_objects(retry_response, max_items=4)
                            if not recovered:
                                recovered = parse_partial_json_array_objects(raw_response, max_items=4)
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
                else:
                    raw_response, parsed_items, prompt = parse_json_array_with_video_retry(
                        backend=backend,
                        video_path=video_path,
                        prompt=prompt,
                        max_items=4,
                    )
                utterances = [
                    Utterance.model_validate(normalize_utterance_payload(item, segment, pov, index))
                    for index, item in enumerate(parsed_items, start=1)
                ]
                annotation = UtteranceAnnotation(
                    game_id=segment.game_id,
                    segment_id=segment.segment_id,
                    player_id=pov.player_id,
                    video_file=pov.video_file,
                    aligned_start_sec=segment.aligned_start_sec,
                    aligned_end_sec=segment.aligned_end_sec,
                    utterances=utterances,
                    raw_response=raw_response,
                )
                write_json(output_path, annotation)
                append_review_items(
                    args.dataset_root,
                    annotation_root=args.output_root,
                    stage="utterances",
                    segment=segment,
                    items=utterances,
                )
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(
                    dataset_root=args.dataset_root,
                    annotation_root=args.output_root,
                    stage="utterances",
                    segment=segment,
                    pov=pov,
                    prompt=prompt,
                    raw_response=raw_response,
                    error=exc,
                )
                stats["error"] += 1
    print(stats)


if __name__ == "__main__":
    main()
