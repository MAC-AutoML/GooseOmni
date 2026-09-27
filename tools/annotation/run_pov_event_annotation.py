from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import (
    annotation_path,
    append_review_items,
    filter_povs,
    filter_segments,
    normalize_pov_event_payload,
    parse_json_array_with_video_retry,
    save_error,
    video_path_for,
)
from gooseomni.benchmark.prompts import pov_event_prompt
from gooseomni.benchmark.schema import POVEvent, POVEventAnnotation

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate aligned Omni Goose POV events."
    )
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = create_backend(
        args.backend,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        server_url=args.server_url,
    )
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
            output_path = annotation_path(
                args.dataset_root,
                "pov_events",
                segment,
                pov.player_id,
                args.output_root,
            )
            if output_path.exists() and args.resume and not args.overwrite:
                stats["skipped"] += 1
                continue
            prompt = pov_event_prompt(segment, pov)
            raw_response = ""
            try:
                raw_response, parsed_items, prompt = parse_json_array_with_video_retry(
                    backend=backend,
                    video_path=video_path_for(args.dataset_root, pov),
                    prompt=prompt,
                    max_items=4,
                )
                events = [
                    POVEvent.model_validate(
                        normalize_pov_event_payload(item, segment, pov, index)
                    )
                    for index, item in enumerate(parsed_items, start=1)
                ]
                annotation = POVEventAnnotation(
                    game_id=segment.game_id,
                    segment_id=segment.segment_id,
                    player_id=pov.player_id,
                    video_file=pov.video_file,
                    aligned_start_sec=segment.aligned_start_sec,
                    aligned_end_sec=segment.aligned_end_sec,
                    events=events,
                    raw_response=raw_response,
                )
                write_json(output_path, annotation)
                append_review_items(
                    args.dataset_root,
                    annotation_root=args.output_root,
                    stage="pov_events",
                    segment=segment,
                    items=events,
                )
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(
                    dataset_root=args.dataset_root,
                    annotation_root=args.output_root,
                    stage="pov_events",
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
