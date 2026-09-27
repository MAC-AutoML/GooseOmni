from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import (
    annotate_text_with_segment_context,
    annotation_path,
    append_review_items,
    filter_segments,
    normalize_phase_event_payload,
    parse_json_array,
    retry_prompt_for_compact_json,
    save_error,
)
from gooseomni.benchmark.prompts import phase_event_prompt
from gooseomni.benchmark.schema import PhaseEvent, PhaseEventAnnotation

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate public phase events for Omni Goose."
    )
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
    for segment in filter_segments(
        load_segments_jsonl(segments_path),
        game_id=args.game_id,
        segment_id=args.segment_id,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
    ):
        output_path = annotation_path(
            args.dataset_root, "phase_events", segment, annotation_root=args.output_root
        )
        if output_path.exists() and args.resume and not args.overwrite:
            stats["skipped"] += 1
            continue
        sources = [
            {
                "player_id": pov.player_id,
                "round_boundary_candidates": pov.qwen_round_boundary_candidates,
            }
            for pov in segment.povs
        ]
        prompt = phase_event_prompt(segment, sources)
        raw_response = ""
        try:
            raw_response = annotate_text_with_segment_context(
                backend, prompt, args.dataset_root, segment
            )
            try:
                parsed_items = parse_json_array(raw_response)
            except Exception as first_error:  # noqa: BLE001
                prompt = retry_prompt_for_compact_json(prompt, max_items=3)
                retry_response = annotate_text_with_segment_context(
                    backend, prompt, args.dataset_root, segment
                )
                try:
                    parsed_items = parse_json_array(retry_response)
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
            phase_events = [
                PhaseEvent.model_validate(
                    normalize_phase_event_payload(item, segment, index)
                )
                for index, item in enumerate(parsed_items, start=1)
            ]
            annotation = PhaseEventAnnotation(
                game_id=segment.game_id,
                segment_id=segment.segment_id,
                aligned_start_sec=segment.aligned_start_sec,
                aligned_end_sec=segment.aligned_end_sec,
                phase_events=phase_events,
                raw_response=raw_response,
            )
            write_json(output_path, annotation)
            append_review_items(
                args.dataset_root,
                annotation_root=args.output_root,
                stage="phase_events",
                segment=segment,
                items=phase_events,
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            save_error(
                dataset_root=args.dataset_root,
                annotation_root=args.output_root,
                stage="phase_events",
                segment=segment,
                prompt=prompt,
                raw_response=raw_response,
                error=exc,
            )
            stats["error"] += 1
    print(stats)


if __name__ == "__main__":
    main()
