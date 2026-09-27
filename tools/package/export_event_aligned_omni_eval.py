from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.eval_export import (
    EventAlignedExportConfig,
    export_event_aligned_omni_eval,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export event-aligned cropped videos and Omni eval annotations."
    )
    parser.add_argument(
        "--trials",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--global-events",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--sync-offsets",
        required=True,
        type=Path,
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--pre-context-sec", default=8.0, type=float)
    parser.add_argument("--post-context-sec", default=4.0, type=float)
    parser.add_argument("--max-duration-sec", default=120.0, type=float)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stream-copy",
        action="store_true",
        help="Use ffmpeg stream copy. Faster, but less accurate near non-keyframe cuts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_event_aligned_omni_eval(
        EventAlignedExportConfig(
            trials_path=args.trials,
            global_events_path=args.global_events,
            sync_offsets_path=args.sync_offsets,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            pre_context_sec=args.pre_context_sec,
            post_context_sec=args.post_context_sec,
            max_duration_sec=args.max_duration_sec,
            limit=args.limit,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            reencode=not args.stream_copy,
        )
    )
    print(
        f"samples={stats['samples']} skipped={stats['skipped']} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
