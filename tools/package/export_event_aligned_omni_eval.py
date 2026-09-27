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

from gooseomni.annotation.eval_export import (
    EventAlignedExportConfig,
    export_event_aligned_omni_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export event-aligned cropped videos and Omni eval annotations."
    )
    parser.add_argument(
        "--trials",
        default="data/processed/candidate_trials_manual_sync_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--global-events",
        default="data/processed/global_events_manual_sync_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--sync-offsets",
        default="data/processed/sync_offsets_manual_v1.json",
        type=Path,
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--output-dir",
        default="data/level_2_event_aligned_manual_sync_v1",
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
    print(f"samples={stats['samples']} skipped={stats['skipped']} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
