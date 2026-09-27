from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.splitting import SplitConfig, run_split

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw multi-POV videos into aligned clips."
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--output-dir", default="data/processed/clips", type=Path)
    parser.add_argument(
        "--manifest-path",
        default="data/processed/clip_manifest.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--sync-offsets",
        default="data/processed/sync_offsets.json",
        type=Path,
        help="JSON file with first-round start offsets per player.",
    )
    parser.add_argument("--segment-sec", default=90, type=int)
    parser.add_argument("--overlap-sec", default=10, type=int)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit-clips", default=None, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = run_split(
        SplitConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            segment_sec=args.segment_sec,
            overlap_sec=args.overlap_sec,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            sync_offsets_path=args.sync_offsets,
            game_id=args.game_id,
            player_id=args.player_id,
            limit_clips=args.limit_clips,
        )
    )
    print(f"planned_clips={len(clips)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
