from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.data_pipeline.local_review import (
    merge_review_shards,
    prepare_review_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or merge local Codex review shards."
    )
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--candidates", required=True, type=Path)
    prepare.add_argument("--review-root", required=True, type=Path)
    prepare.add_argument("--shard-size", type=int, default=5)
    prepare.add_argument("--limit", type=int)
    merge = actions.add_parser("merge")
    merge.add_argument("--review-root", required=True, type=Path)
    merge.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        result = prepare_review_shards(
            args.candidates, args.review_root, args.shard_size, args.limit
        )
    else:
        result = merge_review_shards(args.review_root, args.output)
    print(result)


if __name__ == "__main__":
    main()
