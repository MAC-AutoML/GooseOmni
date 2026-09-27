from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.backends.qwen_omni import QwenBackendConfig, create_backend
from gooseomni.annotation.sync import SyncConfig, infer_sync_offsets

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer aligned first-round start offsets with Qwen3-Omni."
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--output-path",
        default="data/processed/sync_offsets.json",
        type=Path,
    )
    parser.add_argument(
        "--review-dir",
        default="data/processed/sync_review",
        type=Path,
    )
    parser.add_argument(
        "--error-dir",
        default="annotations/errors",
        type=Path,
    )
    parser.add_argument(
        "--backend", default="mock", choices=["mock", "local", "openai"]
    )
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--review-window-sec", type=int, default=600)
    parser.add_argument("--overwrite-review", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-infer offsets even if existing confidence is greater than 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = create_backend(
        QwenBackendConfig(
            backend=args.backend,
            model=args.model,
            server_url=args.server_url,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
        )
    )
    stats = infer_sync_offsets(
        SyncConfig(
            raw_dir=args.raw_dir,
            output_path=args.output_path,
            review_dir=args.review_dir,
            error_dir=args.error_dir,
            backend=backend,
            game_id=args.game_id,
            player_id=args.player_id,
            review_window_sec=args.review_window_sec,
            overwrite_review=args.overwrite_review,
            resume=args.resume and not args.no_resume,
            limit=args.limit,
        )
    )
    print(stats)


if __name__ == "__main__":
    main()
