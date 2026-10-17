#!/usr/bin/env python3
"""CLI: continuous episode split for one long POV recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_preprocess.common.backend import BackendConfig, create_backend
from data_preprocess.paths import WORKSPACE
from data_preprocess.split.scan import scan_player_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one long POV into episode_XXX_<result>.mp4 clips."
    )
    parser.add_argument("--session", required=True, help="date folder under raw/")
    parser.add_argument("--player-id", required=True)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="source mp4 (default: workspace/raw/<session>/<player>.mp4)",
    )
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--backend", default="mock", choices=["mock", "local"])
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    # Rebind logical roots relative to chosen workspace.
    raw_root = workspace / "raw"
    episodes_root = workspace / "episodes"
    runs_root = workspace / "runs"
    for path in (raw_root, episodes_root, runs_root):
        path.mkdir(parents=True, exist_ok=True)

    video = args.video or (raw_root / args.session / f"{args.player_id}.mp4")
    if not video.is_file():
        raise SystemExit(f"source video missing: {video}")

    backend = create_backend(
        BackendConfig(
            backend=args.backend,
            server_url=args.server_url,
            use_video=True,
            use_audio=False,
        )
    )
    run_dir = runs_root / "split" / args.session / args.player_id
    export_dir = episodes_root / args.session
    result = scan_player_continuous(
        source_video=video,
        player_id=args.player_id,
        game_id=args.game_id,
        backend=backend,
        run_dir=run_dir,
        export_dir=export_dir,
        max_episodes=args.max_episodes,
        resume=not args.no_resume,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
