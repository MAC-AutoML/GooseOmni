#!/usr/bin/env python3
"""CLI: align packed games to first in-game frame after role reveal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# src/data_preprocess/align/run_align.py -> src is parents[2]
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_preprocess.align.runner import AlignConfig, run_session_align
from data_preprocess.common.backend import BackendConfig, create_backend
from data_preprocess.paths import ALIGNED_ROOT, GAMES_ROOT, START_TEMPLATE, WORKSPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align packed GGD games to the first in-game frame after "
            "the role-reveal start screen."
        )
    )
    parser.add_argument("--session", required=True, help="date folder name under games/")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=GAMES_ROOT,
        help="packed games root (default: data_preprocess_workspace/games)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ALIGNED_ROOT,
        help="aligned output root (default: data_preprocess_workspace/aligned)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="workspace root for gate checks",
    )
    parser.add_argument(
        "--start-template",
        type=Path,
        default=START_TEMPLATE,
        help="role-reveal reference image (place under src/data_preprocess/assets/)",
    )
    parser.add_argument("--backend", default="mock", choices=["mock", "local"])
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--role-search-sec", type=float, default=30.0)
    parser.add_argument("--enter-search-sec", type=float, default=10.0)
    parser.add_argument("--coarse-fps", type=float, default=4.0)
    parser.add_argument("--max-refine-fps", type=float, default=30.0)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--min-povs-per-game", type=int, default=4)
    parser.add_argument("--no-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.start_template.is_file() and args.backend != "mock":
        raise SystemExit(
            f"start template missing: {args.start_template}\n"
            "Put a role-reveal reference PNG at src/data_preprocess/assets/start_template.png"
        )
    # mock may run without a real template file only if detect is never hit;
    # still require path for local runs.
    if args.backend == "mock" and not args.start_template.is_file():
        args.start_template.parent.mkdir(parents=True, exist_ok=True)
        args.start_template.write_bytes(b"")

    backend = create_backend(
        BackendConfig(
            backend=args.backend,
            server_url=args.server_url,
            use_video=False,
            use_audio=False,
        )
    )
    result = run_session_align(
        AlignConfig(
            session=args.session,
            source_root=args.source_root,
            output_root=args.output_root,
            start_template=args.start_template,
            backend=backend,
            role_search_sec=args.role_search_sec,
            enter_search_sec=args.enter_search_sec,
            coarse_fps=args.coarse_fps,
            refine=not args.no_refine,
            max_refine_fps=args.max_refine_fps,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            game_id=args.game_id,
            player_id=args.player_id,
            limit=args.limit,
            min_povs_per_game=args.min_povs_per_game,
            workspace_root=args.workspace,
            enforce_gate=not args.no_gate,
        )
    )
    print(result)
    return 0 if int(result["stats"].get("error", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
