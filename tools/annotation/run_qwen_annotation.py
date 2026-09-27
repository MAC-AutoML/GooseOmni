from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.backends.qwen_omni import (  # noqa: E402
    QwenBackendConfig,
    create_backend,
)
from gooseomni.annotation.runner import (  # noqa: E402
    annotate_pov_events,
    filter_clips,
    load_manifest,
)

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Omni initial POV annotation."
    )
    parser.add_argument(
        "--manifest-path", default="data/processed/clip_manifest.jsonl", type=Path
    )
    parser.add_argument("--output-dir", default="annotations/pov_events", type=Path)
    parser.add_argument("--error-dir", default="annotations/errors", type=Path)
    parser.add_argument(
        "--backend", default="mock", choices=["mock", "local", "openai"]
    )
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pass-kind", default="combined", choices=["combined", "visual", "audio"]
    )
    parser.add_argument("--canonical-players", default="")
    parser.add_argument("--speaker-confidence-min", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = filter_clips(
        load_manifest(args.manifest_path),
        game_id=args.game_id,
        player_id=args.player_id,
        limit=args.limit,
    )
    backend = create_backend(
        QwenBackendConfig(
            backend=args.backend,
            model=args.model,
            server_url=args.server_url,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            use_video=args.pass_kind != "audio",
            use_audio=args.pass_kind != "visual",
        )
    )
    stats = annotate_pov_events(
        clips=clips,
        backend=backend,
        output_dir=args.output_dir,
        error_dir=args.error_dir,
        resume=args.resume,
        pass_kind=args.pass_kind,
        canonical_players={
            value.strip()
            for value in args.canonical_players.split(",")
            if value.strip()
        }
        or None,
        speaker_confidence_min=args.speaker_confidence_min,
    )
    print(stats)


if __name__ == "__main__":
    main()
