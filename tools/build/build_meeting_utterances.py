from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.postprocess import build_meeting_utterances

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build meeting utterances from POV events."
    )
    parser.add_argument("--pov-events-dir", required=True, type=Path)
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utterances = build_meeting_utterances(args.pov_events_dir, args.output_path)
    print(f"utterances={len(utterances)}")


if __name__ == "__main__":
    main()
