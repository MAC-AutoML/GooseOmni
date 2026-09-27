from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.postprocess import merge_global_events

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge POV events into initial global events."
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
    events = merge_global_events(args.pov_events_dir, args.output_path)
    print(f"global_events={len(events)}")


if __name__ == "__main__":
    main()
