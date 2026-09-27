from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.postprocess import build_information_states

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial information states.")
    parser.add_argument("--pov-events-dir", default="annotations/pov_events", type=Path)
    parser.add_argument(
        "--output-path",
        default="data/processed/information_states.json",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = build_information_states(args.pov_events_dir, args.output_path)
    print(f"information_states={len(states)}")


if __name__ == "__main__":
    main()
