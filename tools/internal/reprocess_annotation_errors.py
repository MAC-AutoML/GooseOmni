from __future__ import annotations

import argparse
from pathlib import Path

from gooseomni.annotation.runner import reprocess_error_files

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse saved Qwen raw responses.")
    parser.add_argument("--error-dir", default="annotations/errors", type=Path)
    parser.add_argument("--output-dir", default="annotations/pov_events", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(reprocess_error_files(args.error_dir, args.output_dir))


if __name__ == "__main__":
    main()
