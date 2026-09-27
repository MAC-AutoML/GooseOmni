from __future__ import annotations

import argparse
import json
from pathlib import Path

from gooseomni.benchmark.decrypto_diagnostics import build_oracle_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GooseOmni oracle trajectory ledger."
    )
    parser.add_argument(
        "--release-root", type=Path, required=True, help="Path to release_benchmark_v2."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root containing oracle_ledger/.",
    )
    parser.add_argument("--game-id", default="g001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_oracle_ledger(
        args.release_root, args.output_root, game_id=args.game_id
    )
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
