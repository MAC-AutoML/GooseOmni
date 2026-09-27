from __future__ import annotations

import argparse
import json
from pathlib import Path

from gooseomni.benchmark.decrypto_diagnostics import build_decrypto_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Decrypto-style A/B/C/D ToM diagnostic probes."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        required=True,
        help="Root containing oracle_ledger/.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="Defaults to --annotation-root."
    )
    parser.add_argument("--limit", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.annotation_root
    counts = build_decrypto_diagnostics(
        args.annotation_root, output_root, limit=args.limit
    )
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
