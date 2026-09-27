from __future__ import annotations

import argparse
import json
from pathlib import Path

from gooseomni.benchmark.decrypto_diagnostics import (
    validate_decrypto_outputs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GooseOmni Decrypto-style outputs."
    )
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_decrypto_outputs(args.annotation_root, args.benchmark_root)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
