from __future__ import annotations

import argparse
import json
from pathlib import Path

from gooseomni.benchmark.decrypto_diagnostics import score_decrypto_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Decrypto-style GooseOmni diagnostics."
    )
    parser.add_argument(
        "--responses", type=Path, required=True, help="JSONL model responses."
    )
    parser.add_argument(
        "--hidden-gold", type=Path, required=True, help="JSONL hidden gold file."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = score_decrypto_diagnostics(
        args.responses, args.hidden_gold, args.output
    )
    print(
        json.dumps({"ok": True, "aggregate": aggregate}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
