#!/usr/bin/env python3
"""CLI: pack per-POV episodes into multi-POV game folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_preprocess.pack.gate import evaluate_date, write_gate
from data_preprocess.pack.pack_games import pack_date
from data_preprocess.paths import WORKSPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack episodes into games/<session>/")
    parser.add_argument("--session", required=True)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="pack even if gate says not ready_to_pack",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="only evaluate/write gate, do not pack",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    if args.gate_only:
        report = evaluate_date(args.session, workspace)
        path = write_gate(report, workspace)
        print({"gate": str(path), **report})
        return 0
    info = pack_date(
        args.session,
        workspace,
        overwrite=args.overwrite,
        require_gate=not args.force,
    )
    print(info)
    return 0 if not info.get("skipped") or info.get("reason") == "already_packed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
