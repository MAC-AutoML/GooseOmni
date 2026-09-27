from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gooseomni.benchmark.decrypto_diagnostics import build_decrypto_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Decrypto-style A/B/C/D ToM diagnostic probes.")
    parser.add_argument("--annotation-root", type=Path, required=True, help="Root containing oracle_ledger/.")
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to --annotation-root.")
    parser.add_argument("--limit", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.annotation_root
    counts = build_decrypto_diagnostics(args.annotation_root, output_root, limit=args.limit)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
