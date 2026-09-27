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

from gooseomni.benchmark.decrypto_diagnostics import build_oracle_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GooseOmni oracle trajectory ledger.")
    parser.add_argument("--release-root", type=Path, required=True, help="Path to release_benchmark_v2.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root containing oracle_ledger/.")
    parser.add_argument("--game-id", default="g001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_oracle_ledger(args.release_root, args.output_root, game_id=args.game_id)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
