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

from gooseomni.benchmark.decrypto_diagnostics import export_gooseomni_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GooseOmni-v1 benchmark files.")
    parser.add_argument("--annotation-root", type=Path, required=True, help="Root containing diagnostics/.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = export_gooseomni_benchmark(args.annotation_root, args.benchmark_root)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
