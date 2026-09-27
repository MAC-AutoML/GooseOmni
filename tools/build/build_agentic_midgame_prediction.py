from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(parent for parent in _Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from gooseomni.benchmark.agentic_midgame import build_agentic_midgame_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GooseOmni Agentic Midgame Prediction track.")
    parser.add_argument("--annotation-root", type=_Path, default=_Path("annotations"))
    parser.add_argument("--benchmark-root", type=_Path, default=_Path("benchmark/gooseomni_v1"))
    parser.add_argument("--limit", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = build_agentic_midgame_prediction(args.annotation_root, args.benchmark_root, limit=args.limit)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
