from __future__ import annotations

import argparse
import json
from pathlib import Path as _Path

from gooseomni.benchmark.agentic_midgame import score_agentic_midgame_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score GooseOmni Agentic Midgame Prediction responses."
    )
    parser.add_argument(
        "--responses", type=_Path, required=True, help="JSONL model responses."
    )
    parser.add_argument(
        "--hidden-gold",
        type=_Path,
        default=_Path(
            "benchmark/gooseomni_v1/private/agentic_midgame_prediction/hidden_gold.jsonl"
        ),
    )
    parser.add_argument("--output", type=_Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = score_agentic_midgame_prediction(
        args.responses, args.hidden_gold, args.output
    )
    print(
        json.dumps({"ok": True, "aggregate": aggregate}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
