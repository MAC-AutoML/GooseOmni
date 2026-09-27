from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mock ToM eval predictions.")
    parser.add_argument("--trials", default="benchmark/weak/trials.jsonl", type=Path)
    parser.add_argument(
        "--output", default="benchmark/weak/predictions_mock.jsonl", type=Path
    )
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        args.trials.open("r", encoding="utf-8") as inp,
        args.output.open("w", encoding="utf-8") as out,
    ):
        for line in inp:
            if args.limit is not None and count >= args.limit:
                break
            trial = json.loads(line)
            pred = {
                "trial_id": trial["trial_id"],
                "prediction": trial["gold"]["label"],
                "evidence": "mock prediction copies weak label",
                "used_forbidden_information": False,
                "json_parse_success": True,
                "schema_validation_success": True,
            }
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")
            count += 1
    print({"predictions": count})


if __name__ == "__main__":
    main()
