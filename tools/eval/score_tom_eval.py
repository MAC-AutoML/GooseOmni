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
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ToM eval predictions.")
    parser.add_argument("--trials", default="benchmark/weak/trials.jsonl", type=Path)
    parser.add_argument("--predictions", default="benchmark/weak/predictions_mock.jsonl", type=Path)
    parser.add_argument("--output", default="benchmark/reports/eval_scores.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = {}
    for line in args.trials.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            trials[row["trial_id"]] = row
    preds = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(preds)
    correct = sum(1 for pred in preds if pred.get("prediction") == trials.get(pred["trial_id"], {}).get("gold", {}).get("label"))
    leakage = sum(1 for pred in preds if pred.get("used_forbidden_information"))
    parse_ok = sum(1 for pred in preds if pred.get("json_parse_success"))
    schema_ok = sum(1 for pred in preds if pred.get("schema_validation_success"))
    scores = {
        "label_accuracy": correct / total if total else 0.0,
        "evidence_validity": None,
        "perspective_leakage_rate": leakage / total if total else 0.0,
        "forbidden_fact_usage_count": leakage,
        "next_action_accuracy": None,
        "json_parse_success": parse_ok / total if total else 0.0,
        "schema_validation_success": schema_ok / total if total else 0.0,
        "total": total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(scores)


if __name__ == "__main__":
    main()

