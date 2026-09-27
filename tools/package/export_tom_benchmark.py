from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from gooseomni.benchmark.io import write_json, write_jsonl
from gooseomni.benchmark.pipeline import output_root
from gooseomni.benchmark.schema import BenchmarkGold, BenchmarkTrial

ROOT = Path(__file__).resolve().parents[1]


CONDITIONS = [
    "single_pov_video",
    "single_pov_events",
    "multi_pov_global",
    "multi_pov_perspective",
    "text_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export weak GooseOmni ToM benchmark files."
    )
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--annotation-root", default=None, type=Path)
    parser.add_argument("--benchmark-root", default="benchmark", type=Path)
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _as_info_rows(values: list[str]) -> list[dict[str, str]]:
    return [{"content": str(item)} for item in values]


def _risk(row: dict[str, Any], condition: str) -> str:
    base = row.get("risk_of_perspective_leakage", "low")
    if condition == "multi_pov_perspective" and base == "low":
        return "medium"
    return base if base in {"low", "medium", "high"} else "medium"


def _review_reasons(row: dict[str, Any], condition: str, risk: str) -> list[str]:
    reasons: list[str] = []
    if row.get("needs_human_review"):
        reasons.append("candidate_needs_human_review")
    if risk in {"medium", "high"}:
        reasons.append(f"perspective_leakage_risk_{risk}")
    if row.get("certainty", 1.0) < 0.5:
        reasons.append("low_certainty")
    if condition == "multi_pov_perspective":
        reasons.append("global_context_perspective_projection")
    return reasons


def _trial_id(row: dict[str, Any], condition: str) -> str:
    base_trial_id = str(row["trial_id"])
    if not base_trial_id.startswith(f"{row['segment_id']}_"):
        base_trial_id = f"{row['segment_id']}_{base_trial_id}"
    return f"{base_trial_id}__{condition}"


def main() -> None:
    args = parse_args()
    ann_root = args.annotation_root or output_root(args.dataset_root)
    candidate_path = ann_root / "candidate_trials" / "g001_candidate_trials.jsonl"
    rows = _load_jsonl(candidate_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    trials = []
    gold_rows = []
    review_rows = []
    candidate_risk_counts = collections.Counter()
    exported_risk_counts = collections.Counter()
    for row in rows:
        candidate_risk_counts[row.get("risk_of_perspective_leakage", "unknown")] += 1
        for condition in CONDITIONS:
            risk = _risk(row, condition)
            review_reasons = _review_reasons(row, condition, risk)
            exported_risk_counts[risk] += 1
            trial = BenchmarkTrial(
                trial_id=_trial_id(row, condition),
                game_id=row["game_id"],
                segment_id=row["segment_id"],
                trial_type=row.get("trial_type") or row.get("question_type", "unknown"),
                target_player=row["target_player"],
                cutoff_abs_sec=row["cutoff_abs_sec"],
                input_condition=condition,
                input_video_files=[
                    f"videos/{row['game_id']}/{row['segment_id']}/{row['target_player']}.mp4"
                ]
                if condition == "single_pov_video"
                else [],
                question=row["question"],
                available_information=_as_info_rows(
                    row.get("available_information", [])
                ),
                hidden_information=_as_info_rows(row.get("hidden_information", [])),
                gold=BenchmarkGold(
                    label=row.get("answer", "unknown"),
                    acceptable_reasoning=[
                        row.get("expected_answer_basis") or row.get("evidence", "")
                    ],
                    forbidden_reasoning=row.get("hidden_information", []),
                    gold_source="qwen_weak",
                ),
                metrics=["label_accuracy", "evidence_validity", "perspective_leakage"],
            )
            payload = trial.model_dump()
            payload.update(
                {
                    "dataset": row.get("dataset", "gooseomni"),
                    "question_type": row.get("question_type"),
                    "answer_format": "json",
                    "source_candidate_trial_id": row.get("trial_id"),
                    "source_pov": row.get("source_pov", []),
                    "certainty": row.get("certainty"),
                    "evidence": row.get("evidence", ""),
                    "risk_of_perspective_leakage": risk,
                    "needs_human_review": bool(review_reasons),
                    "review_reasons": review_reasons,
                    "supporting_global_event_ids": row.get(
                        "supporting_global_event_ids", []
                    ),
                    "supporting_information_state_ids": row.get(
                        "supporting_information_state_ids", []
                    ),
                }
            )
            trials.append(payload)
            gold_rows.append(
                {
                    "trial_id": payload["trial_id"],
                    "gold": trial.gold.model_dump(),
                    "risk_of_perspective_leakage": risk,
                    "review_reasons": review_reasons,
                }
            )
            if review_reasons:
                review_rows.append(payload)

    metadata = {
        "trial_count": len(trials),
        "candidate_count": len(rows),
        "conditions": CONDITIONS,
        "source": str(candidate_path),
        "candidate_risk_distribution": dict(candidate_risk_counts),
        "exported_risk_distribution": dict(exported_risk_counts),
        "gold_source": "qwen_weak",
    }
    write_jsonl(args.benchmark_root / "weak" / "trials.jsonl", trials)
    write_jsonl(args.benchmark_root / "weak" / "gold_weak.jsonl", gold_rows)
    write_json(args.benchmark_root / "weak" / "metadata.json", metadata)
    write_jsonl(args.benchmark_root / "human_review_queue.jsonl", review_rows)
    print({"trials": len(trials), "review_queue": len(review_rows)})


if __name__ == "__main__":
    main()
