#!/usr/bin/env python3
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
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data" / "level_2_extended" / "annotations.json"
DEFAULT_RESULTS = ROOT / "results" / "results_qwen3_omni_level2_extended_audio-video_qwen3-judge.json"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "qwen3_omni_level2_extended_by_language.md"


def _language_of(sample: dict[str, Any]) -> str:
    note = str(sample.get("language_note") or "").strip()
    if note:
        return note
    question = str(sample.get("question_1", {}).get("question") or "")
    if any(ch in question for ch in "¿¡") or any(word in question.lower() for word in ["debería", "debe hablar", "segundo"]):
        return "西班牙语"
    if any(word in question.lower() for word in ["mulher", "homem", "deve ", "segundo"]):
        return "葡萄牙语"
    if any("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in question):
        return "俄语"
    if any(word in question.lower() for word in ["sollte", "soll ", "sprechen", "sekunde"]):
        return "德语"
    if any(word in question.lower() for word in ["doit-il", "doit-elle", "est-ce", "seconde"]):
        return "法语"
    if any(word in question.lower() for word in ["bør", "kvinden", "manden", "tale ved"]):
        return "丹麦语"
    return "未知"


def _pct(num: float, den: float) -> str:
    return "0.00%" if den == 0 else f"{num / den * 100:.2f}%"


def build_report(annotations_path: Path, results_path: Path) -> str:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["data"]
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = results_payload.get("results", [])
    sample_by_id = {sample["video_id"]: sample for sample in annotations}

    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "q1_correct": 0,
        "q2_scores": [],
        "need_q2": 0,
        "q2_answered": 0,
    })
    rows: list[dict[str, Any]] = []

    for row in results:
        sample = sample_by_id.get(row.get("video_id"), {})
        language = _language_of(sample)
        group = groups[language]
        group["total"] += 1
        if row.get("q1_correct"):
            group["q1_correct"] += 1
        needs_q2 = str(row.get("q1_answer") or "").upper() == "A"
        if needs_q2:
            group["need_q2"] += 1
        score = row.get("q2_score")
        if needs_q2 and row.get("q1_correct") and isinstance(score, (int, float)):
            group["q2_scores"].append(float(score))
            group["q2_answered"] += 1
        rows.append({
            "video_id": row.get("video_id"),
            "language": language,
            "q1_answer": row.get("q1_answer"),
            "q1_prediction": row.get("q1_prediction"),
            "q1_correct": row.get("q1_correct"),
            "q2_score": row.get("q2_score"),
            "q2_reference": row.get("q2_reference"),
            "q2_response": row.get("q2_response"),
        })

    lines = [
        "# Qwen3-Omni Level 2 Extended Results by Language",
        "",
        f"- Result file: `{results_path}`",
        f"- Judge model for Q2: `{results_payload.get('q2_judge_model', 'none')}`",
        f"- Overall Q1: {results_payload.get('q1_correct', 0)}/{results_payload.get('q1_total', 0)} ({results_payload.get('q1_accuracy', 0):.2f}%)",
        f"- Overall Q2: count={results_payload.get('q2_count', 0)}, avg={results_payload.get('q2_avg_score', 0):.2f}",
        "",
        "## Summary",
        "",
        "| Language | Total | Q1 Correct | Q1 Acc | Need Q2 | Q2 Scored | Q2 Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for language in sorted(groups):
        group = groups[language]
        scores = group["q2_scores"]
        q2_avg = sum(scores) / len(scores) if scores else 0.0
        lines.append(
            f"| {language} | {group['total']} | {group['q1_correct']} | "
            f"{_pct(group['q1_correct'], group['total'])} | {group['need_q2']} | "
            f"{group['q2_answered']} | {q2_avg:.2f} |"
        )

    lines.extend([
        "",
        "## Per Sample",
        "",
        "| Video ID | Language | Q1 Gold | Q1 Pred | Q1 Correct | Q2 Score | Q2 Reference | Q2 Response |",
        "|---|---|---|---|---:|---:|---|---|",
    ])
    for row in rows:
        ref = str(row["q2_reference"] or "").replace("\n", " ").replace("|", "\\|")
        resp = str(row["q2_response"] or "").replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {row['video_id']} | {row['language']} | {row['q1_answer']} | {row['q1_prediction']} | "
            f"{row['q1_correct']} | {row['q2_score']} | {ref} | {resp} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build language-wise report for Level 2 extended results")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.annotations, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
