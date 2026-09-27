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
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ResultSet:
    path: Path
    model: str
    setting: str
    include_asr: bool
    rows: dict[int, dict]


SPEECH_VERBS = (
    "said",
    "asked",
    "felt",
    "hopes",
    "hope",
    "explains",
    "wants",
    "thinks",
    "says",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_choice(value: str) -> str:
    text = str(value or "").strip().upper()
    return text[:1] if text[:1] in {"A", "B", "C", "D"} else ""


def option_map(options: Iterable[str]) -> dict[str, str]:
    mapped = {}
    for raw in options or []:
        text = str(raw).strip()
        match = re.match(r"^([A-Da-d])\s*[.)]\s*(.*)$", text)
        if match:
            mapped[match.group(1).upper()] = match.group(2).strip()
    return mapped


def canonical_text(text: str) -> str:
    text = re.sub(r"[\"'“”‘’]", "", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_speaker_content(option_text: str) -> tuple[str, str, bool]:
    text = re.sub(r"^([A-Da-d])\s*[.)]\s*", "", option_text or "").strip()
    lowered = text.lower()
    positions = []
    for verb in SPEECH_VERBS:
        match = re.search(rf"\b{re.escape(verb)}\b", lowered)
        if match:
            positions.append(match.start())
    if not positions:
        return "", canonical_text(text), False
    split_at = min(positions)
    return canonical_text(text[:split_at]), canonical_text(text[split_at:]), True


def infer_result_sets(result_dir: Path) -> list[ResultSet]:
    sets = []
    seen = set()
    for path in sorted(result_dir.glob("results_*level1*.json")):
        data = load_json(path)
        rows = data.get("results", []) if isinstance(data, dict) else []
        if not rows:
            continue
        model = str(data.get("model") or path.name.split("_level1")[0].replace("results_", ""))
        setting = str(data.get("modality") or "unknown")
        task_tag = task_tag_from_name(path.name)
        if "asr_tag" in data:
            include_asr = bool(data.get("include_asr"))
            setting = f"{task_tag}{setting}/{data.get('asr_tag')}"
        else:
            include_asr = any(bool(row.get("asr_content")) for row in rows)
            setting = f"{task_tag}{setting}/legacy-with-asr"
        scored = {
            int(row["id"]): row
            for row in rows
            if row.get("id") is not None and bool(row.get("scored", True))
        }
        signature = (
            model,
            setting,
            len(scored),
            sum(1 for row in scored.values() if row.get("is_correct")),
            tuple(sorted(scored)[:10]),
        )
        if signature in seen:
            continue
        seen.add(signature)
        sets.append(ResultSet(path=path, model=model, setting=setting, include_asr=include_asr, rows=scored))
    return sets


def task_tag_from_name(name: str) -> str:
    if "identity-only" in name:
        return "identity-only:"
    if "content-only" in name:
        return "content-only:"
    if "identity-unique" in name:
        return "identity-unique:"
    if "content-unique" in name:
        return "content-unique:"
    return ""


def split_accuracy(result: ResultSet, dataset_by_id: dict[int, dict]) -> dict[str, tuple[int, int, float]]:
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sample_id, row in result.rows.items():
        split = dataset_by_id.get(sample_id, {}).get("metadata", {}).get("consistency", "unknown")
        bucket = buckets[str(split or "unknown")]
        bucket[1] += 1
        if bool(row.get("is_correct")):
            bucket[0] += 1
    buckets["all"] = [sum(v[0] for v in buckets.values()), sum(v[1] for v in buckets.values())]
    return {k: (v[0], v[1], 100.0 * v[0] / v[1] if v[1] else 0.0) for k, v in buckets.items()}


def content_speaker_metrics(result: ResultSet, dataset_by_id: dict[int, dict]) -> dict[str, float | int]:
    total = exact = content = speaker = wrong_speaker_content = parsed = 0
    for sample_id, row in result.rows.items():
        sample = dataset_by_id.get(sample_id)
        if not sample:
            continue
        options = option_map(row.get("options") or sample.get("options") or [])
        correct_choice = normalize_choice(sample.get("correct_answer"))
        pred_choice = normalize_choice(row.get("prediction"))
        if not correct_choice or not pred_choice or correct_choice not in options or pred_choice not in options:
            continue
        correct_speaker, correct_content, ok1 = parse_speaker_content(options[correct_choice])
        pred_speaker, pred_content, ok2 = parse_speaker_content(options[pred_choice])
        total += 1
        exact += int(pred_choice == correct_choice)
        if ok1 and ok2:
            parsed += 1
            same_content = pred_content == correct_content
            same_speaker = pred_speaker == correct_speaker
            content += int(same_content)
            speaker += int(same_speaker)
            wrong_speaker_content += int(same_content and not same_speaker)
    denom = parsed or 1
    return {
        "total": total,
        "parsed": parsed,
        "exact_acc": 100.0 * exact / total if total else 0.0,
        "correct_content_rate": 100.0 * content / denom,
        "correct_speaker_rate": 100.0 * speaker / denom,
        "wrong_speaker_correct_content_rate": 100.0 * wrong_speaker_content / denom,
    }


def mcnemar(a: ResultSet, b: ResultSet) -> tuple[int, int, float]:
    only_a = only_b = 0
    for sample_id in set(a.rows) & set(b.rows):
        ac = bool(a.rows[sample_id].get("is_correct"))
        bc = bool(b.rows[sample_id].get("is_correct"))
        only_a += int(ac and not bc)
        only_b += int(bc and not ac)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    stat = (abs(only_a - only_b) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(stat / 2))
    return only_a, only_b, p_value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(split_rows: list[dict], metrics_rows: list[dict], pair_rows: list[dict]) -> str:
    lines = [
        "# Level 1 Modality Ablation Analysis",
        "",
        "Note: content/speaker decomposition is heuristic. It parses option text into a speaker prefix and an utterance/content suffix using common speech verbs.",
        "",
    ]
    lines += ["## Consistent/Inconsistent Split", ""]
    lines += ["| model | setting | all | consistent | inconsistent | gap cons-incons | n |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|"]
    for row in split_rows:
        lines.append(
            f"| {row['model']} | {row['setting']} | {row['all_acc']:.2f} | "
            f"{row['consistent_acc']:.2f} | {row['inconsistent_acc']:.2f} | "
            f"{row['gap_cons_minus_incons']:.2f} | {row['total']} |"
        )
    lines += ["", "## Content vs Speaker Decomposition", ""]
    lines += ["| model | setting | exact | content | speaker | wrong-speaker/content | parsed |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows:
        lines.append(
            f"| {row['model']} | {row['setting']} | {row['exact_acc']:.2f} | "
            f"{row['correct_content_rate']:.2f} | {row['correct_speaker_rate']:.2f} | "
            f"{row['wrong_speaker_correct_content_rate']:.2f} | {row['parsed']} |"
        )
    lines += ["", "## Paired McNemar Diagnostics", ""]
    lines += ["| model | setting_a | setting_b | only_a | only_b | p_value |"]
    lines += ["|---|---|---|---:|---:|---:|"]
    for row in pair_rows:
        lines.append(
            f"| {row['model']} | {row['setting_a']} | {row['setting_b']} | "
            f"{row['only_a']} | {row['only_b']} | {row['p_value']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/level_1/dataset.json")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()

    dataset = load_json(ROOT / args.dataset)
    dataset_by_id = {int(row["id"]): row for row in dataset}
    result_sets = infer_result_sets(ROOT / args.results_dir)

    split_rows = []
    metrics_rows = []
    for result in result_sets:
        splits = split_accuracy(result, dataset_by_id)
        all_row = splits.get("all", (0, 0, 0.0))
        cons = splits.get("consistent", (0, 0, 0.0))
        incons = splits.get("inconsistent", (0, 0, 0.0))
        split_rows.append({
            "model": result.model,
            "setting": result.setting,
            "all_acc": all_row[2],
            "consistent_acc": cons[2],
            "inconsistent_acc": incons[2],
            "gap_cons_minus_incons": cons[2] - incons[2],
            "total": all_row[1],
            "path": str(result.path.relative_to(ROOT)),
        })
        metrics = content_speaker_metrics(result, dataset_by_id)
        metrics_rows.append({"model": result.model, "setting": result.setting, **metrics})

    pair_rows = []
    by_model: dict[str, list[ResultSet]] = defaultdict(list)
    for result in result_sets:
        by_model[result.model].append(result)
    for model, sets in sorted(by_model.items()):
        for i, left in enumerate(sets):
            for right in sets[i + 1 :]:
                if len(set(left.rows) & set(right.rows)) < 100:
                    continue
                only_a, only_b, p_value = mcnemar(left, right)
                pair_rows.append({
                    "model": model,
                    "setting_a": left.setting,
                    "setting_b": right.setting,
                    "only_a": only_a,
                    "only_b": only_b,
                    "p_value": p_value,
                })

    out_dir = ROOT / args.output_dir
    write_csv(out_dir / "level1_split.csv", split_rows)
    write_csv(out_dir / "level1_content_speaker.csv", metrics_rows)
    write_csv(out_dir / "level1_mcnemar.csv", pair_rows)
    (out_dir / "level1_ablation_analysis.md").write_text(
        render_markdown(split_rows, metrics_rows, pair_rows),
        encoding="utf-8",
    )
    print(out_dir / "level1_ablation_analysis.md")


if __name__ == "__main__":
    main()
