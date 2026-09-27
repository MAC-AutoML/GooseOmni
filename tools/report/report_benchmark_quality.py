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
import collections
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write GooseOmni benchmark quality reports.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--benchmark-root", default="benchmark", type=Path)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(collections.Counter(str(row.get(key, "unknown")) for row in rows))


def _markdown_dict(values: dict[str, Any]) -> str:
    if not values:
        return "{}"
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()
    manifest = _json(args.dataset_root / "manifest.json")
    segments = _jsonl(args.dataset_root / "segments.jsonl")
    metadata = _json(args.benchmark_root / "weak" / "metadata.json")
    trials = _jsonl(args.benchmark_root / "weak" / "trials.jsonl")
    review = _jsonl(args.benchmark_root / "human_review_queue.jsonl")
    report_dir = args.benchmark_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    type_counts = _counter(trials, "trial_type")
    condition_counts = _counter(trials, "input_condition")
    risk_counts = _counter(trials, "risk_of_perspective_leakage")
    review_reason_counts = collections.Counter()
    for row in review:
        for reason in row.get("review_reasons", []):
            review_reason_counts[str(reason)] += 1

    segment_game_ids = sorted({str(row.get("game_id")) for row in segments if row.get("game_id")})
    players = sorted(
        {
            str(pov.get("player_id"))
            for row in segments
            for pov in row.get("povs", [])
            if pov.get("player_id")
        }
    )
    game_id = manifest.get("strict_game_id") or manifest.get("game_id") or ",".join(segment_game_ids)

    dataset_card = f"""# GooseOmni Dataset Card

- dataset: {manifest.get('dataset')}
- format: {manifest.get('format')}
- game_id: {game_id}
- segments: {manifest.get('segment_count')}
- videos: {manifest.get('video_count')}
- strict_6pov: {manifest.get('strict_6pov')}
- players: {_markdown_dict({'players': players})}
"""

    annotation_quality = f"""# Annotation Quality

- weak trials: {len(trials)}
- candidate trials: {metadata.get('candidate_count', 'unknown')}
- human review queue: {len(review)}
- qwen_weak: {len(trials)}
- qwen_checked: 0
- human_verified: 0
- perspective leakage risk distribution: {_markdown_dict(risk_counts)}
- review reason distribution: {_markdown_dict(dict(review_reason_counts))}
"""

    benchmark_card = f"""# ToM Benchmark Card

- trial_count: {len(trials)}
- candidate_count: {metadata.get('candidate_count', 'unknown')}
- input_conditions: {_markdown_dict({'conditions': metadata.get('conditions', [])})}
- trial_type_distribution: {_markdown_dict(type_counts)}
- input_condition_distribution: {_markdown_dict(condition_counts)}
- perspective_leakage_risk_distribution: {_markdown_dict(risk_counts)}
- source: {metadata.get('source')}
- gold_source: {metadata.get('gold_source', 'qwen_weak')}
"""
    (report_dir / "dataset_card.md").write_text(dataset_card, encoding="utf-8")
    (report_dir / "annotation_quality.md").write_text(annotation_quality, encoding="utf-8")
    (report_dir / "tom_benchmark_card.md").write_text(benchmark_card, encoding="utf-8")
    print({"reports": 3})


if __name__ == "__main__":
    main()
