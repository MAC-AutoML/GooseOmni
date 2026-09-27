from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ("Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu")
PLAYER_STAGES = (
    "pov_events",
    "utterances",
    "information_states",
    "memory_states",
    "belief_states",
)
SINGLE_STAGES = ("phase_events", "global_events")
UPSTREAM_PREFIXES = ("og-pov-", "og-utt-", "og-phase-", "og-up-")
DOWNSTREAM_PREFIXES = (
    "og-glob-",
    "og-info-",
    "og-mem-",
    "og-belief-",
    "og-trial-",
    "og-chain-",
)
DOWNSTREAM_PLAYER_STAGES = ("information_states", "memory_states", "belief_states")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Omni Goose annotation and benchmark progress."
    )
    parser.add_argument("--dataset-root", default=Path("data/gooseomni"), type=Path)
    parser.add_argument(
        "--annotation-root",
        default=Path("runs/gooseomni_oracle_pass1/annotations"),
        type=Path,
    )
    parser.add_argument(
        "--benchmark-root",
        default=Path("runs/gooseomni_oracle_pass1/benchmark"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("runs/gooseomni_oracle_pass1/progress_report.json"),
        type=Path,
    )
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _count_player_files(root: Path, stage: str, segment_id: str) -> int:
    path = root / stage / "g001" / segment_id
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if _valid_json_file(item))


def _single_exists(root: Path, stage: str, segment_id: str) -> bool:
    return _valid_json_file(root / stage / "g001" / f"{segment_id}.json")


def _valid_json_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        return bool(text.strip()) and isinstance(json.loads(text), dict)
    except Exception:
        return False


def _segment_index(segment_id: str) -> int:
    return int(segment_id.split("_seg_")[-1].split("_")[0])


def _active_jobs_for_segment(
    active_jobs: set[str],
    segment_index: int,
    prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    matched = []
    for name in active_jobs:
        if prefixes is not None and not name.startswith(prefixes):
            continue
        parts = name.split("-")
        if len(parts) < 3 or parts[0] != "og":
            continue
        try:
            start = int(parts[2])
            end = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else start
        except ValueError:
            continue
        if start <= segment_index <= end:
            matched.append(name)
    return sorted(matched)


def _missing_upstream_stages(row: dict[str, Any]) -> list[str]:
    missing = []
    if row["pov_events"] < len(PLAYERS):
        missing.append("pov_events")
    if row["utterances"] < len(PLAYERS):
        missing.append("utterances")
    if row["phase_events"] != 1:
        missing.append("phase_events")
    return missing


def _missing_downstream_stages(row: dict[str, Any]) -> list[str]:
    missing = []
    if row["global_events"] != 1:
        missing.append("global_events")
    for stage in DOWNSTREAM_PLAYER_STAGES:
        if row[stage] < len(PLAYERS):
            missing.append(stage)
    if row["candidate_trials"] <= 0:
        missing.append("candidate_trials")
    return missing


def _build_next_actions(
    rows: list[dict[str, Any]], active_jobs: set[str]
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in rows:
        missing_downstream = _missing_downstream_stages(row)
        missing_upstream = _missing_upstream_stages(row)
        if row["upstream_complete"] and missing_downstream:
            active_jobs = row["active_downstream_jobs"]
            actions.append(
                {
                    "priority": 1 if active_jobs else 2,
                    "type": "complete_downstream_chain"
                    if active_jobs
                    else "submit_downstream_chain",
                    "index": row["index"],
                    "segment_id": row["segment_id"],
                    "missing_stages": missing_downstream,
                    "active_jobs": active_jobs,
                }
            )
        elif missing_upstream:
            actions.append(
                {
                    "priority": 3,
                    "type": "submit_upstream",
                    "index": row["index"],
                    "segment_id": row["segment_id"],
                    "missing_stages": missing_upstream,
                    "active_jobs": row["active_downstream_jobs"],
                }
            )
    actions.sort(key=lambda item: (item["priority"], item["index"]))
    return actions[:30]


def main() -> None:
    args = parse_args()
    segments = _jsonl(args.dataset_root / "segments.jsonl")
    active_job_list = _active_job_names()
    active_jobs = set(active_job_list)
    candidate_rows = _jsonl(
        args.annotation_root / "candidate_trials" / "g001_candidate_trials.jsonl"
    )
    candidate_counts = collections.Counter(
        row.get("segment_id") for row in candidate_rows
    )
    trials = _jsonl(args.benchmark_root / "weak" / "trials.jsonl")
    review = _jsonl(args.benchmark_root / "human_review_queue.jsonl")

    rows = []
    totals = collections.Counter()
    for index, segment in enumerate(segments, start=1):
        segment_id = segment["segment_id"]
        item: dict[str, Any] = {"index": index, "segment_id": segment_id}
        for stage in PLAYER_STAGES:
            item[stage] = _count_player_files(args.annotation_root, stage, segment_id)
        for stage in SINGLE_STAGES:
            item[stage] = int(_single_exists(args.annotation_root, stage, segment_id))
        item["candidate_trials"] = candidate_counts.get(segment_id, 0)
        item["downstream_marker"] = int(
            (
                args.annotation_root
                / "job_markers"
                / "downstream"
                / f"{segment_id}.json"
            ).exists()
        )
        segment_index = _segment_index(segment_id)
        item["active_upstream_jobs"] = _active_jobs_for_segment(
            active_jobs, segment_index, UPSTREAM_PREFIXES
        )
        item["active_downstream_jobs"] = _active_jobs_for_segment(
            active_jobs, segment_index, DOWNSTREAM_PREFIXES
        )
        item["upstream_complete"] = (
            item["pov_events"] == len(PLAYERS)
            and item["utterances"] == len(PLAYERS)
            and item["phase_events"] == 1
        )
        item["downstream_complete"] = (
            item["global_events"] == 1
            and item["information_states"] == len(PLAYERS)
            and item["memory_states"] == len(PLAYERS)
            and item["belief_states"] == len(PLAYERS)
            and item["candidate_trials"] > 0
        )
        item["ready_for_global"] = (
            item["upstream_complete"] and not item["global_events"]
        )
        item["ready_unsubmitted"] = (
            item["ready_for_global"]
            and not item["downstream_marker"]
            and not item["active_downstream_jobs"]
        )
        item["missing_upstream_stages"] = _missing_upstream_stages(item)
        item["missing_downstream_stages"] = _missing_downstream_stages(item)
        rows.append(item)
        totals["segments"] += 1
        totals["upstream_complete"] += int(item["upstream_complete"])
        totals["downstream_complete"] += int(item["downstream_complete"])
        totals["ready_for_global"] += int(item["ready_for_global"])
        totals["ready_unsubmitted"] += int(item["ready_unsubmitted"])
        totals["global_complete"] += int(item["global_events"])

    payload = {
        "dataset_root": args.dataset_root.as_posix(),
        "annotation_root": args.annotation_root.as_posix(),
        "benchmark_root": args.benchmark_root.as_posix(),
        "totals": dict(totals),
        "candidate_trials": len(candidate_rows),
        "benchmark_trials": len(trials),
        "review_queue": len(review),
        "active_job_counts": dict(collections.Counter(active_job_list)),
        "next_actions": _build_next_actions(rows, active_jobs),
        "segments": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = args.output.with_suffix(".md")
    md_lines = [
        "# Omni Goose Progress",
        "",
        f"- segments: {totals['segments']}",
        f"- upstream_complete: {totals['upstream_complete']}",
        f"- global_complete: {totals['global_complete']}",
        f"- downstream_complete: {totals['downstream_complete']}",
        f"- ready_for_global: {totals['ready_for_global']}",
        f"- ready_unsubmitted: {totals['ready_unsubmitted']}",
        f"- candidate_trials: {len(candidate_rows)}",
        f"- benchmark_trials: {len(trials)}",
        f"- review_queue: {len(review)}",
        "",
        "## Next Actions",
        "",
        "| priority | type | idx | segment_id | missing_stages | active_jobs |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ]
    for action in payload["next_actions"][:20]:
        md_lines.append(
            f"| {action['priority']} | {action['type']} | {action['index']} | {action['segment_id']} | "
            f"{','.join(action['missing_stages'])} | {','.join(action['active_jobs'])} |"
        )
    md_lines.extend(
        [
            "",
            "| idx | segment_id | pov | utt | phase | global | info | memory | belief | trials | marker | active_upstream | active_downstream |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        md_lines.append(
            f"| {row['index']} | {row['segment_id']} | {row['pov_events']} | {row['utterances']} | "
            f"{row['phase_events']} | {row['global_events']} | {row['information_states']} | "
            f"{row['memory_states']} | {row['belief_states']} | {row['candidate_trials']} | "
            f"{row['downstream_marker']} | {','.join(row['active_upstream_jobs'])} | "
            f"{','.join(row['active_downstream_jobs'])} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["totals"], ensure_ascii=False, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {md_path}")


def _active_job_names() -> list[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%j"], check=True, capture_output=True, text=True
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    main()
