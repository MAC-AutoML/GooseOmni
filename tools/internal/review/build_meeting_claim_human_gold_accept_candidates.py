from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in read_jsonl(path) if key in row}


def build_accept_candidates(
    groups_path: Path,
    probes_path: Path,
    hidden_gold_path: Path,
    merge_records_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = read_jsonl(groups_path)
    probes = read_jsonl(probes_path)
    hidden_gold = read_jsonl(hidden_gold_path)
    merge_by_source = load_by_key(merge_records_path, "source_review_item_id")

    accepted_groups: list[dict[str, Any]] = []
    accepted_probes: list[dict[str, Any]] = []
    accepted_hidden: list[dict[str, Any]] = []
    accepted_group_ids: set[str] = set()

    for group in groups:
        source_id = group.get("source_review_item_id")
        merge = merge_by_source.get(source_id)
        if not merge:
            continue
        if (
            merge.get("codex_human_merge_decision")
            != "accept_for_qwen_checked_merge_candidate"
        ):
            continue
        transcript = merge.get("confirmed_transcript") or {}
        speaker = merge.get("confirmed_speaker") or {}
        if (
            transcript.get("transcript_match") != "exact"
            or transcript.get("text_confidence") != "high"
        ):
            continue
        if speaker.get("speaker_confidence") != "high":
            continue
        if merge.get("remaining_uncertainties"):
            # Keep uncertain alias/display-only cases out of human-gold accept candidates.
            continue
        accepted = dict(group)
        accepted["quality"] = {
            **(accepted.get("quality") or {}),
            "human_gold_accept_candidate": True,
            "promotion_to_human_verified_gold": False,
        }
        accepted["needs_human_review"] = True
        accepted["review_reasons"] = [
            "Candidate satisfies exact-transcript and speaker-confidence gates.",
            "Still requires explicit final promotion pass before human_verified release.",
        ]
        accepted_groups.append(accepted)
        accepted_group_ids.add(accepted["probe_group_id"])

    for probe in probes:
        if probe.get("probe_group_id") in accepted_group_ids:
            accepted = dict(probe)
            accepted["gold_source"] = "human_gold_accept_candidate"
            accepted["needs_human_review"] = True
            accepted["promotion_to_human_verified_gold"] = False
            accepted_probes.append(accepted)

    for row in hidden_gold:
        if row.get("probe_group_id") in accepted_group_ids:
            accepted = dict(row)
            accepted["gold_source"] = "human_gold_accept_candidate"
            accepted["needs_human_review"] = True
            accepted["promotion_to_human_verified_gold"] = False
            accepted_hidden.append(accepted)

    return accepted_groups, accepted_probes, accepted_hidden


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict human-gold accept candidates from qwen_checked meeting-claim drafts."
    )
    parser.add_argument("--probe-draft-root", type=Path, required=True)
    parser.add_argument("--merge-review-records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    groups, probes, hidden = build_accept_candidates(
        args.probe_draft_root / "probe_groups.qwen_checked_draft.jsonl",
        args.probe_draft_root / "probes.qwen_checked_draft.jsonl",
        args.probe_draft_root / "hidden_gold.qwen_checked_draft.jsonl",
        args.merge_review_records,
    )
    write_jsonl(
        args.output_root / "probe_groups.human_gold_accept_candidates.jsonl", groups
    )
    write_jsonl(args.output_root / "probes.human_gold_accept_candidates.jsonl", probes)
    write_jsonl(
        args.output_root / "hidden_gold.human_gold_accept_candidates.jsonl", hidden
    )
    probe_type_counts = Counter(row.get("probe_type") for row in probes)
    summary = {
        "ok": True,
        "source_probe_draft_root": args.probe_draft_root.as_posix(),
        "source_merge_review_records": args.merge_review_records.as_posix(),
        "probe_groups": len(groups),
        "probes": len(probes),
        "hidden_gold": len(hidden),
        "probe_type_counts": dict(sorted(probe_type_counts.items())),
        "gold_source": "human_gold_accept_candidate",
        "promotion_to_human_verified_gold": False,
        "note": "This is an accept-candidate layer only. It does not write human_verified benchmark gold.",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
