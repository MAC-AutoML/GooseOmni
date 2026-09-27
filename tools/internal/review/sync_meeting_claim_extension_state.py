from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def refresh_audio_confirmation_queue(records_path: Path, output_root: Path) -> dict[str, Any]:
    rows = read_jsonl(records_path)
    selected: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    for record in rows:
        decision = record.get("codex_human_decision")
        decision_counts[decision or "unknown"] += 1
        if decision not in {"needs_audio_confirmation", "needs_alias_and_audio_confirmation"}:
            continue
        utterance = record.get("candidate_utterance") or {}
        selected.append(
            {
                "audio_confirmation_item_id": record["review_item_id"],
                "source_result_id": record["source_result_id"],
                "video_file": record["video_file"],
                "contact_sheet": record["contact_sheet"],
                "next_required_gate": record["next_required_gate"],
                "canonical_speaker": utterance.get("canonical_speaker"),
                "speaker_display_name": utterance.get("speaker_display_name"),
                "abs_start_sec": utterance.get("abs_start_sec"),
                "abs_end_sec": utterance.get("abs_end_sec"),
                "claim_type": utterance.get("claim_type"),
                "strategic_function": utterance.get("strategic_function"),
                "claim_target_players": utterance.get("claim_target_players") or [],
                "claim_target_display_names": utterance.get("claim_target_display_names") or [],
                "qwen_transcript_to_confirm": utterance.get("utterance_text"),
                "promotion_policy": (
                    "Only promote after exact transcript and any alias mapping are independently confirmed; "
                    "Qwen-only output remains qwen_checked."
                ),
            }
        )
    summary = {
        "ok": True,
        "source_records": records_path.as_posix(),
        "audio_confirmation_items": len(selected),
        "decision_counts_in_source": dict(sorted(decision_counts.items())),
        "promotion_to_human_verified_gold": False,
    }
    write_jsonl(output_root / "audio_confirmation_queue.jsonl", selected)
    write_json(output_root / "audio_confirmation_queue_summary.json", summary)
    return summary


def write_coverage_report(stable_root: Path, records_path: Path, output_path: Path) -> dict[str, Any]:
    probe_groups = read_jsonl(stable_root / "annotations/diagnostics/probe_groups.jsonl")
    trials = read_jsonl(stable_root / "benchmark/gooseomni_v1/static_trials/trials.jsonl")
    records = read_jsonl(records_path)

    query_variables = Counter(
        ((row.get("query_variable") or {}).get("type") or row.get("query_variable_type") or "unknown")
        for row in probe_groups
    )
    probe_types = Counter(row.get("probe_type", "unknown") for row in trials)
    decisions: Counter[str] = Counter()
    claim_types: Counter[str] = Counter()
    strategic_functions: Counter[str] = Counter()
    audio_candidates = 0
    for record in records:
        decision = record.get("codex_human_decision") or "unknown"
        decisions[decision] += 1
        utterance = record.get("candidate_utterance") or {}
        claim_types[utterance.get("claim_type") or "unknown"] += 1
        strategic_functions[utterance.get("strategic_function") or "unknown"] += 1
        if decision in {"needs_audio_confirmation", "needs_alias_and_audio_confirmation"}:
            audio_candidates += 1

    report = {
        "stable_pass115": {
            "probe_groups": len(probe_groups),
            "static_trials": len(trials),
            "query_variable_counts": dict(sorted(query_variables.items())),
            "probe_type_counts": dict(sorted(probe_types.items())),
        },
        "meeting_claim_extension_current": {
            "strict_review_records": len(records),
            "decision_counts": dict(sorted(decisions.items())),
            "audio_confirmation_candidates": audio_candidates,
            "claim_type_counts": dict(sorted(claim_types.items())),
            "strategic_function_counts": dict(sorted(strategic_functions.items())),
        },
        "main_gaps": [
            "pass115 claim_truth_vs_claim_awareness and D_perspective_taking are still thin relative to route_belief/private_witness coverage",
            "meeting claim candidates are not human gold until audio/alias confirmation plus final merge gate",
            "audio-confirmation outputs must be audited before any Codex-human gold merge pack is accepted",
        ],
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync meeting-claim extension audit/review/audio queues and coverage report.")
    parser.add_argument("--results-root", type=Path, default=Path("runs/gooseomni_meeting_claim_grounding_pass123_qwen_batches/results"))
    parser.add_argument("--review-root", type=Path, default=Path("runs/gooseomni_meeting_claim_grounding_pass123_qwen_batches/review"))
    parser.add_argument(
        "--strict-review-root",
        type=Path,
        default=Path("runs/gooseomni_meeting_claim_grounding_pass123_qwen_batches/codex_human_review_strict_high"),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("runs/gooseomni_meeting_claim_grounding_pass123_qwen_batches/codex_human_review_strict_high/codex_review_records_pass124_full_queue"),
    )
    parser.add_argument(
        "--stable-root",
        type=Path,
        default=Path("runs/gooseomni_decrypto_human_verified_combined_pass115_private_witness_safe_route"),
    )
    args = parser.parse_args()

    run(
        [
            ".venv/bin/python",
            "work/audit_meeting_claim_qwen_outputs.py",
            "--results-root",
            args.results_root.as_posix(),
            "--output-root",
            args.review_root.as_posix(),
        ]
    )
    run(
        [
            ".venv/bin/python",
            "work/build_meeting_claim_codex_review_pack.py",
            "--audit-root",
            args.review_root.as_posix(),
            "--results-root",
            args.results_root.as_posix(),
            "--output-root",
            args.strict_review_root.as_posix(),
        ]
    )
    run(
        [
            ".venv/bin/python",
            "work/write_codex_meeting_claim_review_records.py",
            "--review-queue",
            (args.strict_review_root / "codex_human_review_queue.jsonl").as_posix(),
            "--output-root",
            args.records_root.as_posix(),
        ]
    )
    audio_summary = refresh_audio_confirmation_queue(
        args.records_root / "codex_human_review_records.jsonl",
        args.records_root,
    )
    coverage = write_coverage_report(
        args.stable_root,
        args.records_root / "codex_human_review_records.jsonl",
        args.results_root.parent / "reports/current_coverage_gap_report.json",
    )
    summary = {
        "ok": True,
        "audio_confirmation_items": audio_summary["audio_confirmation_items"],
        "strict_review_records": coverage["meeting_claim_extension_current"]["strict_review_records"],
        "stable_probe_groups": coverage["stable_pass115"]["probe_groups"],
        "promotion_to_human_verified_gold": False,
    }
    write_json(args.results_root.parent / "reports/sync_meeting_claim_extension_state_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
