from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report current GooseOmni meeting-claim benchmark expansion progress.")
    parser.add_argument(
        "--stable-root",
        type=Path,
        default=Path("runs/gooseomni_decrypto_human_verified_combined_pass115_private_witness_safe_route"),
    )
    parser.add_argument(
        "--meeting-root",
        type=Path,
        default=Path("runs/gooseomni_meeting_claim_grounding_pass123_qwen_batches"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("runs/gooseomni_meeting_claim_audio_confirmation_pass125_qwen"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stable_validation = read_json(args.stable_root / "benchmark/gooseomni_v1/reports/validation.json")
    stable_trials = read_jsonl(args.stable_root / "benchmark/gooseomni_v1/static_trials/trials.jsonl")
    stable_groups = read_jsonl(args.stable_root / "annotations/diagnostics/probe_groups.jsonl")
    stable_probe_types: dict[str, int] = {}
    for row in stable_trials:
        key = row.get("probe_type", "unknown")
        stable_probe_types[key] = stable_probe_types.get(key, 0) + 1
    stable_query_variables: dict[str, int] = {}
    for row in stable_groups:
        qv = (row.get("query_variable") or {}).get("type") or row.get("query_variable_type") or "unknown"
        stable_query_variables[qv] = stable_query_variables.get(qv, 0) + 1

    meeting_audit = read_json(args.meeting_root / "review/qwen_meeting_claim_audit_summary.json")
    meeting_sync = read_json(args.meeting_root / "reports/sync_meeting_claim_extension_state_summary.json")
    audio_audit = read_json(args.audio_root / "audit/audio_confirmation_audit_summary.json")
    merge_review = read_json(args.audio_root / "codex_human_gold_merge_review_records/summary.json")
    drafts = read_json(args.audio_root / "probe_drafts_qwen_checked/summary.json")
    accept = read_json(args.audio_root / "human_gold_accept_candidates/summary.json")

    report = {
        "ok": True,
        "stable_human_verified_core": {
            "validation_ok": stable_validation.get("ok"),
            "issue_count": stable_validation.get("issue_count"),
            "counts": stable_validation.get("counts", {}),
            "probe_type_counts": dict(sorted(stable_probe_types.items())),
            "query_variable_counts": dict(sorted(stable_query_variables.items())),
        },
        "meeting_claim_candidate_layer": {
            "qwen_results": meeting_audit.get("results"),
            "candidate_utterances": meeting_audit.get("candidate_utterances"),
            "high_priority_candidates": meeting_audit.get("high_priority_candidates"),
            "strict_review_records": meeting_sync.get("strict_review_records"),
            "audio_confirmation_items": meeting_sync.get("audio_confirmation_items"),
            "promotion_to_human_verified_gold": False,
        },
        "audio_confirmation_layer": {
            "results": audio_audit.get("results"),
            "merge_gate_candidates": audio_audit.get("merge_gate_candidates"),
            "decision_counts": audio_audit.get("decision_counts", {}),
            "issue_counts": audio_audit.get("issue_counts", {}),
            "promotion_to_human_verified_gold": False,
        },
        "probe_draft_layer": {
            "merge_review_records": merge_review.get("records"),
            "safe_for_probe_draft_generation": merge_review.get("safe_for_probe_draft_generation"),
            "qwen_checked_probe_groups": drafts.get("probe_groups"),
            "qwen_checked_probes": drafts.get("probes"),
            "human_gold_accept_candidate_groups": accept.get("probe_groups"),
            "human_gold_accept_candidate_probes": accept.get("probes"),
            "promotion_to_human_verified_gold": False,
        },
        "remaining_gates": [
            "Finish pass125/pass128 audio-confirmation coverage.",
            "Run final audio spot-check or independent transcript review for accept candidates.",
            "Create a new non-overwriting human_verified pass only after explicit final accept records.",
            "Validate no hidden_gold leakage and query/probe coverage after merge.",
        ],
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
