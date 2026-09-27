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


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "unknown"
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def summarize_stable_core(stable_root: Path) -> dict[str, Any]:
    validation = read_json(
        stable_root / "benchmark/gooseomni_v1/reports/validation.json"
    )
    trials = read_jsonl(
        stable_root / "benchmark/gooseomni_v1/static_trials/trials.jsonl"
    )
    groups = read_jsonl(stable_root / "annotations/diagnostics/probe_groups.jsonl")
    qv_counts: dict[str, int] = {}
    for group in groups:
        qv = (
            (group.get("query_variable") or {}).get("type")
            or group.get("query_variable_type")
            or "unknown"
        )
        qv_counts[str(qv)] = qv_counts.get(str(qv), 0) + 1
    return {
        "root": stable_root.as_posix(),
        "validation_ok": validation.get("ok"),
        "issue_count": validation.get("issue_count"),
        "counts": validation.get("counts", {}),
        "probe_type_counts": count_by(trials, "probe_type"),
        "query_variable_counts": dict(sorted(qv_counts.items())),
    }


def summarize_audio_root(root: Path) -> dict[str, Any]:
    audio_audit = read_json(root / "audit/audio_confirmation_audit_summary.json")
    merge_review = read_json(
        root / "codex_human_gold_merge_review_records/summary.json"
    )
    drafts = read_json(root / "probe_drafts_qwen_checked/summary.json")
    accept = read_json(root / "human_gold_accept_candidates/summary.json")
    combined = read_json(root / "combined_results_summary.json")
    draft_probes = read_jsonl(
        root / "probe_drafts_qwen_checked/probes.qwen_checked_draft.jsonl"
    )
    accept_probes = read_jsonl(
        root / "human_gold_accept_candidates/probes.human_gold_accept_candidates.jsonl"
    )
    return {
        "root": root.as_posix(),
        "combined_results": combined.get("combined_results"),
        "expected_tasks": combined.get("expected_tasks"),
        "missing_results": combined.get("missing_results"),
        "audio_results": audio_audit.get("results"),
        "merge_gate_candidates": audio_audit.get("merge_gate_candidates"),
        "audio_decision_counts": audio_audit.get("decision_counts", {}),
        "audio_issue_counts": audio_audit.get("issue_counts", {}),
        "merge_review_records": merge_review.get("records"),
        "safe_for_probe_draft_generation": merge_review.get(
            "safe_for_probe_draft_generation"
        ),
        "qwen_checked_probe_groups": drafts.get("probe_groups"),
        "qwen_checked_probes": drafts.get("probes"),
        "qwen_checked_probe_type_counts": count_by(draft_probes, "probe_type"),
        "human_gold_accept_candidate_groups": accept.get("probe_groups"),
        "human_gold_accept_candidate_probes": accept.get("probes"),
        "human_gold_accept_candidate_probe_type_counts": count_by(
            accept_probes, "probe_type"
        ),
        "promotion_to_human_verified_gold": False,
    }


def build_total_report(stable_root: Path, audio_roots: list[Path]) -> dict[str, Any]:
    layers = [summarize_audio_root(root) for root in audio_roots]
    totals = {
        "audio_results": sum(layer.get("audio_results") or 0 for layer in layers),
        "merge_gate_candidates": sum(
            layer.get("merge_gate_candidates") or 0 for layer in layers
        ),
        "safe_for_probe_draft_generation": sum(
            layer.get("safe_for_probe_draft_generation") or 0 for layer in layers
        ),
        "qwen_checked_probe_groups": sum(
            layer.get("qwen_checked_probe_groups") or 0 for layer in layers
        ),
        "qwen_checked_probes": sum(
            layer.get("qwen_checked_probes") or 0 for layer in layers
        ),
        "human_gold_accept_candidate_groups": sum(
            layer.get("human_gold_accept_candidate_groups") or 0 for layer in layers
        ),
        "human_gold_accept_candidate_probes": sum(
            layer.get("human_gold_accept_candidate_probes") or 0 for layer in layers
        ),
    }
    missing_results = [layer for layer in layers if layer.get("missing_results")]
    return {
        "ok": True,
        "stable_human_verified_core": summarize_stable_core(stable_root),
        "audio_confirmation_layers": layers,
        "extension_totals": totals,
        "remaining_gates": [
            "Finish all audio-confirmation jobs; combined layers with missing_results are not final.",
            "Convert only exact/high-confidence accept candidates through an explicit human_verified promotion pass.",
            "Create a new non-overwriting expanded human_verified benchmark pass.",
            "Run no-leakage, hidden_gold isolation, schema, and query/probe coverage validation before release.",
        ],
        "layers_with_missing_results": [layer["root"] for layer in missing_results],
        "promotion_to_human_verified_gold": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report total GooseOmni meeting-claim expansion coverage across audio layers."
    )
    parser.add_argument(
        "--stable-root",
        type=Path,
        default=Path(
            "runs/gooseomni_decrypto_human_verified_combined_pass115_private_witness_safe_route"
        ),
    )
    parser.add_argument("--audio-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_total_report(args.stable_root, args.audio_root)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
