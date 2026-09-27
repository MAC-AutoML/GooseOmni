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


def build_scope_report(pass_root: Path) -> dict[str, Any]:
    annotation_root = pass_root / "annotations"
    benchmark_root = pass_root / "benchmark/gooseomni_v1"
    groups = read_jsonl(annotation_root / "diagnostics/probe_groups.jsonl")
    quality = read_jsonl(annotation_root / "diagnostics/diagnostic_quality.jsonl")
    hidden = read_jsonl(annotation_root / "diagnostics/hidden_gold.jsonl")
    trials = read_jsonl(benchmark_root / "static_trials/trials.jsonl")
    validation = read_json(benchmark_root / "reports/validation.json")

    scope_groups = [
        row
        for row in groups
        if (row.get("quality") or {}).get("scope_limited_public_speech_gold")
    ]
    scope_group_ids = {row["probe_group_id"] for row in scope_groups}
    scope_quality = [
        row for row in quality if row.get("probe_group_id") in scope_group_ids
    ]
    scope_hidden = [
        row for row in hidden if row.get("probe_group_id") in scope_group_ids
    ]
    missing_scope_limitations = [
        row.get("probe_group_id")
        for row in scope_groups
        if not row.get("scope_limitations")
    ]
    quality_missing_scope_flag = [
        row.get("probe_group_id")
        for row in scope_quality
        if not row.get("scope_limited_public_speech_gold")
    ]
    hidden_missing_scope_flag = [
        row.get("probe_group_id")
        for row in scope_hidden
        if not row.get("scope_limited_public_speech_gold")
    ]
    trial_text = (
        (benchmark_root / "static_trials/trials.jsonl").read_text(encoding="utf-8")
        if (benchmark_root / "static_trials/trials.jsonl").exists()
        else ""
    )

    issues: list[dict[str, Any]] = []
    if validation.get("ok") is not True:
        issues.append({"code": "base_validation_not_ok", "validation": validation})
    if missing_scope_limitations:
        issues.append(
            {
                "code": "scope_groups_missing_scope_limitations",
                "ids": missing_scope_limitations[:20],
            }
        )
    if quality_missing_scope_flag:
        issues.append(
            {
                "code": "scope_quality_missing_flag",
                "ids": quality_missing_scope_flag[:20],
            }
        )
    if hidden_missing_scope_flag:
        issues.append(
            {"code": "scope_hidden_missing_flag", "ids": hidden_missing_scope_flag[:20]}
        )
    if "hidden_gold" in trial_text or "forbidden_event_ids" in trial_text:
        issues.append({"code": "public_static_trials_leak_hidden_fields"})

    qv_counts: dict[str, int] = {}
    for row in groups:
        qv = (
            (row.get("query_variable") or {}).get("type")
            or row.get("query_variable_type")
            or "unknown"
        )
        qv_counts[str(qv)] = qv_counts.get(str(qv), 0) + 1
    scope_qv_counts: dict[str, int] = {}
    for row in scope_groups:
        qv = (
            (row.get("query_variable") or {}).get("type")
            or row.get("query_variable_type")
            or "unknown"
        )
        scope_qv_counts[str(qv)] = scope_qv_counts.get(str(qv), 0) + 1

    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "pass_root": pass_root.as_posix(),
        "validation_ok": validation.get("ok"),
        "counts": {
            "probe_groups": len(groups),
            "trials": len(trials),
            "scope_limited_groups": len(scope_groups),
            "scope_limited_hidden_rows": len(scope_hidden),
            "scope_limited_quality_rows": len(scope_quality),
        },
        "query_variable_counts": dict(sorted(qv_counts.items())),
        "scope_limited_query_variable_counts": dict(sorted(scope_qv_counts.items())),
        "probe_type_counts": count_by(trials, "probe_type"),
        "scope_policy": {
            "scope_limited_public_speech_gold": "certifies transcript, speaker, and public meeting-speech interpretation probes only",
            "not_certified_by_scope_limited_gold": [
                "canonical identity of unresolved display names",
                "global truth of a spoken claim unless independently supported by oracle ledger",
                "private POV observations not present in the public meeting clip",
            ],
            "public_trials_hidden_gold_leak_check": "passed"
            if not any(
                i["code"] == "public_static_trials_leak_hidden_fields" for i in issues
            )
            else "failed",
        },
    }


def write_benchmark_card(pass_root: Path, report: dict[str, Any]) -> None:
    reports = pass_root / "benchmark/gooseomni_v1/reports"
    counts = report["counts"]
    qv = report["query_variable_counts"]
    scope_qv = report["scope_limited_query_variable_counts"]
    probe_types = report["probe_type_counts"]
    text = f"""# GooseOmni-v1 Benchmark Card

GooseOmni-v1 is a trajectory-derived Theory-of-Mind diagnostic benchmark built from a strictly aligned 6-POV Goose Goose Duck replay. It uses oracle trajectory ledgers, visibility projections, claim-truth links, and Decrypto-style A/B/C/D probe groups to evaluate representational change, false belief, claim verification, perspective taking, strategy communication, and perspective leakage.

## Release Scope

This pass contains {counts["probe_groups"]} human-verified probe groups and {counts["trials"]} public static trials. Of these, {counts["scope_limited_groups"]} groups are `scope_limited_public_speech_gold`: they certify public meeting-speech transcript, speaker identity, and public interpretation probes, while preserving unresolved display-name or context uncertainty as explicit scope limitations.

Scope-limited meeting-speech gold does not certify unresolved display names as canonical identities, does not certify a spoken claim as globally true unless independently supported by the oracle ledger, and does not add private POV evidence not present in the public meeting clip.

## Diagnostic Mechanism

Segments are storage units only. The benchmark unit is a trajectory node: `cutoff_abs_sec + target_player + query_variable + information_gap`. Probe groups follow a Decrypto-style diagnostic mechanism: A asks for pre-reveal belief, B reconstructs earlier belief after truth/context reveal, C asks for another player's false or incomplete belief, and D asks a speaker to predict listener interpretation from speaker-safe context.

## Counts

- probe_groups: {counts["probe_groups"]}
- static_trials/prompts: {counts["trials"]}
- scope_limited_public_speech_groups: {counts["scope_limited_groups"]}
- query_variable_counts: `{json.dumps(qv, ensure_ascii=False)}`
- scope_limited_query_variable_counts: `{json.dumps(scope_qv, ensure_ascii=False)}`
- probe_type_counts: `{json.dumps(probe_types, ensure_ascii=False)}`

## Leakage Policy

Static `trials.jsonl` contains public prompts only. Hidden gold and forbidden evidence stay in scorer-only files. The scope audit checks that public trials do not expose `hidden_gold` or `forbidden_event_ids`.

## Gold Source

All groups in this pass have `gold_source=human_verified`. Scope-limited groups are still human verified, but their verified scope is public meeting-speech interpretation rather than unrestricted oracle truth.
"""
    (reports / "benchmark_card.md").write_text(text, encoding="utf-8")
    (reports / "annotation_quality.md").write_text(
        "# Annotation Quality\n\n"
        f"- validation_ok: {report['validation_ok']}\n"
        f"- scope_limited_public_speech_groups: {counts['scope_limited_groups']}\n"
        "- scope-limited rows preserve unresolved alias/context uncertainty and do not certify those uncertainties as oracle truth.\n",
        encoding="utf-8",
    )
    (reports / "diagnostic_quality.md").write_text(
        "# Diagnostic Quality\n\n"
        f"- probe_groups: {counts['probe_groups']}\n"
        f"- prompts: {counts['trials']}\n"
        f"- scope_limited_public_speech_groups: {counts['scope_limited_groups']}\n"
        f"- issue_count: {report['issue_count']}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and document scope-limited human gold release semantics."
    )
    parser.add_argument("--pass-root", type=Path, required=True)
    parser.add_argument("--write-card", action="store_true")
    args = parser.parse_args()
    report = build_scope_report(args.pass_root)
    reports = args.pass_root / "benchmark/gooseomni_v1/reports"
    write_json(reports / "scope_limited_gold_audit.json", report)
    if args.write_card:
        write_benchmark_card(args.pass_root, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
