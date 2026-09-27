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
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gooseomni.benchmark.decrypto_diagnostics import write_json, write_jsonl


PROBE_FILES = [
    "probes_A_pre_reveal.jsonl",
    "probes_B_reconstruct.jsonl",
    "probes_C_false_belief.jsonl",
    "probes_D_perspective_taking.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote Codex-reviewed GooseOmni probe groups to human_verified.")
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--review-records", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_input_pass(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise SystemExit(f"output exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def accepted_review_records(records: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    accepted: set[str] = set()
    gate_rows: list[dict[str, Any]] = []
    for record in records:
        group_id = record.get("probe_group_id")
        decision = record.get("gate_decision")
        reasons: list[str] = []
        if not group_id:
            reasons.append("missing_probe_group_id")
        if decision != "accept_human_verified":
            reasons.append("decision_not_accept_human_verified")
        if record.get("reviewer") != "codex_human_reviewer":
            reasons.append("reviewer_not_codex_human_reviewer")
        if not record.get("evidence_checked"):
            reasons.append("missing_evidence_checked")
        if record.get("remaining_uncertainties"):
            reasons.append("remaining_uncertainties_nonempty")
        if record.get("needs_human_review_after"):
            reasons.append("still_needs_human_review")
        ok = not reasons
        if ok and group_id:
            accepted.add(str(group_id))
        gate_rows.append(
            {
                "probe_group_id": group_id,
                "gate_decision": "accept" if ok else "reject",
                "gate_reasons": reasons,
                "source_decision": decision,
                "reviewer": record.get("reviewer"),
                "evidence_checked": record.get("evidence_checked", []),
                "remaining_uncertainties": record.get("remaining_uncertainties", []),
                "review_summary": record.get("review_summary"),
            }
        )
    return accepted, gate_rows


def promote_rows(path: Path, accepted: set[str], review_by_group: dict[str, dict[str, Any]]) -> None:
    rows = []
    for row in read_jsonl(path):
        group_id = row.get("probe_group_id")
        if group_id in accepted:
            promoted = dict(row)
            if "gold_source" in promoted:
                promoted["gold_source"] = "human_verified"
            if "recommended_gold_source" in promoted:
                promoted["recommended_gold_source"] = "human_verified"
            promoted["review_status"] = "pass11_codex_human_verified"
            promoted["human_review_record_id"] = review_by_group[group_id].get("review_record_id")
            promoted["human_reviewer"] = "codex_human_reviewer"
            promoted["needs_human_review"] = False
            rows.append(promoted)
        else:
            rows.append(row)
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    copy_input_pass(args.input_pass_root, args.output_pass_root, args.overwrite)

    records = read_jsonl(args.review_records)
    accepted, gate_rows = accepted_review_records(records)
    review_by_group = {row["probe_group_id"]: row for row in records if row.get("probe_group_id") in accepted}

    known_groups = {row["probe_group_id"] for row in read_jsonl(args.output_pass_root / "annotations/diagnostics/probe_groups.jsonl")}
    unknown = sorted(accepted - known_groups)
    if unknown:
        raise SystemExit(f"review records reference unknown groups: {unknown}")

    diag = args.output_pass_root / "annotations/diagnostics"
    promote_rows(diag / "probe_groups.jsonl", accepted, review_by_group)
    promote_rows(diag / "hidden_gold.jsonl", accepted, review_by_group)
    promote_rows(diag / "diagnostic_quality.jsonl", accepted, review_by_group)
    for filename in PROBE_FILES:
        promote_rows(diag / filename, accepted, review_by_group)

    review_dir = args.output_pass_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.review_records, review_dir / "codex_human_review_records.jsonl")
    write_jsonl(review_dir / "codex_human_review_gate.jsonl", gate_rows)
    summary = {
        "input_pass_root": args.input_pass_root.as_posix(),
        "output_pass_root": args.output_pass_root.as_posix(),
        "review_records": args.review_records.as_posix(),
        "records": len(records),
        "accepted_groups": len(accepted),
        "rejected_groups": len(records) - len(accepted),
        "accepted_probe_group_ids": sorted(accepted),
        "reject_reason_counts": dict(collections.Counter(reason for row in gate_rows for reason in row["gate_reasons"])),
        "status": "pass11_codex_human_verified_gate_complete",
    }
    write_json(review_dir / "pass11_codex_human_review_summary.json", summary)
    (args.output_pass_root / "README.md").write_text(
        "# Omni Goose Decrypto Diagnostic Pass 11 Codex-human-reviewed\n\n"
        "This pass promotes only probe groups with explicit Codex-human review records. "
        "Rejected or uncertain records are preserved in the gate log but are not promoted.\n\n"
        f"- records: {summary['records']}\n"
        f"- accepted_groups: {summary['accepted_groups']}\n"
        f"- rejected_groups: {summary['rejected_groups']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
