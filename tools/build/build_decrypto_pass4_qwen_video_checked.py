from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import write_json, write_jsonl

PROBE_FILES = [
    "probes_A_pre_reveal.jsonl",
    "probes_B_reconstruct.jsonl",
    "probes_C_false_belief.jsonl",
    "probes_D_perspective_taking.jsonl",
]

RED_FLAG_PATTERNS = [
    "not properly grounded",
    "not leakage-safe",
    "heuristic link",
    "not a strong enough basis",
    "not directly visible",
    "remaining uncertainty",
    "uncertain",
    "泄漏",
    "不够",
    "不确定",
    "启发式",
]

PROMPT_LEAK_PATTERNS = [
    "ORACLE_TRUTH_JSON",
    "hidden_event_ids",
    "forbidden_event_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qwen-video-reviewed pass while preserving Codex-generated "
            "leakage-safe prompts from the input pass."
        )
    )
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--qwen-results-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        row = read_json(path)
        row["_result_file"] = path.name
        rows.append(row)
    return rows


def source_group_id(row: dict[str, Any]) -> str | None:
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    corrected_group = (
        parsed.get("corrected_probe_group")
        if isinstance(parsed.get("corrected_probe_group"), dict)
        else {}
    )
    source_group = (
        (row.get("source_task", {}).get("probe_group", {}) or {})
        if isinstance(row.get("source_task"), dict)
        else {}
    )
    group_id = corrected_group.get("probe_group_id") or source_group.get(
        "probe_group_id"
    )
    return str(group_id) if group_id else None


def has_red_flag(parsed: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    joined_reasons = " ".join(str(item) for item in parsed.get("review_reasons", []))
    for pattern in RED_FLAG_PATTERNS:
        if re.search(re.escape(pattern), joined_reasons, flags=re.IGNORECASE):
            reasons.append(f"review_reason_red_flag:{pattern}")
    return reasons


def qwen_video_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    if not row.get("parse_ok"):
        reasons.append("parse_failed")
    if parsed.get("decision") not in {"keep", "repair"}:
        reasons.append("decision_not_keep_or_repair")
    if parsed.get("gold_source_after_review") != "qwen_checked":
        reasons.append("not_qwen_checked")
    if parsed.get("needs_human_review") is True:
        reasons.append("qwen_marked_human_review")
    if parsed.get("remaining_uncertainties"):
        reasons.append("remaining_uncertainties_nonempty")
    reasons.extend(has_red_flag(parsed))
    if not isinstance(parsed.get("corrected_event"), dict):
        reasons.append("missing_corrected_event")
    if not isinstance(parsed.get("corrected_probe_group"), dict):
        reasons.append("missing_corrected_probe_group")
    return not reasons, reasons


def prompt_is_safe(probe: dict[str, Any]) -> bool:
    text = str(probe.get("prompt") or "")
    if (
        probe.get("probe_type") == "A_pre_reveal_belief"
        and "QUERY_VARIABLE_PUBLIC_FORM_JSON" not in text
    ):
        return False
    if (
        probe.get("probe_type") == "D_perspective_taking_prediction"
        and "TARGET_LISTENER_CONTEXT_JSON" in text
    ):
        return False
    return not any(
        pattern in text
        for pattern in PROMPT_LEAK_PATTERNS
        if probe.get("probe_type") == "A_pre_reveal_belief"
    )


def accepted_rows(
    pass_root: Path, qwen_results_root: Path
) -> tuple[set[str], list[dict[str, Any]]]:
    groups = read_jsonl(
        pass_root / "annotations" / "diagnostics" / "probe_groups.jsonl"
    )
    group_ids = {row["probe_group_id"] for row in groups}
    gate_rows: list[dict[str, Any]] = []
    accepted: set[str] = set()
    for row in result_rows(qwen_results_root):
        group_id = source_group_id(row)
        ok, reasons = qwen_video_gate(row)
        if group_id not in group_ids:
            ok = False
            reasons.append("group_not_in_input_pass")
        if ok and group_id:
            accepted.add(group_id)
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        gate_rows.append(
            {
                "result_file": row["_result_file"],
                "review_task_id": row.get("review_task_id"),
                "probe_group_id": group_id,
                "gate_decision": "accept" if ok else "reject",
                "gate_reasons": reasons,
                "qwen_decision": parsed.get("decision"),
                "qwen_gold_source_after_review": parsed.get("gold_source_after_review"),
                "needs_human_review": parsed.get("needs_human_review"),
                "prompt_policy": "preserve_input_pass_codex_safe_prompts",
            }
        )
    return accepted, gate_rows


def promote_with_safe_prompts(
    pass_root: Path,
    output_root: Path,
    accepted_group_ids: set[str],
    gate_rows: list[dict[str, Any]],
) -> None:
    src_diag = pass_root / "annotations" / "diagnostics"
    dst_diag = output_root / "annotations" / "diagnostics"
    dst_diag.mkdir(parents=True, exist_ok=True)
    accepted_by_group = {
        row["probe_group_id"]: row
        for row in gate_rows
        if row.get("gate_decision") == "accept"
    }

    groups = []
    for group in read_jsonl(src_diag / "probe_groups.jsonl"):
        if group["probe_group_id"] in accepted_group_ids:
            promoted = dict(group)
            promoted["gold_source"] = "qwen_checked"
            promoted["review_status"] = "pass4_qwen_video_checked_codex_safe_prompts"
            promoted["qwen_review_file"] = accepted_by_group[group["probe_group_id"]][
                "result_file"
            ]
            promoted["prompt_policy"] = "input_pass_codex_safe_prompts_preserved"
            groups.append(promoted)
        else:
            groups.append(group)
    write_jsonl(dst_diag / "probe_groups.jsonl", groups)

    unsafe_prompts: list[dict[str, Any]] = []
    for filename in PROBE_FILES:
        rows = []
        for probe in read_jsonl(src_diag / filename):
            if probe["probe_group_id"] in accepted_group_ids:
                if not prompt_is_safe(probe):
                    unsafe_prompts.append(
                        {
                            "probe_id": probe.get("probe_id"),
                            "probe_type": probe.get("probe_type"),
                        }
                    )
                promoted = dict(probe)
                promoted["gold_source"] = "qwen_checked"
                promoted["review_status"] = "pass4_qwen_video_checked_prompt_preserved"
                promoted["prompt_policy"] = "input_pass_codex_safe_prompts_preserved"
                rows.append(promoted)
            else:
                rows.append(probe)
        write_jsonl(dst_diag / filename, rows)

    if unsafe_prompts:
        raise SystemExit(
            f"accepted groups contain unsafe input-pass prompts: {unsafe_prompts[:5]}"
        )

    for filename in ["hidden_gold.jsonl", "diagnostic_quality.jsonl"]:
        rows = []
        for row in read_jsonl(src_diag / filename):
            if row["probe_group_id"] in accepted_group_ids:
                promoted = dict(row)
                if "gold_source" in promoted:
                    promoted["gold_source"] = "qwen_checked"
                if "recommended_gold_source" in promoted:
                    promoted["recommended_gold_source"] = "qwen_checked"
                promoted["review_status"] = (
                    "pass4_qwen_video_checked_codex_safe_prompts"
                )
                promoted["prompt_policy"] = "input_pass_codex_safe_prompts_preserved"
                rows.append(promoted)
            else:
                rows.append(row)
        write_jsonl(dst_diag / filename, rows)


def main() -> None:
    args = parse_args()
    src = args.input_pass_root
    dst = args.output_pass_root
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {dst}")
        shutil.rmtree(dst)

    (dst / "annotations").mkdir(parents=True)
    shutil.copytree(
        src / "annotations" / "oracle_ledger", dst / "annotations" / "oracle_ledger"
    )
    for dirname in ["docs", "scripts", "slurm", "tests"]:
        if (src / dirname).exists():
            shutil.copytree(src / dirname, dst / dirname, dirs_exist_ok=True)

    accepted_group_ids, gate_rows = accepted_rows(src, args.qwen_results_root)
    promote_with_safe_prompts(src, dst, accepted_group_ids, gate_rows)

    review_dir = dst / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        args.qwen_results_root,
        review_dir / "qwen3_omni_high_quality_results",
        dirs_exist_ok=True,
    )
    write_jsonl(review_dir / "qwen_video_review_gate.jsonl", gate_rows)
    accepted = [row for row in gate_rows if row["gate_decision"] == "accept"]
    rejected = [row for row in gate_rows if row["gate_decision"] == "reject"]
    summary = {
        "input_pass_root": src.as_posix(),
        "output_pass_root": dst.as_posix(),
        "qwen_results_root": args.qwen_results_root.as_posix(),
        "qwen_results": len(gate_rows),
        "accepted_groups": len(accepted),
        "rejected_groups": len(rejected),
        "accepted_probe_group_ids": sorted(accepted_group_ids),
        "reject_reason_counts": dict(
            collections.Counter(
                reason for row in rejected for reason in row["gate_reasons"]
            )
        ),
        "prompt_policy": "preserve_input_pass_codex_safe_prompts_do_not_use_qwen_corrected_prompts",
        "status": "pass4_qwen_video_checked_gate_complete",
    }
    write_json(review_dir / "pass4_qwen_video_review_summary.json", summary)
    (dst / "README.md").write_text(
        "# Omni Goose Decrypto Diagnostic Pass 4 Qwen-video-reviewed\n\n"
        "This pass uses Qwen3-Omni only as a high-quality video review assistant. "
        "It preserves the leakage-safe Codex prompt templates from the input pass and "
        "does not merge Qwen-generated corrected prompts.\n\n"
        f"- qwen_results: {summary['qwen_results']}\n"
        f"- accepted_groups: {summary['accepted_groups']}\n"
        f"- rejected_groups: {summary['rejected_groups']}\n"
        f"- prompt_policy: {summary['prompt_policy']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
