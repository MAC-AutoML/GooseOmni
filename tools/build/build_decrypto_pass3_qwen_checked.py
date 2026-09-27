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
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pass3_qwen_checked with a conservative Qwen-review merge gate.")
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--qwen-results-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def compact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def prompt_texts(parsed: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for item in parsed.get("corrected_prompts") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("prompt_id") or item.get("probe_type") or next(iter(item.keys()), "unknown"))
        text = compact_text(item.get("prompt") or item.get("prompt_text") or item.get("content") or item)
        if key in {"A", "A_pre_reveal_belief"}:
            texts["A"] = text
        elif key in {"B", "B_post_reveal_reconstruct_previous_belief"}:
            texts["B"] = text
        elif key in {"C", "C_other_agent_false_belief"}:
            texts["C"] = text
        elif key in {"D", "D_perspective_taking_prediction"}:
            texts["D"] = text
    return texts


def hidden_description_leaked(parsed: dict[str, Any]) -> bool:
    texts = prompt_texts(parsed)
    a_text = texts.get("A", "")
    if not a_text:
        return True
    if "QUERY_VARIABLE_PUBLIC_FORM_JSON" not in a_text:
        return True
    event = parsed.get("corrected_event") if isinstance(parsed.get("corrected_event"), dict) else {}
    description = str(event.get("description") or "").strip()
    if description and description in a_text:
        return True
    for field in ["description", "direct_visual_evidence", "direct_audio_evidence"]:
        value = event.get(field)
        if isinstance(value, str) and len(value) >= 12 and value in a_text:
            return True
        if isinstance(value, list):
            for item in value:
                item_text = str(item)
                if len(item_text) >= 12 and item_text in a_text:
                    return True
    return False


def qwen_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
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
    joined_reasons = " ".join(str(item) for item in parsed.get("review_reasons", []))
    for pattern in RED_FLAG_PATTERNS:
        if re.search(re.escape(pattern), joined_reasons, flags=re.IGNORECASE):
            reasons.append(f"review_reason_red_flag:{pattern}")
    if hidden_description_leaked(parsed):
        reasons.append("corrected_A_prompt_leaks_or_missing_public_query")
    if len(parsed.get("corrected_prompts") or []) < 4:
        reasons.append("corrected_prompts_incomplete")
    return not reasons, reasons


def promote_rows(
    pass_root: Path,
    output_root: Path,
    accepted_group_ids: set[str],
    gate_rows: list[dict[str, Any]],
) -> None:
    src_diag = pass_root / "annotations" / "diagnostics"
    dst_diag = output_root / "annotations" / "diagnostics"
    dst_diag.mkdir(parents=True, exist_ok=True)
    accepted_by_group = {row["probe_group_id"]: row for row in gate_rows if row.get("gate_decision") == "accept"}

    groups = []
    for group in read_jsonl(src_diag / "probe_groups.jsonl"):
        if group["probe_group_id"] in accepted_group_ids:
            promoted = dict(group)
            promoted["gold_source"] = "qwen_checked"
            promoted["review_status"] = "pass3_qwen_checked_gate_accepted"
            promoted["qwen_review_file"] = accepted_by_group[group["probe_group_id"]]["result_file"]
            groups.append(promoted)
        else:
            groups.append(group)
    write_jsonl(dst_diag / "probe_groups.jsonl", groups)

    for filename in PROBE_FILES:
        rows = []
        for probe in read_jsonl(src_diag / filename):
            if probe["probe_group_id"] in accepted_group_ids:
                promoted = dict(probe)
                promoted["gold_source"] = "qwen_checked"
                promoted["review_status"] = "pass3_qwen_checked_gate_accepted"
                rows.append(promoted)
            else:
                rows.append(probe)
        write_jsonl(dst_diag / filename, rows)

    for filename in ["hidden_gold.jsonl", "diagnostic_quality.jsonl"]:
        rows = []
        for row in read_jsonl(src_diag / filename):
            if row["probe_group_id"] in accepted_group_ids:
                promoted = dict(row)
                if "gold_source" in promoted:
                    promoted["gold_source"] = "qwen_checked"
                if "recommended_gold_source" in promoted:
                    promoted["recommended_gold_source"] = "qwen_checked"
                promoted["review_status"] = "pass3_qwen_checked_gate_accepted"
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
    shutil.copytree(src / "annotations" / "oracle_ledger", dst / "annotations" / "oracle_ledger")
    for dirname in ["docs", "scripts", "slurm", "tests"]:
        if (src / dirname).exists():
            shutil.copytree(src / dirname, dst / dirname, dirs_exist_ok=True)

    groups = read_jsonl(src / "annotations" / "diagnostics" / "probe_groups.jsonl")
    group_ids = {row["probe_group_id"] for row in groups}
    gate_rows = []
    accepted_group_ids: set[str] = set()
    for row in result_rows(args.qwen_results_root):
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        group = parsed.get("corrected_probe_group") if isinstance(parsed.get("corrected_probe_group"), dict) else {}
        group_id = group.get("probe_group_id") or (row.get("source_task", {}).get("probe_group", {}) or {}).get("probe_group_id")
        accepted, reasons = qwen_gate(row)
        if group_id not in group_ids:
            accepted = False
            reasons.append("group_not_in_input_pass")
        if accepted:
            accepted_group_ids.add(group_id)
        gate_rows.append(
            {
                "result_file": row["_result_file"],
                "review_task_id": row.get("review_task_id"),
                "probe_group_id": group_id,
                "gate_decision": "accept" if accepted else "reject",
                "gate_reasons": reasons,
                "qwen_decision": parsed.get("decision"),
                "qwen_gold_source_after_review": parsed.get("gold_source_after_review"),
                "needs_human_review": parsed.get("needs_human_review"),
            }
        )

    promote_rows(src, dst, accepted_group_ids, gate_rows)
    review_dir = dst / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.qwen_results_root, review_dir / "qwen3_omni_high_quality_results", dirs_exist_ok=True)
    write_jsonl(review_dir / "qwen_merge_gate.jsonl", gate_rows)
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
        "reject_reason_counts": dict(collections.Counter(reason for row in rejected for reason in row["gate_reasons"])),
        "status": "pass3_qwen_checked_gate_complete",
    }
    write_json(review_dir / "pass3_qwen_merge_summary.json", summary)
    (dst / "README.md").write_text(
        "# Omni Goose Decrypto Diagnostic Pass 3 Qwen-checked\n\n"
        "This pass applies a conservative merge gate over Qwen3-Omni high-quality review results.\n\n"
        f"- qwen_results: {summary['qwen_results']}\n"
        f"- accepted_groups: {summary['accepted_groups']}\n"
        f"- rejected_groups: {summary['rejected_groups']}\n\n"
        "Rows are promoted to `qwen_checked` only when they pass parse, decision, uncertainty, red-flag, and A-prompt leakage checks.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
