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
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CORE_REQUIRED_FILES = [
    "README.md",
    "summary.json",
    "probe_groups.jsonl",
    "interactive_prompts.jsonl",
    "trials.jsonl",
    "gold.jsonl",
    "hidden_gold.jsonl",
    "excluded_D_ungrounded.jsonl",
]

PUBLIC_LEAK_PATTERNS = [
    "hidden_gold",
    "forbidden_event_ids",
    "behavior_grounded",
    "leaderboard_core_D_adjudication",
    "adjudication_status",
    "review_decision",
    "observed_vote_ui_outcomes",
    "verbal_response_observations",
    "behavior_evidence_review_item_ids",
    "scope_limited_public",
    "behavior_modality",
    "behavior_outcome_gold_status",
    "human_review_record_id",
    "human_reviewer",
    "review_status",
    "prompt_policy",
    "qwen_review_file",
    "qwen",
    "Qwen",
    "codex_",
    "pass11",
    "pass241",
    "pass242",
]

PROBE_TYPES_ABC = {
    "A_pre_reveal_belief",
    "B_post_reveal_reconstruct_previous_belief",
    "C_other_agent_false_belief",
}
D_PROBE_TYPE = "D_perspective_taking_prediction"
PUBLIC_PROMPT_ALLOWED_KEYS = {
    "acceptable_evidence_ids",
    "cutoff_abs_sec",
    "expected_output_schema",
    "game_id",
    "gold_source",
    "input_condition",
    "probe_group_id",
    "probe_id",
    "probe_type",
    "prompt",
    "target_player",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    value = row.get("trial_id") or row.get("probe_id")
    return str(value or "")


def duplicate_ids(rows: list[dict[str, Any]], id_field_name: str) -> list[str]:
    counts = Counter(str(row.get(id_field_name) or row_id(row)) for row in rows)
    return sorted(item for item, count in counts.items() if item and count > 1)


def probe_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("probe_type", "unknown")) for row in rows).items()))


def public_leak_hits(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    hits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern in PUBLIC_LEAK_PATTERNS:
            if pattern in line:
                hits.append({"path": path.as_posix(), "line": line_no, "pattern": pattern})
    return hits


def expected_core_prompt_ids(full_prompts: list[dict[str, Any]], behavior_grounded_d_groups: set[str]) -> set[str]:
    expected = set()
    for prompt in full_prompts:
        probe_type = prompt.get("probe_type")
        probe_id = str(prompt.get("probe_id") or "")
        group_id = str(prompt.get("probe_group_id") or "")
        if not probe_id:
            continue
        if probe_type in PROBE_TYPES_ABC:
            expected.add(probe_id)
        elif probe_type == D_PROBE_TYPE and group_id in behavior_grounded_d_groups:
            expected.add(probe_id)
    return expected


def d_groups_from_prompts(prompts: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("probe_group_id"))
        for row in prompts
        if row.get("probe_type") == D_PROBE_TYPE and row.get("probe_group_id")
    }


def behavior_grounded_groups(diagnostic_hidden_gold: list[dict[str, Any]], full_d_groups: set[str]) -> set[str]:
    return {
        str(row.get("probe_group_id"))
        for row in diagnostic_hidden_gold
        if row.get("behavior_grounded_d_gold") and row.get("probe_group_id") in full_d_groups
    }


def adjudicated_extended_groups(diagnostic_hidden_gold: list[dict[str, Any]]) -> set[str]:
    groups = set()
    for row in diagnostic_hidden_gold:
        adjudication = row.get("leaderboard_core_D_adjudication")
        if adjudication and row.get("probe_group_id"):
            groups.add(str(row["probe_group_id"]))
    return groups


def validate_leaderboard_core(annotation_root: Path, benchmark_root: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    core_root = benchmark_root / "leaderboard_core"

    for rel in CORE_REQUIRED_FILES:
        path = core_root / rel
        if not path.exists():
            issues.append({"code": "missing_core_file", "path": path.as_posix()})

    full_prompts = read_jsonl(benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
    diagnostic_hidden_gold = read_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl")
    core_prompts = read_jsonl(core_root / "interactive_prompts.jsonl")
    core_trials = read_jsonl(core_root / "trials.jsonl")
    core_gold = read_jsonl(core_root / "gold.jsonl")
    core_hidden_gold = read_jsonl(core_root / "hidden_gold.jsonl")
    excluded = read_jsonl(core_root / "excluded_D_ungrounded.jsonl")

    full_d_groups = d_groups_from_prompts(full_prompts)
    grounded_d_groups = behavior_grounded_groups(diagnostic_hidden_gold, full_d_groups)
    expected_excluded_d_groups = full_d_groups - grounded_d_groups
    expected_prompt_ids = expected_core_prompt_ids(full_prompts, grounded_d_groups)
    core_prompt_ids = {str(row.get("probe_id")) for row in core_prompts if row.get("probe_id")}
    core_trial_ids = {str(row.get("trial_id")) for row in core_trials if row.get("trial_id")}
    core_gold_ids = {str(row.get("trial_id")) for row in core_gold if row.get("trial_id")}
    core_hidden_gold_ids = {str(row.get("trial_id")) for row in core_hidden_gold if row.get("trial_id")}
    excluded_d_groups = {str(row.get("probe_group_id")) for row in excluded if row.get("probe_group_id")}
    diagnostic_adjudicated_groups = adjudicated_extended_groups(diagnostic_hidden_gold)

    if core_prompt_ids != expected_prompt_ids:
        issues.append(
            {
                "code": "core_prompt_ids_mismatch",
                "missing": sorted(expected_prompt_ids - core_prompt_ids)[:20],
                "unexpected": sorted(core_prompt_ids - expected_prompt_ids)[:20],
                "missing_count": len(expected_prompt_ids - core_prompt_ids),
                "unexpected_count": len(core_prompt_ids - expected_prompt_ids),
            }
        )

    expected_trial_ids = expected_prompt_ids
    for name, ids in [
        ("trials.jsonl", core_trial_ids),
        ("gold.jsonl", core_gold_ids),
        ("hidden_gold.jsonl", core_hidden_gold_ids),
    ]:
        if ids != expected_trial_ids:
            issues.append(
                {
                    "code": "core_trial_set_mismatch",
                    "file": name,
                    "missing": sorted(expected_trial_ids - ids)[:20],
                    "unexpected": sorted(ids - expected_trial_ids)[:20],
                    "missing_count": len(expected_trial_ids - ids),
                    "unexpected_count": len(ids - expected_trial_ids),
                }
            )

    for name, rows, id_field in [
        ("interactive_prompts.jsonl", core_prompts, "probe_id"),
        ("trials.jsonl", core_trials, "trial_id"),
        ("gold.jsonl", core_gold, "trial_id"),
        ("hidden_gold.jsonl", core_hidden_gold, "trial_id"),
    ]:
        duplicates = duplicate_ids(rows, id_field)
        if duplicates:
            issues.append({"code": "duplicate_core_ids", "file": name, "ids": duplicates[:20], "count": len(duplicates)})

    if excluded_d_groups != expected_excluded_d_groups:
        issues.append(
            {
                "code": "excluded_d_groups_mismatch",
                "missing": sorted(expected_excluded_d_groups - excluded_d_groups)[:20],
                "unexpected": sorted(excluded_d_groups - expected_excluded_d_groups)[:20],
                "missing_count": len(expected_excluded_d_groups - excluded_d_groups),
                "unexpected_count": len(excluded_d_groups - expected_excluded_d_groups),
            }
        )

    if diagnostic_adjudicated_groups != expected_excluded_d_groups:
        issues.append(
            {
                "code": "diagnostic_adjudication_coverage_mismatch",
                "missing": sorted(expected_excluded_d_groups - diagnostic_adjudicated_groups)[:20],
                "unexpected": sorted(diagnostic_adjudicated_groups - expected_excluded_d_groups)[:20],
                "missing_count": len(expected_excluded_d_groups - diagnostic_adjudicated_groups),
                "unexpected_count": len(diagnostic_adjudicated_groups - expected_excluded_d_groups),
            }
        )

    required_excluded_fields = {
        "review_decision",
        "reasons",
        "missing_evidence_for_promotion",
        "promotion_requirements",
    }
    for row in excluded:
        missing = sorted(field for field in required_excluded_fields if not row.get(field))
        if missing:
            issues.append(
                {
                    "code": "excluded_adjudication_missing_fields",
                    "probe_group_id": row.get("probe_group_id"),
                    "missing": missing,
                }
            )

    for row in core_prompts:
        if row.get("probe_type") == D_PROBE_TYPE and row.get("probe_group_id") not in grounded_d_groups:
            issues.append(
                {
                    "code": "ungrounded_d_in_core_prompts",
                    "probe_id": row.get("probe_id"),
                    "probe_group_id": row.get("probe_group_id"),
                }
            )

    for name, rows in [
        ("interactive_diagnostics/prompts.jsonl", full_prompts),
        ("leaderboard_core/interactive_prompts.jsonl", core_prompts),
    ]:
        for row in rows:
            extra_keys = sorted(set(row) - PUBLIC_PROMPT_ALLOWED_KEYS)
            if extra_keys:
                issues.append(
                    {
                        "code": "public_prompt_unexpected_keys",
                        "file": name,
                        "probe_id": row.get("probe_id"),
                        "extra_keys": extra_keys,
                    }
                )
                break

    for row in core_trials:
        if row.get("probe_type") == D_PROBE_TYPE and row.get("probe_group_id") not in grounded_d_groups:
            issues.append(
                {
                    "code": "ungrounded_d_in_core_trials",
                    "trial_id": row.get("trial_id"),
                    "probe_group_id": row.get("probe_group_id"),
                }
            )

    leak_hits = []
    for path in [
        benchmark_root / "interactive_diagnostics" / "prompts.jsonl",
        benchmark_root / "static_trials" / "trials.jsonl",
        core_root / "interactive_prompts.jsonl",
        core_root / "trials.jsonl",
    ]:
        leak_hits.extend(public_leak_hits(path))
    if leak_hits:
        issues.append({"code": "core_public_leak_pattern", "hits": leak_hits[:50], "hit_count": len(leak_hits)})

    summary: dict[str, Any] = {}
    summary_path = core_root / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("source_pass"):
            inferred_pass = str(benchmark_root).split("/benchmark/gooseomni_v1")[0]
            if inferred_pass and summary.get("source_pass") != inferred_pass:
                issues.append(
                    {
                        "code": "summary_source_pass_mismatch",
                        "summary_source_pass": summary.get("source_pass"),
                        "inferred_pass": inferred_pass,
                    }
                )

    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "core_counts": {
            "interactive_prompts.jsonl": probe_counts(core_prompts),
            "trials.jsonl": probe_counts(core_trials),
        },
        "expected_core_prompt_count": len(expected_prompt_ids),
        "trial_count": len(core_trials),
        "full_D_total": len(full_d_groups),
        "behavior_grounded_D_core": len(grounded_d_groups),
        "excluded_D_ungrounded": len(expected_excluded_d_groups),
        "core_excluded_adjudicated": len(excluded_d_groups),
        "diagnostic_adjudicated": len(diagnostic_adjudicated_groups),
        "summary_source_pass_matches_current": not any(issue["code"] == "summary_source_pass_mismatch" for issue in issues),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GooseOmni leaderboard-core split.")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_leaderboard_core(args.annotation_root, args.benchmark_root)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
