from __future__ import annotations

import json
from pathlib import Path

from tools.validate.validate_gooseomni_leaderboard_core import validate_leaderboard_core

PROBE_TYPES = [
    "A_pre_reveal_belief",
    "B_post_reveal_reconstruct_previous_belief",
    "C_other_agent_false_belief",
    "D_perspective_taking_prediction",
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt(group_id: str, probe_type: str) -> dict:
    suffix = probe_type[0]
    return {
        "probe_id": f"{group_id}_{suffix}",
        "probe_group_id": group_id,
        "probe_type": probe_type,
        "target_player": "Gemini",
        "cutoff_abs_sec": 100.0,
        "input_condition": "target_available_events",
        "prompt": f"Prompt for {group_id} {probe_type}",
        "expected_output_schema": {"answer": "string"},
        "acceptable_evidence_ids": ["claim_001"],
        "gold_source": "human_verified",
    }


def _trial_from_prompt(prompt: dict) -> dict:
    row = {key: value for key, value in prompt.items() if key != "probe_id"}
    row["trial_id"] = prompt["probe_id"]
    return row


def _gold_from_prompt(prompt: dict) -> dict:
    return {
        "trial_id": prompt["probe_id"],
        "probe_group_id": prompt["probe_group_id"],
        "gold_source": "human_verified",
        "acceptable_evidence_ids": ["claim_001"],
        "metrics": ["RC_weak", "FB_weak", "perspective_leakage"],
    }


def _hidden_gold_from_prompt(prompt: dict) -> dict:
    return {
        "trial_id": prompt["probe_id"],
        "probe_group_id": prompt["probe_group_id"],
        "gold_source": "human_verified",
        "acceptable_evidence_ids": ["claim_001"],
        "forbidden_event_ids": ["ge_hidden"],
        "hidden_gold": {"probe_group_id": prompt["probe_group_id"]},
    }


def _excluded_row(group_id: str) -> dict:
    return {
        "probe_group_id": group_id,
        "probe_id": f"{group_id}_D",
        "adjudication_status": "extended_only_excluded_from_leaderboard_core",
        "review_decision": "do_not_promote_D_to_leaderboard_core_without_new_behavior_evidence",
        "reasons": ["No independent later listener behavior was verified."],
        "missing_evidence_for_promotion": ["direct listener response"],
        "promotion_requirements": [
            "independent public behavior evidence after the source claim"
        ],
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    annotation_root = tmp_path / "pass001" / "annotations"
    benchmark_root = tmp_path / "pass001" / "benchmark" / "gooseomni_v1"
    core_root = benchmark_root / "leaderboard_core"
    full_prompts = [
        _prompt(group, probe_type)
        for group in ["pg_bg", "pg_ext"]
        for probe_type in PROBE_TYPES
    ]
    core_prompts = [
        row
        for row in full_prompts
        if row["probe_type"] != "D_perspective_taking_prediction"
        or row["probe_group_id"] == "pg_bg"
    ]
    core_trials = [_trial_from_prompt(row) for row in core_prompts]

    _write_jsonl(
        benchmark_root / "interactive_diagnostics" / "prompts.jsonl", full_prompts
    )
    _write_jsonl(
        annotation_root / "diagnostics" / "hidden_gold.jsonl",
        [
            {"probe_group_id": "pg_bg", "behavior_grounded_d_gold": True},
            {
                "probe_group_id": "pg_ext",
                "leaderboard_core_D_adjudication": {
                    "adjudication_status": "extended_only_excluded_from_leaderboard_core",
                    "review_decision": "do_not_promote_D_to_leaderboard_core_without_new_behavior_evidence",
                },
            },
        ],
    )
    _write_json(
        core_root / "summary.json",
        {
            "source_pass": str(tmp_path / "pass001"),
            "total_core_static_trials": len(core_trials),
        },
    )
    (core_root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (core_root / "README.md").write_text("leaderboard core\n", encoding="utf-8")
    _write_jsonl(
        core_root / "probe_groups.jsonl",
        [{"probe_group_id": "pg_bg"}, {"probe_group_id": "pg_ext"}],
    )
    _write_jsonl(core_root / "interactive_prompts.jsonl", core_prompts)
    _write_jsonl(core_root / "trials.jsonl", core_trials)
    _write_jsonl(
        core_root / "gold.jsonl", [_gold_from_prompt(row) for row in core_prompts]
    )
    _write_jsonl(
        core_root / "hidden_gold.jsonl",
        [_hidden_gold_from_prompt(row) for row in core_prompts],
    )
    _write_jsonl(core_root / "excluded_D_ungrounded.jsonl", [_excluded_row("pg_ext")])
    return annotation_root, benchmark_root


def test_leaderboard_core_validation_accepts_valid_split(tmp_path: Path) -> None:
    annotation_root, benchmark_root = _build_fixture(tmp_path)

    report = validate_leaderboard_core(annotation_root, benchmark_root)

    assert report["ok"]
    assert report["behavior_grounded_D_core"] == 1
    assert report["excluded_D_ungrounded"] == 1
    assert report["core_counts"]["trials.jsonl"]["D_perspective_taking_prediction"] == 1


def test_leaderboard_core_validation_rejects_ungrounded_d_in_core(
    tmp_path: Path,
) -> None:
    annotation_root, benchmark_root = _build_fixture(tmp_path)
    full_prompts = [
        json.loads(line)
        for line in (benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    ext_d = next(
        row
        for row in full_prompts
        if row["probe_group_id"] == "pg_ext"
        and row["probe_type"] == "D_perspective_taking_prediction"
    )
    core_prompts_path = (
        benchmark_root / "leaderboard_core" / "interactive_prompts.jsonl"
    )
    core_trials_path = benchmark_root / "leaderboard_core" / "trials.jsonl"
    with core_prompts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ext_d, ensure_ascii=False) + "\n")
    with core_trials_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_trial_from_prompt(ext_d), ensure_ascii=False) + "\n")

    report = validate_leaderboard_core(annotation_root, benchmark_root)

    codes = {issue["code"] for issue in report["issues"]}
    assert not report["ok"]
    assert "core_prompt_ids_mismatch" in codes
    assert "ungrounded_d_in_core_prompts" in codes
    assert "ungrounded_d_in_core_trials" in codes


def test_leaderboard_core_validation_rejects_missing_adjudication(
    tmp_path: Path,
) -> None:
    annotation_root, benchmark_root = _build_fixture(tmp_path)
    _write_jsonl(
        annotation_root / "diagnostics" / "hidden_gold.jsonl",
        [{"probe_group_id": "pg_bg", "behavior_grounded_d_gold": True}],
    )

    report = validate_leaderboard_core(annotation_root, benchmark_root)

    codes = {issue["code"] for issue in report["issues"]}
    assert not report["ok"]
    assert "diagnostic_adjudication_coverage_mismatch" in codes


def test_leaderboard_core_validation_rejects_public_leak(tmp_path: Path) -> None:
    annotation_root, benchmark_root = _build_fixture(tmp_path)
    trials_path = benchmark_root / "leaderboard_core" / "trials.jsonl"
    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["hidden_gold"] = {"answer": "leak"}
    _write_jsonl(trials_path, rows)

    report = validate_leaderboard_core(annotation_root, benchmark_root)

    codes = {issue["code"] for issue in report["issues"]}
    assert not report["ok"]
    assert "core_public_leak_pattern" in codes
