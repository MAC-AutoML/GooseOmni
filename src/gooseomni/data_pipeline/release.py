from __future__ import annotations

from pathlib import Path
from typing import Any

from gooseomni.benchmark.agentic_midgame import build_agentic_midgame_prediction

from .provenance import write_json
from .release_pipeline import (
    _model_verified,
    _repair_raw_video,
    build_manifest,
    read_jsonl,
    write_jsonl,
)


def _public_group(value: dict[str, Any]) -> dict[str, Any]:
    forbidden = {
        "anchor_event_ids",
        "hidden_event_ids_for_target",
        "forbidden_event_ids",
        "hidden_gold",
    }
    return {
        key: _model_verified(item)
        for key, item in value.items()
        if key not in forbidden
    }


def _decision_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["trial_id"]): row for row in read_jsonl(path)}


def _reviewed_rows(
    trials: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    hidden: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    public_by_id = {str(row["trial_id"]): row for row in trials}
    for trial_id, decision in decisions.items():
        if decision.get("decision") == "reject":
            continue
        public = public_by_id.get(trial_id)
        final = decision.get("final_value")
        if decision.get("decision") == "accept" and final is None:
            if public is None:
                raise ValueError(f"accepted decision has no candidate: {trial_id}")
            final = {
                "public_trial": public,
                "gold": gold[trial_id],
                "hidden_gold": hidden[trial_id],
            }
        if not isinstance(final, dict):
            raise ValueError(f"review decision has no final object: {trial_id}")
        final_public = final.get("public_trial", final)
        final_gold = final.get("gold", gold.get(trial_id))
        final_hidden = final.get("hidden_gold", hidden.get(trial_id))
        if not all(
            isinstance(row, dict) for row in (final_public, final_gold, final_hidden)
        ):
            raise ValueError(f"review decision has invalid row objects: {trial_id}")
        if any(
            str(row.get("trial_id")) != trial_id
            for row in (final_public, final_gold, final_hidden)
        ):
            raise ValueError(f"review decision changed trial_id: {trial_id}")
        public_rows.append(_model_verified(final_public))
        gold_rows.append(_model_verified(final_gold))
        hidden_rows.append(_model_verified(final_hidden))
    return public_rows, gold_rows, hidden_rows


def assemble_release(
    target: Path,
    candidate_root: Path,
    annotation_root: Path,
    dataset_root: Path,
    decisions_path: Path,
    agentic_limit: int = 160,
) -> dict[str, Any]:
    """Assemble a new benchmark release from reviewed, run-local candidates."""
    if target.exists():
        raise FileExistsError(f"refusing to overwrite candidate benchmark: {target}")
    decisions = _decision_map(decisions_path)
    old_static = candidate_root / "static_trials"
    trials = read_jsonl(old_static / "trials.jsonl")
    candidate_gold = {
        str(row["trial_id"]): row for row in read_jsonl(old_static / "gold.jsonl")
    }
    candidate_hidden = {
        str(row["trial_id"]): row
        for row in read_jsonl(old_static / "hidden_gold.jsonl")
    }
    public_rows, gold, hidden = _reviewed_rows(
        trials, candidate_gold, candidate_hidden, decisions
    )
    group_ids = {str(row["probe_group_id"]) for row in public_rows}
    cached_groups = {
        str(item["probe_group_id"]): _public_group(item)
        for group_list in (
            decision.get("final_value", {}).get("probe_groups", [])
            for decision in decisions.values()
            if isinstance(decision.get("final_value"), dict)
        )
        for item in group_list
        if isinstance(item, dict) and item.get("probe_group_id")
    }
    fresh_groups = [
        _public_group(row)
        for row in read_jsonl(
            candidate_root / "interactive_diagnostics/probe_groups.jsonl"
        )
        if str(row.get("probe_group_id")) in group_ids
    ]
    fresh_by_id = {str(row["probe_group_id"]): row for row in fresh_groups}
    groups = [
        cached_groups.get(group_id, fresh_by_id[group_id])
        if group_id in fresh_by_id
        else cached_groups[group_id]
        for group_id in sorted(group_ids)
    ]

    write_jsonl(target / "public/leaderboard_core/trials.jsonl", public_rows)
    write_jsonl(target / "public/leaderboard_core/probe_groups_public.jsonl", groups)
    write_jsonl(target / "private/leaderboard_core/gold.jsonl", gold)
    write_jsonl(target / "private/leaderboard_core/hidden_gold.jsonl", hidden)

    raw_rows = [dict(row) for row in public_rows[:48]]
    write_jsonl(target / "public/raw_video_smoke/raw_video_smoke.jsonl", raw_rows)
    raw_summary = _repair_raw_video(
        target, dataset_root, dataset_root / "segments.json"
    )
    agentic = build_agentic_midgame_prediction(
        annotation_root, target, limit=agentic_limit
    )
    for path in sorted(target.rglob("*.jsonl")):
        rows = read_jsonl(path)
        converted = [_model_verified(row) for row in rows]
        if converted != rows:
            write_jsonl(path, converted)
    (target / "README.md").write_text(
        "# GooseOmni v2\n\nFully automated Qwen3-Omni perception and Codex "
        "adjudication release. Gold labels are `model_verified`.\n",
        encoding="utf-8",
    )
    manifest = build_manifest(target, annotation_root)
    manifest["review_models"] = sorted(
        {
            str(row.get("review_model"))
            for row in decisions.values()
            if row.get("review_model")
        }
        | {"qwen3_omni"}
    )
    write_json(target / "manifest.json", manifest)
    return {
        "accepted_trials": len(public_rows),
        "quarantined_trials": len(trials) - len(public_rows),
        "probe_groups": len(groups),
        **raw_summary,
        "agentic": agentic,
    }
