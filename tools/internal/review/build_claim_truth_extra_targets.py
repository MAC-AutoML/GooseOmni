from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    PLAYERS,
    generate_probes_for_group,
    load_ledger,
    write_jsonl,
)


GOOD_CLAIM_TYPES = {"location", "defense", "accusation", "sighting"}
CONTRADICTED_ANCHOR_EVENT_TYPES = {
    "movement",
    "player_movement",
    "interaction",
    "player_interaction",
    "combat",
    "death",
    "player_death",
}
EXCLUDED_EVENT_TYPES = {
    "discussion",
    "meeting",
    "phase_transition",
    "gameplay_phase_transition",
    "scene_transition",
    "transition",
    "voting",
    "vote",
    "vote_result",
    "voting_result",
    "result",
    "game_result",
    "game_over",
    "game_start",
    "gameplay_start",
    "task",
    "task_completion",
    "unknown",
}
BAD_ANCHOR_DESCRIPTION_TOKENS = {
    "讨论",
    "投票",
    "会议",
    "界面",
    "切换",
    "结果",
    "观战",
    "任务",
    "寻找目标",
    "声称",
    "没有看到",
    "没看到",
    "meeting",
    "voting",
    "vote",
    "task",
}
MIN_CLAIM_CONTENT_CHARS = 8
MIN_POST_MEETING_GAMEPLAY_LOCAL_SEC = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manually specified claim-truth / alibi candidates.")
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--spec", action="append", required=True, help="event_id:claim_id:target[,target...]")
    parser.add_argument("--truth-status", default="contradicted", choices=["supported", "contradicted", "unverified", "ambiguous"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def parse_spec(value: str) -> tuple[str, str, list[str]]:
    event_id, claim_id, targets = value.split(":", 2)
    target_list = [target for target in targets.split(",") if target]
    invalid = [target for target in target_list if target not in PLAYERS]
    if invalid:
        raise SystemExit(f"invalid targets in {value}: {invalid}")
    return event_id, claim_id, target_list


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(edge.get("event_id")), str(edge.get("player_id"))): edge for edge in edges}


def claim_content(claim: dict[str, Any]) -> str:
    return str(claim.get("content") or claim.get("normalized_content") or "").strip()


def event_description(event: dict[str, Any]) -> str:
    return str(event.get("description") or "").strip()


def has_bad_anchor_description(event: dict[str, Any]) -> bool:
    description = event_description(event).lower()
    return any(token in description for token in BAD_ANCHOR_DESCRIPTION_TOKENS)


def phase_local_start_sec(event: dict[str, Any]) -> float | None:
    phase_ids = event.get("source_segment_ids") or []
    if not phase_ids:
        return None
    phase_id = str(phase_ids[0])
    parts = phase_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return float(event.get("abs_start_sec", 0.0) or 0.0) - float(int(parts[1]))


def validate_claim_truth_spec(
    event: dict[str, Any],
    claim: dict[str, Any],
    target: str,
    truth_status: str,
    visibility_edges: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    event_id = str(event.get("world_event_id"))
    claim_id = str(claim.get("claim_id"))
    event_type = str(event.get("event_type") or "")
    speaker = str(claim.get("speaker") or "")
    content = claim_content(claim)

    if event.get("phase_type") != "gameplay":
        errors.append("anchor_event_not_gameplay")
    local_start = phase_local_start_sec(event)
    if event.get("phase_type") == "gameplay" and local_start is not None and local_start < MIN_POST_MEETING_GAMEPLAY_LOCAL_SEC:
        errors.append(f"anchor_event_too_close_to_gameplay_phase_start:{local_start:.1f}s")
    if event_type in EXCLUDED_EVENT_TYPES:
        errors.append(f"anchor_event_type_not_clean_counterevidence:{event_type}")
    if truth_status == "contradicted" and event_type not in CONTRADICTED_ANCHOR_EVENT_TYPES:
        errors.append(f"contradicted_anchor_event_type_too_weak:{event_type}")
    if has_bad_anchor_description(event):
        errors.append("anchor_description_has_ui_or_weak_semantics")
    if event.get("needs_human_review"):
        errors.append("anchor_event_already_needs_human_review")
    if float(event.get("certainty", 0.0) or 0.0) < 0.75:
        errors.append("anchor_event_low_certainty")
    if not event.get("source_povs"):
        errors.append("anchor_event_missing_source_pov")

    if speaker not in PLAYERS:
        errors.append("claim_speaker_not_player")
    if claim.get("claim_type") not in GOOD_CLAIM_TYPES:
        errors.append(f"claim_type_not_supported:{claim.get('claim_type')}")
    if len(content) < MIN_CLAIM_CONTENT_CHARS:
        errors.append("claim_content_too_short")
    if float(claim.get("certainty", 0.0) or 0.0) < 0.7:
        errors.append("claim_low_certainty")
    if target not in claim.get("heard_by", []):
        errors.append("target_did_not_hear_claim")
    if float(claim["abs_end_sec"]) < float(event["abs_end_sec"]):
        errors.append("claim_must_end_after_anchor_event")

    visibility = visibility_edges.get((event_id, target), {}).get("visibility")
    if visibility != "not_visible":
        errors.append(f"target_anchor_visibility_not_hidden:{visibility or 'missing'}")

    if truth_status == "contradicted":
        actors = {str(actor) for actor in event.get("actors", [])}
        description = event_description(event)
        if speaker not in actors and speaker not in description:
            errors.append(f"contradicted_claim_speaker_not_anchor_actor:{speaker}")
        if not str(event.get("location") or "").strip() and not description:
            errors.append("contradicted_anchor_missing_location_or_description")

    # Keep the error easy to grep when a manual spec is rejected.
    if errors:
        errors.insert(0, f"invalid_claim_truth_spec:{event_id}:{claim_id}:{target}")
    return errors


def make_group(idx: int, event: dict[str, Any], claim: dict[str, Any], target: str, truth_status: str) -> dict[str, Any]:
    event_id = event["world_event_id"]
    claim_id = claim["claim_id"]
    return {
        "probe_group_id": f"g001_pclaim_{idx:06d}_{target}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": float(claim["abs_end_sec"]) + 2.0,
        "target_player": target,
        "query_variable": {
            "type": "claim_truth_vs_claim_awareness",
            "description": f"Can {target} distinguish global truth of {claim_id} from local awareness, anchored by {event_id}?",
        },
        "anchor_event_ids": [event_id],
        "related_claim_ids": [claim_id],
        "hidden_event_ids_for_target": [event_id],
        "available_evidence_ids_for_target": [claim_id],
        "selection_reason": (
            f"Manual claim-truth candidate: claim {claim_id} is treated as {truth_status} by candidate construction; "
            f"target {target} heard the claim and must not use hidden anchor {event_id} as local evidence unless reviewed."
        ),
        "diagnostic_families": ["false_belief", "representational_change", "claim_verification", "perspective_taking"],
        "template": "contradicted_alibi" if truth_status == "contradicted" else "claim_backed_hidden_event",
        "quality": {
            "visibility_confidence": 0.75,
            "claim_truth_confidence": 0.75,
            "timestamp_confidence": min(float(event.get("certainty", 0.85) or 0.85), float(claim.get("certainty", 0.85) or 0.85)),
            "needs_human_review": True,
            "paper_gold_candidate": True,
        },
        "needs_human_review": True,
        "gold_source": "qwen_checked",
        "_manual_truth_status": truth_status,
    }


def main() -> None:
    args = parse_args()
    if args.output_pass_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output_pass_root}")
        shutil.rmtree(args.output_pass_root)
    (args.output_pass_root / "annotations").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        args.input_pass_root / "annotations" / "oracle_ledger",
        args.output_pass_root / "annotations" / "oracle_ledger",
    )

    ledger = load_ledger(args.output_pass_root / "annotations")
    events = by_id(ledger["world_events"], "world_event_id")
    claims = by_id(ledger["claims"], "claim_id")
    visibility_edges = edge_lookup(ledger["visibility_edges"])
    groups: list[dict[str, Any]] = []
    for spec in args.spec:
        event_id, claim_id, targets = parse_spec(spec)
        event = events.get(event_id)
        claim = claims.get(claim_id)
        if not event:
            raise SystemExit(f"missing event: {event_id}")
        if not claim:
            raise SystemExit(f"missing claim: {claim_id}")
        for target in targets:
            errors = validate_claim_truth_spec(event, claim, target, args.truth_status, visibility_edges)
            if errors:
                raise SystemExit("; ".join(errors))
            groups.append(make_group(len(groups) + 1, event, claim, target, args.truth_status))

    diagnostics = args.output_pass_root / "annotations" / "diagnostics"
    probes_by_type: dict[str, list[dict[str, Any]]] = {
        "A_pre_reveal_belief": [],
        "B_post_reveal_reconstruct_previous_belief": [],
        "C_other_agent_false_belief": [],
        "D_perspective_taking_prediction": [],
    }
    hidden_gold: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    public_groups = []
    for group in groups:
        truth_status = group.pop("_manual_truth_status")
        probes, gold, quality = generate_probes_for_group(group, ledger)
        gold["claim_truth_global"] = truth_status
        gold["gold_source"] = "qwen_checked"
        gold["paper_gold_candidate"] = True
        quality["recommended_gold_source"] = "qwen_checked"
        quality["needs_human_review"] = True
        quality["paper_gold_candidate"] = True
        for probe in probes:
            probe["gold_source"] = "qwen_checked"
            probes_by_type[probe["probe_type"]].append(probe)
        public_groups.append(group)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    write_jsonl(diagnostics / "probe_groups.jsonl", public_groups)
    write_jsonl(diagnostics / "probes_A_pre_reveal.jsonl", probes_by_type["A_pre_reveal_belief"])
    write_jsonl(diagnostics / "probes_B_reconstruct.jsonl", probes_by_type["B_post_reveal_reconstruct_previous_belief"])
    write_jsonl(diagnostics / "probes_C_false_belief.jsonl", probes_by_type["C_other_agent_false_belief"])
    write_jsonl(diagnostics / "probes_D_perspective_taking.jsonl", probes_by_type["D_perspective_taking_prediction"])
    write_jsonl(diagnostics / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diagnostics / "diagnostic_quality.jsonl", quality_rows)
    print(
        json.dumps(
            {
                "ok": True,
                "output_pass_root": args.output_pass_root.as_posix(),
                "probe_groups": len(public_groups),
                "hidden_gold": len(hidden_gold),
                "A": len(probes_by_type["A_pre_reveal_belief"]),
                "B": len(probes_by_type["B_post_reveal_reconstruct_previous_belief"]),
                "C": len(probes_by_type["C_other_agent_false_belief"]),
                "D": len(probes_by_type["D_perspective_taking_prediction"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
