from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    PLAYERS,
    generate_probes_for_group,
    load_ledger,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vote-influence target expansions from reviewed anchor/claim pairs."
    )
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="event_id:claim_id:target[,target...]",
    )
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


def make_group(
    idx: int, event: dict[str, Any], claim: dict[str, Any], target: str
) -> dict[str, Any]:
    event_id = event["world_event_id"]
    claim_id = claim["claim_id"]
    cutoff = float(claim["abs_end_sec"]) + 2.0
    return {
        "probe_group_id": f"g001_pvote_{idx:06d}_{target}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": cutoff,
        "target_player": target,
        "query_variable": {
            "type": "trust_update",
            "description": f"After {claim_id}, how should {claim.get('speaker')} expect {target} to interpret: {event.get('description', '')}",
        },
        "anchor_event_ids": [event_id],
        "related_claim_ids": [claim_id],
        "hidden_event_ids_for_target": [event_id],
        "available_evidence_ids_for_target": [claim_id],
        "selection_reason": (
            f"Reviewed vote-influence expansion: {event_id} is source-visible but hidden from {target}; "
            f"{target} heard strategic claim {claim_id}, so the probe tests speaker-to-listener perspective taking."
        ),
        "diagnostic_families": [
            "false_belief",
            "representational_change",
            "claim_verification",
            "perspective_taking",
            "strategy_communication",
        ],
        "template": "vote_influence",
        "quality": {
            "visibility_confidence": 0.85,
            "claim_truth_confidence": float(claim.get("certainty", 0.85) or 0.85),
            "timestamp_confidence": float(event.get("certainty", 0.85) or 0.85),
            "needs_human_review": True,
            "paper_gold_candidate": True,
        },
        "needs_human_review": True,
        "gold_source": "qwen_checked",
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
    groups: list[dict[str, Any]] = []
    for spec in args.spec:
        event_id, claim_id, targets = parse_spec(spec)
        event = events.get(event_id)
        claim = claims.get(claim_id)
        if not event:
            raise SystemExit(f"missing event: {event_id}")
        if not claim:
            raise SystemExit(f"missing claim: {claim_id}")
        speaker = claim.get("speaker")
        if speaker not in PLAYERS:
            raise SystemExit(f"claim has invalid speaker: {claim_id} speaker={speaker}")
        for target in targets:
            if target == speaker:
                raise SystemExit(f"target cannot be speaker for vote influence: {spec}")
            if target not in claim.get("heard_by", []):
                raise SystemExit(
                    f"target did not hear claim: {target} claim={claim_id}"
                )
            groups.append(make_group(len(groups) + 1, event, claim, target))

    diagnostics = args.output_pass_root / "annotations" / "diagnostics"
    probes_by_type: dict[str, list[dict[str, Any]]] = {
        "A_pre_reveal_belief": [],
        "B_post_reveal_reconstruct_previous_belief": [],
        "C_other_agent_false_belief": [],
        "D_perspective_taking_prediction": [],
    }
    hidden_gold: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for group in groups:
        probes, gold, quality = generate_probes_for_group(group, ledger)
        gold["gold_source"] = "qwen_checked"
        gold["paper_gold_candidate"] = True
        quality["recommended_gold_source"] = "qwen_checked"
        quality["needs_human_review"] = True
        quality["paper_gold_candidate"] = True
        for probe in probes:
            probe["gold_source"] = "qwen_checked"
            probes_by_type[probe["probe_type"]].append(probe)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    write_jsonl(diagnostics / "probe_groups.jsonl", groups)
    write_jsonl(
        diagnostics / "probes_A_pre_reveal.jsonl", probes_by_type["A_pre_reveal_belief"]
    )
    write_jsonl(
        diagnostics / "probes_B_reconstruct.jsonl",
        probes_by_type["B_post_reveal_reconstruct_previous_belief"],
    )
    write_jsonl(
        diagnostics / "probes_C_false_belief.jsonl",
        probes_by_type["C_other_agent_false_belief"],
    )
    write_jsonl(
        diagnostics / "probes_D_perspective_taking.jsonl",
        probes_by_type["D_perspective_taking_prediction"],
    )
    write_jsonl(diagnostics / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diagnostics / "diagnostic_quality.jsonl", quality_rows)
    print(
        json.dumps(
            {
                "ok": True,
                "output_pass_root": args.output_pass_root.as_posix(),
                "probe_groups": len(groups),
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
