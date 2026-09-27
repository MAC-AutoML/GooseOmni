from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    generate_probes_for_group,
    load_ledger,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build private-witness candidates from already human-reviewed anchors."
    )
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--anchor-event-id", action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def event_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["world_event_id"]: event for event in events}


def make_group(idx: int, event: dict[str, Any], witness: str) -> dict[str, Any]:
    event_id = event["world_event_id"]
    return {
        "probe_group_id": f"g001_pwit_{idx:06d}_{witness}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": float(event["abs_end_sec"]) + 3.0,
        "target_player": witness,
        "query_variable": {
            "type": "private_witness",
            "description": f"Whether {witness} directly witnessed and can reason about others not seeing: {event.get('description', '')}",
        },
        "anchor_event_ids": [event_id],
        "related_claim_ids": [],
        "hidden_event_ids_for_target": [],
        "available_evidence_ids_for_target": [event_id],
        "selection_reason": (
            f"{witness} is the direct source POV for {event_id}; prior human reviews verified other target perspectives "
            "did not have the same anchor evidence."
        ),
        "diagnostic_families": [
            "false_belief",
            "representational_change",
            "perspective_taking",
            "private_witness",
        ],
        "template": "private_witness",
        "quality": {
            "visibility_confidence": 0.9,
            "claim_truth_confidence": 0.6,
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
    events = event_by_id(ledger["world_events"])
    groups: list[dict[str, Any]] = []
    for event_id in args.anchor_event_id:
        event = events.get(event_id)
        if not event:
            raise SystemExit(f"missing event: {event_id}")
        source_povs = event.get("source_povs", [])
        if len(source_povs) != 1:
            raise SystemExit(
                f"event must have exactly one source POV for private_witness: {event_id} {source_povs}"
            )
        groups.append(make_group(len(groups) + 1, event, source_povs[0]))

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
