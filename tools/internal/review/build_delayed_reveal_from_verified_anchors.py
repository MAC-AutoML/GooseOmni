from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    generate_probes_for_group,
    load_ledger,
    write_jsonl,
)


MIN_REVEAL_DELAY_SEC = 30.0
MIN_PHASE_LOCAL_SEC = 15.0
REVEAL_KEYWORDS = {"死", "死亡", "尸体", "击杀", "杀", "刀", "倒地", "血", "kill", "killed", "death", "body"}
REVEAL_CLAIM_TYPES = {"accusation", "location", "sighting", "defense"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build delayed-public-reveal candidates from human-verified hidden anchors.")
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--spec", action="append", required=True, help="event_id:target:claim_id[,claim_id...]")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def parse_spec(value: str) -> tuple[str, str, list[str]]:
    event_id, target, claims = value.split(":", 2)
    return event_id, target, [claim_id for claim_id in claims.split(",") if claim_id]


def text(row: dict[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def claim_text(claim: dict[str, Any]) -> str:
    return f"{text(claim, 'content')} {text(claim, 'normalized_content')}".strip()


def event_text(event: dict[str, Any]) -> str:
    return f"{text(event, 'description')} {text(event, 'location')}".strip()


def phase_local_start_sec(row: dict[str, Any]) -> float | None:
    phase_ids = row.get("source_segment_ids") or []
    if not phase_ids:
        return None
    phase_id = str(phase_ids[0])
    parts = phase_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return float(row.get("abs_start_sec", 0.0) or 0.0) - float(int(parts[1]))


def claim_semantically_matches_event(event: dict[str, Any], claim: dict[str, Any]) -> bool:
    content = claim_text(claim)
    speaker = str(claim.get("speaker") or "")
    event_blob = event_text(event)
    if speaker and speaker in event_blob and any(token in event_blob for token in [f"{speaker} 被", f"{speaker}的角色被", f"{speaker} 的角色在"]) and any(
        token in content for token in ["我杀", "我刀", "我击杀", "一刀下去"]
    ):
        return False
    if not any(keyword in content.lower() for keyword in REVEAL_KEYWORDS):
        return False
    event_names = {
        str(value)
        for key in ("actors", "patients", "source_povs")
        for value in event.get(key, [])
        if value
    }
    event_names.update(name for name in ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"] if name in event_blob)
    if event_names and any(name in content for name in event_names):
        return True
    speaker = str(claim.get("speaker") or "")
    if speaker and speaker in event_names and any(keyword in content for keyword in ["我杀", "我刀", "我击杀", "I killed"]):
        return True
    # Generic public death reveal is allowed only when the event itself has no stable named participant.
    return not event_names and any(keyword in content for keyword in ["死亡", "尸体", "killed", "death"])


def validate_delayed_reveal_spec(event: dict[str, Any], target: str, claim: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if target in event.get("source_povs", []):
        errors.append("target_is_source_pov")
    event_local = phase_local_start_sec(event)
    if event_local is not None and event.get("phase_type") == "gameplay" and event_local < MIN_PHASE_LOCAL_SEC:
        errors.append(f"anchor_event_too_close_to_gameplay_phase_start:{event_local:.1f}s")
    claim_local = phase_local_start_sec(claim)
    if claim_local is not None and claim_local < MIN_PHASE_LOCAL_SEC:
        errors.append(f"reveal_claim_too_close_to_phase_start:{claim_local:.1f}s")
    delay = float(claim["abs_start_sec"]) - float(event["abs_end_sec"])
    if delay < MIN_REVEAL_DELAY_SEC:
        errors.append(f"reveal_too_close_to_event:{delay:.1f}s")
    if len(claim.get("heard_by", [])) < 4:
        errors.append("claim_not_public_enough")
    if claim.get("claim_type") not in REVEAL_CLAIM_TYPES:
        errors.append(f"claim_type_not_reveal:{claim.get('claim_type')}")
    if float(claim["abs_start_sec"]) <= float(event["abs_end_sec"]):
        errors.append("claim_not_after_event")
    if not claim_semantically_matches_event(event, claim):
        errors.append("claim_does_not_semantically_match_event")
    return errors


def make_group(idx: int, event: dict[str, Any], target: str, reveal_claims: list[dict[str, Any]]) -> dict[str, Any]:
    event_id = event["world_event_id"]
    reveal_claim_ids = [claim["claim_id"] for claim in reveal_claims]
    reveal_start = min(float(claim["abs_start_sec"]) for claim in reveal_claims)
    return {
        "probe_group_id": f"g001_pdpr_{idx:06d}_{target}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": float(event["abs_end_sec"]) + 3.0,
        "target_player": target,
        "query_variable": {
            "type": "delayed_public_reveal",
            "description": f"Before public reveal at {reveal_start:.1f}s, whether {target} knew: {event.get('description', '')}",
        },
        "anchor_event_ids": [event_id],
        "related_claim_ids": [],
        "later_public_claim_ids": reveal_claim_ids,
        "hidden_event_ids_for_target": [event_id],
        "available_evidence_ids_for_target": [],
        "selection_reason": (
            f"{event_id} is hidden from {target} at cutoff, then later becomes publicly discussable via "
            f"claim(s) {reveal_claim_ids} after {reveal_start:.1f}s."
        ),
        "diagnostic_families": ["false_belief", "representational_change", "delayed_public_reveal"],
        "template": "delayed_public_reveal",
        "quality": {
            "visibility_confidence": 0.9,
            "claim_truth_confidence": 0.75,
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
        event_id, target, claim_ids = parse_spec(spec)
        event = events.get(event_id)
        if not event:
            raise SystemExit(f"missing event: {event_id}")
        reveal_claims = []
        for claim_id in claim_ids:
            claim = claims.get(claim_id)
            if not claim:
                raise SystemExit(f"missing claim: {claim_id}")
            errors = validate_delayed_reveal_spec(event, target, claim)
            if errors:
                raise SystemExit(f"invalid delayed reveal spec {event_id}:{target}:{claim_id}: {'; '.join(errors)}")
            reveal_claims.append(claim)
        groups.append(make_group(len(groups) + 1, event, target, reveal_claims))

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
        gold["later_public_claim_ids"] = group["later_public_claim_ids"]
        gold["delayed_public_reveal"] = {
            "event_hidden_before_cutoff": True,
            "later_public_claim_ids": group["later_public_claim_ids"],
        }
        quality["recommended_gold_source"] = "qwen_checked"
        quality["needs_human_review"] = True
        quality["paper_gold_candidate"] = True
        for probe in probes:
            probe["gold_source"] = "qwen_checked"
            probes_by_type[probe["probe_type"]].append(probe)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    write_jsonl(diagnostics / "probe_groups.jsonl", groups)
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
