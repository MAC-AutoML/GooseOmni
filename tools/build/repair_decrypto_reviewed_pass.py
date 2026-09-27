from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import PLAYERS, write_json, write_jsonl

PROBE_FILES = {
    "A_pre_reveal_belief": "probes_A_pre_reveal.jsonl",
    "B_post_reveal_reconstruct_previous_belief": "probes_B_reconstruct.jsonl",
    "C_other_agent_false_belief": "probes_C_false_belief.jsonl",
    "D_perspective_taking_prediction": "probes_D_perspective_taking.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair pass1 Decrypto diagnostics into a cleaner pass2_reviewed directory."
    )
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--drop-no-target-evidence", action="store_true", default=True)
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


def snapshot_for(
    snapshots: list[dict[str, Any]], player: str, cutoff: float
) -> dict[str, Any]:
    candidates = [
        row
        for row in snapshots
        if row.get("target_player") == player
        and float(row.get("cutoff_abs_sec", -1)) <= cutoff + 1e-6
    ]
    if not candidates:
        return {
            "public_history": [],
            "private_observations": [],
            "heard_claims": [],
            "inferred_beliefs": [],
            "available_evidence_ids": [],
        }
    return max(candidates, key=lambda row: float(row.get("cutoff_abs_sec", 0.0)))


def compact_speaker_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "public_history": snapshot.get("public_history", [])[:12],
        "speaker_private_observations": snapshot.get("private_observations", [])[:12],
        "speaker_heard_claims": snapshot.get("heard_claims", [])[:12],
        "speaker_inferred_beliefs": snapshot.get("inferred_beliefs", [])[:6],
        "available_evidence_ids": snapshot.get("available_evidence_ids", [])[:30],
    }


def compact_listener_public_model(
    snapshot: dict[str, Any], related_claim_ids: list[str]
) -> dict[str, Any]:
    heard_public_claims = [
        claim
        for claim in snapshot.get("heard_claims", [])
        if claim.get("claim_id") in related_claim_ids
    ]
    return {
        "listener_public_history": snapshot.get("public_history", [])[:12],
        "related_claims_listener_heard": heard_public_claims[:8],
        "private_listener_observations_omitted": True,
        "epistemic_note": "This is not the listener's private POV. It is the speaker-safe model of public/listener-heard context.",
    }


def prompt_header(group: dict[str, Any]) -> str:
    return (
        "Dataset: GooseOmni\n"
        "Game: Goose Goose Duck / 鹅鸭杀风格多人社交博弈\n"
        "Players: Gemini, baile, beigang, mojiang, saoyi, xiaolu\n"
        "Time rule: abs_sec = aligned_start_sec + local_sec.\n"
        "Epistemic rule: distinguish oracle truth from what each player could know at cutoff.\n"
        f"Probe group: {group['probe_group_id']}\n"
        f"Target player: {group['target_player']}\n"
        f"Cutoff abs sec: {group['cutoff_abs_sec']}\n"
    )


def public_query_variable(group: dict[str, Any]) -> dict[str, Any]:
    related_claims = group.get("related_claim_ids", [])
    if related_claims:
        public_question = (
            "Based only on target-available evidence before cutoff, can the target player verify, doubt, "
            "or remain uncertain about the related speech claim?"
        )
    else:
        public_question = (
            "Based only on target-available evidence before cutoff, what can the target player infer about "
            "the recent situation? Do not assume any oracle-hidden event occurred."
        )
    return {
        "query_type": group.get("query_variable", {}).get(
            "type", "hidden_event_awareness"
        ),
        "target_player": group["target_player"],
        "cutoff_abs_sec": group["cutoff_abs_sec"],
        "related_claim_ids": related_claims,
        "anchor_event_redacted_from_A_prompt": bool(
            group.get("hidden_event_ids_for_target")
        ),
        "public_question": public_question,
    }


def repair_a_prompt(probe: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(probe)
    original = probe.get("prompt", "")
    query_json = json.dumps(public_query_variable(group), ensure_ascii=False, indent=2)
    if "TARGET_AVAILABLE_CONTEXT_JSON:" in original:
        context = (
            original.split("TARGET_AVAILABLE_CONTEXT_JSON:", 1)[1]
            .split("QUESTION:", 1)[0]
            .strip()
        )
    else:
        context = "{}"
    repaired["prompt"] = (
        prompt_header(group)
        + "\nOnly use the target player's available information before cutoff. Do not use hidden oracle events.\n"
        + "QUERY_VARIABLE_PUBLIC_FORM_JSON:\n"
        + query_json
        + "\nTARGET_AVAILABLE_CONTEXT_JSON:\n"
        + context
        + f"\nQUESTION: From {group['target_player']}'s perspective at cutoff, answer the public query above. "
        + "State what the player knows, does not know, or can only treat as uncertain. "
        + "Return strict JSON matching the schema."
    )
    repaired["repair_notes"] = [
        "added leakage-safe public query variable",
        "kept oracle-hidden event descriptions out of A prompt",
    ]
    return repaired


def repair_d_prompt(
    probe: dict[str, Any],
    group: dict[str, Any],
    claims_by_id: dict[str, dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    repaired = dict(probe)
    related_claims = [
        claims_by_id[cid]
        for cid in group.get("related_claim_ids", [])
        if cid in claims_by_id
    ]
    speaker = (
        related_claims[0].get("speaker", "unknown") if related_claims else "unknown"
    )
    if speaker not in PLAYERS:
        speaker = "unknown"
    cutoff = float(group["cutoff_abs_sec"])
    speaker_snapshot = (
        snapshot_for(snapshots, speaker, cutoff) if speaker in PLAYERS else {}
    )
    listener_snapshot = snapshot_for(snapshots, group["target_player"], cutoff)
    repaired["prompt"] = (
        prompt_header(group)
        + "\nRELATED_CLAIMS_JSON:\n"
        + json.dumps(related_claims, ensure_ascii=False)
        + "\nSPEAKER_AVAILABLE_CONTEXT_JSON:\n"
        + json.dumps(compact_speaker_context(speaker_snapshot), ensure_ascii=False)
        + "\nSPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON:\n"
        + json.dumps(
            compact_listener_public_model(
                listener_snapshot, group.get("related_claim_ids", [])
            ),
            ensure_ascii=False,
        )
        + f"\nQUESTION: From speaker {speaker}'s perspective after making the strategic claim, predict how listener "
        + f"{group['target_player']} would interpret the claim and how their trust/action may change. "
        + "Do not use listener private observations unless they are present in the speaker-safe public model. "
        + "Return strict JSON matching the schema."
    )
    repaired["repair_notes"] = [
        "replaced listener private context with speaker-safe listener public model"
    ]
    return repaired


def should_drop_group(group: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not group.get("available_evidence_ids_for_target") and not group.get(
        "related_claim_ids"
    ):
        reasons.append("no_target_available_evidence_or_related_claim")
    if (
        group.get("template") == "hidden_event_awareness"
        and not group.get("available_evidence_ids_for_target")
        and not group.get("related_claim_ids")
    ):
        reasons.append("weak_hidden_event_awareness_without_partial_information")
    return bool(reasons), reasons


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
    shutil.copytree(src / "docs", dst / "docs", dirs_exist_ok=True)
    shutil.copytree(src / "scripts", dst / "scripts", dirs_exist_ok=True)
    shutil.copytree(src / "slurm", dst / "slurm", dirs_exist_ok=True)
    shutil.copytree(src / "tests", dst / "tests", dirs_exist_ok=True)

    src_diag = src / "annotations" / "diagnostics"
    dst_diag = dst / "annotations" / "diagnostics"
    dst_diag.mkdir(parents=True, exist_ok=True)
    review_dir = dst / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    groups = read_jsonl(src_diag / "probe_groups.jsonl")
    hidden_gold = read_jsonl(src_diag / "hidden_gold.jsonl")
    quality = read_jsonl(src_diag / "diagnostic_quality.jsonl")
    claims = read_jsonl(src / "annotations" / "oracle_ledger" / "claims.jsonl")
    snapshots = read_jsonl(
        src / "annotations" / "oracle_ledger" / "belief_memory_snapshots.jsonl"
    )
    claims_by_id = {row["claim_id"]: row for row in claims}

    probes: dict[str, list[dict[str, Any]]] = {}
    for probe_type, filename in PROBE_FILES.items():
        probes[probe_type] = read_jsonl(src_diag / filename)

    kept_groups = []
    dropped_groups = []
    kept_ids = set()
    for group in groups:
        drop, reasons = should_drop_group(group)
        if drop:
            dropped = dict(group)
            dropped["drop_reasons"] = reasons
            dropped_groups.append(dropped)
            continue
        repaired_group = dict(group)
        repaired_group["review_status"] = "codex_pass2_prompt_repaired"
        repaired_group["gold_source"] = "qwen_weak"
        kept_groups.append(repaired_group)
        kept_ids.add(group["probe_group_id"])

    repaired_by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    repair_actions = []
    for probe_type, rows in probes.items():
        for probe in rows:
            group_id = probe["probe_group_id"]
            if group_id not in kept_ids:
                continue
            group = next(
                row for row in kept_groups if row["probe_group_id"] == group_id
            )
            repaired_probe = dict(probe)
            if probe_type == "A_pre_reveal_belief":
                repaired_probe = repair_a_prompt(probe, group)
                repair_actions.append(
                    {
                        "probe_id": probe["probe_id"],
                        "action": "A_prompt_add_public_query_variable",
                    }
                )
            elif probe_type == "D_perspective_taking_prediction":
                repaired_probe = repair_d_prompt(probe, group, claims_by_id, snapshots)
                repair_actions.append(
                    {
                        "probe_id": probe["probe_id"],
                        "action": "D_prompt_speaker_safe_context",
                    }
                )
            repaired_by_type[probe_type].append(repaired_probe)

    kept_hidden = [row for row in hidden_gold if row["probe_group_id"] in kept_ids]
    kept_quality = []
    for row in quality:
        if row["probe_group_id"] in kept_ids:
            repaired = dict(row)
            repaired["recommended_gold_source"] = "qwen_weak"
            repaired["review_note"] = (
                "codex pass2 repaired prompts and filtered no-evidence weak probes; still requires qwen_checked or human review before final gold"
            )
            kept_quality.append(repaired)

    write_jsonl(dst_diag / "probe_groups.jsonl", kept_groups)
    for probe_type, filename in PROBE_FILES.items():
        write_jsonl(dst_diag / filename, repaired_by_type[probe_type])
    write_jsonl(dst_diag / "hidden_gold.jsonl", kept_hidden)
    write_jsonl(dst_diag / "diagnostic_quality.jsonl", kept_quality)
    write_jsonl(review_dir / "dropped_probe_groups.jsonl", dropped_groups)
    write_jsonl(review_dir / "prompt_repair_actions.jsonl", repair_actions)

    summary = {
        "input_pass_root": src.as_posix(),
        "output_pass_root": dst.as_posix(),
        "groups_before": len(groups),
        "groups_after": len(kept_groups),
        "groups_dropped": len(dropped_groups),
        "probes_after": sum(len(rows) for rows in repaired_by_type.values()),
        "A_after": len(repaired_by_type["A_pre_reveal_belief"]),
        "B_after": len(repaired_by_type["B_post_reveal_reconstruct_previous_belief"]),
        "C_after": len(repaired_by_type["C_other_agent_false_belief"]),
        "D_after": len(repaired_by_type["D_perspective_taking_prediction"]),
        "repair_actions": len(repair_actions),
        "drop_reasons": dict(
            collections.Counter(
                reason for row in dropped_groups for reason in row["drop_reasons"]
            )
        ),
        "status": "qwen_weak_pass2_repaired_not_final_gold",
    }
    write_json(review_dir / "pass2_repair_summary.json", summary)
    (dst / "README.md").write_text(
        "# Omni Goose Decrypto Diagnostic Pass 2 Reviewed\n\n"
        "This pass is derived from pass1 by deterministic Codex review repairs:\n\n"
        "- dropped weak hidden-event probes with no target-available evidence or related claim;\n"
        "- repaired A prompts with leakage-safe public query variables;\n"
        "- repaired D prompts to use speaker-safe context instead of listener private context;\n"
        "- kept all automatic labels as qwen_weak pending Qwen3-Omni high-quality review or manual verification.\n\n"
        f"Groups before: {summary['groups_before']}\n\n"
        f"Groups after: {summary['groups_after']}\n\n"
        f"Prompts after: {summary['probes_after']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
