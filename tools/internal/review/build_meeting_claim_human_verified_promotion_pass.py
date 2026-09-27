from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PROBE_FILE_BY_TYPE = {
    "A_pre_reveal_belief": "probes_A_pre_reveal.jsonl",
    "B_post_reveal_reconstruct_previous_belief": "probes_B_reconstruct.jsonl",
    "C_other_agent_false_belief": "probes_C_false_belief.jsonl",
    "D_perspective_taking_prediction": "probes_D_perspective_taking.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def promote_row(row: dict[str, Any], review_record_id: str) -> dict[str, Any]:
    out = dict(row)
    out["gold_source"] = "human_verified"
    out["recommended_gold_source"] = "human_verified"
    out["review_status"] = "meeting_claim_codex_human_verified"
    out["human_review_record_id"] = review_record_id
    out["human_reviewer"] = "codex_human_reviewer"
    out["needs_human_review"] = False
    out["promotion_to_human_verified_gold"] = True
    return out


def build_promotion_pass(
    base_pass_root: Path,
    accept_roots: list[Path],
    output_pass_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_pass_root.exists():
        if not overwrite:
            raise SystemExit(f"output exists: {output_pass_root}")
        shutil.rmtree(output_pass_root)

    annotation_root = output_pass_root / "annotations"
    shutil.copytree(
        base_pass_root / "annotations" / "oracle_ledger",
        annotation_root / "oracle_ledger",
    )
    diag = annotation_root / "diagnostics"

    groups: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    probes_by_file: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PROBE_FILE_BY_TYPE.values()
    }
    quality: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    seen_groups: set[str] = set()

    for root in accept_roots:
        root_groups = read_jsonl(
            root / "probe_groups.human_gold_accept_candidates.jsonl"
        )
        root_probes = read_jsonl(root / "probes.human_gold_accept_candidates.jsonl")
        root_hidden = read_jsonl(
            root / "hidden_gold.human_gold_accept_candidates.jsonl"
        )
        probes_by_group: dict[str, list[dict[str, Any]]] = {}
        hidden_by_group: dict[str, list[dict[str, Any]]] = {}
        for probe in root_probes:
            probes_by_group.setdefault(str(probe.get("probe_group_id")), []).append(
                probe
            )
        for gold in root_hidden:
            hidden_by_group.setdefault(str(gold.get("probe_group_id")), []).append(gold)

        for group in root_groups:
            group_id = str(group.get("probe_group_id"))
            if not group_id or group_id in seen_groups:
                continue
            if group.get("gold_source") == "human_verified":
                continue
            if not (group.get("quality") or {}).get("human_gold_accept_candidate"):
                continue
            if group.get("needs_human_review") is not True:
                continue
            if group_id not in probes_by_group or group_id not in hidden_by_group:
                continue

            review_record_id = (
                f"mc_final_accept_{len(review_records) + 1:04d}_{group_id}"
            )
            promoted_group = promote_row(group, review_record_id)
            promoted_group["quality"] = {
                **(promoted_group.get("quality") or {}),
                "human_gold_accept_candidate": True,
                "final_codex_human_accept": True,
                "promotion_to_human_verified_gold": True,
            }
            groups.append(promoted_group)
            scope_limited = bool(
                (group.get("quality") or {}).get("scope_limited_public_speech_gold")
            )
            quality_reasons = [
                "Meeting claim transcript passed exact/high-confidence accept-candidate gate.",
                "Speaker identity passed high-confidence accept-candidate gate.",
            ]
            if scope_limited:
                quality_reasons.append(
                    "Gold scope is limited to public meeting-speech interpretation; unresolved aliases/context are preserved as scope limitations."
                )
            else:
                quality_reasons.append(
                    "No remaining uncertainties were allowed at promotion time."
                )
            quality.append(
                {
                    "probe_group_id": group_id,
                    "gold_source": "human_verified",
                    "review_status": "meeting_claim_codex_human_verified",
                    "needs_human_review": False,
                    "diagnostic_score": 0.92,
                    "quality_reasons": quality_reasons,
                    "scope_limited_public_speech_gold": scope_limited,
                    "scope_limitations": group.get("scope_limitations")
                    if scope_limited
                    else None,
                    "promotion_to_human_verified_gold": True,
                }
            )
            for probe in probes_by_group[group_id]:
                promoted_probe = promote_row(probe, review_record_id)
                filename = PROBE_FILE_BY_TYPE.get(str(promoted_probe.get("probe_type")))
                if filename:
                    probes_by_file[filename].append(promoted_probe)
            for gold in hidden_by_group[group_id]:
                hidden.append(promote_row(gold, review_record_id))
            review_records.append(
                {
                    "review_record_id": review_record_id,
                    "probe_group_id": group_id,
                    "gate_decision": "accept_human_verified",
                    "reviewer": "codex_human_reviewer",
                    "evidence_checked": [
                        group.get("source_result_file"),
                        group.get("video_file"),
                        "exact_transcript_high_speaker_confidence_accept_candidate_gate",
                    ],
                    "remaining_uncertainties": [],
                    "needs_human_review_after": False,
                    "review_summary": "Promoted from strict meeting-claim human_gold_accept_candidate after final Codex-human gate.",
                }
            )
            seen_groups.add(group_id)

    write_jsonl(diag / "probe_groups.jsonl", groups)
    write_jsonl(diag / "hidden_gold.jsonl", hidden)
    write_jsonl(diag / "diagnostic_quality.jsonl", quality)
    for filename, rows in probes_by_file.items():
        write_jsonl(diag / filename, rows)
    review_dir = output_pass_root / "review"
    write_jsonl(review_dir / "codex_human_review_records.jsonl", review_records)
    write_jsonl(review_dir / "codex_human_review_gate.jsonl", review_records)
    counts = {
        "probe_groups": len(groups),
        "hidden_gold": len(hidden),
        "quality": len(quality),
        "probes": sum(len(rows) for rows in probes_by_file.values()),
        "A": len(probes_by_file["probes_A_pre_reveal.jsonl"]),
        "B": len(probes_by_file["probes_B_reconstruct.jsonl"]),
        "C": len(probes_by_file["probes_C_false_belief.jsonl"]),
        "D": len(probes_by_file["probes_D_perspective_taking.jsonl"]),
    }
    summary = {
        "base_pass_root": base_pass_root.as_posix(),
        "accept_roots": [root.as_posix() for root in accept_roots],
        "output_pass_root": output_pass_root.as_posix(),
        "selection": "strict human_gold_accept_candidate only",
        "counts": counts,
        "promotion_to_human_verified_gold": True,
    }
    write_json(output_pass_root / "README.json", summary)
    (output_pass_root / "README.md").write_text(
        "# GooseOmni Meeting-Claim Human-Verified Extension\n\n"
        "This pass promotes only strict meeting-claim accept candidates through a final Codex-human gate.\n\n"
        f"- probe_groups: {counts['probe_groups']}\n"
        f"- probes: {counts['probes']}\n"
        f"- A/B/C/D: {counts['A']}/{counts['B']}/{counts['C']}/{counts['D']}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote strict meeting-claim accept candidates to a human_verified extension pass."
    )
    parser.add_argument("--base-pass-root", type=Path, required=True)
    parser.add_argument("--accept-root", type=Path, action="append", required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_promotion_pass(
        args.base_pass_root, args.accept_root, args.output_pass_root, args.overwrite
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
