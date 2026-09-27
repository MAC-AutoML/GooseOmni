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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gooseomni.benchmark.decrypto_diagnostics import read_jsonl, write_jsonl


QUALITY_PROFILE = {
    "QWEN3_OMNI_MAX_TOKENS": 16384,
    "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": 32768,
    "QWEN3_OMNI_VIDEO_FPS": 1.0,
    "QWEN3_OMNI_VIDEO_MAX_FRAMES": 128,
    "QWEN3_OMNI_VIDEO_MAX_PIXELS": 401408,
}

TEMPLATE_PRIORITY = {
    "contradicted_alibi": 50,
    "vote_influence": 45,
    "delayed_public_reveal": 42,
    "private_witness": 38,
    "hidden_event_awareness": 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-value Qwen3-Omni review queue for Decrypto-style probes.")
    parser.add_argument("--pass-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def score_group(group: dict[str, Any], links_by_claim: dict[str, list[dict[str, Any]]]) -> int:
    score = TEMPLATE_PRIORITY.get(str(group.get("template")), 10)
    score += min(len(group.get("related_claim_ids", [])), 8)
    if group.get("hidden_event_ids_for_target"):
        score += 8
    if group.get("needs_human_review"):
        score += 4
    for claim_id in group.get("related_claim_ids", []):
        for link in links_by_claim.get(claim_id, []):
            if link.get("truth_status_global") == "contradicted":
                score += 18
            elif link.get("truth_status_global") == "ambiguous":
                score += 10
            elif link.get("truth_status_global") == "supported":
                score += 4
            if link.get("needs_human_review"):
                score += 4
    return score


def issue_codes(group: dict[str, Any], links: list[dict[str, Any]]) -> list[str]:
    codes = ["verify_video_grounding", "verify_local_awareness", "promote_only_if_leakage_safe"]
    template = group.get("template")
    if template == "contradicted_alibi":
        codes.extend(["verify_claim_truth_contradiction", "repair_claim_type"])
    elif template == "vote_influence":
        codes.extend(["verify_strategy_claim", "verify_speaker_listener_information_difference"])
    elif template == "delayed_public_reveal":
        codes.extend(["verify_private_to_public_reveal", "verify_reveal_timing"])
    elif template == "private_witness":
        codes.extend(["verify_private_witness", "verify_other_players_not_visible"])
    if any(link.get("needs_human_review") for link in links):
        codes.append("claim_truth_link_needs_review")
    return list(dict.fromkeys(codes))


def build_queue(pass_root: Path, limit: int) -> list[dict[str, Any]]:
    diag = pass_root / "annotations" / "diagnostics"
    ledger = pass_root / "annotations" / "oracle_ledger"
    groups = read_jsonl(diag / "probe_groups.jsonl")
    events_by_id = by_id(read_jsonl(ledger / "world_events.jsonl"), "world_event_id")
    claims_by_id = by_id(read_jsonl(ledger / "claims.jsonl"), "claim_id")
    links = read_jsonl(ledger / "claim_truth_links.jsonl")
    links_by_claim: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_claim.setdefault(str(link.get("claim_id")), []).append(link)

    by_template: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_template.setdefault(str(group.get("template")), []).append(group)
    for template_groups in by_template.values():
        template_groups.sort(key=lambda group: (-score_group(group, links_by_claim), group.get("probe_group_id", "")))

    template_order = ["contradicted_alibi", "vote_influence", "delayed_public_reveal", "private_witness", "hidden_event_awareness"]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    min_per_template = max(1, min(8, limit // max(1, len(template_order))))
    for rank in range(min_per_template):
        for template in template_order:
            template_groups = by_template.get(template, [])
            if rank >= len(template_groups):
                continue
            group = template_groups[rank]
            selected.append(group)
            selected_ids.add(group["probe_group_id"])
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    remaining = sorted(groups, key=lambda group: (-score_group(group, links_by_claim), group.get("probe_group_id", "")))
    for group in remaining:
        if len(selected) >= limit:
            break
        if group["probe_group_id"] in selected_ids:
            continue
        selected.append(group)
        selected_ids.add(group["probe_group_id"])

    tasks = []
    for idx, group in enumerate(selected, start=1):
        related_links = [link for claim_id in group.get("related_claim_ids", []) for link in links_by_claim.get(claim_id, [])]
        anchor_events = [events_by_id[event_id] for event_id in group.get("anchor_event_ids", []) if event_id in events_by_id]
        related_claims = [claims_by_id[claim_id] for claim_id in group.get("related_claim_ids", []) if claim_id in claims_by_id]
        tasks.append(
            {
                "review_task_id": f"pass8_hqv3_{idx:04d}",
                "task_type": "decrypto_probe_group_video_review",
                "priority": "high" if idx <= max(8, limit // 4) else "medium",
                "selection_score": score_group(group, links_by_claim),
                "qwen3_omni_quality_profile": QUALITY_PROFILE,
                "probe_group": group,
                "anchor_events": anchor_events,
                "related_claims": related_claims,
                "claim_truth_links": related_links,
                "known_issue_codes": issue_codes(group, related_links),
                "instruction": (
                    "Review this pass8 Decrypto-style probe group using video evidence. "
                    "Do not generate prompts. Verify event grounding, claim grounding, visibility, and local awareness. "
                    "Promote only if deterministic safe-prompt benchmark use is justified."
                ),
            }
        )
    return tasks


def main() -> None:
    args = parse_args()
    tasks = build_queue(args.pass_root, args.limit)
    write_jsonl(args.output, tasks)
    print(json.dumps({"ok": True, "queue": args.output.as_posix(), "tasks": len(tasks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
