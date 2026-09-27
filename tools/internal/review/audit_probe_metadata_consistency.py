from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BROAD_OR_PUBLIC_EVENT_TYPES = {
    "phase_transition",
    "transition",
    "scene_transition",
    "gameplay_phase_transition",
    "meeting",
    "discussion",
    "voting",
    "vote",
    "player_voting",
    "vote_result",
    "voting_result",
    "result",
    "game_result",
    "game_start",
    "gameplay_start",
    "game_over",
    "unknown",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_reviewed(pass_root: Path) -> set[str]:
    path = pass_root / "review/codex_human_review_records.jsonl"
    if not path.exists():
        return set()
    return {row["probe_group_id"] for row in read_jsonl(path)}


def issue_probe(group: dict, event: dict, claims: list[dict], gold: dict | None) -> list[str]:
    issues: list[str] = []
    cutoff = float(group.get("cutoff_abs_sec") or -1)
    event_start = float(event.get("abs_start_sec") or -1)
    event_end = float(event.get("abs_end_sec") or -1)
    event_type = event.get("event_type")
    template = group.get("template")
    target = group.get("target_player")
    source_povs = event.get("source_povs") or []

    if group.get("needs_human_review"):
        issues.append("group_marked_needs_human_review")
    if event.get("needs_human_review"):
        issues.append("anchor_event_marked_needs_human_review")
    if event_start > cutoff:
        issues.append("anchor_event_starts_after_cutoff")
    if event_end > cutoff and template in {"hidden_event_awareness", "contradicted_alibi"}:
        issues.append("anchor_event_extends_after_cutoff")
    if target in source_povs and template in {"hidden_event_awareness", "contradicted_alibi"}:
        issues.append("target_is_anchor_source_pov")
    if event_type in BROAD_OR_PUBLIC_EVENT_TYPES and template in {"hidden_event_awareness", "contradicted_alibi", "vote_influence", "private_witness", "delayed_public_reveal"}:
        issues.append("anchor_event_is_broad_public_or_transition_type")

    related_claim_ids = group.get("related_claim_ids") or []
    available_ids = group.get("available_evidence_ids_for_target") or []
    if template in {"contradicted_alibi", "vote_influence"} and not related_claim_ids:
        issues.append("claim_template_without_related_claim")
    if template == "hidden_event_awareness" and not related_claim_ids and not available_ids:
        issues.append("hidden_event_without_target_available_evidence")

    for claim in claims:
        claim_start = float(claim.get("abs_start_sec") or -1)
        claim_end = float(claim.get("abs_end_sec") or -1)
        if claim_start > cutoff:
            issues.append(f"claim_{claim.get('claim_id')}_starts_after_cutoff")
        if claim_end > cutoff:
            issues.append(f"claim_{claim.get('claim_id')}_ends_after_cutoff")
        if claim.get("needs_human_review"):
            issues.append(f"claim_{claim.get('claim_id')}_marked_needs_human_review")
        if template == "contradicted_alibi" and claim_start < event_end:
            # The prior pipeline often linked arbitrary post-hoc events. This issue is only fatal when paired with
            # unverified claim truth or a broad event, which is handled in the final decision gate below.
            issues.append(f"claim_{claim.get('claim_id')}_not_after_anchor_event")
        if claim.get("claim_type") in {"other", "unknown"} and template in {"contradicted_alibi", "vote_influence"}:
            issues.append(f"claim_{claim.get('claim_id')}_not_verifiable_claim_type")
        if not claim.get("content"):
            issues.append(f"claim_{claim.get('claim_id')}_empty_content")

    if gold:
        if gold.get("claim_truth_global") == "unverified" and template == "contradicted_alibi":
            issues.append("contradicted_alibi_has_unverified_global_claim_truth")
        if not (gold.get("forbidden_event_ids_for_target") or []) and template in {"hidden_event_awareness", "contradicted_alibi"}:
            issues.append("no_forbidden_event_ids_for_target")
        if gold.get("A_expected_weak", {}).get("knows_truth") is True and template == "hidden_event_awareness":
            issues.append("hidden_event_gold_says_target_knows_truth")

    return sorted(set(issues))


def rejectable(issues: list[str]) -> bool:
    issue_set = set(issues)
    fatal_any = {
        "anchor_event_marked_needs_human_review",
        "anchor_event_starts_after_cutoff",
        "target_is_anchor_source_pov",
        "anchor_event_is_broad_public_or_transition_type",
        "no_forbidden_event_ids_for_target",
    }
    if issue_set & fatal_any:
        return True
    if any(issue.startswith("claim_") and issue.endswith("_starts_after_cutoff") for issue in issues):
        return True
    if any(issue.startswith("claim_") and issue.endswith("_ends_after_cutoff") for issue in issues):
        return True
    if "hidden_event_without_target_available_evidence" in issue_set:
        return True
    if "contradicted_alibi_has_unverified_global_claim_truth" in issue_set and any(
        issue.endswith("_not_verifiable_claim_type") for issue in issues
    ):
        return True
    if "contradicted_alibi_has_unverified_global_claim_truth" in issue_set and any(
        issue.endswith("_not_after_anchor_event") for issue in issues
    ):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pass_root = Path(args.pass_root)
    groups = read_jsonl(pass_root / "annotations/diagnostics/probe_groups.jsonl")
    events = {row["world_event_id"]: row for row in read_jsonl(pass_root / "annotations/oracle_ledger/world_events.jsonl")}
    claims = {row["claim_id"]: row for row in read_jsonl(pass_root / "annotations/oracle_ledger/claims.jsonl")}
    gold = {row["probe_group_id"]: row for row in read_jsonl(pass_root / "annotations/diagnostics/hidden_gold.jsonl")}
    reviewed = load_reviewed(pass_root)

    rows = []
    for group in groups:
        gid = group["probe_group_id"]
        if gid in reviewed:
            continue
        event = events[group["anchor_event_ids"][0]]
        group_claims = [claims[cid] for cid in group.get("related_claim_ids", []) if cid in claims]
        issues = issue_probe(group, event, group_claims, gold.get(gid))
        if rejectable(issues):
            rows.append(
                {
                    "probe_group_id": gid,
                    "template": group.get("template"),
                    "target_player": group.get("target_player"),
                    "cutoff_abs_sec": group.get("cutoff_abs_sec"),
                    "anchor_event_id": event.get("world_event_id"),
                    "anchor_event_type": event.get("event_type"),
                    "anchor_abs_start_sec": event.get("abs_start_sec"),
                    "anchor_abs_end_sec": event.get("abs_end_sec"),
                    "source_povs": event.get("source_povs"),
                    "related_claim_ids": group.get("related_claim_ids"),
                    "issues": issues,
                }
            )

    if args.limit and len(rows) > args.limit:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")

    print(json.dumps({
        "pass_root": str(pass_root),
        "reviewed": len(reviewed),
        "rejectable_unreviewed": len(rows),
        "by_template": Counter(row["template"] for row in rows),
        "top_issues": Counter(issue for row in rows for issue in row["issues"]).most_common(30),
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
