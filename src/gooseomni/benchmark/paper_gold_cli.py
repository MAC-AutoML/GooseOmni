import collections
import json
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import (
    generate_probes_for_group,
    load_ledger,
)
from gooseomni.benchmark.io import write_json, write_jsonl
from gooseomni.benchmark.paper_gold import (
    CLAIM_BACKED_EVENT_TYPES,
    CRITICAL_EVENT_TYPES,
    build_links_by_event,
    claim_by_id,
    claim_is_usable,
    claim_speaker_can_ground_anchor,
    description_mentions_player,
    direct_players,
    edge_lookup,
    group_score,
    has_critical_description,
    heard_by_target_before_cutoff,
    hidden_players,
    is_clean_event,
    is_route_event,
    make_group,
    normalized_description,
    parse_args,
)


def select_paper_gold_candidates(
    ledger: dict[str, list[dict[str, Any]]],
    limit: int,
    pure_hidden_critical_only: bool = False,
    route_hidden_only: bool = False,
    max_targets_per_event: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    events = ledger["world_events"]
    claims = claim_by_id(ledger["claims"])
    links_by_event = build_links_by_event(ledger["claim_truth_links"])
    edges = edge_lookup(ledger["visibility_edges"])
    selected: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str]
    ] = []
    used: set[tuple[str, str, str]] = set()
    used_semantic_events: set[tuple[str, int, str, str]] = set()
    targets_per_event: dict[str, int] = collections.defaultdict(int)

    def add(
        event: dict[str, Any],
        target: str,
        template: str,
        related_claim: dict[str, Any] | None,
        cutoff: float,
        truth_status: str,
    ) -> None:
        key = (
            event["world_event_id"],
            target,
            related_claim["claim_id"] if related_claim else "",
        )
        if key in used:
            return
        if route_hidden_only and target in event.get("actors", []):
            return
        if description_mentions_player(event, target) and (
            event.get("event_type") in CRITICAL_EVENT_TYPES
            or has_critical_description(event)
        ):
            return
        semantic_key = (
            related_claim["claim_id"] if related_claim else "",
            int(float(event["abs_start_sec"]) // 30),
            ",".join(sorted(event.get("source_povs", []))),
            normalized_description(event),
        )
        if semantic_key in used_semantic_events:
            return
        if pure_hidden_critical_only or route_hidden_only:
            semantic_key = (*semantic_key, target)
        if targets_per_event[event["world_event_id"]] >= max_targets_per_event:
            return
        used.add(key)
        used_semantic_events.add(semantic_key)
        targets_per_event[event["world_event_id"]] += 1
        group = make_group(
            len(selected) + 1,
            event,
            target,
            template,
            [related_claim] if related_claim else [],
            cutoff,
            truth_status,
        )
        selected.append((group, event, related_claim, truth_status))

    for event in sorted(
        events,
        key=lambda row: (float(row["abs_start_sec"]), str(row["world_event_id"])),
    ):
        if not is_clean_event(event):
            continue
        event_type = str(event.get("event_type"))
        is_critical = event_type in CRITICAL_EVENT_TYPES or has_critical_description(
            event
        )
        is_route = is_route_event(event)
        if pure_hidden_critical_only and not is_critical:
            continue
        if route_hidden_only and not is_route:
            continue
        if not pure_hidden_critical_only and not route_hidden_only and not is_critical:
            continue
        direct = direct_players(event, edges)
        hidden = hidden_players(event, edges)
        if not direct or not hidden:
            continue

        if not pure_hidden_critical_only and not route_hidden_only:
            for link in links_by_event.get(event["world_event_id"], []):
                if event_type not in CLAIM_BACKED_EVENT_TYPES:
                    continue
                truth_status = str(link.get("truth_status_global", "unverified"))
                if truth_status not in {"supported", "contradicted", "ambiguous"}:
                    continue
                if (
                    bool(link.get("needs_human_review"))
                    and truth_status != "contradicted"
                ):
                    continue
                claim = claims.get(str(link.get("claim_id")))
                if not claim or not claim_is_usable(claim):
                    continue
                if not claim_speaker_can_ground_anchor(event, claim, direct):
                    continue
                if float(event["abs_end_sec"]) > float(claim["abs_end_sec"]) + 1.0:
                    continue
                cutoff = float(claim["abs_end_sec"]) + 2.0
                for target in hidden:
                    if not heard_by_target_before_cutoff(claim, target, cutoff):
                        continue
                    template = (
                        "contradicted_alibi"
                        if truth_status == "contradicted"
                        else "claim_backed_hidden_event"
                    )
                    add(event, target, template, claim, cutoff, truth_status)

        if is_critical or is_route:
            cutoff = float(event["abs_end_sec"]) + 3.0
            target_pool = (
                hidden[:max_targets_per_event]
                if (pure_hidden_critical_only or route_hidden_only)
                else hidden[:3]
            )
            for target in target_pool:
                template = (
                    "route_hidden_event"
                    if route_hidden_only
                    else "critical_hidden_event"
                )
                add(event, target, template, None, cutoff, "unverified")

    ranked = sorted(
        selected, key=lambda item: group_score(item[0], item[1], item[2], item[3])
    )
    groups = []
    truth_by_group = {}
    for new_idx, (group, _event, _claim, truth_status) in enumerate(
        ranked[:limit], start=1
    ):
        group = dict(group)
        group["probe_group_id"] = f"g001_pgold_{new_idx:06d}_{group['target_player']}"
        group["_paper_gold_truth_status"] = truth_status
        truth_by_group[group["probe_group_id"]] = truth_status
        groups.append(group)
    return groups, truth_by_group


def write_diagnostics(
    output_root: Path,
    ledger: dict[str, list[dict[str, Any]]],
    groups: list[dict[str, Any]],
    truth_by_group: dict[str, str],
) -> dict[str, int]:
    diagnostics = output_root / "annotations" / "diagnostics"
    all_probes: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    hidden_gold = []
    quality_rows = []
    for group in groups:
        truth_status = str(
            group.get("_paper_gold_truth_status")
            or truth_by_group.get(group["probe_group_id"], "unverified")
        )
        public_group = {
            key: value
            for key, value in group.items()
            if not key.startswith("_paper_gold_")
        }
        probes, gold, quality = generate_probes_for_group(public_group, ledger)
        gold["claim_truth_global"] = truth_status
        gold["gold_source"] = "qwen_checked"
        gold["paper_gold_candidate"] = True
        quality["recommended_gold_source"] = "qwen_checked"
        quality["needs_human_review"] = True
        quality["paper_gold_candidate"] = True
        for probe in probes:
            probe["gold_source"] = "qwen_checked"
        all_probes.extend(probes)
        for probe in probes:
            by_type[probe["probe_type"]].append(probe)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    public_groups = [
        {
            key: value
            for key, value in group.items()
            if not key.startswith("_paper_gold_")
        }
        for group in groups
    ]
    write_jsonl(diagnostics / "probe_groups.jsonl", public_groups)
    write_jsonl(
        diagnostics / "probes_A_pre_reveal.jsonl", by_type["A_pre_reveal_belief"]
    )
    write_jsonl(
        diagnostics / "probes_B_reconstruct.jsonl",
        by_type["B_post_reveal_reconstruct_previous_belief"],
    )
    write_jsonl(
        diagnostics / "probes_C_false_belief.jsonl",
        by_type["C_other_agent_false_belief"],
    )
    write_jsonl(
        diagnostics / "probes_D_perspective_taking.jsonl",
        by_type["D_perspective_taking_prediction"],
    )
    write_jsonl(diagnostics / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diagnostics / "diagnostic_quality.jsonl", quality_rows)
    return {
        "probe_groups": len(groups),
        "probes": len(all_probes),
        "A": len(by_type["A_pre_reveal_belief"]),
        "B": len(by_type["B_post_reveal_reconstruct_previous_belief"]),
        "C": len(by_type["C_other_agent_false_belief"]),
        "D": len(by_type["D_perspective_taking_prediction"]),
        "hidden_gold": len(hidden_gold),
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
    groups, truth_by_group = select_paper_gold_candidates(
        ledger,
        args.limit,
        pure_hidden_critical_only=args.pure_hidden_critical_only,
        route_hidden_only=args.route_hidden_only,
        max_targets_per_event=args.max_targets_per_event,
    )
    counts = write_diagnostics(args.output_pass_root, ledger, groups, truth_by_group)
    summary = {
        "input_pass_root": args.input_pass_root.as_posix(),
        "output_pass_root": args.output_pass_root.as_posix(),
        "selection_policy": "high_precision_paper_gold_candidates_v1",
        "pure_hidden_critical_only": args.pure_hidden_critical_only,
        "route_hidden_only": args.route_hidden_only,
        "max_targets_per_event": args.max_targets_per_event,
        "limit": args.limit,
        "counts": counts,
        "templates": dict(collections.Counter(group["template"] for group in groups)),
        "note": "Candidates are qwen_checked and still require Codex-human or human verification before leaderboard use.",
    }
    write_json(args.output_pass_root / "README.json", summary)
    (args.output_pass_root / "README.md").write_text(
        "# GooseOmni Decrypto Paper-Gold Candidates\n\n"
        "This pass rebuilds diagnostic candidates with stricter paper-gold filters. "
        "It does not overwrite prior passes and does not promote candidates to human_verified.\n\n"
        f"- probe_groups: {counts['probe_groups']}\n"
        f"- probes: {counts['probes']}\n"
        f"- templates: {summary['templates']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
