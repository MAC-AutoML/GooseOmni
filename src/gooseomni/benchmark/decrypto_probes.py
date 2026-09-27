from .decrypto_ledger import *  # noqa: F401,F403

def load_ledger(root: Path) -> dict[str, list[dict[str, Any]]]:
    ledger = root / "oracle_ledger"
    return {
        "world_events": read_jsonl(ledger / "world_events.jsonl"),
        "claims": read_jsonl(ledger / "claims.jsonl"),
        "phase_events": read_jsonl(ledger / "phase_events.jsonl"),
        "visibility_edges": read_jsonl(ledger / "visibility_edges.jsonl"),
        "belief_memory_snapshots": read_jsonl(ledger / "belief_memory_snapshots.jsonl"),
        "claim_truth_links": read_jsonl(ledger / "claim_truth_links.jsonl"),
        "canonical_event_map": read_jsonl(ledger / "canonical_event_map.jsonl"),
    }


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(edge["event_id"], edge["player_id"]): edge for edge in edges}


def claim_by_id(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {claim["claim_id"]: claim for claim in claims}


def event_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["world_event_id"]: event for event in events}


def ledger_players(ledger: dict[str, list[dict[str, Any]]]) -> list[str]:
    players = {
        str(row["player_id"])
        for row in ledger.get("visibility_edges", [])
        if row.get("player_id")
    }
    players.update(
        str(row["target_player"])
        for row in ledger.get("belief_memory_snapshots", [])
        if row.get("target_player")
    )
    return sorted(players)


def diagnostic_event_allowed(event: dict[str, Any]) -> bool:
    if event.get("gold_source") != "qwen_seed":
        return True
    duration = float(event.get("abs_end_sec", 0.0)) - float(
        event.get("abs_start_sec", 0.0)
    )
    private_types = {
        "body_report",
        "encounter",
        "interaction",
        "kill",
        "movement",
        "role_clue",
        "task",
        "task_ui",
    }
    return (
        0.0 <= duration <= 15.0
        and float(event.get("certainty", 0.0)) >= 0.6
        and event.get("event_type") in private_types
        and len(str(event.get("description", "")).strip()) >= 8
    )


def pick_other_player(
    event: dict[str, Any], target_player: str, players: list[str]
) -> str:
    for player in event.get("source_povs", []):
        if player != target_player:
            return player
    for player in players:
        if player != target_player:
            return player
    return target_player


def make_probe_group(
    idx: int,
    event: dict[str, Any],
    target_player: str,
    template: str,
    related_claims: list[dict[str, Any]],
    quality_review: bool = False,
) -> dict[str, Any]:
    cutoff = float(event["abs_end_sec"]) + 3.0
    available = [claim["claim_id"] for claim in related_claims if target_player in claim.get("heard_by", [])]
    if template == "vote_influence":
        qtype = "trust_update"
    elif template == "private_witness":
        qtype = "hidden_event_awareness"
    elif template == "delayed_public_reveal":
        qtype = "hidden_event_awareness"
    else:
        qtype = "claim_truth_vs_claim_awareness" if related_claims else "hidden_event_awareness"
    families = ["false_belief", "representational_change"]
    if related_claims:
        families.append("claim_verification")
    if template in {"vote_influence", "contradicted_alibi", "private_witness"}:
        families.append("perspective_taking")
    if template == "vote_influence":
        families.append("strategy_communication")
    if template == "delayed_public_reveal":
        families.append("delayed_public_reveal")
    return {
        "probe_group_id": f"{event['game_id']}_pg_{idx:06d}_{target_player}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": cutoff,
        "target_player": target_player,
        "query_variable": {
            "type": qtype,
            "description": f"Whether {target_player} knows or can verify: {event.get('description', '')}",
        },
        "anchor_event_ids": [event["world_event_id"]],
        "related_claim_ids": [claim["claim_id"] for claim in related_claims],
        "hidden_event_ids_for_target": [event["world_event_id"]],
        "available_evidence_ids_for_target": available,
        "selection_reason": f"{event['world_event_id']} is visible to {event.get('source_povs', [])} but hidden from {target_player}; related claims={available}.",
        "diagnostic_families": list(dict.fromkeys(families)),
        "template": template,
        "quality": {
            "visibility_confidence": 0.75,
            "claim_truth_confidence": min([claim.get("certainty", 0.5) for claim in related_claims] + [0.6]),
            "timestamp_confidence": event.get("certainty", 0.5),
            "needs_human_review": quality_review or event.get("needs_human_review", False),
        },
        "needs_human_review": quality_review or event.get("needs_human_review", False),
        "gold_source": "qwen_weak",
    }


def select_probe_groups(ledger: dict[str, list[dict[str, Any]]], limit: int = 240) -> list[dict[str, Any]]:
    events = ledger["world_events"]
    claims = ledger["claims"]
    links = ledger["claim_truth_links"]
    edges = edge_lookup(ledger["visibility_edges"])
    claims_by_id = claim_by_id(claims)
    events_by_id = event_by_id(events)
    players = ledger_players(ledger)
    claims_for_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    truth_for_event: dict[str, str] = {}
    for link in links:
        claim = claims_by_id.get(link["claim_id"])
        if not claim:
            continue
        for event_id in link.get("world_event_ids", []):
            claims_for_event[event_id].append(claim)
            truth_for_event[event_id] = link.get("truth_status_global", "unverified")

    candidates: dict[str, list[tuple[dict[str, Any], str, list[dict[str, Any]], bool]]] = collections.defaultdict(list)
    for event in sorted(events, key=lambda row: (-len(row.get("source_povs", [])), row["abs_start_sec"])):
        if not diagnostic_event_allowed(event):
            continue
        source_povs = set(event.get("source_povs", []))
        if not source_povs or event.get("phase_type") in {"meeting", "final"}:
            continue
        hidden_targets = [
            player
            for player in players
            if player not in source_povs and edges.get((event["world_event_id"], player), {}).get("visibility") == "not_visible"
        ]
        related = claims_for_event.get(event["world_event_id"], [])
        for target in hidden_targets[:3]:
            heard_related = [claim for claim in related if target in claim.get("heard_by", [])]
            if related and not heard_related:
                continue
            template = "contradicted_alibi" if truth_for_event.get(event["world_event_id"]) == "contradicted" else "hidden_event_awareness"
            candidates[template].append((event, target, heard_related, False))

    for event in sorted(events, key=lambda row: row["abs_start_sec"]):
        if not diagnostic_event_allowed(event):
            continue
        if len(event.get("source_povs", [])) != 1:
            continue
        witness = event["source_povs"][0]
        candidates["private_witness"].append((event, witness, claims_for_event.get(event["world_event_id"], []), True))

    for link in links:
        if link.get("truth_status_global") not in {"supported", "contradicted", "ambiguous"}:
            continue
        claim = claims_by_id.get(link["claim_id"])
        if not claim or claim.get("strategic_role") not in {"accusation", "defense", "information_sharing"}:
            continue
        speaker = claim.get("speaker")
        if speaker not in players:
            continue
        for event_id in link.get("world_event_ids", [])[:2]:
            event = events_by_id.get(event_id)
            if not event or event.get("phase_type") in {"meeting", "final"}:
                continue
            for listener in claim.get("heard_by", []):
                if listener == speaker or listener not in players:
                    continue
                listener_edge = edges.get((event_id, listener), {})
                speaker_edge = edges.get((event_id, speaker), {})
                if listener_edge.get("visibility") == speaker_edge.get("visibility"):
                    continue
                candidates["vote_influence"].append((event, listener, [claim], bool(link.get("needs_human_review", False))))
                break

    meeting_claims = [claim for claim in claims if len(claim.get("heard_by", [])) >= 4]
    for event in sorted(events, key=lambda row: row["abs_start_sec"]):
        if not diagnostic_event_allowed(event):
            continue
        source_povs = set(event.get("source_povs", []))
        if not source_povs or len(source_povs) >= len(players) or event.get("phase_type") in {"meeting", "final"}:
            continue
        later_public_claims = [
            claim
            for claim in meeting_claims
            if event["abs_end_sec"] < claim["abs_start_sec"] <= event["abs_end_sec"] + 900
        ][:3]
        if not later_public_claims:
            continue
        for target in players:
            if target in source_povs:
                continue
            if edges.get((event["world_event_id"], target), {}).get("visibility") != "not_visible":
                continue
            candidates["delayed_public_reveal"].append((event, target, later_public_claims, True))
            break

    template_order = ["contradicted_alibi", "hidden_event_awareness", "private_witness", "vote_influence", "delayed_public_reveal"]
    minimums = {
        "contradicted_alibi": max(1, int(limit * 0.10)),
        "hidden_event_awareness": max(1, int(limit * 0.35)),
        "private_witness": max(1, int(limit * 0.15)),
        "vote_influence": max(1, int(limit * 0.15)),
        "delayed_public_reveal": max(1, int(limit * 0.10)),
    }
    groups = []
    idx = 0
    used_keys: set[tuple[str, str, str]] = set()

    def append_candidate(template: str, candidate: tuple[dict[str, Any], str, list[dict[str, Any]], bool]) -> None:
        nonlocal idx
        event, target, related_claims, quality_review = candidate
        key = (template, event["world_event_id"], target)
        if key in used_keys or len(groups) >= limit:
            return
        used_keys.add(key)
        idx += 1
        group = make_probe_group(idx, event, target, template, related_claims, quality_review=quality_review)
        if template == "private_witness":
            group["hidden_event_ids_for_target"] = []
            group["available_evidence_ids_for_target"] = [event["world_event_id"]]
            group["selection_reason"] = f"{target} directly saw {event['world_event_id']} while most other players did not."
        elif template == "vote_influence":
            claim_ids = [claim["claim_id"] for claim in related_claims]
            group["cutoff_abs_sec"] = max(float(claim["abs_end_sec"]) for claim in related_claims) + 2.0
            edge = edges.get((event["world_event_id"], target), {})
            if edge.get("visibility") in {"direct_visual", "direct_audio", "public_ui"}:
                group["hidden_event_ids_for_target"] = []
                group["available_evidence_ids_for_target"] = sorted(
                    set(group.get("available_evidence_ids_for_target", []) + edge.get("evidence_ids", []))
                )
            group["selection_reason"] = f"Strategic claim(s) {claim_ids} may influence listener {target}, whose information state differs from the speaker."
        elif template == "delayed_public_reveal":
            claim_ids = [claim["claim_id"] for claim in related_claims]
            group["selection_reason"] = f"{event['world_event_id']} is private before cutoff and later enters public discussion via claim(s) {claim_ids}."
        groups.append(group)

    for template in template_order:
        for candidate in candidates.get(template, [])[: minimums[template]]:
            append_candidate(template, candidate)
            if len(groups) >= limit:
                break
    for template in template_order:
        for candidate in candidates.get(template, []):
            append_candidate(template, candidate)
            if len(groups) >= limit:
                break
        if len(groups) >= limit:
            break

    return groups[:limit]


def snapshot_for(snapshots: list[dict[str, Any]], player: str, cutoff: float) -> dict[str, Any]:
    candidates = [s for s in snapshots if s["target_player"] == player and s["cutoff_abs_sec"] <= cutoff + 1e-6]
    if not candidates:
        return {"available_evidence_ids": [], "forbidden_event_ids": [], "public_history": [], "private_observations": [], "heard_claims": []}
    return max(candidates, key=lambda row: row["cutoff_abs_sec"])


def compact_context(snapshot: dict[str, Any]) -> str:
    return json.dumps(
        {
            "public_history": snapshot.get("public_history", [])[:12],
            "private_observations": snapshot.get("private_observations", [])[:12],
            "heard_claims": snapshot.get("heard_claims", [])[:12],
            "inferred_beliefs": snapshot.get("inferred_beliefs", [])[:6],
        },
        ensure_ascii=False,
    )


def public_query_form(group: dict[str, Any]) -> str:
    query = group.get("query_variable", {}) if isinstance(group.get("query_variable"), dict) else {}
    return json.dumps(
        {
            "query_type": query.get("type", "unknown"),
            "proposition": query.get("description", ""),
            "target_player": group.get("target_player"),
            "cutoff_abs_sec": group.get("cutoff_abs_sec"),
            "related_claim_ids": group.get("related_claim_ids", []),
            "available_evidence_ids_for_target": group.get("available_evidence_ids_for_target", []),
            "task": "Judge what the target player could know, verify, doubt, or remain uncertain about from target-available evidence only.",
        },
        ensure_ascii=False,
        indent=2,
    )


def speaker_listener_public_model(listener_snapshot: dict[str, Any], listener: str) -> str:
    return json.dumps(
        {
            "listener": listener,
            "shared_public_history": listener_snapshot.get("public_history", [])[:12],
            "public_or_heard_claims": listener_snapshot.get("heard_claims", [])[:12],
            "private_observations_excluded": True,
        },
        ensure_ascii=False,
    )


def expected_schema_for(probe_type: str) -> dict[str, Any]:
    if probe_type == "A_pre_reveal_belief":
        return {"knows_truth": "boolean", "belief_label": "believes_true|believes_false|uncertain|does_not_know|unknown", "likely_belief": "string", "suspicion_update": "increase|decrease|unchanged|unknown", "evidence_ids": "array[string]", "confidence": "number"}
    if probe_type == "B_post_reveal_reconstruct_previous_belief":
        return {"target_knew_truth_at_cutoff": "boolean", "reconstructed_prior_belief": "string", "must_not_use_revealed_truth_as_prior_evidence": "boolean", "evidence_ids_available_at_cutoff": "array[string]", "confidence": "number"}
    if probe_type == "C_other_agent_false_belief":
        return {"other_player": "string", "other_player_knew_truth_at_cutoff": "boolean", "other_player_likely_belief": "string", "evidence_ids_available_to_other_player": "array[string]", "confidence": "number"}
    return {"speaker": "string", "listener": "string", "predicted_listener_trust_update": "increase|decrease|unchanged|unknown", "predicted_listener_next_action": "accuse|vote|defend|ignore|follow|avoid|unknown", "reason_from_speaker_perspective": "string", "confidence": "number"}


def probe_prompt_header(group: dict[str, Any]) -> str:
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
