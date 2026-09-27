from pathlib import Path

from gooseomni.benchmark.decrypto_canonical import (
    PLAYERS,
    build_belief_snapshots,
    build_candidate_claims,
    build_candidate_events,
    build_claim_truth_links,
    build_phase_events,
    build_visibility_edges,
    canonicalize_claims,
    canonicalize_events,
    load_gold_annotations,
    write_jsonl,
)


def build_oracle_ledger(
    release_root: Path, output_root: Path, game_id: str = "g001"
) -> dict[str, int]:
    annotations = load_gold_annotations(release_root, game_id)
    candidate_events = build_candidate_events(annotations)
    candidate_claims = build_candidate_claims(annotations)
    world_events, event_maps, _ = canonicalize_events(candidate_events)
    claims, _ = canonicalize_claims(candidate_claims)
    phase_events = build_phase_events(annotations)
    visibility_edges = build_visibility_edges(world_events, claims)
    claim_truth_links = build_claim_truth_links(claims, world_events, visibility_edges)

    cutoffs = []
    for link in claim_truth_links[:200]:
        claim = next((c for c in claims if c["claim_id"] == link["claim_id"]), None)
        if not claim:
            continue
        for player in PLAYERS:
            cutoffs.append((player, float(claim["abs_end_sec"]) + 1.0))
    for event in world_events[:200]:
        for player in PLAYERS:
            if player not in event.get("source_povs", []):
                cutoffs.append((player, float(event["abs_end_sec"]) + 3.0))
    seen_cutoffs = []
    seen = set()
    for player, cutoff in sorted(cutoffs, key=lambda x: (x[1], x[0])):
        key = (player, round(cutoff, 1))
        if key not in seen:
            seen.add(key)
            seen_cutoffs.append((player, cutoff))
    snapshots = build_belief_snapshots(
        world_events, claims, visibility_edges, seen_cutoffs[:360]
    )

    ledger = output_root / "oracle_ledger"
    write_jsonl(ledger / "world_events.jsonl", world_events)
    write_jsonl(ledger / "claims.jsonl", claims)
    write_jsonl(ledger / "phase_events.jsonl", phase_events)
    write_jsonl(ledger / "visibility_edges.jsonl", visibility_edges)
    write_jsonl(ledger / "belief_memory_snapshots.jsonl", snapshots)
    write_jsonl(ledger / "claim_truth_links.jsonl", claim_truth_links)
    write_jsonl(ledger / "canonical_event_map.jsonl", event_maps)
    return {
        "world_events": len(world_events),
        "claims": len(claims),
        "phase_events": len(phase_events),
        "visibility_edges": len(visibility_edges),
        "belief_memory_snapshots": len(snapshots),
        "claim_truth_links": len(claim_truth_links),
        "canonical_event_map": len(event_maps),
    }
