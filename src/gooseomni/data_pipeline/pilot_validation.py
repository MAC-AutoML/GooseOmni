from __future__ import annotations

from collections import Counter
from typing import Any

from .alignment import validate_alignment
from .tom_trials import TOM_LAYERS, public_trial, validate_trials


def validate_pilot(
    alignment: dict[str, Any],
    episodes: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    states: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    players: list[str],
    enforce_release_gates: bool = True,
    minimum_episodes: int = 5,
    minimum_trials_per_layer: int = 30,
) -> dict[str, Any]:
    issues = validate_alignment(alignment, players)
    issues.extend(validate_trials(trials))
    state_by_id = {str(row["information_state_id"]): row for row in states}
    evidence_ids = {
        str(value) for node in nodes for value in node.get("evidence_asset_ids", [])
    }
    for row in trials:
        if set(row.get("evidence_ids", [])) - evidence_ids:
            issues.append(f"missing trial evidence: {row.get('trial_id')}")
        public = public_trial(row)
        if any(
            key in public
            for key in (
                "gold",
                "future_behavior",
                "oracle_private_facts",
                "review_only_hidden_evidence",
            )
        ):
            issues.append(f"hidden gold leak: {row.get('trial_id')}")
    layer_counts = Counter(str(row.get("tom_layer")) for row in trials)
    target_players = {str(row.get("target_player")) for row in trials}
    coverage_gaps: list[str] = []
    release_profile = minimum_episodes >= 5 and minimum_trials_per_layer >= 30
    if enforce_release_gates:
        if len(episodes) < minimum_episodes:
            coverage_gaps.append(
                f"pilot requires at least {minimum_episodes} valid episodes"
            )
        for layer in TOM_LAYERS:
            if layer_counts[layer] < minimum_trials_per_layer:
                coverage_gaps.append(
                    "pilot layer has fewer than "
                    f"{minimum_trials_per_layer} trials: {layer}"
                )
        missing_targets = set(players) - target_players
        if missing_targets:
            coverage_gaps.append(
                f"players never used as target: {sorted(missing_targets)}"
            )
        if release_profile:
            issues.extend(coverage_gaps)
    release_eligible = (
        len(episodes) >= 5
        and all(layer_counts[layer] >= 30 for layer in TOM_LAYERS)
        and set(players) <= target_players
        and not issues
    )
    return {
        "ok": not issues,
        "issues": issues,
        "episode_count": len(episodes),
        "trajectory_node_count": len(nodes),
        "information_state_count": len(state_by_id),
        "trial_count": len(trials),
        "layer_counts": dict(layer_counts),
        "target_players": sorted(target_players),
        "public_file_leak_hits": [],
        "coverage_gaps": coverage_gaps,
        "release_eligible": release_eligible,
    }
