from __future__ import annotations

import hashlib
from typing import Any

TOM_LAYERS = (
    "perception_access",
    "knowledge_state",
    "belief_and_false_belief",
    "intent_and_action_prediction",
    "stance_trust_and_agreement",
)
CERTAINTY_LEVELS = {"explicit", "behaviorally_supported", "inferred_distribution", "unknown"}
LAYER_EVENT_TYPES = {
    "perception_access": {
        "movement", "encounter", "task", "task_ui", "interaction", "meeting",
        "vote", "kill", "body_report", "action_outcome", "utterance", "claim",
        "reaction", "accusation", "defense", "audio_cue",
    },
    "knowledge_state": {
        "encounter", "meeting", "vote", "kill", "body_report", "action_outcome",
        "utterance", "claim", "accusation", "defense",
    },
    "belief_and_false_belief": {"utterance", "claim", "accusation", "defense"},
    "intent_and_action_prediction": {
        "movement", "task", "task_ui", "interaction", "vote", "action_outcome",
    },
    "stance_trust_and_agreement": {"reaction", "accusation", "defense", "vote"},
}
PLAYER_CENTRIC_EVENT_TYPES = {"movement", "task", "task_ui", "interaction"}


def subject_only_evidence(
    evidence_ids: list[str], target_state: dict[str, Any]
) -> list[str]:
    """Keep evidence available to the subject but unavailable to the target."""
    target_available = {
        str(row) for row in target_state.get("available_evidence_ids", [])
    }
    return sorted(set(evidence_ids) - target_available)


def evidence_for_layer(
    state: dict[str, Any],
    nodes: list[dict[str, Any]],
    target_player: str,
    layer: str,
) -> list[str]:
    """Return only cutoff-safe, subject-accessible evidence relevant to one ToM layer."""
    if layer not in TOM_LAYERS:
        raise ValueError(f"unknown ToM layer: {layer}")
    available = {str(row) for row in state.get("available_evidence_ids", [])}
    cutoff = float(state["cutoff_abs_sec"])
    evidence: set[str] = set()
    for node in nodes:
        if float(node.get("abs_start_sec", 0.0)) > cutoff:
            continue
        node_evidence = {
            str(row) for row in node.get("evidence_asset_ids", [])
        } & available
        if not node_evidence or node.get("event_type") not in LAYER_EVENT_TYPES[layer]:
            continue
        related_players = {
            str(node.get("speaker_id", "")),
            *(str(row) for row in node.get("visible_players", [])),
            *(str(row) for row in node.get("mentioned_players", [])),
        }
        if node.get("event_type") in PLAYER_CENTRIC_EVENT_TYPES:
            related_players.add(str(node.get("player_id", "")))
        if target_player in related_players:
            evidence.update(node_evidence)
    return sorted(evidence)


def structured_gold(
    layer: str,
    state: dict[str, Any],
    target_player: str,
    nodes: list[dict[str, Any]],
    evidence_ids: list[str],
) -> dict[str, Any]:
    """Build a scorer-ready gold value without promoting speculation to fact."""
    evidence = set(evidence_ids)
    relevant = [
        node
        for node in nodes
        if evidence & {str(value) for value in node.get("evidence_asset_ids", [])}
    ]
    base = {"subject_only_evidence_ids": sorted(evidence)}
    event_types = sorted({str(node.get("event_type", "unknown")) for node in relevant})
    if layer == "perception_access":
        access = "direct_visual" if any(
            target_player in {str(value) for value in node.get("visible_players", [])}
            for node in relevant
        ) else "public_ui"
        return {
            **base,
            "certainty": "behaviorally_supported",
            "answer": {
                "access": access,
                "subject_player": str(state["player_id"]),
                "target_player": target_player,
                "observed_event_types": event_types,
            },
        }
    if layer == "knowledge_state":
        return {
            **base,
            "certainty": "behaviorally_supported",
            "answer": {
                "knowledge_status": "known_from_direct_observation",
                "target_player": target_player,
                "observed_event_types": event_types,
            },
        }
    explicit = next(
        (
            node
            for node in relevant
            if node.get("speaker_id") == target_player and node.get("utterance")
        ),
        None,
    )
    if layer in {"belief_and_false_belief", "stance_trust_and_agreement"} and explicit:
        return {
            **base,
            "certainty": "explicit",
            "answer": {
                "speaker_id": target_player,
                "utterance": str(explicit["utterance"]),
                "event_type": str(explicit.get("event_type", "utterance")),
            },
        }
    if layer == "intent_and_action_prediction":
        cutoff = float(state["cutoff_abs_sec"])
        future = next(
            (
                node
                for node in sorted(nodes, key=lambda row: float(row["abs_start_sec"]))
                if float(node["abs_start_sec"]) > cutoff
                and target_player
                in {
                    str(node.get("player_id", "")),
                    str(node.get("speaker_id", "")),
                    *(str(value) for value in node.get("visible_players", [])),
                }
            ),
            None,
        )
        if future is not None:
            return {
                **base,
                "certainty": "behaviorally_supported",
                "answer": {
                    "next_observed_event_type": str(future.get("event_type", "unknown")),
                    "location": str(future.get("location") or "unknown"),
                },
                "future_evidence_ids": sorted(
                    str(value) for value in future.get("evidence_asset_ids", [])
                ),
            }
    return {**base, "certainty": "unknown"}


def split_for_group(group_id: str) -> str:
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if value < 7 else "dev" if value < 9 else "test"


def build_trial_group(
    state: dict[str, Any],
    target_player: str,
    query_variable: str,
    information_gap: str,
    layer: str,
    evidence_ids: list[str],
    gold: dict[str, Any],
    query: str | None = None,
    review_only_hidden_evidence: list[str] | None = None,
) -> list[dict[str, Any]]:
    if layer not in TOM_LAYERS:
        raise ValueError(f"unknown ToM layer: {layer}")
    available = set(state.get("available_evidence_ids", []))
    if set(evidence_ids) - available:
        raise ValueError("trial cites evidence outside the subject information state")
    certainty = str(gold.get("certainty", "unknown"))
    if certainty not in CERTAINTY_LEVELS:
        raise ValueError("invalid ToM gold certainty")
    if layer in {"intent_and_action_prediction", "stance_trust_and_agreement"}:
        if certainty == "inferred_distribution" and "answer" in gold:
            raise ValueError("inferred distributions cannot become a single factual answer")
    subject = str(state["player_id"])
    episode_id = str(state["episode_id"])
    cutoff = float(state["cutoff_abs_sec"])
    group_id = f"{episode_id}:{cutoff:.3f}:{subject}:{target_player}:{query_variable}"
    split = split_for_group(group_id)
    return [
        {
            "trial_id": f"{group_id}:{variant}",
            "probe_group_id": group_id,
            "variant": variant,
            "episode_id": episode_id,
            "cutoff_abs_sec": cutoff,
            "subject_player": subject,
            "target_player": target_player,
            "query_variable": query_variable,
            "information_gap": information_gap,
            "query": query or f"Infer {query_variable} for {target_player}.",
            "tom_layer": layer,
            "split": split,
            "evidence_ids": evidence_ids,
            "gold": gold,
            "gold_source": "model_verified",
            "review_only_hidden_evidence": sorted(
                set(review_only_hidden_evidence or [])
            ),
        }
        for variant in ("A", "B", "C", "D")
    ]


def public_trial(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = {
        "gold",
        "future_behavior",
        "oracle_private_facts",
        "hidden_gold",
        "review_only_hidden_evidence",
    }
    return {key: value for key, value in row.items() if key not in forbidden}


def validate_trials(rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    splits: dict[str, set[str]] = {}
    layer_counts = dict.fromkeys(TOM_LAYERS, 0)
    for row in rows:
        if row.get("gold_source") != "model_verified":
            issues.append(f"invalid gold source: {row.get('trial_id')}")
        layer = str(row.get("tom_layer"))
        if layer not in layer_counts:
            issues.append(f"invalid ToM layer: {row.get('trial_id')}")
        else:
            layer_counts[layer] += 1
        splits.setdefault(str(row.get("probe_group_id")), set()).add(str(row.get("split")))
        if float(row.get("cutoff_abs_sec", -1)) < 0:
            issues.append(f"invalid cutoff: {row.get('trial_id')}")
        gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
        certainty = str(gold.get("certainty", "unknown"))
        if certainty == "unknown":
            issues.append(f"unresolved ToM gold: {row.get('trial_id')}")
        if certainty == "inferred_distribution" and not gold.get("distribution"):
            issues.append(f"empty ToM gold distribution: {row.get('trial_id')}")
    issues.extend(
        f"probe group crosses splits: {group_id}"
        for group_id, values in splits.items()
        if len(values) != 1
    )
    return issues
