from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    PLAYERS,
)

CRITICAL_EVENT_TYPES = {"death", "player_death", "combat"}
CLAIM_BACKED_EVENT_TYPES = {
    "death",
    "player_death",
    "combat",
    "movement",
    "player_movement",
}
MIN_CLAIM_CONTENT_CHARS = 8
MIN_POST_MEETING_GAMEPLAY_LOCAL_SEC = 15.0
EXCLUDED_EVENT_TYPES = {
    "discussion",
    "meeting",
    "phase_transition",
    "gameplay_phase_transition",
    "scene_transition",
    "transition",
    "voting",
    "vote",
    "vote_result",
    "voting_result",
    "result",
    "game_result",
    "game_over",
    "game_start",
    "gameplay_start",
    "task",
    "task_completion",
    "unknown",
}
GOOD_CLAIM_TYPES = {"location", "defense", "accusation", "sighting"}
BAD_DESCRIPTION_TOKENS = {
    "讨论",
    "投票",
    "阶段",
    "界面",
    "切换",
    "结果",
    "大厅界面",
    "观战",
    "lobby",
    "meeting phase",
    "voting",
    "没有看到",
    "没看到",
    "no body",
    "no kill",
    "no killing",
}
CRITICAL_DESCRIPTION_TOKENS = {
    "击杀",
    "被杀",
    "死亡",
    "倒地",
    "血迹",
    "尸体",
    "攻击",
    "killed",
    "death",
    "body",
    "blood",
    "attack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a high-precision GooseOmni Decrypto candidate pass for paper-grade human review."
    )
    parser.add_argument("--input-pass-root", type=Path, required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--pure-hidden-critical-only", action="store_true")
    parser.add_argument("--route-hidden-only", action="store_true")
    parser.add_argument("--max-targets-per-event", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(edge["event_id"], edge["player_id"]): edge for edge in edges}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["world_event_id"]: event for event in events}


def claim_by_id(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {claim["claim_id"]: claim for claim in claims}


def text(row: dict[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def has_bad_description(event: dict[str, Any]) -> bool:
    description = text(event, "description").lower()
    return any(token in description for token in BAD_DESCRIPTION_TOKENS)


def has_critical_description(event: dict[str, Any]) -> bool:
    description = text(event, "description").lower()
    return any(token in description for token in CRITICAL_DESCRIPTION_TOKENS)


def phase_local_start_sec(event: dict[str, Any]) -> float | None:
    phase_ids = event.get("source_segment_ids") or []
    if not phase_ids:
        return None
    phase_id = str(phase_ids[0])
    parts = phase_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return float(event.get("abs_start_sec", 0.0) or 0.0) - float(int(parts[1]))


def is_after_gameplay_phase_warmup(event: dict[str, Any]) -> bool:
    local_start = phase_local_start_sec(event)
    return local_start is None or local_start >= MIN_POST_MEETING_GAMEPLAY_LOCAL_SEC


def is_route_event(event: dict[str, Any]) -> bool:
    if event.get("phase_type") != "gameplay":
        return False
    if not is_after_gameplay_phase_warmup(event):
        return False
    if event.get("event_type") not in {
        "movement",
        "player_movement",
        "task",
        "task_completion",
        "gameplay",
    }:
        return False
    if len(event.get("source_povs", [])) != 1:
        return False
    if event.get("needs_human_review"):
        return False
    if float(event.get("certainty", 0.0) or 0.0) < 0.8:
        return False
    description = text(event, "description")
    if has_bad_description(event) or has_critical_description(event):
        return False
    if any(
        token in description
        for token in [
            "开始",
            "寻找目标",
            "周围有其他角色",
            "地图上",
            "声称",
            "击中",
            "倒下",
        ]
    ):
        return False
    if not any(player in description for player in PLAYERS):
        return False
    return any(
        location in description
        for location in [
            "街道",
            "下水道",
            "学校",
            "实验室",
            "游戏厅",
            "警署",
            "餐厅",
            "电站",
            "公园",
            "大厅",
        ]
    )


def normalized_description(event: dict[str, Any]) -> str:
    value = text(event, "description").lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[。.,，'\"“”‘’\\[\\]（）()\\s]+", "", value)
    for token in [
        "看到了",
        "看到",
        "角色",
        "的",
        "了",
        "在街道上",
        "周围有血迹",
        "尸体",
        "倒地",
    ]:
        value = value.replace(token, "")
    return value[:40]


def description_mentions_player(event: dict[str, Any], player: str) -> bool:
    description = text(event, "description")
    return player in description


def is_clean_event(event: dict[str, Any]) -> bool:
    if event.get("phase_type") in {"meeting", "final"}:
        return False
    if event.get("phase_type") == "gameplay" and not is_after_gameplay_phase_warmup(
        event
    ):
        return False
    if event.get("event_type") in EXCLUDED_EVENT_TYPES:
        return False
    if has_bad_description(event):
        return False
    if float(event.get("certainty", 0.0) or 0.0) < 0.65:
        return False
    if event.get("needs_human_review"):
        return False
    source_povs = event.get("source_povs", [])
    return bool(source_povs) and all(player in PLAYERS for player in source_povs)


def direct_players(
    event: dict[str, Any], edges: dict[tuple[str, str], dict[str, Any]]
) -> list[str]:
    event_id = event["world_event_id"]
    return [
        player
        for player in PLAYERS
        if edges.get((event_id, player), {}).get("visibility") == "direct_visual"
    ]


def hidden_players(
    event: dict[str, Any], edges: dict[tuple[str, str], dict[str, Any]]
) -> list[str]:
    event_id = event["world_event_id"]
    return [
        player
        for player in PLAYERS
        if edges.get((event_id, player), {}).get("visibility") == "not_visible"
    ]


def heard_by_target_before_cutoff(
    claim: dict[str, Any], target: str, cutoff: float
) -> bool:
    return target in claim.get("heard_by", []) and float(claim["abs_end_sec"]) <= cutoff


def claim_is_usable(claim: dict[str, Any]) -> bool:
    if any(
        str(seg).endswith("_final_003530_003540") or "_final_" in str(seg)
        for seg in claim.get("source_segment_ids", [])
    ):
        return False
    if claim.get("speaker") not in PLAYERS:
        return False
    if claim.get("claim_type") not in GOOD_CLAIM_TYPES:
        return False
    if float(claim.get("certainty", 0.0) or 0.0) < 0.55:
        return False
    content = text(claim, "content")
    if any(token in content for token in ["没用", "没有什么作用", "开玩笑", "哈哈"]):
        return False
    return len(content) >= MIN_CLAIM_CONTENT_CHARS


def claim_speaker_can_ground_anchor(
    event: dict[str, Any], claim: dict[str, Any], direct: list[str]
) -> bool:
    speaker = str(claim.get("speaker") or "")
    if speaker in direct:
        return True
    if speaker in {str(actor) for actor in event.get("actors", [])}:
        description = text(event, "description")
        if any(
            token in description
            for token in [f"{speaker}的角色倒地", f"{speaker} 被杀", f"{speaker}的尸体"]
        ):
            return False
        return True
    return False


def build_links_by_event(
    links: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for link in links:
        for event_id in link.get("world_event_ids", []):
            by_event[str(event_id)].append(link)
    return by_event


def group_score(
    group: dict[str, Any],
    event: dict[str, Any],
    claim: dict[str, Any] | None,
    truth_status: str,
) -> tuple[int, str]:
    score = 0
    event_type = str(event.get("event_type"))
    if event_type in CRITICAL_EVENT_TYPES:
        score += 80
    elif has_critical_description(event):
        score += 70
    elif claim is not None:
        score += 45
    if truth_status == "contradicted":
        score += 30
    elif truth_status == "supported":
        score += 18
    elif truth_status == "ambiguous":
        score += 10
    if claim is not None:
        if claim.get("claim_type") in {"location", "defense", "sighting"}:
            score += 14
        score += min(10, int(float(claim.get("certainty", 0.0) or 0.0) * 10))
    score += min(10, int(float(event.get("certainty", 0.0) or 0.0) * 10))
    return (-score, str(group["probe_group_id"]))


def make_group(
    idx: int,
    event: dict[str, Any],
    target: str,
    template: str,
    related_claims: list[dict[str, Any]],
    cutoff: float,
    truth_status: str,
) -> dict[str, Any]:
    claim_ids = [claim["claim_id"] for claim in related_claims]
    if template == "contradicted_alibi":
        query_type = "claim_truth_vs_claim_awareness"
    elif template == "claim_backed_hidden_event":
        query_type = "claim_truth_vs_claim_awareness"
    elif template == "route_hidden_event":
        query_type = "route_belief"
    else:
        query_type = "hidden_event_awareness"
    return {
        "probe_group_id": f"g001_pgold_{idx:06d}_{target}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": cutoff,
        "target_player": target,
        "query_variable": {
            "type": query_type,
            "description": f"At cutoff, what can {target} know about: {event.get('description', '')}",
        },
        "anchor_event_ids": [event["world_event_id"]],
        "related_claim_ids": claim_ids,
        "hidden_event_ids_for_target": [event["world_event_id"]],
        "available_evidence_ids_for_target": claim_ids,
        "selection_reason": (
            f"Paper-gold candidate: {event['world_event_id']} is directly visible to {event.get('source_povs', [])}, "
            f"not visible to {target}, and target-available claims={claim_ids} before cutoff={cutoff}."
        ),
        "diagnostic_families": [
            family
            for family in dict.fromkeys(
                [
                    "false_belief",
                    "representational_change",
                    "perspective_taking",
                    "claim_verification" if related_claims else "",
                ]
            )
            if family
        ],
        "template": template,
        "quality": {
            "visibility_confidence": 0.85,
            "claim_truth_confidence": 0.8
            if truth_status in {"supported", "contradicted"}
            else 0.65,
            "timestamp_confidence": float(event.get("certainty", 0.75) or 0.75),
            "needs_human_review": True,
            "paper_gold_candidate": True,
        },
        "needs_human_review": True,
        "gold_source": "qwen_checked",
    }
