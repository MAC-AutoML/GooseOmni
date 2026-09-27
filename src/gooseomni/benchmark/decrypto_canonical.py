from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
LOW_CERTAINTY = 0.55


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def certainty(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().lower()
    if text in {"high", "高", "很高"}:
        return 0.85
    if text in {"medium", "中", "中等"}:
        return 0.6
    if text in {"low", "低"}:
        return 0.35
    return 0.5


def normalized_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", "", text)


def phase_id_from_path(path: Path) -> str:
    return path.parent.name


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def event_key(event: dict[str, Any]) -> tuple[str, str, str]:
    actor = ",".join(sorted(str(x) for x in event.get("actors", []))) or "unknown"
    return (str(event.get("event_type", "other")), actor, str(event.get("location", "unknown")))


def should_merge_event(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if event_key(a) != event_key(b):
        return False
    if interval_iou(float(a["abs_start_sec"]), float(a["abs_end_sec"]), float(b["abs_start_sec"]), float(b["abs_end_sec"])) > 0.5:
        return True
    return abs(float(a["abs_start_sec"]) - float(b["abs_start_sec"])) <= 3.0 and abs(float(a["abs_end_sec"]) - float(b["abs_end_sec"])) <= 3.0


def claim_key(claim: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(claim.get("speaker", "unknown")),
        str(claim.get("claim_type", "other")),
        normalized_text(claim.get("normalized_content") or claim.get("content"))[:80],
    )


def should_merge_claim(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if claim_key(a) != claim_key(b):
        return False
    return abs(float(a["abs_start_sec"]) - float(b["abs_start_sec"])) <= 3.0


def infer_claim_type(text: str) -> str:
    if any(token in text for token in ["身份", "角色", "刺客", "鸭", "鹅", "警长", "拆弹", "验尸"]):
        return "role"
    if any(token in text for token in ["在", "上面", "下面", "右边", "左边", "大厅", "下水道", "街道", "位置"]):
        return "location"
    if any(token in text for token in ["杀", "刀", "打", "投", "票", "放逐"]):
        return "accusation"
    if any(token in text for token in ["我没", "不是我", "一直", "没有", "别投"]):
        return "defense"
    if any(token in text for token in ["看到", "看见", "目击"]):
        return "sighting"
    return "other"


def infer_strategic_role(claim_type: str, text: str) -> str:
    if claim_type in {"defense", "location"} or any(token in text for token in ["我没", "不是我", "一直", "没有"]):
        return "defense"
    if claim_type == "accusation" or any(token in text for token in ["投", "杀", "刀"]):
        return "accusation"
    if claim_type == "sighting":
        return "information_sharing"
    return "other"


def mentioned_players(text: str) -> list[str]:
    return [player for player in PLAYERS if player in text]


def load_gold_annotations(release_root: Path, game_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((release_root / "gold_annotations" / game_id).glob("*/*.json")):
        row = read_json(path)
        row["_path"] = path.as_posix()
        rows.append(row)
    return rows


def build_candidate_events(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    counter = 0
    for ann in annotations:
        for obs in ann.get("observations", []):
            counter += 1
            source_pov = ann["player_id"]
            actor = obs.get("actor")
            actors = [actor] if actor in PLAYERS else []
            source_types = set(obs.get("source_types", []))
            events.append(
                {
                    "local_event_id": f"{ann['phase_id']}_{source_pov}_le_{counter:06d}",
                    "game_id": ann["game_id"],
                    "source_segment_ids": [ann["phase_id"]],
                    "source_povs": [source_pov],
                    "abs_start_sec": float(obs["abs_start_sec"]),
                    "abs_end_sec": float(obs["abs_end_sec"]),
                    "phase_type": ann["phase_type"],
                    "event_type": obs.get("event_type", "other"),
                    "actors": actors,
                    "patients": [],
                    "location": obs.get("location", "unknown"),
                    "description": obs.get("description", ""),
                    "direct_visual_evidence": [obs.get("evidence", "")] if "direct_visual_observation" in source_types else [],
                    "direct_audio_evidence": [obs.get("evidence", "")] if "speech_claim" in source_types else [],
                    "public_evidence": [obs.get("public_result_text", "")] if "public_result" in source_types else [],
                    "inferred_fields": [obs.get("inferred_belief_text", "")] if obs.get("inferred_belief_text") else [],
                    "certainty": certainty(obs.get("certainty")),
                    "needs_human_review": bool(obs.get("needs_human_review", False)),
                }
            )
    return events


def build_candidate_claims(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    counter = 0
    for ann in annotations:
        for utt in ann.get("utterances", []):
            text = next(
                (
                    value.strip()
                    for value in [utt.get("claim_text"), utt.get("transcript"), utt.get("text"), utt.get("content")]
                    if isinstance(value, str) and value.strip()
                ),
                "",
            )
            if not normalized_text(text):
                continue
            speaker = utt.get("speaker", "unknown")
            if speaker not in PLAYERS:
                speaker = "unknown"
            claim_type = infer_claim_type(text)
            counter += 1
            claims.append(
                {
                    "local_claim_id": f"{ann['phase_id']}_{ann['player_id']}_lc_{counter:06d}",
                    "game_id": ann["game_id"],
                    "source_segment_ids": [ann["phase_id"]],
                    "speaker": speaker,
                    "heard_by": PLAYERS[:] if ann["phase_type"] in {"meeting", "final"} else [ann["player_id"]],
                    "abs_start_sec": float(utt["abs_start_sec"]),
                    "abs_end_sec": float(utt["abs_end_sec"]),
                    "claim_type": claim_type,
                    "content": next((value.strip() for value in [utt.get("transcript"), text] if isinstance(value, str) and value.strip()), text),
                    "normalized_content": text,
                    "time_referred": {"type": "relative_recent" if any(t in text for t in ["刚", "刚才", "上一轮"]) else "unknown"},
                    "target_entities": mentioned_players(text),
                    "related_event_ids": [],
                    "strategic_role": infer_strategic_role(claim_type, text),
                    "certainty": certainty(utt.get("certainty")),
                    "needs_human_review": bool(utt.get("needs_human_review", False)) or speaker == "unknown",
                }
            )
    return claims


def canonicalize_events(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    canonical: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    local_to_world: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda row: (row["abs_start_sec"], row["event_type"])):
        match = next((event for event in canonical if should_merge_event(event, cand)), None)
        if match is None:
            world_id = f"ge_{len(canonical) + 1:06d}"
            match = dict(cand)
            match["world_event_id"] = world_id
            match.pop("local_event_id", None)
            match["_duplicates"] = [cand["local_event_id"]]
            canonical.append(match)
        else:
            match["abs_start_sec"] = min(float(match["abs_start_sec"]), float(cand["abs_start_sec"]))
            match["abs_end_sec"] = max(float(match["abs_end_sec"]), float(cand["abs_end_sec"]))
            match["source_segment_ids"] = sorted(set(match.get("source_segment_ids", []) + cand.get("source_segment_ids", [])))
            match["source_povs"] = sorted(set(match.get("source_povs", []) + cand.get("source_povs", [])))
            for key in ["direct_visual_evidence", "direct_audio_evidence", "public_evidence", "inferred_fields"]:
                match[key] = list(dict.fromkeys(match.get(key, []) + cand.get(key, [])))
            match["certainty"] = max(float(match.get("certainty", 0.0)), float(cand.get("certainty", 0.0)))
            match["needs_human_review"] = bool(match.get("needs_human_review", False) or cand.get("needs_human_review", False))
            match.setdefault("_duplicates", []).append(cand["local_event_id"])
        local_to_world[cand["local_event_id"]] = match["world_event_id"]

    for event in canonical:
        duplicates = event.pop("_duplicates", [])
        maps.append(
            {
                "canonical_event_id": event["world_event_id"],
                "duplicate_local_event_ids": duplicates,
                "abs_time_cluster": [event["abs_start_sec"], event["abs_end_sec"]],
                "canonical_source": duplicates[0] if duplicates else event["world_event_id"],
                "merge_reason": "merged by event_type/actor/location and temporal overlap or close absolute times",
            }
        )
    return canonical, maps, local_to_world


def canonicalize_claims(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    canonical: list[dict[str, Any]] = []
    local_to_claim: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda row: (row["abs_start_sec"], row["speaker"], row["claim_type"])):
        match = next((claim for claim in canonical if should_merge_claim(claim, cand)), None)
        if match is None:
            claim_id = f"claim_{len(canonical) + 1:06d}"
            match = dict(cand)
            match["claim_id"] = claim_id
            match.pop("local_claim_id", None)
            match["_duplicates"] = [cand["local_claim_id"]]
            canonical.append(match)
        else:
            match["abs_start_sec"] = min(float(match["abs_start_sec"]), float(cand["abs_start_sec"]))
            match["abs_end_sec"] = max(float(match["abs_end_sec"]), float(cand["abs_end_sec"]))
            match["source_segment_ids"] = sorted(set(match.get("source_segment_ids", []) + cand.get("source_segment_ids", [])))
            match["heard_by"] = sorted(set(match.get("heard_by", []) + cand.get("heard_by", [])))
            match["target_entities"] = sorted(set(match.get("target_entities", []) + cand.get("target_entities", [])))
            match["certainty"] = max(float(match.get("certainty", 0.0)), float(cand.get("certainty", 0.0)))
            match["needs_human_review"] = bool(match.get("needs_human_review", False) or cand.get("needs_human_review", False))
            match.setdefault("_duplicates", []).append(cand["local_claim_id"])
        local_to_claim[cand["local_claim_id"]] = match["claim_id"]
    for claim in canonical:
        claim.pop("_duplicates", None)
    return canonical, local_to_claim


def build_phase_events(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for ann in annotations:
        if ann["phase_id"] not in seen:
            seen[ann["phase_id"]] = {
                "phase_event_id": f"phase_{ann['phase_index_global']:06d}",
                "game_id": ann["game_id"],
                "episode_id": ann["episode_id"],
                "phase_id": ann["phase_id"],
                "phase_type": ann["phase_type"],
                "phase_order_label_zh": ann["phase_order_label_zh"],
                "abs_start_sec": ann["aligned_start_sec"],
                "abs_end_sec": ann["aligned_end_sec"],
                "previous_phase_id": ann.get("previous_phase_id"),
                "next_phase_id": ann.get("next_phase_id"),
            }
    return sorted(seen.values(), key=lambda row: row["abs_start_sec"])


def build_visibility_edges(events: list[dict[str, Any]], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for event in events:
        source_povs = set(event.get("source_povs", []))
        is_public = event.get("phase_type") in {"meeting", "final"} or bool(event.get("public_evidence"))
        for player in PLAYERS:
            if player in source_povs:
                visibility = "direct_visual"
                evidence_ids = [event["world_event_id"]]
                explanation = f"{player} POV directly observed this event."
                confidence = event.get("certainty", 0.5)
            elif is_public:
                visibility = "public_ui"
                evidence_ids = [event["world_event_id"]]
                explanation = "The event is public in meeting/final UI or public evidence."
                confidence = min(0.8, event.get("certainty", 0.5))
            else:
                visibility = "not_visible"
                evidence_ids = []
                explanation = f"No evidence that {player} saw or heard this event before cutoff."
                confidence = 0.75
            edges.append(
                {
                    "edge_id": f"vis_{event['world_event_id']}_{player}",
                    "game_id": event["game_id"],
                    "event_id": event["world_event_id"],
                    "player_id": player,
                    "cutoff_abs_sec": event["abs_end_sec"],
                    "visibility": visibility,
                    "evidence_ids": evidence_ids,
                    "explanation": explanation,
                    "confidence": confidence,
                }
            )
    for claim in claims:
        for player in PLAYERS:
            edges.append(
                {
                    "edge_id": f"vis_{claim['claim_id']}_{player}",
                    "game_id": claim["game_id"],
                    "event_id": claim["claim_id"],
                    "player_id": player,
                    "cutoff_abs_sec": claim["abs_end_sec"],
                    "visibility": "heard_claim" if player in claim.get("heard_by", []) else "not_visible",
                    "evidence_ids": [claim["claim_id"]] if player in claim.get("heard_by", []) else [],
                    "explanation": "Player heard the claim." if player in claim.get("heard_by", []) else "No evidence the player heard this claim.",
                    "confidence": claim.get("certainty", 0.5),
                }
            )
    return edges


def event_claim_similarity(event: dict[str, Any], claim: dict[str, Any]) -> bool:
    text = normalized_text(" ".join([claim.get("content", ""), claim.get("normalized_content", "")]))
    desc = normalized_text(event.get("description", ""))
    if claim.get("speaker") in event.get("actors", []):
        return True
    if any(player in event.get("actors", []) for player in claim.get("target_entities", [])):
        return True
    if event.get("location") and normalized_text(event.get("location")) in text:
        return True
    if any(token in text and token in desc for token in ["杀", "刀", "尸体", "投", "右", "左", "上", "下", "任务"]):
        return True
    return False


def build_claim_truth_links(claims: list[dict[str, Any]], events: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event_player = {(e["event_id"], e["player_id"]): e for e in edges}
    links = []
    for claim in claims:
        nearby = [
            event
            for event in events
            if event["abs_start_sec"] <= claim["abs_end_sec"] + 180
            and event["abs_end_sec"] >= claim["abs_start_sec"] - 180
            and event_claim_similarity(event, claim)
        ][:4]
        if not nearby:
            continue
        truth = "unverified"
        text = normalized_text(claim.get("content"))
        if claim.get("claim_type") in {"defense", "location"} and any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "ambiguous"
        if any(token in text for token in ["没有", "没去", "不是", "一直"]) and any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "contradicted"
        elif any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "supported"
        local_awareness = {}
        for player in PLAYERS:
            player_edges = [by_event_player.get((event["world_event_id"], player), {}) for event in nearby]
            if any(edge.get("visibility") == "direct_visual" for edge in player_edges):
                local_awareness[player] = "has_contradictory_visual_evidence" if truth == "contradicted" else "has_supporting_visual_evidence"
            elif player in claim.get("heard_by", []):
                local_awareness[player] = "not_enough_information"
            else:
                local_awareness[player] = "unknown"
        links.append(
            {
                "claim_truth_link_id": f"ctl_{claim['claim_id']}_{nearby[0]['world_event_id']}",
                "claim_id": claim["claim_id"],
                "world_event_ids": [event["world_event_id"] for event in nearby],
                "truth_status_global": truth,
                "local_awareness_by_player": local_awareness,
                "explanation": "Heuristic link between claim and nearby canonical world events sharing speaker/actor/location/action cues.",
                "confidence": min(claim.get("certainty", 0.5), max(event.get("certainty", 0.5) for event in nearby)),
                "needs_human_review": truth in {"ambiguous", "unverified"},
            }
        )
    return links


def build_belief_snapshots(
    events: list[dict[str, Any]], claims: list[dict[str, Any]], edges: list[dict[str, Any]], probe_cutoffs: list[tuple[str, float]]
) -> list[dict[str, Any]]:
    edge_by_event_player = collections.defaultdict(list)
    for edge in edges:
        edge_by_event_player[(edge["event_id"], edge["player_id"])].append(edge)
    snapshots = []
    for target_player, cutoff in probe_cutoffs:
        public_history = []
        private_observations = []
        heard_claims = []
        hidden = []
        available_ids = []
        forbidden_ids = []
        for event in events:
            if event["abs_start_sec"] > cutoff:
                continue
            player_edges = edge_by_event_player.get((event["world_event_id"], target_player), [])
            visibility = player_edges[0]["visibility"] if player_edges else "unknown"
            if visibility in {"direct_visual", "direct_audio", "public_ui"}:
                evidence_id = f"obs_{target_player}_{event['world_event_id']}"
                private_observations.append(
                    {
                        "evidence_id": evidence_id,
                        "world_event_id": event["world_event_id"],
                        "visibility": visibility,
                        "content": event.get("description", ""),
                        "confidence": event.get("certainty", 0.5),
                    }
                )
                available_ids.append(evidence_id)
                if visibility == "public_ui":
                    public_history.append({"evidence_id": evidence_id, "abs_sec": event["abs_start_sec"], "type": "public_ui", "content": event.get("description", "")})
            elif visibility == "not_visible":
                hidden.append(
                    {
                        "world_event_id": event["world_event_id"],
                        "reason_hidden": "not_visible",
                        "visible_to": event.get("source_povs", []),
                    }
                )
                forbidden_ids.append(event["world_event_id"])
        for claim in claims:
            if claim["abs_start_sec"] <= cutoff and target_player in claim.get("heard_by", []):
                heard_claims.append(
                    {
                        "evidence_id": claim["claim_id"],
                        "claim_id": claim["claim_id"],
                        "speaker": claim.get("speaker", "unknown"),
                        "content": claim.get("content", ""),
                        "local_truth_awareness": "not_enough_information",
                    }
                )
                available_ids.append(claim["claim_id"])
                public_history.append({"evidence_id": claim["claim_id"], "abs_sec": claim["abs_start_sec"], "type": "meeting_statement", "content": claim.get("content", "")})
        inferred = [
            {
                "belief_id": f"belief_{target_player}_{int(cutoff):06d}",
                "about": "current hidden events and heard claims",
                "belief_label": "uncertain" if hidden else "unknown",
                "basis_evidence_ids": available_ids[:10],
                "confidence": 0.5,
            }
        ]
        snapshots.append(
            {
                "snapshot_id": f"mem_g001_{target_player}_{int(cutoff):06d}",
                "game_id": "g001",
                "target_player": target_player,
                "cutoff_abs_sec": cutoff,
                "public_history": public_history,
                "private_observations": private_observations,
                "heard_claims": heard_claims,
                "inferred_beliefs": inferred,
                "hidden_events_for_target": hidden,
                "forbidden_event_ids": sorted(set(forbidden_ids)),
                "available_evidence_ids": sorted(set(available_ids)),
            }
        )
    return snapshots


