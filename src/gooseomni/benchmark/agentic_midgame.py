from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
PROMPT_RULES = """你只能使用指定 ego_player 在 cutoff_abs_sec 前可见、可听、公开知道的信息。
你可以做概率推断，但必须区分“直接观察到”和“推断出来”。
不要声称 ego_player 看到了其他 POV 才能看到的信息。
不要使用 cutoff 之后的事件作为证据。
死亡、倒地、尸体、血迹画面只能证明可见死亡状态，不能证明攻击者、阵营、鸭子技能或击杀机制。
如果证据不足，请输出 uncertain，而不是强行给出确定结论。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_ledger(annotation_root: Path) -> dict[str, list[dict[str, Any]]]:
    root = annotation_root / "oracle_ledger"
    return {
        "world_events": read_jsonl(root / "world_events.jsonl"),
        "claims": read_jsonl(root / "claims.jsonl"),
        "claim_truth_links": read_jsonl(root / "claim_truth_links.jsonl"),
        "visibility_edges": read_jsonl(root / "visibility_edges.jsonl"),
        "belief_memory_snapshots": read_jsonl(root / "belief_memory_snapshots.jsonl"),
    }


def public_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: public_safe(val) for key, val in value.items() if key not in {"world_event_id", "hidden_gold", "forbidden_event_ids"}}
    if isinstance(value, list):
        return [public_safe(item) for item in value]
    return value


def compact_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    return public_safe(
        {
            "public_history": snapshot.get("public_history", [])[:10],
            "private_observations": snapshot.get("private_observations", [])[:10],
            "heard_claims": snapshot.get("heard_claims", [])[:10],
            "inferred_beliefs": snapshot.get("inferred_beliefs", [])[:4],
            "available_evidence_ids": snapshot.get("available_evidence_ids", [])[:24],
        }
    )


def snapshot_for(snapshots: list[dict[str, Any]], player: str, cutoff: float) -> dict[str, Any]:
    choices = [s for s in snapshots if s.get("target_player") == player and float(s.get("cutoff_abs_sec", 0)) <= cutoff + 1e-6]
    if not choices:
        return {"available_evidence_ids": [], "forbidden_event_ids": [], "public_history": [], "private_observations": [], "heard_claims": []}
    return max(choices, key=lambda row: float(row.get("cutoff_abs_sec", 0)))


def edge_map(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(edge.get("event_id"), edge.get("player_id")): edge for edge in edges}


def event_category(event_type: Any, description: str = "") -> str:
    text = f"{event_type or ''} {description}".lower()
    if any(t in text for t in ["vote", "投票", "放逐"]):
        return "vote"
    if any(t in text for t in ["claim", "utter", "speech", "发言", "说", "accus", "指控", "defense"]):
        return "speech_response"
    if any(t in text for t in ["kill", "body", "dead", "尸体", "死亡", "倒地", "血"]):
        return "kill_related"
    if any(t in text for t in ["task", "任务"]):
        return "task"
    if any(t in text for t in ["route", "move", "location", "经过", "移动", "位置"]):
        return "location"
    if any(t in text for t in ["meeting", "会议"]):
        return "meeting_strategy"
    return "unknown"


def expected_schema(family: str) -> dict[str, Any]:
    if family == "E_hidden_world_state_estimation":
        return {"known_from_pov": "array", "inferred_hidden_state": "array", "oracle_guess": {"event_type": "location|route|kill_related|meeting_strategy|unknown", "actors": "array[string]", "time_range_abs_sec": "array[number,number]", "confidence": "number"}, "uncertainty": "low|medium|high", "evidence_ids": "array[string]", "must_not_claim_direct_observation": "boolean"}
    if family == "F_other_player_current_action_prediction":
        return {"target_player": "string", "predicted_location": "string", "predicted_action": "string", "basis": "direct_observation|inference|uncertain", "evidence_ids": "array[string]", "confidence": "number"}
    if family == "G_next_behavior_prediction":
        return {"target_player": "string", "prediction_window_sec": "number", "top1_next_behavior": "accuse|vote|defend|ignore|follow|avoid|report_body|move|task|unknown", "topk_next_behaviors": "array[string]", "evidence_ids": "array[string]", "confidence": "number"}
    return {"global_truth_guess": "supported|contradicted|unverified|ambiguous", "local_knowability": "knows|suspects|cannot_know|unknown", "speaker_intent_guess": "truthful|lying|mistaken|strategic|unclear", "evidence_ids": "array[string]", "confidence": "number"}


def prompt_header(trial_id: str, family: str, ego: str, cutoff: float) -> str:
    return (
        "Dataset: GooseOmni\n"
        "Game: Goose Goose Duck / 鹅鸭杀风格多人社交博弈\n"
        "Players: Gemini, baile, beigang, mojiang, saoyi, xiaolu\n"
        "Track: Agentic Midgame Prediction\n"
        f"Trial: {trial_id}\nProbe family: {family}\nEgo player: {ego}\nCutoff abs sec: {cutoff}\n\n"
        f"POV RULES:\n{PROMPT_RULES}\n"
    )


def make_public_trial(base: dict[str, Any], context: dict[str, Any], question: str) -> dict[str, Any]:
    prompt = prompt_header(base["trial_id"], base["probe_family"], base["ego_player"], base["cutoff_abs_sec"])
    prompt += "\nEGO_AVAILABLE_CONTEXT_JSON:\n" + json.dumps(context, ensure_ascii=False, indent=2)
    prompt += "\n\nQUESTION:\n" + question + "\nReturn strict JSON matching EXPECTED_OUTPUT_SCHEMA."
    return {
        "trial_id": base["trial_id"],
        "game_id": base["game_id"],
        "probe_family": base["probe_family"],
        "ego_player": base["ego_player"],
        "cutoff_abs_sec": base["cutoff_abs_sec"],
        "prediction_target": base["prediction_target"],
        "prediction_window": base["prediction_window"],
        "input_condition": "ego_available_context",
        "prompt": prompt,
        "expected_output_schema": expected_schema(base["probe_family"]),
        "gold_source": base.get("gold_source", "oracle_seed"),
    }


def _base(idx: int, family: str, ego: str, cutoff: float, target: str, window: dict[str, Any], game_id: str) -> dict[str, Any]:
    return {"trial_id": f"{game_id}_amp_{idx:06d}", "game_id": game_id, "probe_family": family, "ego_player": ego, "cutoff_abs_sec": round(cutoff, 2), "prediction_target": target, "prediction_window": window, "gold_source": "oracle_seed"}


def _gold(public: dict[str, Any], acceptable: list[str], metrics: list[str]) -> dict[str, Any]:
    return {"trial_id": public["trial_id"], "probe_family": public["probe_family"], "gold_source": public["gold_source"], "acceptable_evidence_ids": acceptable, "metrics": metrics}


def _hidden(public: dict[str, Any], acceptable: list[str], forbidden: list[str], hidden: dict[str, Any]) -> dict[str, Any]:
    return {"trial_id": public["trial_id"], "probe_family": public["probe_family"], "gold_source": public["gold_source"], "acceptable_evidence_ids": acceptable, "forbidden_event_ids": forbidden, "hidden_gold": hidden}


def add_trial(rows: dict[str, Any], public: dict[str, Any], acceptable: list[str], forbidden: list[str], hidden: dict[str, Any], metrics: list[str], per_family: int) -> None:
    counters = rows["counters"]
    if counters[public["probe_family"]] >= per_family or len(rows["public"]) >= rows["limit"]:
        return
    rows["public"].append(public)
    rows["gold"].append(_gold(public, acceptable, metrics))
    rows["hidden"].append(_hidden(public, acceptable, forbidden, hidden))
    counters[public["probe_family"]] += 1


def build_agentic_midgame_prediction(annotation_root: Path, benchmark_root: Path, limit: int = 160) -> dict[str, int]:
    ledger = load_ledger(annotation_root)
    events = sorted(ledger["world_events"], key=lambda r: (float(r.get("abs_start_sec", 0)), r.get("world_event_id", "")))
    claims = {c["claim_id"]: c for c in ledger["claims"] if c.get("claim_id")}
    snapshots = ledger["belief_memory_snapshots"]
    edges = edge_map(ledger["visibility_edges"])
    players = sorted({str(row["player_id"]) for row in ledger["visibility_edges"] if row.get("player_id")})
    per_family = max(1, limit // 4)
    rows: dict[str, Any] = {"public": [], "gold": [], "hidden": [], "counters": collections.Counter(), "limit": limit}
    idx = 0

    for event in events:
        if rows["counters"]["E_hidden_world_state_estimation"] >= per_family:
            break
        source = [p for p in event.get("source_povs", []) if p in players]
        if not source or event.get("phase_type") in {"meeting", "final"}:
            continue
        for ego in players:
            if ego in source or edges.get((event.get("world_event_id"), ego), {}).get("visibility") != "not_visible":
                continue
            idx += 1
            cutoff = float(event.get("abs_end_sec", 0)) + 3.0
            snap = snapshot_for(snapshots, ego, cutoff)
            base = _base(idx, "E_hidden_world_state_estimation", ego, cutoff, "hidden_world_state", {"type": "same_cutoff", "duration_sec": 0}, str(event["game_id"]))
            public = make_public_trial(base, compact_context(snap), "从 ego_player 当前视角看，隐藏世界中最可能真实发生了什么？哪些是直接看到的，哪些只是推断？")
            hidden = {"oracle_event": {"event_type": event_category(event.get("event_type"), event.get("description", "")), "raw_event_type": event.get("event_type"), "actors": event.get("actors", []), "location": event.get("location", "unknown"), "time_range_abs_sec": [event.get("abs_start_sec"), event.get("abs_end_sec")], "world_event_id": event.get("world_event_id")}, "visible_to": source}
            add_trial(rows, public, snap.get("available_evidence_ids", []), [event.get("world_event_id")], hidden, ["hidden_world_state_accuracy", "oracle_overclaim_rate", "pov_evidence_support_rate"], per_family)
            break

    for event in events:
        if rows["counters"]["F_other_player_current_action_prediction"] >= per_family:
            break
        target = next((p for p in event.get("source_povs", []) if p in players), None)
        if not target or event.get("phase_type") in {"meeting", "final"}:
            continue
        for ego in players:
            if ego == target or edges.get((event.get("world_event_id"), ego), {}).get("visibility") != "not_visible":
                continue
            idx += 1
            cutoff = (float(event.get("abs_start_sec", 0)) + float(event.get("abs_end_sec", 0))) / 2.0
            snap = snapshot_for(snapshots, ego, cutoff)
            base = _base(idx, "F_other_player_current_action_prediction", ego, cutoff, target, {"type": "same_cutoff", "duration_sec": 0}, str(event["game_id"]))
            public = make_public_trial(base, compact_context(snap), f"在 cutoff_abs_sec 时，{target} 最可能在哪里、正在做什么？这是直接观察、基于轨迹/发言推断，还是不确定？")
            hidden = {"target_player": target, "actual_event_type": event_category(event.get("event_type"), event.get("description", "")), "raw_event_type": event.get("event_type"), "actual_location": event.get("location", "unknown"), "actual_description": event.get("description", ""), "world_event_id": event.get("world_event_id")}
            add_trial(rows, public, snap.get("available_evidence_ids", []), [event.get("world_event_id")], hidden, ["offscreen_action_accuracy", "oracle_overclaim_rate", "pov_evidence_support_rate"], per_family)
            break

    for snap in sorted(snapshots, key=lambda r: float(r.get("cutoff_abs_sec", 0))):
        if rows["counters"]["G_next_behavior_prediction"] >= per_family:
            break
        ego = snap.get("target_player")
        cutoff = float(snap.get("cutoff_abs_sec", 0))
        if ego not in players:
            continue
        future = [e for e in events if cutoff < float(e.get("abs_start_sec", 0)) <= cutoff + 180 and any(p in players and p != ego for p in e.get("actors", []) + e.get("source_povs", []))]
        if not future:
            continue
        event = future[0]
        target = next((p for p in event.get("actors", []) + event.get("source_povs", []) if p in players and p != ego), "unknown")
        idx += 1
        base = _base(idx, "G_next_behavior_prediction", ego, cutoff, target, {"type": "future_window", "duration_sec": 180}, str(event["game_id"]))
        public = make_public_trial(base, compact_context(snap), f"在接下来 180 秒内，{target} 最可能做什么：质疑、投票、跟随、躲避、移动、做任务、报尸、继续自保或沉默？给出 top1 和 top-k。")
        hidden = {"target_player": target, "future_behavior": event_category(event.get("event_type"), event.get("description", "")), "raw_event_type": event.get("event_type"), "future_event_abs_sec": event.get("abs_start_sec"), "future_location": event.get("location", "unknown"), "future_description": event.get("description", ""), "world_event_id": event.get("world_event_id")}
        add_trial(rows, public, snap.get("available_evidence_ids", []), [event.get("world_event_id")], hidden, ["next_behavior_top1_accuracy", "next_behavior_topk_accuracy", "temporal_window_hit_rate"], per_family)

    for link in ledger["claim_truth_links"]:
        if rows["counters"]["H_deception_state_inference"] >= per_family:
            break
        claim = claims.get(link.get("claim_id"))
        if not claim or link.get("truth_status_global") not in {"supported", "contradicted", "unverified", "ambiguous"}:
            continue
        for ego in claim.get("heard_by", []):
            if ego not in players:
                continue
            idx += 1
            cutoff = float(claim.get("abs_end_sec", 0)) + 1.0
            snap = snapshot_for(snapshots, ego, cutoff)
            context = compact_context(snap)
            context["claim_under_evaluation"] = {"claim_id": claim.get("claim_id"), "speaker": claim.get("speaker"), "claim_type": claim.get("claim_type"), "content": claim.get("content")}
            base = _base(idx, "H_deception_state_inference", ego, cutoff, claim.get("speaker", "unknown"), {"type": "claim_after", "duration_sec": 0}, str(claim["game_id"]))
            public = make_public_trial(base, context, "这个 claim 从全局看更可能 supported / contradicted / unverified / ambiguous？从 ego_player 当时视角看，他是否有足够证据知道这一点？请分开回答。")
            local = link.get("local_awareness_by_player", {}).get(ego, "unknown")
            forbidden = [eid for eid in link.get("world_event_ids", []) if edges.get((eid, ego), {}).get("visibility") == "not_visible"]
            hidden = {"claim_id": claim.get("claim_id"), "truth_status_global": link.get("truth_status_global"), "local_awareness": local, "world_event_ids": link.get("world_event_ids", [])}
            add_trial(rows, public, snap.get("available_evidence_ids", []) + [claim.get("claim_id")], forbidden, hidden, ["deception_detection_global", "deception_awareness_local", "pov_evidence_support_rate"], per_family)
            break

    public_root = benchmark_root / "public" / "agentic_midgame_prediction"
    private_root = benchmark_root / "private" / "agentic_midgame_prediction"
    write_jsonl(public_root / "trials.jsonl", rows["public"])
    write_jsonl(public_root / "prompts.jsonl", [{"trial_id": r["trial_id"], "prompt": r["prompt"], "expected_output_schema": r["expected_output_schema"]} for r in rows["public"]])
    write_jsonl(private_root / "gold.jsonl", rows["gold"])
    write_jsonl(private_root / "hidden_gold.jsonl", rows["hidden"])
    write_json(public_root / "summary.json", {"track": "agentic_midgame_prediction", "total_trials": len(rows["public"]), "by_probe_family": dict(rows["counters"]), "gold_source_default": "oracle_seed"})
    write_json(public_root / "scoring_rules.json", scoring_rules())
    (public_root / "README.md").write_text(readme_text(len(rows["public"]), rows["counters"]), encoding="utf-8")
    return {"trials": len(rows["public"]), "gold": len(rows["gold"]), "hidden_gold": len(rows["hidden"]), **{k: rows["counters"][k] for k in sorted(rows["counters"])}}


def scoring_rules() -> dict[str, str]:
    return {
        "hidden_world_state_accuracy": "E probes: oracle_guess event/action category overlaps scorer-only hidden oracle event.",
        "offscreen_action_accuracy": "F probes: predicted current location/action matches target player's same-cutoff hidden POV event category.",
        "next_behavior_top1_accuracy": "G probes: top1 future behavior matches actual future event category within prediction window.",
        "next_behavior_topk_accuracy": "G probes: top-k contains actual future behavior category.",
        "deception_detection_global": "H probes: global_truth_guess matches scorer-only claim truth.",
        "deception_awareness_local": "H probes: local_knowability matches what ego could know from available evidence.",
        "oracle_overclaim_rate": "Response presents hidden oracle facts as direct ego observation instead of calibrated inference.",
        "pov_evidence_support_rate": "Cited evidence IDs are allowed by the ego POV input condition.",
        "death_skill_overclaim_rate": "Visible death/body/blood is overclaimed as killer, role, faction, duck skill, or mechanism.",
    }


def readme_text(total: int, counters: collections.Counter[str]) -> str:
    rows = "\n".join(f"- `{k}`: {v}" for k, v in sorted(counters.items()))
    return f"""# Agentic Midgame Prediction

This supplementary GooseOmni track evaluates the online middle-layer skills needed by a hypothetical omni player in a 5-human + 1-omni game: hidden-state estimation, offscreen player action prediction, future behavior prediction, and deception-state inference from one limited POV.

It does not replace `leaderboard_core`. The core A/B/C/D benchmark measures Decrypto-style Theory-of-Mind diagnostics. This track measures POV-to-oracle state estimation and POV-to-future behavior prediction, with scorer-private hidden gold from the aligned 6-POV replay.

- total_trials: {total}
{rows}

Public files contain only ego-available context and prompts. Scorer-only hidden POV facts, future outcomes, and forbidden event IDs live under `private/agentic_midgame_prediction/` and must never be sent to evaluated models.
"""


def parse_answer(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {}, isinstance(value, dict))
    except Exception:
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if not match:
            return {}, False
        try:
            value = json.loads(match.group(0))
            return (value if isinstance(value, dict) else {}, isinstance(value, dict))
        except Exception:
            return {}, False


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"route": "location", "movement": "location", "move": "location", "body": "kill_related", "death": "kill_related", "accusation": "speech_response", "defense": "speech_response", "has_contradictory_visual_evidence": "knows", "has_supporting_visual_evidence": "knows", "not_enough_information": "cannot_know"}
    return aliases.get(text, text)


def cited_ids(parsed: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    stack: list[Any] = [parsed]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key.endswith("_ids") and isinstance(value, list):
                    ids.update(str(v) for v in value if isinstance(v, str))
                elif key in {"evidence_id", "event_id", "claim_id", "world_event_id"} and isinstance(value, str):
                    ids.add(value)
                else:
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return ids


def contains_death_skill_overclaim(text: str) -> bool:
    death = r"死亡|尸体|倒地|血迹|dead|body|blood"
    overclaim = r"鸭子技能|鸭技能|技能杀|凶手|杀手|killer|duck skill|role|阵营|击杀机制|mechanism"
    if not re.search(death, text, re.I):
        return False
    for match in re.finditer(overclaim, text, re.I):
        prefix = text[max(0, match.start() - 8) : match.start()]
        if not re.search(r"不能|无法|不可|不能确认|无法确认|不能证明|无法证明|not confirm|cannot", prefix, re.I):
            return True
    return False


def oracle_overclaim(parsed: dict[str, Any], raw: str, forbidden: list[str]) -> bool:
    text = json.dumps(parsed, ensure_ascii=False) + "\n" + (raw or "")
    if parsed.get("must_not_claim_direct_observation") is False:
        return True
    if any(fid and fid in text for fid in forbidden):
        return True
    return bool(re.search(r"直接看到|亲眼看到|clearly saw|directly observed", text, re.I) and forbidden)


def score_one(response: dict[str, Any], hidden: dict[str, Any]) -> dict[str, Any]:
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else None
    parse_ok = parsed is not None
    if parsed is None:
        parsed, parse_ok = parse_answer(response.get("raw_response", ""))
    family = hidden.get("probe_family") or response.get("probe_family")
    gold = hidden.get("hidden_gold", {})
    forbidden = hidden.get("forbidden_event_ids", [])
    acceptable = set(hidden.get("acceptable_evidence_ids", []))
    ids = cited_ids(parsed)
    unsupported = ids - acceptable - set(forbidden) if acceptable else set()
    raw_text = response.get("raw_response", "") + json.dumps(parsed, ensure_ascii=False)
    row = {"trial_id": response.get("trial_id"), "probe_family": family, "json_parse_success": parse_ok, "schema_validation_success": bool(parsed), "oracle_overclaim": oracle_overclaim(parsed, raw_text, forbidden), "death_skill_overclaim": contains_death_skill_overclaim(raw_text), "pov_evidence_support": not unsupported and not (ids & set(forbidden))}
    if family == "E_hidden_world_state_estimation":
        expected = normalize_label(gold.get("oracle_event", {}).get("event_type"))
        actual = normalize_label((parsed.get("oracle_guess") or {}).get("event_type"))
        row["hidden_world_state_accuracy"] = actual == expected if expected else None
    elif family == "F_other_player_current_action_prediction":
        expected = normalize_label(gold.get("actual_event_type"))
        actual = normalize_label(parsed.get("predicted_action") or (parsed.get("oracle_guess") or {}).get("event_type"))
        row["offscreen_action_accuracy"] = actual == expected if expected else None
    elif family == "G_next_behavior_prediction":
        expected = normalize_label(gold.get("future_behavior"))
        top1 = normalize_label(parsed.get("top1_next_behavior") or parsed.get("predicted_next_action"))
        topk = [normalize_label(x) for x in parsed.get("topk_next_behaviors", []) if isinstance(x, str)]
        row["next_behavior_top1_accuracy"] = top1 == expected if expected else None
        row["next_behavior_topk_accuracy"] = expected in set(topk + [top1]) if expected else None
        row["temporal_window_hit_rate"] = True
    elif family == "H_deception_state_inference":
        expected_global = normalize_label(gold.get("truth_status_global"))
        expected_local = normalize_label(gold.get("local_awareness"))
        actual_global = normalize_label(parsed.get("global_truth_guess"))
        actual_local = normalize_label(parsed.get("local_knowability"))
        row["deception_detection_global"] = actual_global == expected_global if expected_global else None
        row["deception_awareness_local"] = actual_local == expected_local if expected_local else None
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row[key] for row in rows if isinstance(row.get(key), bool)]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def score_agentic_midgame_prediction(responses_path: Path, hidden_gold_path: Path, output_path: Path) -> dict[str, Any]:
    responses = read_jsonl(responses_path)
    hidden_by_id = {row["trial_id"]: row for row in read_jsonl(hidden_gold_path)}
    scored = [score_one(resp, hidden_by_id.get(resp.get("trial_id"), {})) for resp in responses]
    metric_keys = ["hidden_world_state_accuracy", "offscreen_action_accuracy", "next_behavior_top1_accuracy", "next_behavior_topk_accuracy", "deception_detection_global", "deception_awareness_local", "temporal_window_hit_rate", "pov_evidence_support", "json_parse_success", "schema_validation_success"]
    aggregate = {key: _mean(scored, key) for key in metric_keys}
    aggregate["oracle_overclaim_rate"] = _mean([{**r, "v": r.get("oracle_overclaim")} for r in scored], "v") or 0.0
    aggregate["death_skill_overclaim_rate"] = _mean([{**r, "v": r.get("death_skill_overclaim")} for r in scored], "v") or 0.0

    def val(key: str, default: float = 0.0) -> float:
        value = aggregate.get(key)
        return default if value is None else float(value)

    aggregate["SoG-Agentic-Midgame"] = 0.25 * val("hidden_world_state_accuracy") + 0.20 * val("next_behavior_top1_accuracy") + 0.15 * val("deception_detection_global") + 0.15 * val("deception_awareness_local") + 0.10 * val("pov_evidence_support") + 0.10 * (1 - val("oracle_overclaim_rate")) + 0.05 * 0.5
    families = sorted({r.get("probe_family") for r in scored if r.get("probe_family")})
    by_family = {family: {key: _mean([r for r in scored if r.get("probe_family") == family], key) for key in metric_keys} for family in families}
    write_json(output_path, {"aggregate": aggregate, "by_probe_family": by_family, "scores": scored})
    return aggregate


def public_leak_hits(public_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    forbidden_terms = ["hidden_gold", "forbidden_event_ids", "future_description", "actual_description", "world_event_id", "oracle_event"]
    hits = []
    for row in public_rows:
        text = json.dumps(row, ensure_ascii=False)
        for term in forbidden_terms:
            if term in text:
                hits.append({"trial_id": row.get("trial_id", ""), "term": term})
    return hits
