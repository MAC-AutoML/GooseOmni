import collections
import json
import re
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_canonical import PLAYERS, read_jsonl, write_json
from gooseomni.benchmark.decrypto_probes import load_ledger


def parse_answer(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        return json.loads(raw), True
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), True
            except Exception:
                return {}, False
    return {}, False


def uses_forbidden_evidence(parsed: dict[str, Any], forbidden: list[str]) -> bool:
    text = json.dumps(parsed, ensure_ascii=False)
    return any(item and item in text for item in forbidden)


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalize_enum(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "true": "supported",
        "false": "contradicted",
        "not_enough_info": "not_enough_information",
        "not_enough_evidence": "not_enough_information",
        "unknown": "unknown",
    }
    return aliases.get(text, text)


def answer_claim_truth(answers: dict[str, dict[str, Any]]) -> str | None:
    keys = [
        "claim_truth_global",
        "global_claim_truth",
        "truth_status_global",
        "claim_global_truth",
    ]
    for letter in ["A", "B", "C", "D"]:
        value = first_present(answers.get(letter, {}), keys)
        normalized = normalize_enum(value)
        if normalized:
            return normalized
    return None


def answer_local_awareness(answers: dict[str, dict[str, Any]]) -> str | None:
    keys = [
        "claim_awareness_local_target",
        "local_claim_awareness",
        "target_claim_awareness",
        "target_can_know_claim_truth",
        "target_has_evidence_for_claim_truth",
    ]
    for letter in ["A", "B", "C", "D"]:
        answer = answers.get(letter, {})
        value = first_present(answer, keys)
        if isinstance(value, bool):
            return "has_evidence" if value else "not_enough_information"
        normalized = normalize_enum(value)
        if normalized:
            return normalized
    return None


def evidence_ids_from_answer(parsed: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    stack: list[Any] = [parsed]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {
                    "evidence_id",
                    "event_id",
                    "claim_id",
                    "world_event_id",
                } and isinstance(value, str):
                    ids.add(value)
                elif key.endswith("_ids") and isinstance(value, list):
                    ids.update(str(entry) for entry in value if isinstance(entry, str))
                else:
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return ids


def schema_ok_for_answer(letter: str, parsed: dict[str, Any]) -> bool:
    if not parsed:
        return False
    if letter == "A":
        return "knows_truth" in parsed and bool(
            parsed.get("belief_label") or parsed.get("likely_belief")
        )
    if letter == "B":
        return "target_knew_truth_at_cutoff" in parsed and bool(
            parsed.get("reconstructed_prior_belief") or parsed.get("belief_label")
        )
    if letter == "C":
        return "other_player_knew_truth_at_cutoff" in parsed and bool(
            parsed.get("other_player_likely_belief") or parsed.get("belief_label")
        )
    if letter == "D":
        return bool(parsed.get("listener")) and bool(
            parsed.get("predicted_listener_trust_update")
            or parsed.get("predicted_trust_update")
        )
    return bool(parsed)


def d_reference(hidden_gold: dict[str, Any]) -> dict[str, Any]:
    value = hidden_gold.get("D_PT_reference")
    if isinstance(value, dict):
        return value
    nested_gold = hidden_gold.get("hidden_gold")
    if isinstance(nested_gold, dict) and isinstance(
        nested_gold.get("D_PT_reference"), dict
    ):
        return nested_gold["D_PT_reference"]
    return {}


def d_perspective_scores(
    answer: dict[str, Any], hidden_gold: dict[str, Any]
) -> tuple[bool | None, bool | None]:
    if not answer:
        return None, None
    trust_update = answer.get("predicted_listener_trust_update") or answer.get(
        "predicted_trust_update"
    )
    next_action = answer.get("predicted_listener_next_action") or answer.get(
        "predicted_next_action"
    )
    if not trust_update:
        return False, False
    reference = d_reference(hidden_gold)
    expected_listener = reference.get("listener")
    expected_speaker = reference.get("speaker")
    listener_ok = not expected_listener or answer.get("listener") == expected_listener
    speaker_ok = not expected_speaker or answer.get("speaker") == expected_speaker
    weak = bool(answer.get("listener") and listener_ok)
    strong = bool(weak and speaker_ok and next_action not in {None, "unknown"})
    return weak, strong


def score_group_answers(
    group_id: str,
    answers: dict[str, dict[str, Any]],
    hidden_gold: dict[str, Any],
    forbidden: list[str],
) -> dict[str, Any]:
    a = answers.get("A", {})
    b = answers.get("B", {})
    c = answers.get("C", {})
    d = answers.get("D", {})
    acceptable = hidden_gold.get(
        "acceptable_evidence_ids_for_target", []
    ) or hidden_gold.get("hidden_gold", {}).get(
        "acceptable_evidence_ids_for_target", []
    )
    leakage = any(uses_forbidden_evidence(ans, forbidden) for ans in [a, b, c, d])
    cited_ids: set[str] = set()
    for answer in [a, b, c, d]:
        cited_ids.update(evidence_ids_from_answer(answer))
    acceptable_set = set(acceptable)
    forbidden_set = set(forbidden)
    unsupported_ids = (
        cited_ids - acceptable_set - forbidden_set if acceptable_set else set()
    )
    a.get("knows_truth")
    b_knows = b.get("target_knew_truth_at_cutoff")
    c_knows = c.get("other_player_knew_truth_at_cutoff")
    a_label = a.get("belief_label") or a.get("likely_belief")
    b_label = b.get("belief_label") or b.get("reconstructed_prior_belief")
    expected_global = normalize_enum(hidden_gold.get("claim_truth_global"))
    expected_local = normalize_enum(hidden_gold.get("claim_awareness_local_target"))
    actual_global = answer_claim_truth(answers)
    actual_local = answer_local_awareness(answers)
    pt_weak, pt_strong = d_perspective_scores(d, hidden_gold)
    return {
        "probe_group_id": group_id,
        "RC_weak": b_knows is False if b else None,
        "RC_strong": bool(
            b and a and b_knows is False and str(a_label)[:40] in str(b_label)
        ),
        "FB_weak": c_knows is not True if c else None,
        "FB_strong": bool(
            c and c_knows is not True and c.get("other_player_likely_belief")
        ),
        "PT_weak": pt_weak,
        "PT_strong": pt_strong,
        "claim_verification_global": actual_global == expected_global
        if expected_global
        else None,
        "claim_verification_local": actual_local == expected_local
        if expected_local
        else None,
        "perspective_leakage": leakage,
        "forbidden_evidence_usage": leakage,
        "evidence_support": not leakage and not unsupported_ids,
        "json_parse_success": all(
            answer.get("_parse_ok", True) for answer in answers.values()
        ),
        "schema_validation_success": all(
            schema_ok_for_answer(letter, answer) for letter, answer in answers.items()
        ),
    }


def score_decrypto_diagnostics(
    responses_path: Path, hidden_gold_path: Path, output_path: Path
) -> dict[str, Any]:
    responses = read_jsonl(responses_path)
    hidden_rows = read_jsonl(hidden_gold_path)
    hidden_by_group = {
        row["probe_group_id"]: row for row in hidden_rows if "probe_group_id" in row
    }
    answers_by_group: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(
        dict
    )
    parse_success = 0
    for row in responses:
        parsed = row.get("parsed")
        ok = isinstance(parsed, dict)
        if not ok:
            parsed, ok = parse_answer(row.get("raw_response", ""))
        parse_success += int(ok)
        parsed["_parse_ok"] = ok
        probe_type = row.get("probe_type", "")
        letter = probe_type[:1] if probe_type else row.get("probe_id", "")[-1:]
        answers_by_group[row["probe_group_id"]][letter] = parsed
    scores = []
    for group_id, answers in answers_by_group.items():
        hidden = hidden_by_group.get(group_id, {})
        forbidden = hidden.get("forbidden_event_ids_for_target", []) or hidden.get(
            "hidden_gold", {}
        ).get("forbidden_event_ids_for_target", [])
        scores.append(score_group_answers(group_id, answers, hidden, forbidden))
    aggregate = {
        "groups_scored": len(scores),
        "json_parse_success": parse_success / len(responses) if responses else 0.0,
        "perspective_leakage_rate": sum(
            1 for row in scores if row["perspective_leakage"]
        )
        / len(scores)
        if scores
        else 0.0,
        "RC_weak": sum(1 for row in scores if row["RC_weak"]) / len(scores)
        if scores
        else 0.0,
        "FB_weak": sum(1 for row in scores if row["FB_weak"]) / len(scores)
        if scores
        else 0.0,
        "claim_verification_global": sum(
            1 for row in scores if row["claim_verification_global"]
        )
        / len(scores)
        if scores
        else 0.0,
        "claim_verification_local": sum(
            1 for row in scores if row["claim_verification_local"]
        )
        / len(scores)
        if scores
        else 0.0,
        "forbidden_evidence_usage_rate": sum(
            1 for row in scores if row["forbidden_evidence_usage"]
        )
        / len(scores)
        if scores
        else 0.0,
        "evidence_support_rate": sum(1 for row in scores if row["evidence_support"])
        / len(scores)
        if scores
        else 0.0,
        "schema_validation_success": sum(
            1 for row in scores if row["schema_validation_success"]
        )
        / len(scores)
        if scores
        else 0.0,
    }
    write_json(output_path, {"aggregate": aggregate, "scores": scores})
    return aggregate


def validate_decrypto_outputs(
    annotation_root: Path, benchmark_root: Path
) -> dict[str, Any]:
    issues = []
    ledger = load_ledger(annotation_root)
    for row in ledger["world_events"]:
        if row.get("abs_start_sec") is None or row.get("abs_end_sec") is None:
            issues.append(
                {"code": "event_missing_abs_time", "id": row.get("world_event_id")}
            )
    for row in ledger["claims"]:
        if row.get("abs_start_sec") is None or row.get("abs_end_sec") is None:
            issues.append({"code": "claim_missing_abs_time", "id": row.get("claim_id")})
    edge_counts = collections.Counter(
        edge["event_id"]
        for edge in ledger["visibility_edges"]
        if edge["event_id"].startswith("ge_")
    )
    for event in ledger["world_events"]:
        if edge_counts[event["world_event_id"]] < len(PLAYERS):
            issues.append(
                {
                    "code": "visibility_not_6pov",
                    "id": event["world_event_id"],
                    "count": edge_counts[event["world_event_id"]],
                }
            )
    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    hidden_gold = {
        row.get("probe_group_id"): row
        for row in read_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl")
    }
    probes = read_jsonl(benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
    probes_by_group = collections.defaultdict(set)
    for probe in probes:
        probes_by_group[probe["probe_group_id"]].add(probe["probe_type"])
        prompt = probe.get("prompt", "")
        if probe["probe_type"] == "A_pre_reveal_belief":
            if "QUERY_VARIABLE_PUBLIC_FORM_JSON" not in prompt:
                issues.append(
                    {
                        "code": "A_prompt_missing_public_query_form",
                        "id": probe["probe_id"],
                    }
                )
            if "ORACLE_TRUTH_JSON" in prompt:
                issues.append(
                    {"code": "A_prompt_contains_oracle_truth", "id": probe["probe_id"]}
                )
            if any(fid in prompt for fid in probe.get("forbidden_event_ids", [])):
                issues.append(
                    {"code": "A_prompt_leaks_hidden_truth", "id": probe["probe_id"]}
                )
        if (
            probe["probe_type"] == "B_post_reveal_reconstruct_previous_belief"
            and "answer to probe a" in prompt.lower()
        ):
            issues.append(
                {"code": "B_prompt_mentions_A_answer", "id": probe["probe_id"]}
            )
        if probe["probe_type"] == "D_perspective_taking_prediction":
            if "TARGET_LISTENER_CONTEXT_JSON" in prompt:
                issues.append(
                    {
                        "code": "D_prompt_contains_listener_private_context",
                        "id": probe["probe_id"],
                    }
                )
            if "SPEAKER_AVAILABLE_CONTEXT_JSON" not in prompt:
                issues.append(
                    {
                        "code": "D_prompt_missing_speaker_context",
                        "id": probe["probe_id"],
                    }
                )
            if "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON" not in prompt:
                issues.append(
                    {
                        "code": "D_prompt_missing_listener_public_model",
                        "id": probe["probe_id"],
                    }
                )
    for group in groups:
        types = probes_by_group[group["probe_group_id"]]
        for required in [
            "A_pre_reveal_belief",
            "B_post_reveal_reconstruct_previous_belief",
            "C_other_agent_false_belief",
        ]:
            if required not in types:
                issues.append(
                    {
                        "code": "probe_group_missing_required_probe",
                        "id": group["probe_group_id"],
                        "missing": required,
                    }
                )
        if (
            group.get("gold_source") == "human_verified"
            and "D_perspective_taking_prediction" in types
        ):
            group_hidden = hidden_gold.get(group["probe_group_id"], {})
            d_ref = group_hidden.get("D_PT_reference", {})
            if not group_hidden.get("D_PT_reference_enriched_by") or not d_ref.get(
                "human_verified_scope"
            ):
                issues.append(
                    {
                        "code": "human_verified_D_missing_enriched_gold",
                        "id": group["probe_group_id"],
                    }
                )
    trials_text = (
        (benchmark_root / "static_trials" / "trials.jsonl").read_text(encoding="utf-8")
        if (benchmark_root / "static_trials" / "trials.jsonl").exists()
        else ""
    )
    if "hidden_gold" in trials_text or "forbidden_event_ids" in trials_text:
        issues.append({"code": "static_trials_leak_hidden_gold"})
    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "counts": {
            "world_events": len(ledger["world_events"]),
            "claims": len(ledger["claims"]),
            "visibility_edges": len(ledger["visibility_edges"]),
            "probe_groups": len(groups),
            "prompts": len(probes),
        },
    }
