from .decrypto_probes import *  # noqa: F401,F403

def generate_probes_for_group(group: dict[str, Any], ledger: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    events = event_by_id(ledger["world_events"])
    claims = claim_by_id(ledger["claims"])
    snapshots = ledger["belief_memory_snapshots"]
    target = group["target_player"]
    cutoff = group["cutoff_abs_sec"]
    target_snapshot = snapshot_for(snapshots, target, cutoff)
    anchor_events = [events[event_id] for event_id in group.get("anchor_event_ids", []) if event_id in events]
    related_claims = [claims[claim_id] for claim_id in group.get("related_claim_ids", []) if claim_id in claims]
    players = ledger_players(ledger)
    other_player = (
        pick_other_player(anchor_events[0], target, players)
        if anchor_events
        else next((player for player in players if player != target), target)
    )
    other_snapshot = snapshot_for(snapshots, other_player, cutoff)
    oracle_truth = json.dumps(anchor_events, ensure_ascii=False)
    related_claim_text = json.dumps(related_claims, ensure_ascii=False)
    header = probe_prompt_header(group)
    target_context = compact_context(target_snapshot)
    other_context = compact_context(other_snapshot)
    query_form = public_query_form(group)
    forbidden = group.get("hidden_event_ids_for_target", [])
    acceptable = group.get("available_evidence_ids_for_target", [])
    anchor_event_ids = [event["world_event_id"] for event in anchor_events]
    target_has_anchor_evidence = bool(set(anchor_event_ids) & set(acceptable)) or any(
        target in event.get("source_povs", []) for event in anchor_events
    )
    target_knows_truth_at_cutoff = target_has_anchor_evidence and not forbidden
    target_belief_label = "knows_truth" if target_knows_truth_at_cutoff else ("does_not_know" if forbidden else "uncertain")

    probes = [
        {
            "probe_id": f"{group['probe_group_id']}_A",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "A_pre_reveal_belief",
            "input_condition": "target_available_events",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nOnly use the target player's available information before cutoff. Do not use hidden oracle events.\nQUERY_VARIABLE_PUBLIC_FORM_JSON:\n{query_form}\nTARGET_AVAILABLE_CONTEXT_JSON:\n{target_context}\nQUESTION: From {target}'s perspective at cutoff, can {target} verify, doubt, believe, or remain uncertain about the public-form query using only target-available evidence?\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("A_pre_reveal_belief"),
            "forbidden_event_ids": forbidden,
            "acceptable_evidence_ids": acceptable,
            "gold_source": "qwen_weak",
        },
        {
            "probe_id": f"{group['probe_group_id']}_B",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "B_post_reveal_reconstruct_previous_belief",
            "input_condition": "oracle_truth_revealed",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nORACLE_TRUTH_JSON:\n{oracle_truth}\nTARGET_AVAILABLE_CONTEXT_BEFORE_REVEAL_JSON:\n{target_context}\nQUESTION: Now that oracle truth is revealed, reconstruct {target}'s belief at the earlier cutoff. Do not treat revealed truth as evidence {target} had at cutoff. Do not rely on prior probe responses.\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("B_post_reveal_reconstruct_previous_belief"),
            "forbidden_event_ids": forbidden,
            "acceptable_evidence_ids": acceptable,
            "gold_source": "qwen_weak",
        },
        {
            "probe_id": f"{group['probe_group_id']}_C",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "C_other_agent_false_belief",
            "input_condition": "global_to_perspective",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nORACLE_TRUTH_JSON:\n{oracle_truth}\nOTHER_PLAYER: {other_player}\nOTHER_PLAYER_AVAILABLE_CONTEXT_BEFORE_REVEAL_JSON:\n{other_context}\nQUESTION: From {other_player}'s perspective at cutoff, did {other_player} know the oracle truth? What would {other_player} likely believe before reveal?\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("C_other_agent_false_belief"),
            "forbidden_event_ids": [],
            "acceptable_evidence_ids": other_snapshot.get("available_evidence_ids", []),
            "gold_source": "qwen_weak",
        },
    ]

    if related_claims:
        speaker = related_claims[0].get("speaker", "unknown")
        speaker_snapshot = snapshot_for(snapshots, speaker, cutoff) if speaker in players else {"available_evidence_ids": [], "public_history": [], "private_observations": [], "heard_claims": []}
        speaker_context = compact_context(speaker_snapshot)
        listener_public_model = speaker_listener_public_model(target_snapshot, target)
        probes.append(
            {
                "probe_id": f"{group['probe_group_id']}_D",
                "probe_group_id": group["probe_group_id"],
                "probe_type": "D_perspective_taking_prediction",
                "input_condition": "speaker_perspective",
                "target_player": target,
                "cutoff_abs_sec": cutoff,
                "prompt": header + f"\nRELATED_CLAIMS_JSON:\n{related_claim_text}\nSPEAKER_AVAILABLE_CONTEXT_JSON:\n{speaker_context}\nSPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON:\n{listener_public_model}\nQUESTION: From speaker {speaker}'s perspective after making the strategic claim, predict how listener {target} would interpret the claim and how their trust/action may change. Use only what the speaker could know or reasonably model from public listener history.\nReturn strict JSON matching the schema.",
                "expected_output_schema": expected_schema_for("D_perspective_taking_prediction"),
                "forbidden_event_ids": forbidden,
                "acceptable_evidence_ids": acceptable,
                "gold_source": "qwen_weak",
            }
        )

    hidden_gold = {
        "probe_group_id": group["probe_group_id"],
        "A_expected_weak": {"knows_truth": target_knows_truth_at_cutoff, "must_not_use_event_ids": forbidden},
        "B_RC_weak": {
            "must_not_answer_as_if_target_saw_hidden_events": bool(forbidden),
            "target_had_anchor_evidence_at_cutoff": target_has_anchor_evidence,
        },
        "B_RC_strong_reference": {"belief_label": target_belief_label, "key_evidence_class": "available_claim_or_partial_observation"},
        "C_FB_weak": {"other_player": other_player, "other_player_knows_truth": other_player in (anchor_events[0].get("source_povs", []) if anchor_events else [])},
        "C_FB_strong_reference": {"other_player": other_player, "key_evidence_class": "other_player_available_context"},
        "D_PT_reference": {"listener": target, "requires_information_state_difference": bool(related_claims)},
        "forbidden_event_ids_for_target": forbidden,
        "acceptable_evidence_ids_for_target": acceptable,
        "claim_truth_global": "unverified",
        "claim_awareness_local_target": "can_verify_from_direct_evidence"
        if target_knows_truth_at_cutoff
        else ("not_enough_information" if related_claims else "unknown"),
        "gold_source": "qwen_weak",
    }
    quality = {
        "probe_group_id": group["probe_group_id"],
        "keep": len(probes) >= 3 and bool(group.get("hidden_event_ids_for_target") or group.get("available_evidence_ids_for_target")),
        "diagnostic_score": 0.8 if len(probes) >= 4 else 0.7,
        "failure_reasons": [],
        "needs_human_review": bool(group.get("needs_human_review", False) or group.get("quality", {}).get("needs_human_review", False)),
        "recommended_gold_source": "qwen_weak",
    }
    return probes, hidden_gold, quality


def build_decrypto_diagnostics(ledger_root: Path, output_root: Path, limit: int = 240) -> dict[str, int]:
    ledger = load_ledger(ledger_root)
    groups = select_probe_groups(ledger, limit=limit)
    all_probes: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    hidden_gold: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for group in groups:
        group = dict(group)
        snapshot = snapshot_for(
            ledger["belief_memory_snapshots"],
            str(group["target_player"]),
            float(group["cutoff_abs_sec"]),
        )
        group["available_evidence_ids_for_target"] = sorted(
            set(group.get("available_evidence_ids_for_target", []))
            | set(snapshot.get("available_evidence_ids", []))
        )
        probes, gold, quality = generate_probes_for_group(group, ledger)
        all_probes.extend(probes)
        for probe in probes:
            by_type[probe["probe_type"]].append(probe)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    diag = output_root / "diagnostics"
    write_jsonl(diag / "probe_groups.jsonl", groups)
    write_jsonl(diag / "probes_A_pre_reveal.jsonl", by_type["A_pre_reveal_belief"])
    write_jsonl(diag / "probes_B_reconstruct.jsonl", by_type["B_post_reveal_reconstruct_previous_belief"])
    write_jsonl(diag / "probes_C_false_belief.jsonl", by_type["C_other_agent_false_belief"])
    write_jsonl(diag / "probes_D_perspective_taking.jsonl", by_type["D_perspective_taking_prediction"])
    write_jsonl(diag / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diag / "diagnostic_quality.jsonl", quality_rows)
    return {
        "probe_groups": len(groups),
        "probes": len(all_probes),
        "A": len(by_type["A_pre_reveal_belief"]),
        "B": len(by_type["B_post_reveal_reconstruct_previous_belief"]),
        "C": len(by_type["C_other_agent_false_belief"]),
        "D": len(by_type["D_perspective_taking_prediction"]),
        "hidden_gold": len(hidden_gold),
    }


def export_gooseomni_benchmark(annotation_root: Path, benchmark_root: Path) -> dict[str, int]:
    diag = annotation_root / "diagnostics"
    groups = read_jsonl(diag / "probe_groups.jsonl")
    probes = (
        read_jsonl(diag / "probes_A_pre_reveal.jsonl")
        + read_jsonl(diag / "probes_B_reconstruct.jsonl")
        + read_jsonl(diag / "probes_C_false_belief.jsonl")
        + read_jsonl(diag / "probes_D_perspective_taking.jsonl")
    )
    hidden_gold = read_jsonl(diag / "hidden_gold.jsonl")
    quality = read_jsonl(diag / "diagnostic_quality.jsonl")
    hidden_by_group = {row["probe_group_id"]: row for row in hidden_gold}

    interactive = benchmark_root / "interactive_diagnostics"
    static = benchmark_root / "static_trials"
    reports = benchmark_root / "reports"
    write_jsonl(interactive / "probe_groups.jsonl", groups)
    write_jsonl(interactive / "prompts.jsonl", probes)
    write_jsonl(interactive / "hidden_gold.jsonl", hidden_gold)
    scoring_rules = {
        "RC_weak": "B must not attribute hidden revealed truth to target_player at cutoff.",
        "RC_strong": "B must match A on main belief label and key evidence class.",
        "FB_weak": "C must identify whether other player knew hidden truth at cutoff.",
        "FB_strong": "C must reconstruct other player's concrete pre-reveal belief.",
        "PT_weak": "D must distinguish speaker and listener information states.",
        "PT_strong": "D must predict listener reaction consistent with later evidence when available.",
        "claim_verification_global": "Model identifies whether a claim is globally supported, contradicted, unverified, or ambiguous.",
        "claim_verification_local": "Model distinguishes global claim truth from what the target player could know at cutoff.",
        "perspective_leakage": "Model references forbidden_event_ids while simulating a limited perspective.",
        "forbidden_evidence_usage_rate": "Fraction of scored groups whose answers cite hidden or forbidden event IDs.",
        "evidence_support_rate": "Fraction of scored groups that avoid forbidden evidence and cite only allowed evidence classes.",
        "json_parse_success": "Response can be parsed as JSON.",
        "schema_validation_success": "Parsed response matches the probe's expected output schema sufficiently for scoring.",
    }
    write_json(interactive / "scoring_rules.json", scoring_rules)

    trials = []
    gold_rows = []
    hidden_rows = []
    input_conditions: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for probe in probes:
        group_gold = hidden_by_group.get(probe["probe_group_id"], {})
        public_trial = {
            "trial_id": probe["probe_id"],
            "probe_group_id": probe["probe_group_id"],
            "game_id": "g001",
            "probe_type": probe["probe_type"],
            "target_player": probe["target_player"],
            "cutoff_abs_sec": probe["cutoff_abs_sec"],
            "input_condition": probe["input_condition"],
            "prompt": probe["prompt"],
            "expected_output_schema": probe["expected_output_schema"],
            "gold_source": probe["gold_source"],
        }
        trials.append(public_trial)
        input_conditions[probe["input_condition"]].append(public_trial)
        gold_rows.append(
            {
                "trial_id": probe["probe_id"],
                "probe_group_id": probe["probe_group_id"],
                "gold_source": probe["gold_source"],
                "acceptable_evidence_ids": probe.get("acceptable_evidence_ids", []),
                "metrics": [
                    "RC_weak",
                    "RC_strong",
                    "FB_weak",
                    "FB_strong",
                    "PT_weak",
                    "PT_strong",
                    "claim_verification_global",
                    "claim_verification_local",
                    "perspective_leakage",
                    "forbidden_evidence_usage_rate",
                    "evidence_support_rate",
                    "json_parse_success",
                    "schema_validation_success",
                ],
            }
        )
        hidden_rows.append(
            {
                "trial_id": probe["probe_id"],
                "probe_group_id": probe["probe_group_id"],
                "gold_source": group_gold.get("gold_source", probe["gold_source"]),
                "hidden_gold": group_gold,
                "forbidden_event_ids": probe.get("forbidden_event_ids", []),
                "acceptable_evidence_ids": probe.get("acceptable_evidence_ids", []),
            }
        )

    write_jsonl(static / "trials.jsonl", trials)
    write_jsonl(static / "gold.jsonl", gold_rows)
    write_jsonl(static / "hidden_gold.jsonl", hidden_rows)
    for condition, rows in input_conditions.items():
        write_jsonl(static / "input_conditions" / f"{condition}.jsonl", rows)
    track_rows = {
        "raw_pov_video": [
            {
                "trial_id": trial["trial_id"],
                "probe_group_id": trial["probe_group_id"],
                "track": "raw_pov_video_generalist",
                "target_player": trial["target_player"],
                "cutoff_abs_sec": trial["cutoff_abs_sec"],
                "input_condition": trial["input_condition"],
                "requires_target_pov_video": True,
                "prompt": trial["prompt"],
            }
            for trial in trials
        ],
        "structured_perspective": [
            trial
            for trial in trials
            if trial["input_condition"] in {"target_available_events", "public_history_only", "speaker_perspective"}
        ],
        "global_to_perspective": [
            trial
            for trial in trials
            if trial["input_condition"] in {"oracle_truth_revealed", "global_to_perspective"}
        ],
        "specialist_agent": [
            {
                **trial,
                "allowed_resources": ["oracle_ledger", "visibility_edges", "belief_memory_snapshots", "claim_truth_links"],
            }
            for trial in trials
        ],
    }
    for track_name, rows in track_rows.items():
        write_jsonl(static / "input_conditions" / f"{track_name}.jsonl", rows)
    review_queue = [q for q in quality if q.get("needs_human_review") or q.get("diagnostic_score", 1.0) < 0.65]
    write_jsonl(benchmark_root / "human_review_queue.jsonl", review_queue)
    (reports / "diagnostic_quality.md").parent.mkdir(parents=True, exist_ok=True)
    (reports / "benchmark_card.md").write_text(
        "# GooseOmni-v1 Benchmark Card\n\n"
        "GooseOmni-v1 is a trajectory-derived Theory-of-Mind diagnostic benchmark "
        "built from a strictly aligned 6-POV Goose Goose Duck replay. It uses oracle "
        "trajectory ledgers, visibility projections, claim-truth links, and Decrypto-style "
        "A/B/C/D probe groups to evaluate representational change, false belief, claim "
        "verification, perspective taking, strategy communication, and perspective leakage.\n\n"
        "Segments are storage units only. The benchmark unit is a trajectory node: "
        "`cutoff_abs_sec + target_player + query_variable + information_gap`. "
        "Each node is selected from an oracle trajectory ledger and projected into player-local "
        "knowledge states before probe generation.\n\n"
        "Probe groups follow a Decrypto-style diagnostic mechanism. A asks for the target player's "
        "pre-reveal belief using only target-available evidence. B reveals oracle truth but asks the "
        "model to reconstruct the target's earlier belief without truth contamination. C asks for "
        "another player's false or incomplete belief. D asks a speaker to predict how a listener "
        "will interpret a strategic claim from speaker-safe context.\n\n"
        "Interactive diagnostics evaluate A/B/C/D consistency and representational change across "
        "multiple prompts. Static trials provide frozen public prompts and separate hidden gold for "
        "leaderboard-style scoring. `trials.jsonl` must not expose hidden gold or forbidden event IDs; "
        "`hidden_gold.jsonl` is scorer-only.\n\n"
        "Perspective leakage means that an answer simulating a limited player perspective cites "
        "hidden oracle facts, forbidden event IDs, or evidence only available to another POV. "
        "`gold_source` is `qwen_weak` by default, `qwen_checked` only after high-quality video review "
        "and Codex merge gating, and `human_verified` only after explicit human review.\n\n"
        f"- probe_groups: {len(groups)}\n"
        f"- prompts: {len(probes)}\n"
        f"- human_review_queue: {len(review_queue)}\n"
        "- gold_source: qwen_weak unless later promoted by explicit checking or human review\n",
        encoding="utf-8",
    )
    (reports / "diagnostic_quality.md").write_text(
        f"# Diagnostic Quality\n\n- probe_groups: {len(groups)}\n- prompts: {len(probes)}\n- review_queue: {len(review_queue)}\n",
        encoding="utf-8",
    )
    (reports / "annotation_quality.md").write_text(
        "# Annotation Quality\n\nAll labels are weak automatic labels unless promoted to qwen_checked or human_verified.\n",
        encoding="utf-8",
    )
    return {"probe_groups": len(groups), "prompts": len(probes), "static_trials": len(trials), "review_queue": len(review_queue)}
