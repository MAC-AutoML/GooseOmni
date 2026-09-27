from __future__ import annotations

import json
from pathlib import Path

import pytest

from gooseomni.benchmark.decrypto_export import build_decrypto_diagnostics
from gooseomni.data_pipeline.config import load_data_config
from gooseomni.data_pipeline.ledger_seed import build_seed_ledger
from gooseomni.data_pipeline.local_review import (
    merge_review_shards,
    prepare_review_shards,
)
from gooseomni.data_pipeline.provenance import sha256_tree
from gooseomni.data_pipeline.release import _reviewed_rows
from gooseomni.data_pipeline.runner import DataPipelineRunner
from gooseomni.data_pipeline.stages import (
    PILOT_STAGE_NAMES,
    STAGE_NAMES,
    stage_names,
    validate_judgement,
)


def _config(tmp_path: Path, duplicate: bool = False) -> Path:
    video = tmp_path / "raw" / "g100" / "player-a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    players = [
        {"player_id": "player-a", "video_path": str(video)},
        {
            "player_id": "player-a" if duplicate else "player-b",
            "video_path": str(video),
        },
    ]
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "version: test_v2\n"
        f"games:\n  - game_id: g100\n    players: {json.dumps(players)}\n"
        "    valid_range: [1, 10]\n    excluded_ranges: [[2, 3]]\n"
        "models: {perception: qwen3_omni, judge: gpt-5.6-sol}\n"
        f"outputs: {{run_root: {tmp_path / 'run'}, benchmark_root: {tmp_path / 'benchmark'}}}\n",
        encoding="utf-8",
    )
    return path


def test_data_config_accepts_manifest_players(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path))
    assert [row.player_id for row in config.games[0].players] == [
        "player-a",
        "player-b",
    ]
    assert config.validate_inputs() == []


def test_strict_pilot_uses_new_stage_sequence(tmp_path: Path) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "version: test_v2",
            "version: gooseomni_v2_pilot\npipeline: strict_tom_pilot",
        )
        + (
            "perception: {dual_pass: true, max_visual_chunk_sec: 30, "
            "max_audio_chunk_sec: 30, split_on_phase_boundaries: true}\n"
        ),
        encoding="utf-8",
    )
    config = load_data_config(path)
    assert stage_names(config) == PILOT_STAGE_NAMES
    assert "sync" not in stage_names(config)
    assert config.validate_inputs() == []


def test_strict_pilot_rejects_invalid_phase_chunk_config(tmp_path: Path) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "version: test_v2",
            "version: gooseomni_v2_pilot\npipeline: strict_tom_pilot",
        )
        + "perception: {dual_pass: true, max_audio_chunk_sec: 0}\n",
        encoding="utf-8",
    )
    issues = load_data_config(path).validate_inputs()
    assert "max_audio_chunk_sec must be positive" in issues
    assert "strict pilot requires phase-boundary perception splitting" in issues


def test_data_config_resolves_stage_cache(tmp_path: Path) -> None:
    path = _config(tmp_path)
    cache = tmp_path / "sync.json"
    cache.write_text("{}\n", encoding="utf-8")
    path.write_text(
        path.read_text(encoding="utf-8") + f"cache: {{sync_offsets: {cache}}}\n",
        encoding="utf-8",
    )
    assert load_data_config(path).stage_cache == {"sync_offsets": cache}


def test_data_config_rejects_duplicates_and_bad_ranges(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path, duplicate=True))
    assert any("duplicate player_id" in issue for issue in config.validate_inputs())


def test_pipeline_resume_and_config_hash(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path))
    calls: list[str] = []

    def handler(context):
        calls.append(context.stage)
        return {"stage": context.stage}

    handlers = dict.fromkeys(STAGE_NAMES, handler)
    runner = DataPipelineRunner(config, handlers)
    manifest = runner.run()
    assert all(row["status"] == "complete" for row in manifest["stages"].values())
    runner.run(resume=True)
    assert calls == list(STAGE_NAMES)


def test_pipeline_can_stop_at_a_stage(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path))
    calls: list[str] = []

    def handler(context):
        calls.append(context.stage)
        return {"stage": context.stage}

    runner = DataPipelineRunner(config, dict.fromkeys(STAGE_NAMES, handler))
    runner.run(to_stage="segment")
    assert calls == ["ingest", "sync", "segment"]


def test_failed_stage_keeps_summary_in_sync(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path))

    def fail(_context):
        raise RuntimeError("expected failure")

    runner = DataPipelineRunner(config, {"ingest": fail})
    with pytest.raises(RuntimeError, match="expected failure"):
        runner.run(to_stage="ingest")
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((config.run_root / "summary.json").read_text(encoding="utf-8"))
    assert manifest == summary
    assert summary["stages"]["ingest"]["status"] == "failed"


def test_resume_stops_when_completed_stage_matches_to_stage(tmp_path: Path) -> None:
    config = load_data_config(_config(tmp_path))
    calls: list[str] = []

    def handler(context):
        calls.append(context.stage)
        return {"stage": context.stage}

    runner = DataPipelineRunner(config, dict.fromkeys(STAGE_NAMES, handler))
    runner.run(to_stage="segment")
    calls.clear()
    runner.run(resume=True, from_stage="sync", to_stage="segment")
    assert calls == []


@pytest.mark.parametrize("decision", ["accept", "repair", "reject"])
def test_codex_judgement_decisions(decision: str) -> None:
    candidate = {
        "value": "qwen",
        "available_evidence_ids": ["e1"],
        "gold_source": "qwen_weak",
    }
    parsed = {
        "decision": decision,
        "final_value": {"value": "codex"} if decision == "repair" else None,
        "reason": "checked",
        "evidence_ids": ["e1"],
    }
    result = validate_judgement(candidate, parsed, "gpt-5.6-sol")
    assert result["decision"] == decision
    if decision != "reject":
        assert result["final_value"]["gold_source"] == "model_verified"
    assert result["input_hash"]
    assert result["retry_count"] == 0


def test_codex_cannot_emit_human_verified() -> None:
    result = validate_judgement(
        {"available_evidence_ids": ["e1"]},
        {
            "decision": "repair",
            "final_value": {"gold_source": "human_verified", "value": "fixed"},
            "evidence_ids": ["e1"],
        },
        "gpt-5.6-sol",
    )
    assert result["final_value"]["gold_source"] == "model_verified"


def test_codex_judgement_fails_closed_on_unknown_evidence() -> None:
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_judgement(
            {"available_evidence_ids": ["e1"]},
            {
                "decision": "accept",
                "final_value": None,
                "reason": "bad",
                "evidence_ids": ["hidden"],
            },
            "gpt-5.6-sol",
        )


def test_review_cache_can_restore_a_curated_trial_absent_from_fresh_candidates() -> (
    None
):
    final = {
        "public_trial": {
            "trial_id": "curated-1",
            "probe_group_id": "group-1",
            "gold_source": "human_verified",
        },
        "gold": {"trial_id": "curated-1", "gold_source": "human_verified"},
        "hidden_gold": {
            "trial_id": "curated-1",
            "gold_source": "human_verified",
            "human_verified_scope": "human_verified_annotation_gate",
        },
    }
    public, gold, hidden = _reviewed_rows(
        [],
        {},
        {},
        {"curated-1": {"decision": "repair", "final_value": final}},
    )
    assert [row["trial_id"] for row in public] == ["curated-1"]
    assert {
        public[0]["gold_source"],
        gold[0]["gold_source"],
        hidden[0]["gold_source"],
    } == {"model_verified"}
    assert hidden[0]["model_verified_scope"] == "model_verified_annotation_gate"


def test_tree_hash_ignores_appledouble_files(tmp_path: Path) -> None:
    (tmp_path / "value.json").write_text('{"ok": true}\n', encoding="utf-8")
    before = sha256_tree(tmp_path)
    (tmp_path / "._value.json").write_bytes(b"finder metadata")
    assert sha256_tree(tmp_path) == before


def test_seed_ledger_supports_manifest_defined_players(tmp_path: Path) -> None:
    fusion = tmp_path / "fusion"
    fusion.mkdir()
    events = [
        {
            "global_event_id": "local-1",
            "game_id": "custom-game",
            "start_sec": 1,
            "end_sec": 2,
            "event_type": "movement",
            "description": "Alpha moved",
            "source_clip_ids": ["clip-1"],
            "source_player_ids": ["Alpha"],
            "confidence": 0.9,
        },
        {
            "global_event_id": "local-2",
            "game_id": "custom-game",
            "start_sec": 3,
            "end_sec": 4,
            "event_type": "movement",
            "description": "Beta moved",
            "source_clip_ids": ["clip-2"],
            "source_player_ids": ["Beta"],
            "confidence": 0.8,
        },
    ]
    utterances = [
        {
            "game_id": "custom-game",
            "player_id": "Alpha",
            "clip_id": "clip-1",
            "start_sec": 5,
            "end_sec": 6,
            "speaker_id": "Alpha",
            "text": "I saw Beta",
            "speech_act": "information_sharing",
            "confidence": 0.8,
        }
    ]
    (fusion / "global_events.json").write_text(json.dumps(events), encoding="utf-8")
    (fusion / "meeting_utterances.json").write_text(
        json.dumps(utterances), encoding="utf-8"
    )
    output = tmp_path / "annotations"
    stats = build_seed_ledger(fusion, output / "oracle_ledger", ["Alpha", "Beta"])
    diagnostics = build_decrypto_diagnostics(output, output, limit=4)
    groups = [
        json.loads(line)
        for line in (output / "diagnostics/probe_groups.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert stats["world_events"] == 2
    assert diagnostics["probe_groups"] > 0
    assert all(row["probe_group_id"].startswith("custom-game_") for row in groups)
    first_probe = json.loads(
        (output / "diagnostics/probes_A_pre_reveal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert '"proposition"' in first_probe["prompt"]


def test_local_codex_review_prepare_and_merge(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates/static_trials"
    candidate_root.mkdir(parents=True)
    rows = [{"trial_id": "trial-1", "gold_source": "qwen_seed"}]
    gold = [
        {
            "trial_id": "trial-1",
            "gold_source": "qwen_seed",
            "acceptable_evidence_ids": ["e1"],
        }
    ]
    hidden = [
        {
            "trial_id": "trial-1",
            "gold_source": "qwen_seed",
            "acceptable_evidence_ids": ["e1"],
        }
    ]
    for name, values in (("trials", rows), ("gold", gold), ("hidden_gold", hidden)):
        (candidate_root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )
    review_root = tmp_path / "review"
    manifest = prepare_review_shards(candidate_root.parent, review_root, shard_size=1)
    output = Path(manifest["shards"][0]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "trial_id": "trial-1",
                        "decision": "accept",
                        "reason": "evidence is internally consistent",
                        "evidence_ids": ["e1"],
                        "final_value": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = merge_review_shards(review_root, review_root / "decisions.jsonl")
    assert summary["complete"] is True
    assert summary["validated"] == 1


def test_local_codex_review_decodes_repair_json(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates/static_trials"
    candidate_root.mkdir(parents=True)
    rows = [{"trial_id": "trial-1", "gold_source": "qwen_seed"}]
    gold = [{"trial_id": "trial-1", "acceptable_evidence_ids": []}]
    hidden = [{"trial_id": "trial-1", "acceptable_evidence_ids": []}]
    for name, values in (("trials", rows), ("gold", gold), ("hidden_gold", hidden)):
        (candidate_root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )
    review_root = tmp_path / "review"
    manifest = prepare_review_shards(candidate_root.parent, review_root, shard_size=1)
    output = Path(manifest["shards"][0]["output"])
    repaired = {
        "trial_id": "trial-1",
        "public_trial": rows[0],
        "gold": gold[0],
        "hidden_gold": hidden[0],
    }
    output.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "trial_id": "trial-1",
                        "decision": "repair",
                        "reason": "repair candidate",
                        "evidence_ids": [],
                        "final_value": json.dumps(repaired),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    merged = review_root / "decisions.jsonl"
    summary = merge_review_shards(review_root, merged)
    assert summary["complete"] is True
    decision = json.loads(merged.read_text(encoding="utf-8"))
    assert decision["final_value"]["public_trial"]["trial_id"] == "trial-1"
    assert decision["final_value"]["gold_source"] == "model_verified"
