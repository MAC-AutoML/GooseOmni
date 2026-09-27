from __future__ import annotations

import json
from pathlib import Path

import pytest

from gooseomni.evaluation.runner import (
    EvaluationConfig,
    parse_json_answer,
    plan_evaluation,
    run_evaluation,
    supports_modality,
)
from gooseomni.models.pipeline.types import InferenceResult


class FakeClient:
    model_name = "gpt4o"

    def predict(self, request):
        assert "hidden_gold" not in request.prompt
        return InferenceResult(
            text=json.dumps(
                {
                    "knows_truth": False,
                    "belief_label": "does_not_know",
                    "confidence": 0.8,
                }
            ),
            model=self.model_name,
        )


def _benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "gooseomni_v2"
    (root / "public/leaderboard_core").mkdir(parents=True)
    (root / "public/raw_video_smoke").mkdir(parents=True)
    (root / "public/agentic_midgame_prediction").mkdir(parents=True)
    row = {
        "trial_id": "t1",
        "probe_group_id": "p1",
        "probe_type": "A_pre_reveal_belief",
        "prompt": "public prompt",
        "expected_output_schema": {
            "knows_truth": "boolean",
            "belief_label": "string",
            "confidence": "number",
        },
    }
    for path in [
        root / "public/leaderboard_core/trials.jsonl",
        root / "public/raw_video_smoke/raw_video_smoke.jsonl",
        root / "public/agentic_midgame_prediction/trials.jsonl",
    ]:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text('{"name":"gooseomni_v2"}\n')
    return root


def test_modality_matrix_uses_registry() -> None:
    assert supports_modality("gpt4o", "v") == (True, None)
    assert supports_modality("gpt4o", "a")[0] is False
    assert supports_modality("gpt_audio", "a") == (True, None)
    assert supports_modality("gpt_audio", "v")[0] is False
    assert supports_modality("gpt_audio", "text")[0] is False


def test_plan_covers_all_registered_models(tmp_path: Path) -> None:
    plan = plan_evaluation(
        EvaluationConfig(
            benchmark=_benchmark(tmp_path),
            run_root=tmp_path / "run",
            models=("all",),
            tracks=("raw_video_smoke",),
            limit=1,
        )
    )
    assert len(plan["task_groups"]) == 14 * 4
    assert any(row["status"] == "skipped" for row in plan["task_groups"])


def test_run_uses_unified_client_and_resume(tmp_path: Path) -> None:
    root = _benchmark(tmp_path)
    config = EvaluationConfig(
        benchmark=root,
        run_root=tmp_path / "run",
        models=("gpt4o",),
        tracks=("leaderboard_core",),
        modalities=("text",),
        limit=1,
    )
    result = run_evaluation(config, client_factory=lambda _: FakeClient())
    assert result["ok"] == 1
    row = json.loads((config.run_root / "responses.jsonl").read_text())
    assert row["task_key"] == "t1|gpt4o|text|gooseomni_eval_v2"
    assert row["parse_ok"] is True
    assert row["input_asset_hash"]
    assert (config.run_root / "events.jsonl").is_file()
    assert (config.run_root / "errors.jsonl").is_file()
    assert (config.run_root / "summary.json").is_file()
    resumed = run_evaluation(
        EvaluationConfig(**{**config.__dict__, "resume": True}),
        client_factory=lambda _: FakeClient(),
    )
    assert resumed["submitted"] == 0


def test_run_refuses_changed_input_in_same_directory(tmp_path: Path) -> None:
    root = _benchmark(tmp_path)
    run_root = tmp_path / "run"
    config = EvaluationConfig(
        benchmark=root,
        run_root=run_root,
        models=("gpt4o",),
        tracks=("leaderboard_core",),
        modalities=("text",),
        limit=1,
    )
    run_evaluation(config, client_factory=lambda _: FakeClient())
    with pytest.raises(ValueError, match="config or benchmark input changed"):
        run_evaluation(
            EvaluationConfig(**{**config.__dict__, "prompt_version": "changed"}),
            client_factory=lambda _: FakeClient(),
        )


def test_parse_json_answer_recovers_fenced_content() -> None:
    parsed, ok = parse_json_answer("answer: ```json\n{\"a\": 1}\n```")
    assert ok is True
    assert parsed == {"a": 1}
