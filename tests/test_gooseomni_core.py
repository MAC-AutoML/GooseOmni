from pathlib import Path

import pytest

from gooseomni.cli import main
from gooseomni.models.model_server.clients import CLIENTS
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.registry import get_model_spec, list_model_specs


def test_registry_contains_all_supported_models() -> None:
    specs = list_model_specs()
    assert len(specs) == 14
    assert sum(spec.kind == "local" for spec in specs) == 8
    assert sum(spec.kind == "api" for spec in specs) == 6
    assert all("text" in spec.native_inputs for spec in specs)
    assert get_model_spec("gpt_audio").pipeline_inputs == ("text", "audio")
    assert all(
        spec.pipeline_inputs == ("text", "video", "audio")
        for spec in specs
        if spec.name not in {"gpt4o", "gpt_audio"}
    )
    assert get_model_spec("ming").native_outputs == ("text", "image", "audio")
    assert all(spec.pipeline_outputs == ("text",) for spec in specs)
    assert "audio" not in get_model_spec("gpt4o").native_inputs
    assert get_model_spec("gpt4o").pipeline_inputs == ("text", "video")
    assert get_model_spec("gpt_audio").native_inputs == ("text", "audio")


@pytest.mark.parametrize("model_name", sorted(CLIENTS))
def test_every_model_adapter_implements_contract(model_name: str) -> None:
    client = CLIENTS[model_name]()
    assert client.model_name == model_name
    assert callable(client.predict)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("missing")


def test_inference_contract_supports_text_only_and_video() -> None:
    text_request = InferenceRequest(prompt="hello", use_video=False, use_audio=False)
    assert text_request.video_path is None
    with pytest.raises(ValueError, match="requires video_path"):
        text_request.require_video_path()

    video_request = InferenceRequest(prompt="inspect", video_path=Path("sample.mp4"))
    assert video_request.require_video_path() == "sample.mp4"
    result = InferenceResult(text="raw", parsed_answer="A", model="mock")
    assert result.answer == "A"
    assert result.raw_response == "raw"


def test_models_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "list", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"qwen3_omni"' in output
    assert '"gpt4o"' in output
