from __future__ import annotations

import pytest

from gooseomni.models.model_server.baichuan_omni.client import BaichuanOmni15Client
from gooseomni.models.model_server.clients import CLIENTS
from gooseomni.models.pipeline.types import InferenceRequest
from gooseomni.models.utils.omni_http_client import OmniHttpClient
from gooseomni.models.utils.openai_compat_tester import OpenAICompatTester

OMNI_HTTP_MODELS = {
    "qwen2_5_omni": lambda video, audio: (video, audio),
    "qwen3_omni": lambda video, audio: (video, audio),
    "qwen3_omni_thinking": lambda video, audio: (video, audio),
    "miniomni_2": lambda video, audio: (video, audio),
    "omnivinci": lambda video, audio: (video, audio),
    "vita_1_5": lambda video, audio: (video, audio),
    "ming": lambda video, audio: (video, audio),
}

API_MODELS = {
    "gemini_2_5_flash",
    "gemini_2_5_pro",
    "gemini_3_flash_preview",
    "gemini_3_pro_preview",
}


@pytest.mark.parametrize("model_name", sorted(OMNI_HTTP_MODELS))
@pytest.mark.parametrize("use_video,use_audio", [(True, True), (True, False), (False, True), (False, False)])
def test_local_http_clients_forward_media_switches(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    use_video: bool,
    use_audio: bool,
) -> None:
    captured: dict[str, bool] = {}

    def fake_call(self: OmniHttpClient, *args: object, **kwargs: object) -> str:
        captured["use_video"] = bool(kwargs["use_video"])
        captured["use_audio"] = bool(kwargs["use_audio"])
        return "A"

    monkeypatch.setattr(OmniHttpClient, "call_api", fake_call)
    request = InferenceRequest(
        prompt="question",
        video_path="sample.mp4",
        use_video=use_video,
        use_audio=use_audio,
    )

    CLIENTS[model_name]().predict(request)

    assert (captured["use_video"], captured["use_audio"]) == OMNI_HTTP_MODELS[
        model_name
    ](use_video, use_audio)


@pytest.mark.parametrize("use_video", [True, False])
def test_baichuan_forwards_video_switch(
    monkeypatch: pytest.MonkeyPatch, use_video: bool
) -> None:
    captured: dict[str, bool] = {}

    def fake_call(self: BaichuanOmni15Client, *args: object, **kwargs: object) -> str:
        captured["use_video"] = bool(kwargs["use_video"])
        captured["use_audio"] = bool(kwargs["use_audio"])
        return "A"

    monkeypatch.setattr(BaichuanOmni15Client, "_call_api", fake_call)
    request = InferenceRequest(
        prompt="question",
        video_path="sample.mp4",
        use_video=use_video,
        use_audio=not use_video,
    )

    BaichuanOmni15Client().predict(request)

    assert captured["use_video"] is use_video
    assert captured["use_audio"] is (not use_video)


@pytest.mark.parametrize("model_name", sorted(API_MODELS))
@pytest.mark.parametrize(
    "use_video,use_audio", [(True, True), (True, False), (False, True), (False, False)]
)
def test_api_clients_map_media_to_gateway_content(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    use_video: bool,
    use_audio: bool,
) -> None:
    captured: dict[str, bool] = {}

    monkeypatch.setattr(OpenAICompatTester, "__init__", lambda self, *args, **kwargs: None)

    def fake_call(self: OpenAICompatTester, *args: object, **kwargs: object) -> str:
        captured["include_images"] = bool(kwargs["include_images"])
        captured["include_audio"] = bool(kwargs["include_audio"])
        captured["has_media_path"] = args[0] is not None
        return "A"

    monkeypatch.setattr(OpenAICompatTester, "call", fake_call)
    request = InferenceRequest(
        prompt="question",
        video_path="sample.mp4" if (use_video or use_audio) else None,
        use_video=use_video,
        use_audio=use_audio,
    )

    CLIENTS[model_name]().predict(request)

    assert captured["include_images"] is use_video
    assert captured["include_audio"] is use_audio
    assert captured["has_media_path"] is (use_video or use_audio)


@pytest.mark.parametrize("use_video", [True, False])
def test_gpt4o_maps_text_and_visual_without_audio(
    monkeypatch: pytest.MonkeyPatch, use_video: bool
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(OpenAICompatTester, "__init__", lambda self, *args, **kwargs: None)

    def fake_call(self: OpenAICompatTester, *args: object, **kwargs: object) -> str:
        captured["media_path"] = args[0]
        captured["include_images"] = kwargs["include_images"]
        captured["include_audio"] = kwargs["include_audio"]
        return "A"

    monkeypatch.setattr(OpenAICompatTester, "call", fake_call)
    CLIENTS["gpt4o"]().predict(
        InferenceRequest(
            prompt="question",
            video_path="sample.mp4" if use_video else None,
            use_video=use_video,
            use_audio=False,
        )
    )

    assert captured == {
        "media_path": "sample.mp4" if use_video else None,
        "include_images": use_video,
        "include_audio": False,
    }


def test_gpt4o_rejects_audio() -> None:
    with pytest.raises(ValueError, match="use gpt_audio"):
        CLIENTS["gpt4o"]().predict(
            InferenceRequest(
                prompt="question",
                video_path="sample.mp4",
                use_video=False,
                use_audio=True,
            )
        )


def test_gpt_audio_maps_audio_without_visual_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init(self: OpenAICompatTester, model_name: str) -> None:
        captured["model_name"] = model_name

    def fake_call(self: OpenAICompatTester, *args: object, **kwargs: object) -> str:
        captured["media_path"] = args[0]
        captured["include_images"] = kwargs["include_images"]
        captured["include_audio"] = kwargs["include_audio"]
        return "audio answer"

    monkeypatch.setattr(OpenAICompatTester, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatTester, "call", fake_call)

    result = CLIENTS["gpt_audio"]().predict(
        InferenceRequest(
            prompt="question",
            video_path="sample.mp4",
            use_video=False,
            use_audio=True,
        )
    )

    assert captured == {
        "model_name": "gpt-audio-2025-08-28",
        "media_path": "sample.mp4",
        "include_images": False,
        "include_audio": True,
    }
    assert result.text == "audio answer"


@pytest.mark.parametrize(
    "inference_request,error",
    [
        (
            InferenceRequest(
                prompt="question",
                video_path="sample.mp4",
                use_video=True,
                use_audio=True,
            ),
            "does not support visual input",
        ),
        (
            InferenceRequest(prompt="question", use_video=False, use_audio=False),
            "requires audio input",
        ),
    ],
)
def test_gpt_audio_rejects_unverified_modalities(
    inference_request: InferenceRequest, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CLIENTS["gpt_audio"]().predict(inference_request)
