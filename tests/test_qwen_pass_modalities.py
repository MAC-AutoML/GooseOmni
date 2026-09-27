from __future__ import annotations

from gooseomni.annotation.backends.qwen_omni import LocalQwenOmniServerBackend


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call_api(self, video_path: str, prompt: str, **kwargs: bool) -> str:
        self.calls.append({"video_path": video_path, "prompt": prompt, **kwargs})
        return "[]"


def test_visual_backend_disables_audio() -> None:
    backend = LocalQwenOmniServerBackend("http://example", True, False)
    client = RecordingClient()
    backend.client = client
    assert backend.generate("visual", "clip.mp4") == "[]"
    assert client.calls[0]["use_video"] is True
    assert client.calls[0]["use_audio"] is False


def test_audio_backend_disables_video() -> None:
    backend = LocalQwenOmniServerBackend("http://example", False, True)
    client = RecordingClient()
    backend.client = client
    assert backend.generate("audio", "clip.mp4") == "[]"
    assert client.calls[0]["use_video"] is False
    assert client.calls[0]["use_audio"] is True
