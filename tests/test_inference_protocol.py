from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask, request

from gooseomni.models.model_server.local_common.http import parse_infer_request
from gooseomni.models.utils.omni_http_client import OmniHttpClient


class _Response:
    status_code = 200

    @staticmethod
    def json() -> dict[str, str]:
        return {"answer": "ok"}


def test_http_client_uses_json_for_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr("requests.post", fake_post)
    result = OmniHttpClient("http://localhost:5090").call_api(
        None, "question", use_video=False, use_audio=False
    )
    assert result == "ok"
    assert calls[0][1]["json"]["use_video"] is False
    assert "files" not in calls[0][1]


def test_http_client_uses_multipart_for_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr("requests.post", fake_post)
    OmniHttpClient("http://localhost:5090").call_api(
        str(media), "question", use_video=True, use_audio=False
    )
    assert "files" in calls[0][1]
    assert calls[0][1]["data"]["use_audio"] == "false"


def test_parse_infer_request_accepts_text_json() -> None:
    app = Flask(__name__)
    with app.test_request_context(
        "/v1/infer",
        method="POST",
        json={"prompt": "hello", "use_video": False, "use_audio": False},
    ):
        payload = parse_infer_request(request)
    assert payload.prompt == "hello"
    assert payload.upload is None


def test_parse_infer_request_rejects_enabled_media_without_upload() -> None:
    app = Flask(__name__)
    with app.test_request_context(
        "/v1/infer", method="POST", json={"prompt": "hello", "use_video": True}
    ):
        with pytest.raises(ValueError, match="Media file"):
            parse_infer_request(request)
