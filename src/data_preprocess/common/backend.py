"""Vision/language backend: mock or local Qwen HTTP server (no gooseomni)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests


class ModelBackend(Protocol):
    def generate(
        self,
        prompt: str,
        video_path: str | None = None,
        *,
        image_paths: list[str] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "mock"  # mock | local
    server_url: str | None = None
    use_video: bool = True
    use_audio: bool = False
    timeout_sec: float = 300.0


class MockBackend:
    """Deterministic stub for offline tests."""

    def generate(
        self,
        prompt: str,
        video_path: str | None = None,
        *,
        image_paths: list[str] | None = None,
    ) -> str:
        _ = video_path, image_paths
        if "开局对齐审查员" in prompt or "start_screen" in prompt:
            return json.dumps(
                {"state": "start_screen", "reason": "mock"}, ensure_ascii=False
            )
        if "单局开局" in prompt or "game_start" in prompt:
            return "[]"
        if "单局结算" in prompt or "胜利" in prompt and "审查员" in prompt:
            return "[]"
        return "[]"


class LocalOmniBackend:
    """POST multipart to a local Omni HTTP server (same contract as project servers)."""

    def __init__(
        self,
        server_url: str,
        *,
        use_video: bool = True,
        use_audio: bool = False,
        timeout_sec: float = 300.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.use_video = use_video
        self.use_audio = use_audio
        self.timeout_sec = timeout_sec

    def generate(
        self,
        prompt: str,
        video_path: str | None = None,
        *,
        image_paths: list[str] | None = None,
    ) -> str:
        if image_paths and not video_path:
            answer = self._call(
                prompt,
                video_path=None,
                image_paths=image_paths,
                use_video=False,
                use_audio=False,
            )
            if not answer:
                raise RuntimeError("local Omni server returned empty answer")
            return answer
        if not video_path:
            raise ValueError("video_path required when not in image-only mode")
        answer = self._call(
            prompt,
            video_path=video_path,
            image_paths=None,
            use_video=self.use_video,
            use_audio=self.use_audio,
        )
        if not answer:
            raise RuntimeError("local Omni server returned empty answer")
        return answer

    def _call(
        self,
        prompt: str,
        *,
        video_path: str | None,
        image_paths: list[str] | None,
        use_video: bool,
        use_audio: bool,
        max_retries: int = 5,
    ) -> str | None:
        image_paths = [p for p in (image_paths or []) if p]
        for attempt in range(1, max_retries + 1):
            open_handles: list = []
            try:
                data = {
                    "prompt": prompt,
                    "use_video": str(use_video).lower(),
                    "use_audio": str(use_audio).lower(),
                    "visual_mask": "false",
                }
                files = []
                if video_path:
                    handle = open(video_path, "rb")
                    open_handles.append(handle)
                    files.append(
                        (
                            "video",
                            (
                                os.path.basename(video_path),
                                handle,
                                "application/octet-stream",
                            ),
                        )
                    )
                for index, image_path in enumerate(image_paths):
                    handle = open(image_path, "rb")
                    open_handles.append(handle)
                    files.append(
                        (
                            f"image_{index}",
                            (
                                os.path.basename(image_path),
                                handle,
                                "application/octet-stream",
                            ),
                        )
                    )
                response = requests.post(
                    f"{self.server_url}/generate",
                    data=data,
                    files=files or None,
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return str(payload.get("answer") or payload.get("text") or "").strip()
                return str(payload).strip()
            except Exception:  # noqa: BLE001
                if attempt >= max_retries:
                    raise
                time.sleep(3.0 * attempt)
            finally:
                for handle in open_handles:
                    handle.close()
        return None


def create_backend(config: BackendConfig) -> ModelBackend:
    kind = (config.backend or "mock").strip().lower()
    if kind == "mock":
        return MockBackend()
    if kind == "local":
        url = config.server_url or os.environ.get("QWEN3_OMNI_SERVER_URL")
        if not url:
            raise ValueError(
                "local backend needs --server-url or QWEN3_OMNI_SERVER_URL"
            )
        return LocalOmniBackend(
            url,
            use_video=config.use_video,
            use_audio=config.use_audio,
            timeout_sec=config.timeout_sec,
        )
    raise ValueError(f"unknown backend: {config.backend}")
