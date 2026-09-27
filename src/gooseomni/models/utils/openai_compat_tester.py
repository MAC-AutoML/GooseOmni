from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from gooseomni.config.paths import PATHS
from gooseomni.config.settings import CONFIG


class GeminiSafetyBlockedError(RuntimeError):
    """Gemini request blocked by safety policy (skip this sample without retry)."""


class OpenAICompatTester:
    """OpenAI-compatible multimodal caller (text + image)."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = api_base or os.getenv("OPENAI_API_BASE") or CONFIG.api("openai").get("base_url")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or CONFIG.api("openai").get("api_key")
        if not self.base_url or not self.api_key:
            raise ValueError("Missing OpenAI API base_url or api_key. Check .env or configs/gooseomni.yaml.")
        self.logger = logging.getLogger(f"openai_compat_tester.{model_name}")
        if not self.logger.handlers:
            model_log_dir = PATHS.results_logs / model_name
            model_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = model_log_dir / "openai_compat_tester.log"
            handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _extract_frames(
        self,
        video_path: str,
        frame_interval_sec: int,
        max_frames: int | None,
        jpeg_quality: int = 50,
    ) -> list[str]:
        try:
            import cv2
        except Exception as exc:  # noqa: BLE001
            self.logger.error("OpenCV is not available: %s", exc)
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = int(round(fps * frame_interval_sec)) if fps and fps > 0 else 0

        frames_base64: list[str] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if step and idx % step != 0:
                idx += 1
                continue

            frame = cv2.resize(frame, (128, 128))
            quality = max(10, min(95, int(jpeg_quality)))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frames_base64.append(base64.b64encode(buffer).decode('utf-8'))

            if max_frames is not None and len(frames_base64) >= max_frames:
                break

            if step == 0:
                break
            idx += 1

        cap.release()
        return frames_base64

    def _extract_audio(self, video_path: str) -> str:
        with tempfile.TemporaryDirectory(prefix="gooseomni_api_audio_") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    video_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(audio_path),
                ],
                check=True,
                timeout=300,
            )
            if not audio_path.is_file() or not audio_path.stat().st_size:
                raise RuntimeError("No audio extracted from video.")
            return base64.b64encode(audio_path.read_bytes()).decode("ascii")

    def call(
        self,
        video_path: str | None,
        question: str,
        user_prompt: str | None = None,
        model_params: dict | None = None,
        include_images: bool = True,
        include_audio: bool = False,
    ) -> str:
        frame_interval_sec = int(CONFIG.runtime("frame_interval_sec", 1))
        configured_max_frames = CONFIG.runtime("max_frames", None)
        if model_params and model_params.get("max_frames") is not None:
            configured_max_frames = model_params["max_frames"]
        if configured_max_frames is not None:
            try:
                configured_max_frames = int(configured_max_frames)
            except (TypeError, ValueError):
                configured_max_frames = None

        raw_api_retries = os.getenv("GOOSEOMNI_API_MAX_RETRIES")
        if raw_api_retries is None:
            default_retries = 2 if self._is_gemini_model() else CONFIG.runtime("max_retries", 5)
            raw_api_retries = CONFIG.runtime("api_max_retries", default_retries)
        api_max_retries = int(raw_api_retries or 1)

        raw_api_retry_delay = os.getenv("GOOSEOMNI_API_RETRY_DELAY")
        if raw_api_retry_delay is None:
            raw_api_retry_delay = CONFIG.runtime("api_retry_delay", CONFIG.runtime("request_delay", 1.0))
        api_retry_delay = float(raw_api_retry_delay or 1.0)

        raw_request_timeout = os.getenv("GOOSEOMNI_API_REQUEST_TIMEOUT")
        if raw_request_timeout is None:
            default_timeout = 45 if self._is_gemini_model() else 120
            raw_request_timeout = CONFIG.runtime("api_request_timeout", default_timeout)
        api_request_timeout = float(raw_request_timeout or 30)

        if api_max_retries < 1:
            api_max_retries = 1
        if api_request_timeout < 1:
            api_request_timeout = 1.0

        full_question = f"{user_prompt}\n\n{question}" if user_prompt else question
        max_frames_try = configured_max_frames
        jpeg_quality_try = 50
        last_error = "unknown"

        for attempt in range(1, api_max_retries + 1):
            frames: list[str] = []
            if (include_images or include_audio) and not video_path:
                raise ValueError("Image or audio input requires video_path.")
            if include_images:
                frames = self._extract_frames(
                    str(video_path),
                    frame_interval_sec,
                    max_frames_try,
                    jpeg_quality=jpeg_quality_try,
                )
                if not frames:
                    raise RuntimeError("No frames extracted from video.")

            content = [{"type": "text", "text": full_question}]
            if include_images:
                for frame in frames:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                    )
            if include_audio:
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": self._extract_audio(str(video_path)),
                            "format": "wav",
                        },
                    }
                )

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
            }
            if model_params:
                max_tokens = model_params.get("max_tokens")
                temperature = model_params.get("temperature")
                top_p = model_params.get("top_p")
                if max_tokens is not None:
                    payload["max_tokens"] = int(max_tokens)
                if temperature is not None:
                    payload["temperature"] = float(temperature)
                if top_p is not None:
                    payload["top_p"] = float(top_p)

            try:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=api_request_timeout,
                )
                if response.status_code >= 400:
                    body = response.text[:1000]
                    if self._is_gemini_safety_blocked(body):
                        raise GeminiSafetyBlockedError(
                            f"Gemini safety blocked: model={self.model_name}, status={response.status_code}, body={body}"
                        )
                    # 400 often means payload too large or format edge cases:
                    # degrade JPEG quality only, keep frame count unchanged.
                    if response.status_code == 400 and include_images and jpeg_quality_try > 15:
                        jpeg_quality_try = max(15, int(jpeg_quality_try * 0.7))
                        self.logger.warning(
                            "400 for %s, reduce jpeg quality to %s (keep frame count) and retry (attempt %s/%s). body=%s",
                            self.model_name,
                            jpeg_quality_try,
                            attempt,
                            api_max_retries,
                            body[:200],
                        )
                        last_error = f"status=400 body={body}"
                        time.sleep(min(60.0, api_retry_delay * (2 ** (attempt - 1))))
                        continue
                    raise RuntimeError(
                        f"OpenAI-compatible API error: status={response.status_code}, model={self.model_name}, body={body}"
                    )

                data = response.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise RuntimeError(
                        f"OpenAI-compatible API returned empty choices: model={self.model_name}, body={str(data)[:1000]}"
                    )
                message = choices[0].get("message", {})
                response_content = message.get("content", "")
                if isinstance(response_content, list):
                    response_content = " ".join(
                        str(x.get("text", "")) for x in response_content if isinstance(x, dict)
                    )
                return str(response_content or "")
            except GeminiSafetyBlockedError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self.logger.warning(
                    "API call failed for %s (attempt %s/%s): %s",
                    self.model_name,
                    attempt,
                    api_max_retries,
                    last_error,
                )
                if attempt >= api_max_retries:
                    break
                time.sleep(min(60.0, api_retry_delay * (2 ** (attempt - 1))))

        raise RuntimeError(f"OpenAI-compatible API failed after retries: model={self.model_name}, error={last_error}")

    def _is_gemini_safety_blocked(self, message: str) -> bool:
        model_name = self.model_name.lower()
        if "gemini" not in model_name:
            return False
        text = (message or "").lower()
        return (
            "prohibited_content" in text
            or "request_body_blocked" in text
            or "prompt_blocked" in text
            or "content is prohibited under official usage policies" in text
        )

    def _is_gemini_model(self) -> bool:
        return "gemini" in self.model_name.lower()
