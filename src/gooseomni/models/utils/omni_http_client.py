from __future__ import annotations

import logging
import os
import time

import requests


class OmniHttpClient:
    """Minimal HTTP client: upload video + question, return cleaned answer."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.logger = logging.getLogger("gooseomni.models.utils.omni_http_client")
        self.timeout_sec = float(os.getenv("OMNI_HTTP_TIMEOUT_SEC", "300"))

    def _extract_clean_answer(self, raw_answer: str) -> str:
        if not raw_answer:
            return ""

        lines = [line.strip() for line in raw_answer.split("\n") if line.strip()]
        if not lines:
            return raw_answer.strip()

        last_line = lines[-1]
        if last_line[:1].upper() in {"A", "B", "C", "D"} and len(last_line) == 1:
            return last_line.upper()

        return raw_answer.strip()

    def call_api(
        self,
        video_path: str | None,
        question: str,
        user_prompt: str | None = None,
        use_video: bool = True,
        use_audio: bool = True,
        visual_mask: bool = False,
        media_start_sec: float | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ) -> str | None:
        if (use_video or use_audio) and not video_path:
            raise ValueError(
                "video_path is required when video or audio input is enabled"
            )
        video_name = os.path.basename(video_path) if video_path else "text-only"
        self.logger.info("Calling OMNI API: %s", video_name)

        full_question = f"{user_prompt}\n\n{question}" if user_prompt else question

        for attempt in range(1, max_retries + 1):
            try:
                payload = {
                    "prompt": full_question,
                    "use_video": use_video,
                    "use_audio": use_audio,
                    "visual_mask": visual_mask,
                }
                if media_start_sec is not None:
                    payload["media_start_sec"] = media_start_sec
                if video_path:
                    with open(video_path, "rb") as video_file:
                        response = requests.post(
                            f"{self.server_url}/v1/infer",
                            files={"video": video_file},
                            data={
                                key: str(value).lower()
                                if isinstance(value, bool)
                                else value
                                for key, value in payload.items()
                            },
                            timeout=self.timeout_sec,
                        )
                else:
                    response = requests.post(
                        f"{self.server_url}/v1/infer",
                        json=payload,
                        timeout=self.timeout_sec,
                    )

                if response.status_code == 200:
                    raw_answer = response.json().get("answer", "")
                    clean_answer = self._extract_clean_answer(raw_answer)
                    self.logger.info("OMNI API ok (attempt %s)", attempt)
                    return clean_answer

                error_text = response.text.lower()
                if response.status_code == 500 and "out of memory" in error_text:
                    raise RuntimeError("OMNI server CUDA out of memory")

                self.logger.warning(
                    "OMNI API bad status=%s (attempt %s)",
                    response.status_code,
                    attempt,
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, RuntimeError) and "out of memory" in str(exc):
                    raise
                self.logger.warning("OMNI API error (attempt %s): %s", attempt, exc)

            if attempt < max_retries:
                time.sleep(retry_delay)

        return None
