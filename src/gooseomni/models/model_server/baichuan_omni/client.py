from __future__ import annotations

import os

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.utils.omni_http_client import OmniHttpClient


class BaichuanOmni15Client:
    @property
    def model_name(self) -> str:
        return "baichuan_omni_1_5"

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

    def _call_api(
        self,
        server_url: str,
        video_path: str | None,
        question: str,
        use_video: bool = True,
        use_audio: bool = True,
        max_retries: int = 5,
        retry_delay: float = 0.0,
    ) -> str | None:
        return OmniHttpClient(server_url).call_api(
            video_path,
            question,
            use_video=use_video,
            use_audio=use_audio,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def _build_prompt(self, request: InferenceRequest) -> str:
        asr_content = ""
        if request.metadata:
            asr_content = request.metadata.get("asr_content") or ""
        options = request.options or []
        question = request.prompt or ""
        parts = []
        if asr_content:
            parts.append("ASR Transcript:\n" + asr_content.strip())
        if options:
            parts.append("Options:\n" + "\n".join(options))
        if question:
            parts.append(question.strip())
        if options:
            parts.append("Answer ONLY with the option letter (A, B, C, or D).")
        return "\n\n".join(parts)

    def predict(self, request: InferenceRequest) -> InferenceResult:
        model_config = CONFIG.model("baichuan_omni_1_5")
        server_url = request.metadata.get("server_url") if request.metadata else None
        server_url = (
            server_url
            or model_config.get("server_url")
            or os.getenv("BAICHUAN_OMNI_1_5_SERVER_URL")
        )
        if not server_url:
            raise ValueError(
                "Missing Baichuan-Omni-1.5 server_url. Please configure it in configs/gooseomni.yaml or environment variables."
            )

        full_question = self._build_prompt(request)

        use_video = request.use_video

        media_path = (
            request.require_video_path() if (use_video or request.use_audio) else None
        )
        raw_answer = self._call_api(
            server_url,
            media_path,
            full_question,
            use_video=use_video,
            use_audio=request.use_audio,
            max_retries=CONFIG.runtime("max_retries", 5),
            retry_delay=CONFIG.runtime("request_delay", 0.0),
        )
        clean_answer = self._extract_clean_answer(raw_answer or "")
        return InferenceResult(
            text=raw_answer or "", parsed_answer=clean_answer, model=self.model_name
        )
