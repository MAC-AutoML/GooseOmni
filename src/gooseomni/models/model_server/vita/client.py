from __future__ import annotations

import os

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.utils.omni_http_client import OmniHttpClient


class Vita15Client:
    @property
    def model_name(self) -> str:
        return "vita_1_5"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        model_config = CONFIG.model("vita_1_5")
        server_url = request.metadata.get("server_url") if request.metadata else None
        server_url = server_url or model_config.get("server_url") or os.getenv("VITA_1_5_SERVER_URL")
        if not server_url:
            raise ValueError("Missing VITA-1.5 server_url. Please configure it in configs/gooseomni.yaml or environment variables.")

        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        user_prompt = user_prompt or model_config.get("user_prompt")
        use_video = request.use_video

        client = OmniHttpClient(server_url)
        media_path = request.require_video_path() if (use_video or request.use_audio) else None
        raw_answer = client.call_api(
            media_path,
            request.prompt,
            user_prompt=user_prompt,
            use_video=use_video,
            use_audio=request.use_audio,
            max_retries=CONFIG.runtime("max_retries", 5),
            retry_delay=CONFIG.runtime("request_delay", 0.0),
        )
        return InferenceResult(text=raw_answer or "", parsed_answer=raw_answer or "", model=self.model_name)
