from __future__ import annotations

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.utils.openai_compat_tester import OpenAICompatTester


class GPTAudioClient:
    """Audio-only adapter for the gateway's verified GPT Audio route."""

    @property
    def model_name(self) -> str:
        return "gpt_audio"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        if request.use_video:
            raise ValueError("gpt_audio does not support visual input")
        if not request.use_audio:
            raise ValueError("gpt_audio requires audio input")

        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        model_config = CONFIG.model("gpt_audio")
        upstream_model = model_config.get("model_name", "gpt-audio-2025-08-28")
        tester = OpenAICompatTester(model_name=upstream_model)
        raw_answer = tester.call(
            request.require_video_path(),
            request.prompt,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=False,
            include_audio=True,
        )
        return InferenceResult(
            text=raw_answer or "",
            parsed_answer=raw_answer or "",
            model=self.model_name,
        )
