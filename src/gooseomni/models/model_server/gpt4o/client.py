from __future__ import annotations

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.utils.openai_compat_tester import OpenAICompatTester


class GPT4oClient:
    @property
    def model_name(self) -> str:
        return "gpt4o"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        if request.use_audio:
            raise ValueError(
                "gpt4o does not support audio input; use gpt_audio for audio-only requests"
            )

        # level1_pipeline already assembles ASR/options/answer_format into user_prompt; avoid duplicate assembly here.
        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        use_video = request.use_video

        model_config = CONFIG.model("gpt4o")
        model_name = model_config.get("model_name", "gpt-4o")
        tester = OpenAICompatTester(model_name=model_name)
        media_path = request.require_video_path() if request.use_video else None
        raw_answer = tester.call(
            media_path,
            request.prompt,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=use_video,
            include_audio=False,
        )
        return InferenceResult(
            text=raw_answer or "", parsed_answer=raw_answer or "", model=self.model_name
        )
