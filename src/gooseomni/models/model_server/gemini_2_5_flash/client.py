from __future__ import annotations

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.types import InferenceRequest, InferenceResult
from gooseomni.models.utils.openai_compat_tester import OpenAICompatTester


class Gemini25FlashClient:
    @property
    def model_name(self) -> str:
        return "gemini_2_5_flash"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        # level1_pipeline already assembles ASR/options/answer_format into user_prompt; avoid duplicate assembly here.
        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        use_video = request.use_video

        model_config = CONFIG.model("gemini_2_5_flash")
        model_name = model_config.get("model_name", "gemini-2.5-flash")
        tester = OpenAICompatTester(model_name=model_name)
        media_path = (
            request.require_video_path()
            if (request.use_video or request.use_audio)
            else None
        )
        raw_answer = tester.call(
            media_path,
            request.prompt,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=use_video,
            include_audio=request.use_audio,
        )
        return InferenceResult(
            text=raw_answer or "", parsed_answer=raw_answer or "", model=self.model_name
        )
