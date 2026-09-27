from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferPayload:
    prompt: str
    use_video: bool
    use_audio: bool
    visual_mask: bool
    upload: Any | None
    media_start_sec: float | None


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_infer_request(request: Any, default_audio: bool = True) -> InferPayload:
    data = request.get_json(silent=True) if request.is_json else request.form
    data = data or {}
    prompt = str(data.get("prompt", "")).strip()
    use_video = _bool(data.get("use_video"), True)
    use_audio = _bool(data.get("use_audio"), default_audio)
    visual_mask = _bool(data.get("visual_mask"), False)
    media_start_raw = data.get("media_start_sec")
    media_start_sec = (
        float(media_start_raw) if media_start_raw not in {None, ""} else None
    )
    upload = request.files.get("video")
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if (use_video or use_audio) and (upload is None or not upload.filename):
        raise ValueError("Media file is required when video or audio input is enabled")
    return InferPayload(
        prompt, use_video, use_audio, visual_mask, upload, media_start_sec
    )
