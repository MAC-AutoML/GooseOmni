from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceRequest:
    """Model-independent multimodal inference request."""

    prompt: str
    video_path: str | Path | None = None
    use_video: bool = True
    use_audio: bool = True
    options: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_video_path(self) -> str:
        if self.video_path is None:
            raise ValueError("This model request requires video_path")
        return str(self.video_path)


@dataclass(frozen=True)
class InferenceResult:
    """Normalized model inference result."""

    text: str
    parsed_answer: str | None = None
    latency_sec: float | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.parsed_answer if self.parsed_answer is not None else self.text

    @property
    def raw_response(self) -> str:
        return self.text
