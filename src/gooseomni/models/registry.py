"""Static model registry without importing heavyweight model runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gooseomni.config import CONFIG, PATHS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    extra: str
    server_script: str | None = None
    native_inputs: tuple[str, ...] = ("text",)
    pipeline_inputs: tuple[str, ...] = ("text", "video", "audio")
    native_outputs: tuple[str, ...] = ("text",)
    pipeline_outputs: tuple[str, ...] = ("text",)

    @property
    def model_path(self) -> Path | None:
        value = CONFIG.model(self.name).get("model_path")
        return Path(value) if value else None

    @property
    def resolved_server_script(self) -> Path | None:
        return PATHS.root / self.server_script if self.server_script else None


MODEL_SPECS = {
    "gpt4o": ModelSpec(
        "gpt4o",
        "api",
        "api",
        native_inputs=("text", "image"),
        pipeline_inputs=("text", "video"),
    ),
    "gpt_audio": ModelSpec(
        "gpt_audio",
        "api",
        "api",
        native_inputs=("text", "audio"),
        pipeline_inputs=("text", "audio"),
    ),
    "gemini_2_5_flash": ModelSpec("gemini_2_5_flash", "api", "api", native_inputs=("text", "image", "video", "audio")),
    "gemini_2_5_pro": ModelSpec("gemini_2_5_pro", "api", "api", native_inputs=("text", "image", "video", "audio")),
    "gemini_3_flash_preview": ModelSpec("gemini_3_flash_preview", "api", "api", native_inputs=("text", "image", "video", "audio")),
    "gemini_3_pro_preview": ModelSpec("gemini_3_pro_preview", "api", "api", native_inputs=("text", "image", "video", "audio")),
    "qwen2_5_omni": ModelSpec("qwen2_5_omni", "local", "qwen", "src/gooseomni/models/model_server/qwen2_5_omni/qwen_omni_server.py", ("text", "image", "video", "audio"), native_outputs=("text", "audio")),
    "qwen3_omni": ModelSpec("qwen3_omni", "local", "qwen", "src/gooseomni/models/model_server/qwen3_omni/qwen3_omni_server.py", ("text", "image", "video", "audio"), native_outputs=("text", "audio")),
    "qwen3_omni_thinking": ModelSpec("qwen3_omni_thinking", "local", "qwen", "src/gooseomni/models/model_server/qwen3_omni_thinking/qwen3_omni_thinking_server.py", ("text", "image", "video", "audio")),
    "miniomni_2": ModelSpec("miniomni_2", "local", "miniomni", "src/gooseomni/models/model_server/miniomni_2/miniomni2_server.py", ("text", "image", "audio"), native_outputs=("text", "audio")),
    "omnivinci": ModelSpec("omnivinci", "local", "omnivinci", "src/gooseomni/models/model_server/omnivinci/omnivinci_server.py", ("text", "image", "video", "audio")),
    "vita_1_5": ModelSpec("vita_1_5", "local", "vita", "src/gooseomni/models/model_server/vita/vita_server.py", ("text", "image", "video", "audio"), native_outputs=("text", "audio")),
    "baichuan_omni_1_5": ModelSpec("baichuan_omni_1_5", "local", "baichuan", "src/gooseomni/models/model_server/baichuan_omni/baichuan_omni_server.py", ("text", "image", "video", "audio"), native_outputs=("text", "audio")),
    "ming": ModelSpec("ming", "local", "ming", "src/gooseomni/models/model_server/ming/ming_server.py", ("text", "image", "video", "audio"), native_outputs=("text", "image", "audio")),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc


def list_model_specs() -> list[ModelSpec]:
    return [MODEL_SPECS[name] for name in sorted(MODEL_SPECS)]
