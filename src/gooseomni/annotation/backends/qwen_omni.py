from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gooseomni.annotation.json_utils import parse_json
from gooseomni.models.utils.omni_http_client import OmniHttpClient


class ModelBackend(Protocol):
    def generate(self, prompt: str, video_path: str | None = None) -> str: ...


@dataclass(frozen=True)
class QwenBackendConfig:
    backend: str = "mock"
    model: str = "qwen3-omni"
    server_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    use_video: bool = True
    use_audio: bool = True


class MockQwenOmniBackend:
    def generate(self, prompt: str, video_path: str | None = None) -> str:
        if "raw_start_sec" in prompt and "第一局正式游戏开始" in prompt:
            return (
                "{"
                '"game_id":"mock_game",'
                '"player_id":"mock_player",'
                '"raw_start_sec":0.0,'
                '"evidence":"mock sync offset",'
                '"confidence":0.1'
                "}"
            )
        clip_id = Path(video_path).stem if video_path else "mock_clip"
        parts = clip_id.split("_")
        game_id = parts[0] if len(parts) >= 4 else "mock_game"
        player_id = parts[1] if len(parts) >= 4 else "mock_player"
        start_sec = float(parts[-2]) if len(parts) >= 4 and parts[-2].isdigit() else 0.0
        end_sec = float(parts[-1]) if len(parts) >= 4 and parts[-1].isdigit() else 1.0
        return (
            "["
            "{"
            f'"clip_id":"{clip_id}",'
            f'"game_id":"{game_id}",'
            f'"player_id":"{player_id}",'
            f'"start_sec":{start_sec},'
            f'"end_sec":{end_sec},'
            '"event_type":"mock_observation",'
            '"description":"Mock backend placeholder event.",'
            '"visible_players":[],'
            '"mentioned_players":[],'
            '"location":null,'
            '"confidence":0.1,'
            '"evidence":"mock response"'
            "}"
            "]"
        )


class LocalQwenOmniServerBackend:
    def __init__(self, server_url: str, use_video: bool, use_audio: bool) -> None:
        self.client = OmniHttpClient(server_url)
        self.use_video = use_video
        self.use_audio = use_audio

    def generate(self, prompt: str, video_path: str | None = None) -> str:
        if not video_path:
            raise ValueError("Local Qwen3-Omni server backend requires video_path")
        if self.use_audio and not self.use_video:
            return self._generate_chunked_audio(prompt, video_path)
        if self.use_video and not self.use_audio:
            return self._generate_chunked_visual(prompt, video_path)
        return self._generate_once(prompt, video_path)

    def _generate_once(self, prompt: str, media_path: str) -> str:
        match = re.search(
            r"_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$", Path(media_path).stem
        )
        media_start_sec = float(match.group(1)) if match else None
        answer = self.client.call_api(
            media_path,
            prompt,
            use_video=self.use_video,
            use_audio=self.use_audio,
            media_start_sec=media_start_sec,
        )
        if answer is None:
            raise RuntimeError("Qwen3-Omni local server returned no answer")
        return answer

    def _generate_chunked_audio(self, prompt: str, video_path: str) -> str:
        window = _aligned_window(prompt)
        chunk_sec = float(os.getenv("QWEN3_OMNI_AUDIO_CHUNK_SEC", "30"))
        if window is None or window[1] - window[0] <= chunk_sec:
            return self._generate_once(prompt, video_path)

        rows: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="gooseomni-audio-") as temp_dir:
            for start_sec, end_sec in _audio_windows(*window, chunk_sec):
                output = Path(temp_dir) / (f"audio_{start_sec:.3f}_{end_sec:.3f}.wav")
                _extract_audio_chunk(
                    Path(video_path),
                    output,
                    offset_sec=start_sec - window[0],
                    duration_sec=end_sec - start_sec,
                )
                chunk_prompt = re.sub(
                    r"aligned_time=\[[^\]]+\]",
                    f"aligned_time=[{start_sec}, {end_sec}]",
                    prompt,
                    count=1,
                )
                payload = parse_json(self._generate_once(chunk_prompt, str(output)))
                if not isinstance(payload, list):
                    raise ValueError("audio chunk response must be a JSON array")
                rows.extend(payload)
        return json.dumps(rows, ensure_ascii=False)

    def _generate_chunked_visual(self, prompt: str, video_path: str) -> str:
        window = _aligned_window(prompt)
        chunk_sec = float(os.getenv("QWEN3_OMNI_VISUAL_CHUNK_SEC", "30"))
        if window is None or window[1] - window[0] <= chunk_sec:
            return self._generate_once(prompt, video_path)

        rows: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="gooseomni-visual-") as temp_dir:
            for start_sec, end_sec in _audio_windows(*window, chunk_sec):
                output = Path(temp_dir) / (f"video_{start_sec:.3f}_{end_sec:.3f}.mp4")
                _extract_video_chunk(
                    Path(video_path),
                    output,
                    offset_sec=start_sec - window[0],
                    duration_sec=end_sec - start_sec,
                )
                chunk_prompt = re.sub(
                    r"aligned_time=\[[^\]]+\]",
                    f"aligned_time=[{start_sec}, {end_sec}]",
                    prompt,
                    count=1,
                )
                payload = parse_json(self._generate_once(chunk_prompt, str(output)))
                if not isinstance(payload, list):
                    raise ValueError("visual chunk response must be a JSON array")
                rows.extend(payload)
        return json.dumps(rows, ensure_ascii=False)


def _aligned_window(prompt: str) -> tuple[float, float] | None:
    match = re.search(
        r"aligned_time=\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        prompt,
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _audio_windows(
    start_sec: float,
    end_sec: float,
    chunk_sec: float,
) -> list[tuple[float, float]]:
    if chunk_sec <= 0:
        raise ValueError("QWEN3_OMNI_AUDIO_CHUNK_SEC must be positive")
    windows = []
    current = start_sec
    while current < end_sec:
        next_end = min(current + chunk_sec, end_sec)
        windows.append((current, next_end))
        current = next_end
    return windows


def _extract_audio_chunk(
    source: Path,
    output: Path,
    offset_sec: float,
    duration_sec: float,
) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{offset_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        message = completed.stderr.strip() or "ffmpeg did not create audio chunk"
        raise RuntimeError(f"audio chunk extraction failed: {message}")


def _extract_video_chunk(
    source: Path,
    output: Path,
    offset_sec: float,
    duration_sec: float,
) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{offset_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-y",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        message = completed.stderr.strip() or "ffmpeg did not create video chunk"
        raise RuntimeError(f"visual chunk extraction failed: {message}")


class OpenAICompatibleBackend:
    def __init__(self, config: QwenBackendConfig) -> None:
        from openai import OpenAI

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable {config.api_key_env}"
            )
        self.model = config.model
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, prompt: str, video_path: str | None = None) -> str:
        content = prompt
        if video_path:
            content = f"{prompt}\n\n本地视频路径：{video_path}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def create_backend(config: QwenBackendConfig) -> ModelBackend:
    backend = config.backend.lower()
    if backend == "mock":
        return MockQwenOmniBackend()
    if backend in {"local", "local-server", "server"}:
        server_url = config.server_url or os.getenv("QWEN3_OMNI_SERVER_URL")
        if not server_url:
            server_url = "http://127.0.0.1:5090"
        return LocalQwenOmniServerBackend(
            server_url,
            use_video=config.use_video,
            use_audio=config.use_audio,
        )
    if backend in {"openai", "openai-compatible"}:
        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
