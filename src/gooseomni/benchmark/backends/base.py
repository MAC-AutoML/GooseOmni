from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from gooseomni.models.utils.omni_http_client import OmniHttpClient


class BaseBackend(ABC):
    @abstractmethod
    def annotate_video(self, video_path: Path, prompt: str) -> str:
        raise NotImplementedError

    def annotate_audio(self, video_path: Path, prompt: str) -> str:
        return self.annotate_video(video_path, prompt)

    @abstractmethod
    def annotate_text(self, prompt: str) -> str:
        raise NotImplementedError


class MockBackend(BaseBackend):
    def annotate_video(self, video_path: Path, prompt: str) -> str:
        if "TASK: utterance" in prompt:
            return json.dumps(
                [
                    {
                        "dataset": "gooseomni",
                        "game_id": "g001",
                        "segment_id": "mock_segment",
                        "player_id": "Gemini",
                        "utterance_id": "utt_001",
                        "speaker": "unknown",
                        "speaker_confidence": 0.4,
                        "transcript": "mock utterance",
                        "text": "mock utterance",
                        "addressee": [],
                        "mentioned_players": [],
                        "speech_act": "unknown",
                        "claims": [],
                        "possible_intents": [],
                        "local_start_sec": 0.0,
                        "local_end_sec": 1.0,
                        "certainty": 0.4,
                        "evidence": "mock audio evidence",
                        "source_pov": ["Gemini"],
                        "is_direct_observation": False,
                        "is_speech_claim": True,
                        "needs_human_review": True,
                    }
                ],
                ensure_ascii=False,
            )
        return json.dumps(
            [
                {
                    "dataset": "gooseomni",
                    "game_id": "g001",
                    "segment_id": "mock_segment",
                    "player_id": "Gemini",
                    "event_id": "local_e001",
                    "local_start_sec": 0.0,
                    "local_end_sec": 1.0,
                    "event_type": "mock_observation",
                    "description": "Mock POV event.",
                    "actor": "Gemini",
                    "visible_players": [],
                    "mentioned_players": [],
                    "location": "unknown",
                    "speaker": None,
                    "utterance": None,
                    "claim_type": None,
                    "certainty": 0.8,
                    "evidence": "mock visual evidence",
                    "source_pov": ["Gemini"],
                    "is_direct_observation": True,
                    "is_speech_claim": False,
                    "needs_human_review": False,
                }
            ],
            ensure_ascii=False,
        )

    def annotate_text(self, prompt: str) -> str:
        if "TASK: phase_event" in prompt:
            return json.dumps(
                [
                    {
                        "dataset": "gooseomni",
                        "game_id": "g001",
                        "segment_id": "mock_segment",
                        "phase_event_id": "phase_001",
                        "local_start_sec": 0.0,
                        "local_end_sec": 1.0,
                        "phase_type": "discussion",
                        "visible_to_all": True,
                        "evidence_povs": ["Gemini"],
                        "certainty": 0.8,
                        "evidence": "mock phase evidence",
                        "source_pov": ["Gemini"],
                        "is_direct_observation": True,
                        "is_speech_claim": False,
                        "needs_human_review": False,
                    }
                ],
                ensure_ascii=False,
            )
        if "TASK: global_merge" in prompt:
            return json.dumps(
                [
                    {
                        "dataset": "gooseomni",
                        "game_id": "g001",
                        "segment_id": "mock_segment",
                        "target_player": "global",
                        "global_event_id": "mock_global_event_0001",
                        "local_start_sec": 0.0,
                        "local_end_sec": 1.0,
                        "event_type": "mock_global_event",
                        "description": "Mock merged event.",
                        "actors": [],
                        "involved_players": [],
                        "visible_to": ["Gemini"],
                        "heard_by": [],
                        "supporting_pov_event_ids": [],
                        "supporting_utterance_ids": [],
                        "conflict": False,
                        "certainty": 0.8,
                        "evidence": "mock merge evidence",
                        "source_pov": ["Gemini"],
                        "is_direct_observation": True,
                        "is_speech_claim": False,
                        "needs_human_review": False,
                    }
                ],
                ensure_ascii=False,
            )
        if "TASK: information_state" in prompt:
            return json.dumps(
                {
                    "dataset": "gooseomni",
                    "game_id": "g001",
                    "segment_id": "mock_segment",
                    "target_player": "Gemini",
                    "cutoff_abs_sec": 1.0,
                    "available_information": [],
                    "unknown_or_unseen_information": [],
                    "suspicion_state": {},
                    "trust_state": {},
                    "known_facts": ["mock known fact"],
                    "beliefs": [],
                    "unknowns": [],
                    "private_observations": [],
                    "public_information": [],
                    "false_or_uncertain_beliefs": [],
                    "certainty": 0.8,
                    "evidence": "mock state evidence",
                    "source_pov": ["Gemini"],
                    "needs_human_review": False,
                },
                ensure_ascii=False,
            )
        if "TASK: memory_state" in prompt:
            return json.dumps(
                {
                    "dataset": "gooseomni",
                    "game_id": "g001",
                    "segment_id": "mock_segment",
                    "target_player": "Gemini",
                    "cutoff_abs_sec": 1.0,
                    "memory_items": [
                        {
                            "memory_id": "mem_Gemini_001",
                            "first_observed_abs_sec": 0.0,
                            "last_referenced_abs_sec": 1.0,
                            "memory_type": "direct_visual",
                            "content": "mock memory",
                            "source_event_ids": ["local_e001"],
                            "source_claim_ids": [],
                            "confidence": 0.8,
                            "decay_status": "active",
                            "visibility": "private",
                            "needs_human_review": False,
                        }
                    ],
                    "memory_delta": [
                        {
                            "operation": "add",
                            "memory_id": "mem_Gemini_001",
                            "reason": "mock update",
                        }
                    ],
                    "needs_human_review": False,
                },
                ensure_ascii=False,
            )
        if "TASK: belief_state" in prompt:
            return json.dumps(
                {
                    "dataset": "gooseomni",
                    "game_id": "g001",
                    "segment_id": "mock_segment",
                    "target_player": "Gemini",
                    "cutoff_abs_sec": 1.0,
                    "knows": [{"content": "mock known fact", "basis": "direct_visual", "source_ids": ["local_e001"]}],
                    "does_not_know": [],
                    "believes_or_suspects": [],
                    "trust_state": {},
                    "forbidden_information": [],
                    "certainty": 0.8,
                    "evidence": "mock belief evidence",
                    "source_pov": ["Gemini"],
                    "needs_human_review": False,
                },
                ensure_ascii=False,
            )
        if "TASK: checker" in prompt:
            return json.dumps(
                [
                    {
                        "checker_name": "mock_checker",
                        "annotation_id": "mock_annotation",
                        "verdict": "pass",
                        "reason": "mock checker pass",
                        "suggested_fix": None,
                        "confidence": 0.8,
                        "leakage": False,
                        "leaked_items": [],
                    }
                ],
                ensure_ascii=False,
            )
        if "TASK: candidate_trial" in prompt:
            return json.dumps(
                [
                    {
                        "dataset": "gooseomni",
                        "game_id": "g001",
                        "segment_id": "mock_segment",
                        "trial_id": "mock_trial_0001",
                        "question_type": "first_order_belief",
                        "trial_type": "belief_state",
                        "target_player": "Gemini",
                        "cutoff_abs_sec": 1.0,
                        "question": "Gemini 知道什么？",
                        "answer": "mock known fact",
                        "distractors": [],
                        "available_information": ["mock known fact"],
                        "hidden_information": [],
                        "expected_answer_basis": "Only use target player visible/heard information.",
                        "supporting_global_event_ids": ["mock_global_event_0001"],
                        "supporting_information_state_ids": [],
                        "risk_of_perspective_leakage": "low",
                        "certainty": 0.8,
                        "evidence": "mock trial evidence",
                        "source_pov": ["Gemini"],
                        "needs_human_review": False,
                    }
                ],
                ensure_ascii=False,
            )
        return "[]"


@dataclass(frozen=True)
class QwenOmniBackend(BaseBackend):
    model: str = "qwen3-omni"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    server_url: str | None = None

    @property
    def requires_video_for_text(self) -> bool:
        return self._local_server_url is not None

    @property
    def _local_server_url(self) -> str | None:
        return self.server_url or os.getenv("QWEN3_OMNI_SERVER_URL")

    def annotate_video(self, video_path: Path, prompt: str) -> str:
        if self._local_server_url:
            return self._local_video_call(video_path, prompt)
        return self._chat(f"{prompt}\n\nLocal video path: {video_path.as_posix()}")

    def annotate_audio(self, video_path: Path, prompt: str) -> str:
        if self._local_server_url:
            return self._local_video_call(video_path, prompt, use_video=False, use_audio=True)
        return self.annotate_video(video_path, prompt)

    def annotate_text(self, prompt: str) -> str:
        if self._local_server_url:
            raise ValueError("Local Qwen3-Omni server requires annotate_video with a segment context video")
        return self._chat(prompt)

    def _local_video_call(self, video_path: Path, prompt: str, *, use_video: bool = True, use_audio: bool = True) -> str:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video_path not found: {path}")
        server_url = self._local_server_url
        if not server_url:
            raise ValueError("Missing local Qwen3-Omni server URL")
        answer = OmniHttpClient(server_url).call_api(str(path), prompt, use_video=use_video, use_audio=use_audio)
        if answer is None:
            raise RuntimeError("Qwen3-Omni local server returned no answer")
        return answer

    def _chat(self, prompt: str) -> str:
        from openai import OpenAI

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {self.api_key_env}")
        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""



def create_backend(
    backend: str,
    *,
    model: str = "qwen3-omni",
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    server_url: str | None = None,
) -> BaseBackend:
    backend_name = backend.lower()
    if backend_name == "mock":
        return MockBackend()
    if backend_name in {"local", "local-server", "server"}:
        return QwenOmniBackend(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            server_url=server_url or os.getenv("QWEN3_OMNI_SERVER_URL"),
        )
    if backend_name in {"qwen", "openai", "openai-compatible"}:
        return QwenOmniBackend(model=model, api_key_env=api_key_env, base_url=base_url)
    raise ValueError(f"Unsupported backend: {backend}")
