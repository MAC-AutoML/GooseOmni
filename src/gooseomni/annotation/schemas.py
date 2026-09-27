from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Clip(BaseModel):
    game_id: str
    player_id: str
    clip_id: str
    clip_path: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    episode_id: str | None = None
    phase_index: int | None = Field(default=None, ge=0)
    phase_type: str | None = None
    source_clip_id: str | None = None

    @field_validator("clip_path")
    @classmethod
    def normalize_clip_path(cls, value: str) -> str:
        return str(Path(value))

    @field_validator("end_sec")
    @classmethod
    def validate_time_order(cls, value: float, info: Any) -> float:
        start_sec = info.data.get("start_sec")
        if start_sec is not None and value <= start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return value


class POVEvent(BaseModel):
    evidence_id: str | None = None
    clip_id: str
    game_id: str
    player_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    event_type: str
    description: str
    visible_players: list[str] = Field(default_factory=list)
    mentioned_players: list[str] = Field(default_factory=list)
    location: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    timestamp_evidence: str | None = None
    speaker_id: str | None = None
    speaker_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    utterance: str | None = None
    dual_pass_consistent: bool = False
    visual_text_evidence_id: str | None = None
    heard_by: list[str] = Field(default_factory=list)
    meeting_public: bool = False
    episode_id: str | None = None
    phase_index: int | None = Field(default=None, ge=0)
    phase_type: str | None = None
    source_clip_id: str | None = None
    public_ui: bool = False


class Utterance(BaseModel):
    game_id: str
    player_id: str
    clip_id: str | None = None
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    speaker_id: str | None = None
    text: str
    addressee_ids: list[str] = Field(default_factory=list)
    mentioned_players: list[str] = Field(default_factory=list)
    speech_act: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InformationState(BaseModel):
    game_id: str
    player_id: str
    cutoff_time: float = Field(ge=0)
    known_facts: list[str] = Field(default_factory=list)
    beliefs: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    private_observations: list[str] = Field(default_factory=list)
    public_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GlobalEvent(BaseModel):
    game_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    event_type: str
    description: str
    involved_players: list[str] = Field(default_factory=list)
    source_clip_ids: list[str] = Field(default_factory=list)
    source_player_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateTrial(BaseModel):
    game_id: str
    trial_id: str
    question_type: Literal[
        "first_order_belief",
        "second_order_belief",
        "hidden_information",
        "false_belief",
        "intent_inference",
    ]
    target_player_id: str
    cutoff_time: float = Field(ge=0)
    question: str
    answer: str
    distractors: list[str] = Field(default_factory=list)
    supporting_global_event_ids: list[str] = Field(default_factory=list)
    supporting_information_state_ids: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SyncOffset(BaseModel):
    game_id: str
    player_id: str
    raw_start_sec: float = Field(ge=0)
    evidence: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RoundBoundaryCandidate(BaseModel):
    game_id: str
    player_id: str
    clip_id: str
    round_index: int | None = None
    aligned_start_sec: float = Field(ge=0)
    aligned_end_sec: float = Field(gt=0)
    boundary_type: str
    evidence: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_povs: list[str] = Field(default_factory=list)
    needs_review: bool = False

    @field_validator("aligned_end_sec")
    @classmethod
    def validate_boundary_time_order(cls, value: float, info: Any) -> float:
        start_sec = info.data.get("aligned_start_sec")
        if start_sec is not None and value <= start_sec:
            raise ValueError("aligned_end_sec must be greater than aligned_start_sec")
        return value
