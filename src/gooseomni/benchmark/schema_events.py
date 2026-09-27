from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DATASET_NAME = "gooseomni"
VALID_PLAYERS = ("Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu")
VALID_PLAYER_SET = set(VALID_PLAYERS)


def _validate_player(value: str) -> str:
    if value not in VALID_PLAYER_SET:
        raise ValueError(f"player_id must be one of {sorted(VALID_PLAYER_SET)}")
    return value


class POVRef(BaseModel):
    player_id: str
    video_file: str
    source_raw_video: str | None = None
    base_sync_raw_start_sec: float | None = None
    sync_correction_sec: float | None = None
    corrected_sync_raw_start_sec: float | None = None
    raw_start_sec: float | None = None
    raw_end_sec: float | None = None
    aligned_start_sec: float = Field(ge=0)
    aligned_end_sec: float = Field(gt=0)
    qwen_round_boundary_candidates: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        return _validate_player(value)

    @field_validator("video_file")
    @classmethod
    def normalize_video_file(cls, value: str) -> str:
        return Path(value).as_posix()

    @model_validator(mode="after")
    def validate_time_order(self) -> POVRef:
        if self.aligned_end_sec <= self.aligned_start_sec:
            raise ValueError("aligned_end_sec must be greater than aligned_start_sec")
        return self


class Segment(BaseModel):
    dataset: str = DATASET_NAME
    segment_id: str
    original_segment_id: str | None = None
    game_id: str
    aligned_start_sec: float = Field(ge=0)
    aligned_end_sec: float = Field(gt=0)
    duration_sec: float = Field(gt=0)
    pov_count: int = 6
    povs: list[POVRef]

    @model_validator(mode="after")
    def validate_segment(self) -> Segment:
        if self.dataset != DATASET_NAME:
            raise ValueError(f"dataset must be {DATASET_NAME}")
        if self.aligned_end_sec <= self.aligned_start_sec:
            raise ValueError("aligned_end_sec must be greater than aligned_start_sec")
        if (
            abs((self.aligned_end_sec - self.aligned_start_sec) - self.duration_sec)
            > 1e-6
        ):
            raise ValueError(
                "duration_sec must equal aligned_end_sec - aligned_start_sec"
            )
        players = [pov.player_id for pov in self.povs]
        if len(players) != self.pov_count:
            raise ValueError("pov_count must match number of povs")
        if sorted(players) != sorted(VALID_PLAYERS):
            raise ValueError(
                "strict gooseomni segment must contain exactly the 6 valid players"
            )
        for pov in self.povs:
            if pov.aligned_start_sec != self.aligned_start_sec:
                raise ValueError("POV aligned_start_sec must match segment")
            if pov.aligned_end_sec != self.aligned_end_sec:
                raise ValueError("POV aligned_end_sec must match segment")
        return self


class TimedEvidenceModel(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    local_start_sec: float = Field(ge=0)
    local_end_sec: float = Field(gt=0)
    abs_start_sec: float = Field(ge=0)
    abs_end_sec: float = Field(gt=0)
    certainty: float = Field(ge=0.0, le=1.0)
    evidence: str
    source_pov: list[str]
    is_direct_observation: bool
    is_speech_claim: bool
    needs_human_review: bool = False

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        if value != DATASET_NAME:
            raise ValueError(f"dataset must be {DATASET_NAME}")
        return value

    @field_validator("source_pov")
    @classmethod
    def validate_source_pov(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("source_pov must not be empty")
        for player_id in value:
            _validate_player(player_id)
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> TimedEvidenceModel:
        if self.local_end_sec <= self.local_start_sec:
            raise ValueError("local_end_sec must be greater than local_start_sec")
        if self.abs_end_sec <= self.abs_start_sec:
            raise ValueError("abs_end_sec must be greater than abs_start_sec")
        return self


class POVEvent(TimedEvidenceModel):
    event_id: str
    player_id: str
    event_type: str
    description: str
    actor: str | None = None
    visible_players: list[str] = Field(default_factory=list)
    mentioned_players: list[str] = Field(default_factory=list)
    location: str = "unknown"
    speaker: str | None = None
    utterance: str | None = None
    claim_type: str | None = None

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        return _validate_player(value)

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, value: str | None) -> str | None:
        if value in {None, "unknown"}:
            return value
        return _validate_player(value)

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, value: str | None) -> str | None:
        if value in {None, "unknown"}:
            return value
        return _validate_player(value)

    @field_validator("visible_players", "mentioned_players")
    @classmethod
    def validate_player_lists(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class POVEventAnnotation(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    player_id: str
    video_file: str
    aligned_start_sec: float | None = None
    aligned_end_sec: float | None = None
    events: list[POVEvent]
    uncertain_items: list[dict[str, Any]] = Field(default_factory=list)
    raw_response: str | None = None

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        return _validate_player(value)


class Claim(BaseModel):
    claim_id: str | None = None
    speaker: str = "unknown"
    claim_type: str = "unknown"
    content: str
    subject_players: list[str] = Field(default_factory=list)
    object_players: list[str] = Field(default_factory=list)
    mentioned_players: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    time_referred: str | None = None
    is_direct_observation_claim: bool | None = None
    certainty: float = Field(ge=0.0, le=1.0)
    evidence: str = ""

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, value: str) -> str:
        if value == "unknown":
            return value
        return _validate_player(value)

    @field_validator("mentioned_players", "subject_players", "object_players")
    @classmethod
    def validate_player_list(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class Utterance(TimedEvidenceModel):
    utterance_id: str
    player_id: str
    speaker: str = "unknown"
    speaker_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transcript: str = ""
    text: str
    addressee: list[str] = Field(default_factory=list)
    mentioned_players: list[str] = Field(default_factory=list)
    speech_act: str = "unknown"
    claims: list[Claim] = Field(default_factory=list)
    possible_intents: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        return _validate_player(value)

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, value: str) -> str:
        if value == "unknown":
            return value
        return _validate_player(value)

    @field_validator("addressee", "mentioned_players")
    @classmethod
    def validate_player_lists(cls, value: list[str]) -> list[str]:
        normalized = [player for player in value if player != "unknown"]
        for player_id in normalized:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class UtteranceAnnotation(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    player_id: str
    video_file: str
    aligned_start_sec: float | None = None
    aligned_end_sec: float | None = None
    utterances: list[Utterance]
    raw_response: str | None = None

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        return _validate_player(value)


class GlobalEvent(TimedEvidenceModel):
    target_player: str = "global"
    global_event_id: str
    event_type: str
    description: str
    actors: list[str] = Field(default_factory=list)
    involved_players: list[str] = Field(default_factory=list)
    visible_to: list[str] = Field(default_factory=list)
    heard_by: list[str] = Field(default_factory=list)
    not_visible_to: list[str] = Field(default_factory=list)
    related_claim_ids: list[str] = Field(default_factory=list)
    supporting_pov_event_ids: list[str] = Field(default_factory=list)
    supporting_utterance_ids: list[str] = Field(default_factory=list)
    conflict: bool = False

    @field_validator(
        "involved_players", "actors", "visible_to", "heard_by", "not_visible_to"
    )
    @classmethod
    def validate_player_lists(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class GlobalEventAnnotation(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    aligned_start_sec: float | None = None
    aligned_end_sec: float | None = None
    global_events: list[GlobalEvent]
    claims: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    raw_response: str | None = None


class PhaseEvent(TimedEvidenceModel):
    target_player: str = "global"
    phase_event_id: str
    phase_type: Literal[
        "discussion",
        "voting",
        "vote_result",
        "exile",
        "action",
        "body_report",
        "meeting_start",
        "unknown",
    ]
    visible_to_all: bool = False
    evidence_povs: list[str] = Field(default_factory=list)

    @field_validator("evidence_povs")
    @classmethod
    def validate_evidence_povs(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class PhaseEventAnnotation(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    aligned_start_sec: float
    aligned_end_sec: float
    phase_events: list[PhaseEvent]
    raw_response: str | None = None


class MemoryItem(BaseModel):
    memory_id: str
    first_observed_abs_sec: float = Field(ge=0)
    last_referenced_abs_sec: float = Field(ge=0)
    memory_type: Literal[
        "direct_visual",
        "heard_claim",
        "public_result",
        "self_action",
        "inferred",
        "phase",
    ]
    content: str
    source_event_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    decay_status: Literal["active", "stale", "contradicted"] = "active"
    visibility: Literal["private", "public"] = "private"
    needs_human_review: bool = False

    @model_validator(mode="after")
    def validate_sources(self) -> MemoryItem:
        if not self.source_event_ids and not self.source_claim_ids:
            self.needs_human_review = True
        return self


class MemoryDelta(BaseModel):
    operation: Literal["add", "update", "contradict", "forget_low_confidence"]
    memory_id: str
    reason: str


class MemoryState(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    memory_items: list[MemoryItem] = Field(default_factory=list)
    memory_delta: list[MemoryDelta] = Field(default_factory=list)
    needs_human_review: bool = False

    @field_validator("target_player")
    @classmethod
    def validate_target_player(cls, value: str) -> str:
        return _validate_player(value)


class ForbiddenInformation(BaseModel):
    hidden_event_id: str
    reason: str


class BeliefItem(BaseModel):
    player: str | None = None
    belief_type: str
    content: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("player")
    @classmethod
    def validate_player(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_player(value)


class BeliefState(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    knows: list[dict[str, Any]] = Field(default_factory=list)
    does_not_know: list[dict[str, Any]] = Field(default_factory=list)
    believes_or_suspects: list[BeliefItem] = Field(default_factory=list)
    trust_state: dict[str, Any] = Field(default_factory=dict)
    forbidden_information: list[ForbiddenInformation] = Field(default_factory=list)
    certainty: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
    source_pov: list[str] = Field(default_factory=list)
    needs_human_review: bool = False

    @field_validator("target_player")
    @classmethod
    def validate_target_player(cls, value: str) -> str:
        return _validate_player(value)

    @field_validator("source_pov")
    @classmethod
    def validate_source_pov(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))


class InformationState(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    available_information: list[dict[str, Any]] = Field(default_factory=list)
    unknown_or_unseen_information: list[dict[str, Any]] = Field(default_factory=list)
    suspicion_state: dict[str, Any] = Field(default_factory=dict)
    trust_state: dict[str, Any] = Field(default_factory=dict)
    known_facts: list[str] = Field(default_factory=list)
    beliefs: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    private_observations: list[str] = Field(default_factory=list)
    public_information: list[str] = Field(default_factory=list)
    false_or_uncertain_beliefs: list[str] = Field(default_factory=list)
    certainty: float = Field(ge=0.0, le=1.0)
    evidence: str
    source_pov: list[str]
    needs_human_review: bool = False

    @field_validator("target_player")
    @classmethod
    def validate_target_player(cls, value: str) -> str:
        return _validate_player(value)

    @field_validator("source_pov")
    @classmethod
    def validate_source_pov(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("source_pov must not be empty")
        for player_id in value:
            _validate_player(player_id)
        return value
