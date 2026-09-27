from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from gooseomni.benchmark.schema_benchmark import (
    ClaimTruthStatus,
    GoldSource,
    VisibilityLabel,
)

from .schema_events import _validate_player


class OracleWorldEvent(BaseModel):
    world_event_id: str
    game_id: str
    source_segment_ids: list[str] = Field(default_factory=list)
    source_povs: list[str] = Field(default_factory=list)
    abs_start_sec: float = Field(ge=0)
    abs_end_sec: float = Field(gt=0)
    phase_type: str = "unknown"
    event_type: str = "other"
    actors: list[str] = Field(default_factory=list)
    patients: list[str] = Field(default_factory=list)
    location: str = "unknown"
    description: str = ""
    direct_visual_evidence: list[str] = Field(default_factory=list)
    direct_audio_evidence: list[str] = Field(default_factory=list)
    public_evidence: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = False

    @field_validator("source_povs")
    @classmethod
    def validate_source_povs(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_abs_time_order(self) -> "OracleWorldEvent":
        if self.abs_end_sec <= self.abs_start_sec:
            raise ValueError("abs_end_sec must be greater than abs_start_sec")
        return self


class OracleClaim(BaseModel):
    claim_id: str
    game_id: str
    source_segment_ids: list[str] = Field(default_factory=list)
    speaker: str = "unknown"
    heard_by: list[str] = Field(default_factory=list)
    abs_start_sec: float = Field(ge=0)
    abs_end_sec: float = Field(gt=0)
    claim_type: str = "other"
    content: str
    normalized_content: str = ""
    time_referred: dict[str, Any] = Field(default_factory=dict)
    target_entities: list[str] = Field(default_factory=list)
    related_event_ids: list[str] = Field(default_factory=list)
    strategic_role: str = "other"
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = False

    @field_validator("speaker")
    @classmethod
    def validate_oracle_speaker(cls, value: str) -> str:
        if value == "unknown":
            return value
        return _validate_player(value)

    @field_validator("heard_by")
    @classmethod
    def validate_heard_by(cls, value: list[str]) -> list[str]:
        for player_id in value:
            _validate_player(player_id)
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_claim_time_order(self) -> "OracleClaim":
        if self.abs_end_sec <= self.abs_start_sec:
            raise ValueError("abs_end_sec must be greater than abs_start_sec")
        return self


class PhaseEpisode(BaseModel):
    game_id: str
    episode_id: str
    episode_index: int = Field(ge=0)
    phase_id: str
    phase_type: str
    phase_index_global: int = Field(ge=0)
    phase_index_in_episode: int = Field(ge=0)
    phase_order_label_zh: str
    aligned_start_sec: float = Field(ge=0)
    aligned_end_sec: float = Field(gt=0)
    previous_phase_id: str | None = None
    next_phase_id: str | None = None


class VisibilityEdge(BaseModel):
    edge_id: str
    game_id: str
    event_id: str
    player_id: str
    cutoff_abs_sec: float = Field(ge=0)
    visibility: VisibilityLabel
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("player_id")
    @classmethod
    def validate_visibility_player(cls, value: str) -> str:
        return _validate_player(value)


class ClaimTruthLink(BaseModel):
    claim_truth_link_id: str
    claim_id: str
    world_event_ids: list[str] = Field(default_factory=list)
    truth_status_global: ClaimTruthStatus = "unverified"
    local_awareness_by_player: dict[str, str] = Field(default_factory=dict)
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = False


class CanonicalEventMap(BaseModel):
    canonical_event_id: str
    duplicate_local_event_ids: list[str] = Field(default_factory=list)
    abs_time_cluster: list[float]
    canonical_source: str
    merge_reason: str


class BeliefMemorySnapshot(BaseModel):
    snapshot_id: str
    game_id: str
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    public_history: list[dict[str, Any]] = Field(default_factory=list)
    private_observations: list[dict[str, Any]] = Field(default_factory=list)
    heard_claims: list[dict[str, Any]] = Field(default_factory=list)
    inferred_beliefs: list[dict[str, Any]] = Field(default_factory=list)
    hidden_events_for_target: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_event_ids: list[str] = Field(default_factory=list)
    available_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("target_player")
    @classmethod
    def validate_snapshot_player(cls, value: str) -> str:
        return _validate_player(value)


class DiagnosticProbeGroup(BaseModel):
    probe_group_id: str
    game_id: str
    source_segment_ids: list[str] = Field(default_factory=list)
    cutoff_abs_sec: float = Field(ge=0)
    target_player: str
    query_variable: dict[str, Any]
    anchor_event_ids: list[str] = Field(default_factory=list)
    related_claim_ids: list[str] = Field(default_factory=list)
    hidden_event_ids_for_target: list[str] = Field(default_factory=list)
    available_evidence_ids_for_target: list[str] = Field(default_factory=list)
    selection_reason: str
    diagnostic_families: list[str] = Field(default_factory=list)
    template: str = "hidden_event_awareness"
    quality: dict[str, Any] = Field(default_factory=dict)
    needs_human_review: bool = False
    gold_source: GoldSource = "qwen_weak"

    @field_validator("target_player")
    @classmethod
    def validate_probe_group_player(cls, value: str) -> str:
        return _validate_player(value)


class DiagnosticProbe(BaseModel):
    probe_id: str
    probe_group_id: str
    probe_type: Literal[
        "A_pre_reveal_belief",
        "B_post_reveal_reconstruct_previous_belief",
        "C_other_agent_false_belief",
        "D_perspective_taking_prediction",
    ]
    input_condition: Literal[
        "target_pov_only",
        "target_available_events",
        "public_history_only",
        "oracle_truth_revealed",
        "global_to_perspective",
        "speaker_perspective",
    ]
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    prompt: str
    expected_output_schema: dict[str, Any]
    forbidden_event_ids: list[str] = Field(default_factory=list)
    acceptable_evidence_ids: list[str] = Field(default_factory=list)
    gold_source: GoldSource = "qwen_weak"

    @field_validator("target_player")
    @classmethod
    def validate_probe_player(cls, value: str) -> str:
        return _validate_player(value)


class ProbeGold(BaseModel):
    probe_group_id: str
    A_expected_weak: dict[str, Any] = Field(default_factory=dict)
    B_RC_weak: dict[str, Any] = Field(default_factory=dict)
    B_RC_strong_reference: dict[str, Any] = Field(default_factory=dict)
    C_FB_weak: dict[str, Any] = Field(default_factory=dict)
    C_FB_strong_reference: dict[str, Any] = Field(default_factory=dict)
    D_PT_reference: dict[str, Any] = Field(default_factory=dict)
    forbidden_event_ids_for_target: list[str] = Field(default_factory=list)
    acceptable_evidence_ids_for_target: list[str] = Field(default_factory=list)
    claim_truth_global: ClaimTruthStatus | None = None
    claim_awareness_local_target: str = "unknown"
    gold_source: GoldSource = "qwen_weak"


class ProbeAnswer(BaseModel):
    probe_id: str
    raw_response: str | None = None
    parsed: dict[str, Any] = Field(default_factory=dict)


class DiagnosticScore(BaseModel):
    probe_group_id: str
    RC_weak: bool | None = None
    RC_strong: bool | None = None
    FB_weak: bool | None = None
    FB_strong: bool | None = None
    PT_weak: bool | None = None
    PT_strong: bool | None = None
    claim_verification_global: bool | None = None
    claim_verification_local: bool | None = None
    perspective_leakage: bool = False
    forbidden_evidence_usage: bool = False
    evidence_support: bool | None = None
    json_parse_success: bool = True
    schema_validation_success: bool = True
