from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from gooseomni.benchmark.schema_events import DATASET_NAME

from .schema_events import _validate_player


class CandidateTrial(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    trial_id: str
    question_type: Literal[
        "first_order_belief",
        "second_order_belief",
        "hidden_information",
        "false_belief",
        "intent_inference",
        "knowledge_access",
        "belief_state",
    ]
    trial_type: str | None = None
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    question: str
    answer: str
    distractors: list[str] = Field(default_factory=list)
    available_information: list[str] = Field(default_factory=list)
    hidden_information: list[str] = Field(default_factory=list)
    expected_answer_basis: str | None = None
    supporting_global_event_ids: list[str] = Field(default_factory=list)
    supporting_information_state_ids: list[str] = Field(default_factory=list)
    risk_of_perspective_leakage: Literal["low", "medium", "high"] = "low"
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


class CheckerFinding(BaseModel):
    checker_name: str
    annotation_id: str | None = None
    verdict: Literal[
        "supported", "partially_supported", "unsupported", "uncertain", "pass", "fail"
    ]
    reason: str
    suggested_fix: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    leakage: bool = False
    leaked_items: list[dict[str, Any]] = Field(default_factory=list)


class CheckerReport(BaseModel):
    dataset: str = DATASET_NAME
    game_id: str
    segment_id: str
    checker_name: str
    findings: list[CheckerFinding] = Field(default_factory=list)
    passed: bool = True
    needs_human_review: bool = False


class BenchmarkInputCondition(BaseModel):
    condition: Literal[
        "single_pov_video",
        "single_pov_events",
        "multi_pov_global",
        "multi_pov_perspective",
        "text_only",
    ]
    input_video_files: list[str] = Field(default_factory=list)
    input_annotation_files: list[str] = Field(default_factory=list)


class BenchmarkGold(BaseModel):
    label: str
    acceptable_reasoning: list[str] = Field(default_factory=list)
    forbidden_reasoning: list[str] = Field(default_factory=list)
    gold_source: Literal[
        "qwen_weak", "qwen_checked", "model_verified", "human_verified"
    ] = "qwen_weak"


class BenchmarkTrial(BaseModel):
    dataset: str = DATASET_NAME
    trial_id: str
    game_id: str
    segment_id: str
    trial_type: str
    target_player: str
    cutoff_abs_sec: float = Field(ge=0)
    input_condition: Literal[
        "single_pov_video",
        "single_pov_events",
        "multi_pov_global",
        "multi_pov_perspective",
        "text_only",
    ]
    input_video_files: list[str] = Field(default_factory=list)
    question: str
    answer_format: str = "json"
    available_information: list[dict[str, Any]] = Field(default_factory=list)
    hidden_information: list[dict[str, Any]] = Field(default_factory=list)
    gold: BenchmarkGold
    metrics: list[str] = Field(default_factory=list)

    @field_validator("target_player")
    @classmethod
    def validate_target_player(cls, value: str) -> str:
        return _validate_player(value)


class AnnotationError(BaseModel):
    stage: str
    segment_id: str
    player_id: str | None = None
    target_player: str | None = None
    video_file: str | None = None
    raw_response: str | None = None
    error_message: str
    prompt: str


VisibilityLabel = Literal[
    "direct_visual",
    "direct_audio",
    "public_ui",
    "heard_claim",
    "not_visible",
    "post_cutoff",
    "unknown",
]
ClaimTruthStatus = Literal["supported", "contradicted", "unverified", "ambiguous"]
GoldSource = Literal["qwen_weak", "qwen_checked", "model_verified", "human_verified"]
