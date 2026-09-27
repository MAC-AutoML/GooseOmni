from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gooseomni.config import PATHS


@dataclass(frozen=True)
class PlayerInput:
    player_id: str
    video_path: Path


@dataclass(frozen=True)
class GameInput:
    game_id: str
    players: tuple[PlayerInput, ...]
    valid_range: tuple[float, float | None]
    excluded_ranges: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DataPipelineConfig:
    version: str
    games: tuple[GameInput, ...]
    perception_model: str
    judge_model: str
    run_root: Path
    benchmark_root: Path
    config_path: Path
    source_benchmark: Path | None = None
    dataset_root: Path = PATHS.root / "data/gooseomni"
    stage_cache: dict[str, Path] | None = None
    pipeline: str = "legacy_v2"
    reference_player: str = "Gemini"
    minimum_alignment_anchors: int = 3
    median_residual_limit_sec: float = 1.0
    p95_residual_limit_sec: float = 2.0
    minimum_anchor_confidence: float = 0.15
    dual_pass: bool = False
    speaker_confidence_min: float = 0.85
    max_visual_chunk_sec: float = 30.0
    max_audio_chunk_sec: float = 30.0
    split_on_phase_boundaries: bool = False
    minimum_pilot_episodes: int = 5
    minimum_trials_per_layer: int = 30
    registry: dict[str, Any] | None = None

    def validate_inputs(self, require_files: bool = True) -> list[str]:
        issues: list[str] = []
        game_ids: set[str] = set()
        for game in self.games:
            if game.game_id in game_ids:
                issues.append(f"duplicate game_id: {game.game_id}")
            game_ids.add(game.game_id)
            players: set[str] = set()
            if not game.players:
                issues.append(f"game has no players: {game.game_id}")
            for player in game.players:
                if player.player_id in players:
                    issues.append(
                        f"duplicate player_id in {game.game_id}: {player.player_id}"
                    )
                players.add(player.player_id)
                if (
                    require_files
                    and self.source_benchmark is None
                    and not player.video_path.is_file()
                ):
                    issues.append(f"missing video: {player.video_path}")
            start, end = game.valid_range
            if start < 0 or (end is not None and end <= start):
                issues.append(f"invalid valid_range for {game.game_id}")
            for excluded_start, excluded_end in game.excluded_ranges:
                if excluded_start < start or excluded_end <= excluded_start:
                    issues.append(f"invalid excluded_range for {game.game_id}")
                if end is not None and excluded_end > end:
                    issues.append(f"excluded_range exceeds valid_range for {game.game_id}")
        if require_files and self.source_benchmark is not None:
            if not self.source_benchmark.is_dir():
                issues.append(f"missing source benchmark: {self.source_benchmark}")
            if not (self.dataset_root / "segments.json").is_file():
                issues.append(f"missing aligned dataset manifest: {self.dataset_root}")
        if self.pipeline == "strict_tom_pilot":
            if self.version != "gooseomni_v2_pilot":
                issues.append("strict_tom_pilot requires version gooseomni_v2_pilot")
            if self.reference_player != "Gemini":
                issues.append("strict pilot reference_player must be Gemini")
            if not self.dual_pass:
                issues.append("strict pilot requires dual-pass perception")
            if self.max_visual_chunk_sec <= 0:
                issues.append("max_visual_chunk_sec must be positive")
            if self.max_audio_chunk_sec <= 0:
                issues.append("max_audio_chunk_sec must be positive")
            if not self.split_on_phase_boundaries:
                issues.append("strict pilot requires phase-boundary perception splitting")
        return issues


def _rooted_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PATHS.root / path


def _game_from_dict(value: dict[str, Any]) -> GameInput:
    players = tuple(
        PlayerInput(
            player_id=str(row["player_id"]),
            video_path=_rooted_path(str(row["video_path"])),
        )
        for row in value.get("players", [])
    )
    raw_range = value.get("valid_range", [0, None])
    valid_range = (float(raw_range[0]), None if raw_range[1] is None else float(raw_range[1]))
    excluded = tuple((float(row[0]), float(row[1])) for row in value.get("excluded_ranges", []))
    return GameInput(str(value["game_id"]), players, valid_range, excluded)


def load_data_config(path: str | Path) -> DataPipelineConfig:
    config_path = _rooted_path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    outputs = payload.get("outputs", {})
    models = payload.get("models", {})
    alignment = payload.get("alignment", {}) or {}
    perception = payload.get("perception", {}) or {}
    pilot_gates = payload.get("pilot_gates", {}) or {}
    return DataPipelineConfig(
        version=str(payload.get("version", "gooseomni_v2")),
        games=tuple(_game_from_dict(row) for row in payload.get("games", [])),
        perception_model=str(models.get("perception", "qwen3_omni")),
        judge_model=os.getenv(
            "GOOSEOMNI_CODEX_MODEL", str(models.get("judge", "gpt-5.6-sol"))
        ),
        run_root=_rooted_path(outputs.get("run_root", "runs/gooseomni_v2")),
        benchmark_root=_rooted_path(
            outputs.get("benchmark_root", "benchmark/gooseomni_v2")
        ),
        config_path=config_path,
        source_benchmark=(
            _rooted_path(payload["source_benchmark"])
            if payload.get("source_benchmark")
            else None
        ),
        dataset_root=_rooted_path(
            payload.get("dataset_root", "data/gooseomni")
        ),
        stage_cache={
            str(key): _rooted_path(value)
            for key, value in (payload.get("cache", {}) or {}).items()
        },
        pipeline=str(payload.get("pipeline", "legacy_v2")),
        reference_player=str(alignment.get("reference_player", "Gemini")),
        minimum_alignment_anchors=int(
            alignment.get("minimum_anchors_per_episode", 3)
        ),
        median_residual_limit_sec=float(
            alignment.get("median_residual_limit_sec", 1.0)
        ),
        p95_residual_limit_sec=float(alignment.get("p95_residual_limit_sec", 2.0)),
        minimum_anchor_confidence=float(
            alignment.get("minimum_anchor_confidence", 0.15)
        ),
        dual_pass=bool(perception.get("dual_pass", False)),
        speaker_confidence_min=float(perception.get("speaker_confidence_min", 0.85)),
        max_visual_chunk_sec=float(perception.get("max_visual_chunk_sec", 30.0)),
        max_audio_chunk_sec=float(perception.get("max_audio_chunk_sec", 30.0)),
        split_on_phase_boundaries=bool(
            perception.get("split_on_phase_boundaries", False)
        ),
        minimum_pilot_episodes=int(pilot_gates.get("minimum_episodes", 5)),
        minimum_trials_per_layer=int(
            pilot_gates.get("minimum_trials_per_layer", 30)
        ),
        registry=dict(payload.get("registry", {}) or {}),
    )
