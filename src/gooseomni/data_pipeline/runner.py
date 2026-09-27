from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import DataPipelineConfig
from .provenance import append_jsonl, event, read_json, stable_hash, write_json
from .stages import StageContext, StageHandler, handlers_for, stage_names


class DataPipelineRunner:
    def __init__(
        self,
        config: DataPipelineConfig,
        handlers: dict[str, StageHandler] | None = None,
    ) -> None:
        self.config = config
        self.handlers = {**handlers_for(config), **(handlers or {})}
        self.manifest_path = config.run_root / "run_manifest.json"
        self.events_path = config.run_root / "events.jsonl"
        self.errors_path = config.run_root / "errors.jsonl"

    def _config_hash(self) -> str:
        return stable_hash(
            {
                "version": self.config.version,
                "games": [
                    {
                        "game_id": game.game_id,
                        "players": [
                            (player.player_id, str(player.video_path))
                            for player in game.players
                        ],
                        "valid_range": game.valid_range,
                        "excluded_ranges": game.excluded_ranges,
                    }
                    for game in self.config.games
                ],
                "perception_model": self.config.perception_model,
                "judge_model": self.config.judge_model,
                "source_benchmark": (
                    str(self.config.source_benchmark)
                    if self.config.source_benchmark
                    else None
                ),
                "dataset_root": str(self.config.dataset_root),
                "stage_cache": {
                    key: str(value)
                    for key, value in sorted((self.config.stage_cache or {}).items())
                },
                "pipeline": self.config.pipeline,
                "alignment": {
                    "reference_player": self.config.reference_player,
                    "minimum_anchors": self.config.minimum_alignment_anchors,
                    "median_limit": self.config.median_residual_limit_sec,
                    "p95_limit": self.config.p95_residual_limit_sec,
                    "minimum_anchor_confidence": self.config.minimum_anchor_confidence,
                },
                "dual_pass": self.config.dual_pass,
                "speaker_confidence_min": self.config.speaker_confidence_min,
                "perception_chunks": {
                    "max_visual_chunk_sec": self.config.max_visual_chunk_sec,
                    "max_audio_chunk_sec": self.config.max_audio_chunk_sec,
                    "split_on_phase_boundaries": self.config.split_on_phase_boundaries,
                },
                "pilot_gates": {
                    "minimum_episodes": self.config.minimum_pilot_episodes,
                    "minimum_trials_per_layer": self.config.minimum_trials_per_layer,
                },
                "registry": self.config.registry or {},
            }
        )

    def status(self) -> dict[str, Any]:
        return read_json(
            self.manifest_path,
            {
                "version": self.config.version,
                "config_hash": self._config_hash(),
                "stages": {},
            },
        )

    def run(
        self,
        resume: bool = False,
        from_stage: str | None = None,
        to_stage: str | None = None,
    ) -> dict[str, Any]:
        stages = stage_names(self.config)
        if from_stage and from_stage not in stages:
            raise ValueError(f"unknown stage: {from_stage}")
        if to_stage and to_stage not in stages:
            raise ValueError(f"unknown stage: {to_stage}")
        if (
            from_stage
            and to_stage
            and stages.index(from_stage) > stages.index(to_stage)
        ):
            raise ValueError("from_stage must not come after to_stage")
        manifest = self.status()
        current_hash = self._config_hash()
        if resume and manifest.get("config_hash") != current_hash:
            raise ValueError("config hash changed; start a new run instead of resuming")
        manifest["config_hash"] = current_hash
        manifest["version"] = self.config.version
        manifest["config_path"] = str(self.config.config_path)
        manifest.setdefault("stages", {})
        started = from_stage is None
        for stage in stages:
            started = started or stage == from_stage
            if not started:
                continue
            previous = manifest["stages"].get(stage, {})
            if resume and previous.get("status") == "complete":
                if stage == to_stage:
                    break
                continue
            handler = self.handlers.get(stage)
            if handler is None:
                manifest["stages"][stage] = {
                    "status": "pending",
                    "reason": "stage handler is not configured",
                }
                write_json(self.manifest_path, manifest)
                break
            append_jsonl(self.events_path, event(stage, "started"))
            started_at = time.time()
            try:
                output = handler(StageContext(self.config, stage, self.config.run_root))
            except Exception as exc:
                error = event(stage, "failed", error=str(exc))
                append_jsonl(self.errors_path, error)
                append_jsonl(self.events_path, error)
                manifest["stages"][stage] = error
                write_json(self.manifest_path, manifest)
                write_json(self.config.run_root / "summary.json", manifest)
                raise
            record = event(
                stage,
                "complete",
                duration_sec=round(time.time() - started_at, 3),
                output=output,
                output_hash=stable_hash(output),
            )
            manifest["stages"][stage] = record
            append_jsonl(self.events_path, record)
            write_json(self.manifest_path, manifest)
            if stage == to_stage:
                break
        write_json(self.config.run_root / "summary.json", manifest)
        return manifest


def load_run_manifest(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root) / "run_manifest.json"
    return read_json(path, {})
