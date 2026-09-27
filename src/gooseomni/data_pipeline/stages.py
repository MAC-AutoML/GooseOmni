from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gooseomni.config import PATHS
from gooseomni.models.utils.openai_compat_tester import OpenAICompatTester

from .config import DataPipelineConfig
from .pilot_stages import PILOT_HANDLERS  # noqa: E402
from .provenance import sha256_file, stable_hash
from .raw_stages import (
    codex_review_stage as raw_codex_review_stage,
)
from .raw_stages import (
    event_fusion_stage as raw_event_fusion_stage,
)
from .raw_stages import (
    oracle_and_belief_stage as raw_oracle_and_belief_stage,
)
from .raw_stages import (
    qwen_perception_stage as raw_qwen_perception_stage,
)
from .raw_stages import (
    segment_stage as raw_segment_stage,
)
from .raw_stages import (
    sync_stage as raw_sync_stage,
)
from .raw_stages import (
    trial_build_stage as raw_trial_build_stage,
)
from .release_pipeline import build_release, validate_release

STAGE_NAMES = (
    "ingest",
    "sync",
    "segment",
    "qwen_perception",
    "event_fusion",
    "oracle_and_belief",
    "codex_review",
    "trial_build",
    "validate",
    "package",
)

PILOT_STAGE_NAMES = (
    "ingest",
    "align",
    "episode",
    "perceive_visual",
    "perceive_audio",
    "trajectory_fusion",
    "information_state",
    "tom_trial_build",
    "codex_review",
    "validate",
    "package",
)


def stage_names(config: DataPipelineConfig) -> tuple[str, ...]:
    return PILOT_STAGE_NAMES if config.pipeline == "strict_tom_pilot" else STAGE_NAMES


@dataclass(frozen=True)
class StageContext:
    config: DataPipelineConfig
    stage: str
    run_root: Path


StageHandler = Callable[[StageContext], dict[str, Any]]


def run_command(command: list[str], cwd: Path = PATHS.root) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-2000:]}"
        )
    return {"command": command, "stdout_tail": completed.stdout[-2000:]}


def ingest_stage(context: StageContext) -> dict[str, Any]:
    issues = context.config.validate_inputs(require_files=True)
    if issues:
        raise ValueError("; ".join(issues))
    games = []
    for game in context.config.games:
        games.append(
            {
                "game_id": game.game_id,
                "players": [
                    {
                        "player_id": player.player_id,
                        "video_path": str(player.video_path),
                        "sha256": sha256_file(player.video_path),
                    }
                    for player in game.players
                ],
                "valid_range": list(game.valid_range),
                "excluded_ranges": [list(row) for row in game.excluded_ranges],
            }
        )
    return {"games": games, "input_hash": stable_hash(games)}


def _migration_stage(context: StageContext) -> dict[str, Any]:
    source = context.config.source_benchmark
    if source is None or not source.is_dir():
        raise FileNotFoundError("source_benchmark is required for release migration")
    return {
        "mode": "frozen_v1_migration",
        "source_benchmark": str(source),
        "stage": context.stage,
    }


def trial_build_stage(context: StageContext) -> dict[str, Any]:
    if context.config.source_benchmark is None:
        raise RuntimeError(
            "raw-data trial build must be executed after the existing annotation/oracle "
            "stages; configure source_benchmark only for the frozen source-to-release migration"
        )
    staging = context.run_root / "benchmark_staging"
    return build_release(
        staging,
        source=context.config.source_benchmark,
        dataset_root=context.config.dataset_root,
    )


def validate_stage(context: StageContext) -> dict[str, Any]:
    report = validate_release(context.run_root / "benchmark_staging", probe_media=True)
    if not report["ok"]:
        raise RuntimeError(f"release validation failed: {report['issues']}")
    return report


def package_stage(context: StageContext) -> dict[str, Any]:
    source = context.run_root / "benchmark_staging"
    destination = context.config.benchmark_root
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite published benchmark: {destination}"
        )
    link_targets = {
        path.relative_to(source): path.resolve()
        for path in source.rglob("*")
        if path.is_symlink()
    }
    shutil.copytree(source, destination, symlinks=True)
    for relative, target in link_targets.items():
        published_link = destination / relative
        published_link.unlink()
        published_link.symlink_to(os.path.relpath(target, published_link.parent))
    report = validate_release(destination, probe_media=False)
    if not report["ok"]:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(f"published release validation failed: {report['issues']}")
    return {"benchmark_root": str(destination), "validation": report}


class CodexJudge:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def review(self, candidate: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are the final GooseOmni structured-data judge. Review the candidate "
            "against its supplied evidence only. Return strict JSON with keys decision "
            "(accept|repair|reject), final_value, reason, and evidence_ids. Never invent "
            "evidence IDs. Automatic output must use gold_source=model_verified.\n\n"
            f"CANDIDATE_JSON:\n{json.dumps(candidate, ensure_ascii=False, sort_keys=True)}"
        )
        tester = OpenAICompatTester(self.model_name)
        raw = tester.call(
            None,
            prompt,
            model_params={"max_tokens": 2048, "temperature": 0.0, "top_p": 1.0},
            include_images=False,
            include_audio=False,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Codex judge returned invalid JSON") from exc
        return validate_judgement(candidate, parsed, self.model_name, raw)


def _evidence_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_ids" and isinstance(item, list):
                found.update(str(row) for row in item)
            else:
                found.update(_evidence_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_evidence_ids(item))
    return found


def validate_judgement(
    candidate: dict[str, Any], parsed: dict[str, Any], model_name: str, raw: str = ""
) -> dict[str, Any]:
    decision = parsed.get("decision")
    if decision not in {"accept", "repair", "reject"}:
        raise ValueError("Codex judge decision must be accept, repair, or reject")
    allowed = {str(row) for row in candidate.get("available_evidence_ids", [])}
    cited = _evidence_ids(parsed)
    if cited - allowed:
        raise ValueError(
            f"Codex judge cited unknown evidence IDs: {sorted(cited - allowed)}"
        )
    final_value = candidate if decision == "accept" else parsed.get("final_value")
    if decision == "repair" and not isinstance(final_value, dict):
        raise ValueError("repair decision requires object final_value")
    if isinstance(final_value, dict):
        final_value = {**final_value, "gold_source": "model_verified"}
        if final_value.get("gold_source") == "human_verified":
            raise ValueError("automatic judgement cannot create human_verified gold")
    return {
        "decision": decision,
        "candidate": candidate,
        "final_value": final_value,
        "reason": str(parsed.get("reason", "")),
        "evidence_ids": sorted(cited),
        "review_model": model_name,
        "prompt_version": "codex_judge_v1",
        "input_hash": stable_hash(candidate),
        "response_hash": stable_hash(raw),
        "request_time_unix": time.time(),
        "retry_count": 0,
        "raw_response": raw,
        "gold_source": "model_verified" if decision != "reject" else None,
    }


def _raw_or_migration(
    raw_handler: StageHandler, migration_handler: StageHandler = _migration_stage
) -> StageHandler:
    def handler(context: StageContext) -> dict[str, Any]:
        if context.config.source_benchmark is not None:
            return migration_handler(context)
        return raw_handler(context)

    return handler


DEFAULT_HANDLERS: dict[str, StageHandler] = {
    "ingest": ingest_stage,
    "sync": _raw_or_migration(raw_sync_stage),
    "segment": _raw_or_migration(raw_segment_stage),
    "qwen_perception": _raw_or_migration(raw_qwen_perception_stage),
    "event_fusion": _raw_or_migration(raw_event_fusion_stage),
    "oracle_and_belief": _raw_or_migration(raw_oracle_and_belief_stage),
    "codex_review": _raw_or_migration(raw_codex_review_stage),
    "trial_build": _raw_or_migration(raw_trial_build_stage, trial_build_stage),
    "validate": validate_stage,
    "package": package_stage,
}


def handlers_for(config: DataPipelineConfig) -> dict[str, StageHandler]:
    if config.pipeline == "strict_tom_pilot":
        return {"ingest": ingest_stage, **PILOT_HANDLERS}
    return DEFAULT_HANDLERS
