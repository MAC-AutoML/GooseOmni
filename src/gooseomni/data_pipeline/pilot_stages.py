from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .alignment import Anchor, fit_affine_ransac, validate_alignment
from .episode_review import audited_episode_stage
from .episodes import build_episodes, consensus_boundaries, validate_episode_anchors
from .information_state import build_information_state, observed_cutoffs
from .pilot_common import cache_path
from .pilot_perception import perceive_audio_stage, perceive_visual_stage
from .pilot_review import codex_review_stage
from .pilot_validation import validate_pilot
from .provenance import sha256_tree, stable_hash, write_json
from .release_pipeline import read_jsonl, write_jsonl
from .reviewed_trials import accepted_trials
from .tom_trials import (
    TOM_LAYERS,
    build_trial_group,
    evidence_for_layer,
    public_trial,
    structured_gold,
    subject_only_evidence,
)
from .trajectory_review import review_and_write_trajectory


def _read_records(path: Path, list_key: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for source in paths:
        if source.name.startswith("._"):
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = (
            payload.get(list_key, [])
            if list_key and isinstance(payload, dict)
            else payload
        )
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))
    return rows


def align_stage(context: Any) -> dict[str, Any]:
    target = context.run_root / "artifacts/alignment.json"
    cached = cache_path(context, "alignment")
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    if cached is not None:
        payload = json.loads(cached.read_text(encoding="utf-8"))
        mode = "audited_cache"
    else:
        anchors_path = cache_path(context, "alignment_anchors")
        if anchors_path is None:
            raise RuntimeError(
                "strict alignment requires cache.alignment or cache.alignment_anchors; "
                "candidate extraction must run before semantic fitting"
            )
        raw = json.loads(anchors_path.read_text(encoding="utf-8"))
        mappings = {}
        for player in players:
            anchors = [
                Anchor(
                    global_abs_sec=float(row["global_abs_sec"]),
                    raw_sec=float(row["raw_sec"]),
                    event_type=str(row["event_type"]),
                    evidence_id=str(row["evidence_id"]),
                    confidence=float(row.get("confidence", 1.0)),
                )
                for row in raw.get(player, [])
                if float(row.get("confidence", 1.0))
                >= context.config.minimum_anchor_confidence
            ]
            mappings[player] = fit_affine_ransac(anchors)
        common_start = max(
            0.0,
            max(
                -float(row["offset"]) / float(row["scale"]) for row in mappings.values()
            ),
        )
        payload = {
            "schema_version": "gooseomni_alignment_v1",
            "reference_player": context.config.reference_player,
            "mapping_formula": "raw_sec = scale * global_abs_sec + offset",
            "mappings": mappings,
            "common_coverage_start_sec": common_start,
            "anchor_source_hash": sha256_tree(anchors_path),
        }
        mode = "ransac_fit"
    issues = validate_alignment(
        payload,
        players,
        context.config.median_residual_limit_sec,
        context.config.p95_residual_limit_sec,
        context.config.minimum_alignment_anchors,
    )
    if issues:
        quarantine = context.run_root / "quarantine/alignment.json"
        write_json(quarantine, {"issues": issues, "alignment": payload})
        raise RuntimeError(f"alignment failed closed: {issues}")
    write_json(target, payload)
    clip_manifest = cache_path(context, "clip_manifest")
    manifest_target = context.run_root / "artifacts/clip_manifest.jsonl"
    clip_stats: dict[str, Any]
    if clip_manifest is not None:
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip_manifest, manifest_target)
        clip_stats = {"mode": "cache_replay", "records": len(read_jsonl(clip_manifest))}
    else:
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError(
                "aligned clip materialization must run in a Slurm compute job"
            )
        from .aligned_clips import materialize_aligned_clips

        clip_stats = materialize_aligned_clips(
            context.config.games,
            payload,
            context.run_root / "artifacts/clips",
            manifest_target,
        )
    return {
        "mode": mode,
        "output": str(target),
        "players": len(players),
        "clips": clip_stats,
    }


def episode_stage(context: Any) -> dict[str, Any]:
    audited = audited_episode_stage(context)
    if audited is not None:
        return audited
    source = cache_path(context, "boundary_candidates")
    if source is None:
        server_url = os.getenv("QWEN3_OMNI_SERVER_URL")
        if not server_url or not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError(
                "fresh episode reconstruction requires Qwen in Slurm or "
                "cache.boundary_candidates"
            )
        source = context.run_root / "annotations/boundary_candidates"
        from .stages import run_command

        command = [
            sys.executable,
            "tools/annotation/run_qwen_round_boundary_annotation.py",
            "--manifest-path",
            str(context.run_root / "artifacts/clip_manifest.jsonl"),
            "--output-dir",
            str(source),
            "--error-dir",
            str(context.run_root / "errors/boundary_candidates"),
            "--backend",
            "local",
            "--server-url",
            server_url,
            "--model",
            context.config.perception_model,
            "--resume",
        ]
        limit = os.getenv("GOOSEOMNI_PILOT_LIMIT_CLIPS")
        if limit:
            command.extend(["--limit", limit])
        run_command(command)
    rows = _read_records(source, "round_boundary_candidates")
    episodes = []
    for game in context.config.games:
        game_rows = [row for row in rows if str(row.get("game_id")) == game.game_id]
        consensus = consensus_boundaries(game_rows)
        session_end = max(float(row.get("aligned_end_sec", 0)) for row in game_rows)
        game_episodes = build_episodes(game.game_id, consensus, session_end)
        alignment = json.loads(
            (context.run_root / "artifacts/alignment.json").read_text(encoding="utf-8")
        )
        coverage_start = float(alignment.get("common_coverage_start_sec", 0.0))
        incomplete = [
            row for row in game_episodes if float(row["abs_start_sec"]) < coverage_start
        ]
        if incomplete:
            write_json(
                context.run_root
                / f"quarantine/{game.game_id}_incomplete_coverage.json",
                {
                    "common_coverage_start_sec": coverage_start,
                    "episodes": incomplete,
                    "reason": "one or more POV recordings do not cover episode start",
                },
            )
        episodes.extend(row for row in game_episodes if row not in incomplete)
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    report = validate_episode_anchors(
        rows,
        episodes,
        players,
        context.config.minimum_alignment_anchors,
        context.config.median_residual_limit_sec,
        context.config.p95_residual_limit_sec,
    )
    write_json(context.run_root / "artifacts/episode_alignment_report.json", report)
    if not report["ok"]:
        write_json(context.run_root / "quarantine/episode_alignment.json", report)
        raise RuntimeError(f"episode alignment failed closed: {report['issues']}")
    target = context.run_root / "artifacts/episodes.json"
    write_json(target, episodes)
    return {
        "episodes": len(episodes),
        "output": str(target),
        "input_hash": sha256_tree(source),
        "alignment_report": report,
    }


def trajectory_fusion_stage(context: Any) -> dict[str, Any]:
    visual = _read_records(context.run_root / "annotations/perception_visual", "events")
    from .trajectory_review import load_visual_error_candidates

    visual.extend(
        load_visual_error_candidates(
            context.run_root / "errors/perception_visual",
            context.run_root / "annotations/perception_visual",
        )
    )
    audio = _read_records(context.run_root / "annotations/perception_audio", "events")
    episodes = json.loads(
        (context.run_root / "artifacts/episodes.json").read_text(encoding="utf-8")
    )
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    return review_and_write_trajectory(
        context.run_root, visual, audio, episodes, players
    )


def information_state_stage(context: Any) -> dict[str, Any]:
    nodes = read_jsonl(context.run_root / "artifacts/trajectory_nodes.jsonl")
    episodes = json.loads(
        (context.run_root / "artifacts/episodes.json").read_text(encoding="utf-8")
    )
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    states = []
    for episode in episodes:
        episode_nodes = [
            row for row in nodes if row["episode_id"] == episode["episode_id"]
        ]
        cutoffs = observed_cutoffs(episode_nodes)
        for cutoff in cutoffs:
            states.extend(
                build_information_state(
                    episode["episode_id"], player, cutoff, episode_nodes
                )
                for player in players
            )
    target = context.run_root / "artifacts/information_states.jsonl"
    write_jsonl(target, states)
    return {"states": len(states), "output": str(target)}


def tom_trial_build_stage(context: Any) -> dict[str, Any]:
    states = read_jsonl(context.run_root / "artifacts/information_states.jsonl")
    nodes = read_jsonl(context.run_root / "artifacts/trajectory_nodes.jsonl")
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    state_by_key = {
        (row["episode_id"], row["cutoff_abs_sec"], row["player_id"]): row
        for row in states
    }
    trials = []
    for state in states:
        episode_nodes = [
            row for row in nodes if row["episode_id"] == state["episode_id"]
        ]
        for target in (player for player in players if player != state["player_id"]):
            target_state = state_by_key[
                (state["episode_id"], state["cutoff_abs_sec"], target)
            ]
            for layer in TOM_LAYERS:
                evidence = evidence_for_layer(state, episode_nodes, target, layer)[:4]
                evidence = subject_only_evidence(evidence, target_state)
                if not evidence:
                    continue
                hidden = sorted(
                    {
                        str(item)
                        for node in episode_nodes
                        if float(node["abs_start_sec"]) > float(state["cutoff_abs_sec"])
                        and target
                        in {
                            str(node.get("player_id", "")),
                            str(node.get("speaker_id", "")),
                            *(str(value) for value in node.get("visible_players", [])),
                            *(
                                str(value)
                                for value in node.get("mentioned_players", [])
                            ),
                        }
                        for item in node.get("evidence_asset_ids", [])
                    }
                )[:4]
                gold = structured_gold(layer, state, target, episode_nodes, evidence)
                hidden = sorted(set(hidden) | set(gold.get("future_evidence_ids", [])))
                trials.extend(
                    build_trial_group(
                        state,
                        target,
                        f"{layer}_query",
                        "subject_observed_target_gap",
                        layer,
                        evidence,
                        gold,
                        query=(
                            f"在 {state['cutoff_abs_sec']:.3f} 秒时，"
                            f"{state['player_id']} 应如何判断 {target} 的 {layer}？"
                        ),
                        review_only_hidden_evidence=hidden,
                    )
                )
    target = context.run_root / "candidates/tom_trials.jsonl"
    write_jsonl(target, trials)
    return {"trials": len(trials), "groups": len(trials) // 4, "output": str(target)}


def validate_stage(context: Any) -> dict[str, Any]:
    alignment = json.loads(
        (context.run_root / "artifacts/alignment.json").read_text(encoding="utf-8")
    )
    episodes = json.loads(
        (context.run_root / "artifacts/episodes.json").read_text(encoding="utf-8")
    )
    nodes = read_jsonl(context.run_root / "artifacts/trajectory_nodes.jsonl")
    states = read_jsonl(context.run_root / "artifacts/information_states.jsonl")
    candidates = {
        row["trial_id"]: row
        for row in read_jsonl(context.run_root / "candidates/tom_trials.jsonl")
    }
    decisions = read_jsonl(context.run_root / "review_decisions.jsonl")
    trials = accepted_trials(candidates, decisions)
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    report = validate_pilot(
        alignment,
        episodes,
        nodes,
        states,
        trials,
        players,
        minimum_episodes=context.config.minimum_pilot_episodes,
        minimum_trials_per_layer=context.config.minimum_trials_per_layer,
    )
    write_json(context.run_root / "validation_report.json", report)
    if not report["ok"]:
        raise RuntimeError(f"pilot validation failed closed: {report['issues']}")
    return report


def package_stage(context: Any) -> dict[str, Any]:
    target = context.config.benchmark_root
    validation = json.loads(
        (context.run_root / "validation_report.json").read_text(encoding="utf-8")
    )
    if not validation.get("release_eligible"):
        raise RuntimeError("pilot package refused: release gates are not satisfied")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite published pilot: {target}")
    decisions = read_jsonl(context.run_root / "review_decisions.jsonl")
    candidates = {
        row["trial_id"]: row
        for row in read_jsonl(context.run_root / "candidates/tom_trials.jsonl")
    }
    accepted = accepted_trials(candidates, decisions)
    (target / "public").mkdir(parents=True)
    (target / "private").mkdir(parents=True)
    write_jsonl(target / "public/trials.jsonl", [public_trial(row) for row in accepted])
    write_jsonl(target / "private/hidden_gold.jsonl", accepted)
    registry = {
        "name": "gooseomni_v2_pilot",
        "leaderboard_eligible": False,
        "gold_source": "model_verified",
        "frozen_diagnostics": [
            {
                "path": str(
                    (context.config.registry or {}).get("frozen_diagnostic_run", "")
                ),
                "label": "weak_epistemic_diagnostic",
                "diagnostic_only": True,
            }
        ],
        "trial_count": len(accepted),
        "content_hash": stable_hash(accepted),
    }
    write_json(target / "registry.json", registry)
    return {"benchmark_root": str(target), "trials": len(accepted)}


PILOT_HANDLERS = {
    "align": align_stage,
    "episode": episode_stage,
    "perceive_visual": perceive_visual_stage,
    "perceive_audio": perceive_audio_stage,
    "trajectory_fusion": trajectory_fusion_stage,
    "information_state": information_state_stage,
    "tom_trial_build": tom_trial_build_stage,
    "codex_review": codex_review_stage,
    "validate": validate_stage,
    "package": package_stage,
}
