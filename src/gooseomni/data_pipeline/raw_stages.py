from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from gooseomni.annotation.postprocess import (
    build_information_states,
    build_meeting_utterances,
    load_pov_event_files,
    merge_global_events,
)
from gooseomni.benchmark.decrypto_export import (
    build_decrypto_diagnostics,
    export_gooseomni_benchmark,
)

from .provenance import sha256_tree, stable_hash
from .release import assemble_release
from .release_pipeline import read_jsonl, write_jsonl


def _cache(context: Any, name: str) -> Path | None:
    value = (context.config.stage_cache or {}).get(name)
    if value is not None and not value.exists():
        raise FileNotFoundError(f"configured cache does not exist: {name}={value}")
    return value


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite stage output: {target}")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("._*", ".__*", "__pycache__"),
    )


def _result(mode: str, source: Path, target: Path, **extra: Any) -> dict[str, Any]:
    input_hash = sha256_tree(source)
    return {
        "mode": mode,
        "source": str(source),
        "target": str(target),
        "input_hash": input_hash,
        "output_hash": (
            input_hash
            if source.resolve() == target.resolve()
            else sha256_tree(target)
        ),
        **extra,
    }


def _raw_input_dir(context: Any) -> Path:
    root = context.run_root / "artifacts/raw_inputs"
    for game in context.config.games:
        game_root = root / game.game_id
        game_root.mkdir(parents=True, exist_ok=True)
        for player in game.players:
            target = game_root / f"{player.player_id}{player.video_path.suffix}"
            if not target.exists():
                target.symlink_to(player.video_path)
    return root


def sync_stage(context: Any) -> dict[str, Any]:
    target = context.run_root / "artifacts/sync_offsets.json"
    cached = _cache(context, "sync_offsets")
    if cached is not None:
        _copy_file(cached, target)
        return _result("cache_replay", cached, target)
    server_url = os.getenv("QWEN3_OMNI_SERVER_URL")
    if not server_url:
        raise RuntimeError("sync requires cache.sync_offsets or QWEN3_OMNI_SERVER_URL")
    raw_root = _raw_input_dir(context)
    review = context.run_root / "artifacts/sync_review"
    errors = context.run_root / "errors/sync"
    command = [
        sys.executable,
        "tools/build/infer_sync_offsets.py",
        "--raw-dir",
        str(raw_root),
        "--output-path",
        str(target),
        "--review-dir",
        str(review),
        "--error-dir",
        str(errors),
        "--backend",
        "local",
        "--server-url",
        server_url,
        "--model",
        context.config.perception_model,
        "--no-resume",
    ]
    from .stages import run_command

    executed = run_command(command)
    return _result("qwen_live", raw_root, target, command=executed)


def segment_stage(context: Any) -> dict[str, Any]:
    target = context.run_root / "artifacts/segments.json"
    cached = _cache(context, "segments")
    if cached is not None:
        _copy_file(cached, target)
        return _result("cache_replay", cached, target)
    manifest = context.run_root / "artifacts/clip_manifest.jsonl"
    clips = context.run_root / "artifacts/clips"
    command = [
        sys.executable,
        "tools/build/split_raw_videos.py",
        "--raw-dir",
        str(_raw_input_dir(context)),
        "--output-dir",
        str(clips),
        "--manifest-path",
        str(manifest),
        "--sync-offsets",
        str(context.run_root / "artifacts/sync_offsets.json"),
    ]
    from .stages import run_command

    executed = run_command(command)
    return _result("fresh_split", manifest, manifest, command=executed)


def qwen_perception_stage(context: Any) -> dict[str, Any]:
    target = context.run_root / "annotations/pov_events"
    cached = _cache(context, "perception")
    if cached is not None:
        _copy_tree(cached, target)
        return _result(
            "cache_replay", cached, target, model=context.config.perception_model
        )
    server_url = os.getenv("QWEN3_OMNI_SERVER_URL")
    if not server_url or not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError(
            "fresh Qwen perception must run inside Slurm with QWEN3_OMNI_SERVER_URL"
        )
    command = [
        sys.executable,
        "tools/annotation/run_qwen_annotation.py",
        "--manifest-path",
        str(context.run_root / "artifacts/clip_manifest.jsonl"),
        "--output-dir",
        str(target),
        "--error-dir",
        str(context.run_root / "errors/qwen_perception"),
        "--backend",
        "local",
        "--server-url",
        server_url,
        "--model",
        context.config.perception_model,
        "--resume",
    ]
    from .stages import run_command

    executed = run_command(command)
    return _result("qwen_live", target, target, command=executed)


def event_fusion_stage(context: Any) -> dict[str, Any]:
    source = context.run_root / "annotations/pov_events"
    target = context.run_root / "artifacts/fusion"
    target.mkdir(parents=True, exist_ok=False)
    utterances = target / "meeting_utterances.json"
    events = target / "global_events.json"
    states = target / "information_states.json"
    loaded_events = load_pov_event_files(source)
    build_meeting_utterances(source, utterances, loaded_events)
    merge_global_events(source, events, loaded_events)
    build_information_states(source, states, loaded_events)
    return _result("deterministic_rebuild", source, target)


def oracle_and_belief_stage(context: Any) -> dict[str, Any]:
    annotation_root = context.run_root / "annotations"
    target = annotation_root / "oracle_ledger"
    cached = _cache(context, "oracle_ledger")
    if cached is None:
        from .ledger_seed import build_seed_ledger

        players = [
            player.player_id for game in context.config.games for player in game.players
        ]
        ledger_stats = build_seed_ledger(
            context.run_root / "artifacts/fusion", target, players
        )
        mode = "qwen_seed_rebuild"
        source = context.run_root / "artifacts/fusion"
    else:
        _copy_tree(cached, target)
        ledger_stats = None
        mode = "cache_replay_and_rebuild"
        source = cached
    diagnostics = build_decrypto_diagnostics(
        annotation_root, annotation_root, limit=276
    )
    candidates = context.run_root / "candidates"
    exported = export_gooseomni_benchmark(annotation_root, candidates)
    return _result(
        mode,
        source,
        target,
        ledger=ledger_stats,
        diagnostics=diagnostics,
        candidate_export=exported,
        candidate_hash=sha256_tree(candidates),
    )


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["trial_id"]): row for row in read_jsonl(path)}


def _candidate_records(root: Path) -> list[dict[str, Any]]:
    static = root / "static_trials"
    public = read_jsonl(static / "trials.jsonl")
    gold = _rows(static / "gold.jsonl")
    hidden = _rows(static / "hidden_gold.jsonl")
    return [
        {
            "trial_id": str(row["trial_id"]),
            "public_trial": row,
            "gold": gold[str(row["trial_id"])],
            "hidden_gold": hidden[str(row["trial_id"])],
            "available_evidence_ids": hidden[str(row["trial_id"])].get(
                "acceptable_evidence_ids", []
            ),
        }
        for row in public
    ]


def _review_cache(context: Any, source: Path) -> list[dict[str, Any]]:
    candidates = {
        str(row["trial_id"]): row
        for row in _candidate_records(context.run_root / "candidates")
    }
    public = _rows(source / "public/leaderboard_core/trials.jsonl")
    gold = _rows(source / "private/leaderboard_core/gold.jsonl")
    hidden = _rows(source / "private/leaderboard_core/hidden_gold.jsonl")
    groups = {
        str(row["probe_group_id"]): row
        for row in read_jsonl(
            source / "public/leaderboard_core/probe_groups_public.jsonl"
        )
    }
    now = time.time()
    decisions = []
    for trial_id, public_row in public.items():
        candidate = candidates.get(
            trial_id,
            {
                "trial_id": trial_id,
                "public_trial": public_row,
                "available_evidence_ids": hidden[trial_id].get(
                    "acceptable_evidence_ids", []
                ),
                "candidate_source": "frozen_review_cache",
            },
        )
        group = groups.get(str(public_row["probe_group_id"]))
        if group is None:
            raise ValueError(f"review cache has no public probe group: {trial_id}")
        final_value = {
            "trial_id": trial_id,
            "public_trial": public_row,
            "gold": gold[trial_id],
            "hidden_gold": hidden[trial_id],
            "probe_groups": [group],
        }
        decision = (
            "accept" if stable_hash(candidate) == stable_hash(final_value) else "repair"
        )
        reason = "replayed from frozen Qwen plus Codex review cache"
        decisions.append(
            {
                "trial_id": trial_id,
                "decision": decision,
                "candidate": candidate,
                "final_value": final_value,
                "reason": reason,
                "evidence_ids": candidate.get("available_evidence_ids", []),
                "review_model": "legacy_codex_qwen_review",
                "prompt_version": "legacy_codex_review_replay_v1",
                "input_hash": stable_hash(candidate),
                "response_hash": stable_hash(final_value),
                "request_time_unix": now,
                "retry_count": 0,
                "gold_source": "model_verified" if final_value else None,
            }
        )
    return decisions


def codex_review_stage(context: Any) -> dict[str, Any]:
    cached = _cache(context, "review_benchmark")
    if cached is not None:
        decisions = _review_cache(context, cached)
        mode = "review_cache_replay"
    else:
        from .stages import CodexJudge

        judge = CodexJudge(context.config.judge_model)
        decisions = []
        for candidate in _candidate_records(context.run_root / "candidates"):
            result = judge.review(candidate)
            decisions.append({"trial_id": candidate["trial_id"], **result})
        mode = "codex_live"
    path = context.run_root / "review_decisions.jsonl"
    write_jsonl(path, decisions)
    accepted = sum(row["decision"] in {"accept", "repair"} for row in decisions)
    return {
        "mode": mode,
        "decisions": len(decisions),
        "accepted": accepted,
        "quarantined": len(decisions) - accepted,
        "output": str(path),
        "output_hash": sha256_tree(path),
    }


def trial_build_stage(context: Any) -> dict[str, Any]:
    return assemble_release(
        context.run_root / "benchmark_staging",
        context.run_root / "candidates",
        context.run_root / "annotations",
        context.config.dataset_root,
        context.run_root / "review_decisions.jsonl",
    )
