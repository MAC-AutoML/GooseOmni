from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

#!/usr/bin/env python3


PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
PUBLIC_FORBIDDEN = [
    "hidden_gold",
    "forbidden_event_ids",
    "qwen",
    "Qwen",
    "codex_",
    "human_review_record_id",
    "review_status",
    "behavior_grounded",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def link_tree_files(src_root: Path, dst_root: Path) -> dict[str, Any]:
    counts = Counter()
    total_bytes = 0
    sample_hashes: dict[str, str] = {}
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)
        mode = link_or_copy(src, dst_root / rel)
        counts[mode] += 1
        total_bytes += src.stat().st_size
        if len(sample_hashes) < 8:
            h = hashlib.sha256()
            with src.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            sample_hashes[rel.as_posix()] = h.hexdigest()
    return {
        "counts": dict(counts),
        "total_bytes": total_bytes,
        "sample_sha256": sample_hashes,
    }


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return None


def public_leak_hits(path: Path) -> list[dict[str, Any]]:
    hits = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        for pattern in PUBLIC_FORBIDDEN:
            if pattern in line:
                hits.append(
                    {"path": path.as_posix(), "line": line_no, "pattern": pattern}
                )
    return hits


def choose_video_for_trial(
    trial: dict[str, Any], group: dict[str, Any], videos_root: Path
) -> Path | None:
    phase_ids = list(
        dict.fromkeys(str(x) for x in group.get("source_segment_ids", []) if x)
    )
    player_order = []
    target = trial.get("target_player") or group.get("target_player")
    if target in PLAYERS:
        player_order.append(target)
    player_order.extend(PLAYERS)
    player_order = list(dict.fromkeys(player_order))
    for phase_id in phase_ids:
        for player in player_order:
            path = videos_root / "g001" / phase_id / f"{player}.mp4"
            if path.exists():
                return path
    return None


def stratified_smoke_rows(
    trials: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    videos_root: Path,
    per_type: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_type[str(trial.get("probe_type"))].append(trial)
    for probe_type in [
        "A_pre_reveal_belief",
        "B_post_reveal_reconstruct_previous_belief",
        "C_other_agent_false_belief",
        "D_perspective_taking_prediction",
    ]:
        count = 0
        for trial in by_type.get(probe_type, []):
            group = groups.get(trial["probe_group_id"], {})
            video = choose_video_for_trial(trial, group, videos_root)
            if not video:
                missing.append(
                    {
                        "trial_id": trial.get("trial_id"),
                        "probe_group_id": trial.get("probe_group_id"),
                        "probe_type": probe_type,
                        "reason": "no_video_resolved",
                    }
                )
                continue
            row = dict(trial)
            row["track"] = "raw_video_smoke"
            row["video_file"] = video.relative_to(videos_root.parents[2]).as_posix()
            row["video_resolution_basis"] = (
                "target_player_then_players_by_source_segment"
            )
            row["prompt"] = (
                row["prompt"]
                + "\n\nRAW_POV_VIDEO_RULES:\n只根据随本题提供的指定 POV 视频在 cutoff 前可见/可听的信息回答。不要使用其他 POV、后续揭示或 hidden oracle 信息。死亡/倒地/血迹画面只认证可见死亡状态，不认证攻击者、阵营、技能或击杀机制。"
            )
            selected.append(row)
            count += 1
            if count >= per_type:
                break
    return selected, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GooseOmni Qwen3-Omni eval workspace."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-pass", type=Path, default=None)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"),
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path("runs/gooseomni_eval_qwen3omni_pass001")
    )
    parser.add_argument("--raw-smoke-per-type", type=int, default=12)
    parser.add_argument(
        "--model-path", default="/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct"
    )
    parser.add_argument("--overwrite-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    workdir = args.workdir
    source_pass = args.source_pass or Path(
        (repo_root / "current_recommended_pass.txt").read_text(encoding="utf-8").strip()
    )
    benchmark_src = source_pass / "benchmark" / "gooseomni_v1"
    videos_src = args.release_root / "inputs" / "videos"
    manifest_src = args.release_root / "inputs" / "manifest.jsonl"
    if not benchmark_src.exists():
        raise FileNotFoundError(f"benchmark source not found: {benchmark_src}")
    if not videos_src.exists():
        raise FileNotFoundError(f"video source not found: {videos_src}")
    if not manifest_src.exists():
        raise FileNotFoundError(f"video manifest not found: {manifest_src}")

    for rel in [
        "inputs",
        "responses/structured",
        "responses/raw_video_smoke",
        "scores",
        "manifests",
        "logs",
        "data",
    ]:
        (workdir / rel).mkdir(parents=True, exist_ok=True)

    benchmark_dst = workdir / "benchmark" / "gooseomni_v1"
    videos_dst = workdir / "data" / "release_benchmark_v2" / "inputs" / "videos"
    benchmark_link = link_tree_files(benchmark_src, benchmark_dst)
    videos_link = link_tree_files(videos_src, videos_dst)
    manifest_mode = link_or_copy(
        manifest_src,
        workdir / "data" / "release_benchmark_v2" / "inputs" / "manifest.jsonl",
    )

    trials_path = benchmark_dst / "leaderboard_core" / "trials.jsonl"
    hidden_path = benchmark_dst / "leaderboard_core" / "hidden_gold.jsonl"
    groups_path = benchmark_dst / "leaderboard_core" / "probe_groups.jsonl"
    trials = read_jsonl(trials_path)
    hidden = read_jsonl(hidden_path)
    groups = {row["probe_group_id"]: row for row in read_jsonl(groups_path)}

    structured_rows = []
    for row in trials:
        out = dict(row)
        out["track"] = "structured"
        structured_rows.append(out)
    raw_rows, missing_videos = stratified_smoke_rows(
        structured_rows, groups, videos_dst, args.raw_smoke_per_type
    )

    structured_path = workdir / "inputs" / "structured_leaderboard_core.jsonl"
    raw_path = workdir / "inputs" / "raw_video_smoke.jsonl"
    if args.overwrite_inputs or not structured_path.exists():
        write_jsonl(structured_path, structured_rows)
    if args.overwrite_inputs or not raw_path.exists():
        write_jsonl(raw_path, raw_rows)

    migration = {
        "workdir": workdir.as_posix(),
        "source_pass": source_pass.as_posix(),
        "source_benchmark": benchmark_src.as_posix(),
        "source_videos": videos_src.as_posix(),
        "source_video_manifest": manifest_src.as_posix(),
        "benchmark_link": benchmark_link,
        "videos_link": videos_link,
        "manifest_mode": manifest_mode,
        "leaderboard_trials": len(trials),
        "leaderboard_hidden_gold": len(hidden),
        "video_manifest_rows": sum(
            1
            for line in manifest_src.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        "structured_input_rows": len(structured_rows),
        "raw_video_smoke_rows": len(raw_rows),
        "raw_video_missing_candidates": missing_videos[:50],
        "raw_video_probe_type_counts": dict(
            Counter(row["probe_type"] for row in raw_rows)
        ),
        "git_commit": git_commit(repo_root),
    }
    write_json(workdir / "manifests" / "data_migration_manifest.json", migration)
    write_json(
        workdir / "manifests" / "eval_config.json",
        {
            "model": "Qwen3-Omni-30B-A3B-Instruct",
            "model_path": args.model_path,
            "source_pass": source_pass.as_posix(),
            "tracks": {
                "structured": {
                    "input": "inputs/structured_leaderboard_core.jsonl",
                    "rows": len(structured_rows),
                    "official_core": True,
                },
                "raw_video_smoke": {
                    "input": "inputs/raw_video_smoke.jsonl",
                    "rows": len(raw_rows),
                    "official_core": False,
                },
            },
            "qwen_env_defaults": {
                "QWEN3_OMNI_MAX_TOKENS": 16384,
                "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": 32768,
                "QWEN3_OMNI_VIDEO_FPS": 1.0,
                "QWEN3_OMNI_VIDEO_MAX_FRAMES": 128,
                "QWEN3_OMNI_VIDEO_MAX_PIXELS": 401408,
                "OMNI_HTTP_TIMEOUT_SEC": 1200,
                "temperature": 0,
            },
        },
    )
    (workdir / "README.md").write_text(
        f"# GooseOmni Qwen3-Omni Evaluation Pass001\n\nSource benchmark: `{source_pass.as_posix()}`\n\nTracks:\n- `structured`: {len(structured_rows)} leaderboard-core public trials.\n- `raw_video_smoke`: {len(raw_rows)} stratified video smoke trials.\n\nQwen3-Omni is used only as the evaluated model. Hidden gold is scorer-only and must not enter prompts.\n\nQuality defaults avoid 4096-token coarse settings: max tokens 16384, text merge 32768, video max frames 128.\n",
        encoding="utf-8",
    )

    leak_hits = []
    for path in [structured_path, raw_path]:
        leak_hits.extend(public_leak_hits(path))
    issues = []
    if len(trials) != 889:
        issues.append({"code": "unexpected_trial_count", "count": len(trials)})
    if len(hidden) != 889:
        issues.append({"code": "unexpected_hidden_gold_count", "count": len(hidden)})
    if len(raw_rows) != args.raw_smoke_per_type * 4:
        issues.append(
            {"code": "unexpected_raw_video_smoke_count", "count": len(raw_rows)}
        )
    report = {
        "ok": not issues and not leak_hits,
        "issues": issues,
        "public_leak_hits": leak_hits,
        "migration": migration,
    }
    write_json(workdir / "manifests" / "prepare_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
