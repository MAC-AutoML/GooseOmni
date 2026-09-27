from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PLAYERS = ("Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu")
UPSTREAM_STAGES = ("pov_events", "utterances", "phase_events")
UPSTREAM_CHAIN_STAGE = "upstream_chain"
UPSTREAM_ACTIVE_STAGES = UPSTREAM_STAGES + (UPSTREAM_CHAIN_STAGE,)
DOWNSTREAM_STAGES = (
    "global_events",
    "information_states",
    "memory_states",
    "belief_states",
    "candidate_trials",
)
DOWNSTREAM_CHAIN_STAGE = "downstream_chain"
DOWNSTREAM_ACTIVE_STAGES = DOWNSTREAM_STAGES + (DOWNSTREAM_CHAIN_STAGE,)
STAGE_ABBREVIATIONS = {
    "pov_events": "pov",
    "utterances": "utt",
    "phase_events": "phase",
    "upstream_chain": "up",
    "global_events": "glob",
    "information_states": "info",
    "memory_states": "mem",
    "belief_states": "belief",
    "candidate_trials": "trial",
    "downstream_chain": "chain",
}

SUBMIT_LIMIT = "SUBMIT_LIMIT"


@dataclass(frozen=True)
class SegmentStatus:
    segment_id: str
    pov_events: int
    utterances: int
    phase_events: bool
    global_events: bool
    information_states: int
    memory_states: int
    belief_states: int
    candidate_trials: int

    @property
    def upstream_complete(self) -> bool:
        return (
            self.pov_events == len(PLAYERS)
            and self.utterances == len(PLAYERS)
            and self.phase_events
        )

    @property
    def ready_for_global(self) -> bool:
        return self.upstream_complete and not self.global_events

    @property
    def downstream_complete(self) -> bool:
        return (
            self.global_events
            and self.information_states == len(PLAYERS)
            and self.memory_states == len(PLAYERS)
            and self.belief_states == len(PLAYERS)
            and self.candidate_trials > 0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Qwen3-Omni Omni Goose oracle annotation jobs."
    )
    parser.add_argument("--dataset-root", default=Path("data/gooseomni"), type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument(
        "--annotation-root",
        default=Path("runs/gooseomni_oracle_pass1/annotations"),
        type=Path,
    )
    parser.add_argument(
        "--slurm-script",
        default=Path("configs/slurm/qwen3_omni_oracle_2gpu_stage.slurm"),
        type=Path,
    )
    parser.add_argument(
        "--multi-worker-slurm-script",
        default=Path("configs/slurm/qwen3_omni_oracle_4x2_local.slurm"),
        type=Path,
        help="8-GPU script that runs multiple 2-GPU Qwen3-Omni workers inside one Slurm job.",
    )
    parser.add_argument("--game-id", default="g001")
    parser.add_argument(
        "--mode",
        choices=[
            "status",
            "submit-upstream",
            "promote-downstream",
            "repair-downstream",
            "submit-balanced",
            "submit-4x2-balanced",
            "export",
        ],
        default="status",
    )
    parser.add_argument(
        "--start-index", default=1, type=int, help="1-based segment index."
    )
    parser.add_argument(
        "--end-index", default=None, type=int, help="Inclusive 1-based segment index."
    )
    parser.add_argument(
        "--upstream-start-index",
        default=None,
        type=int,
        help="Optional upstream-only start index for submit-balanced mode.",
    )
    parser.add_argument(
        "--upstream-end-index",
        default=None,
        type=int,
        help="Optional upstream-only inclusive end index for submit-balanced mode.",
    )
    parser.add_argument("--stage", choices=list(UPSTREAM_STAGES), default=None)
    parser.add_argument("--max-jobs", default=12, type=int)
    parser.add_argument(
        "--max-total-jobs",
        default=60,
        type=int,
        help="Skip new Slurm submissions when this user already has at least this many queued/running jobs. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-pending-jobs",
        default=55,
        type=int,
        help="Skip new Slurm submissions when this user already has at least this many pending jobs. Use 0 to disable.",
    )
    parser.add_argument(
        "--balanced-upstream-jobs",
        default=4,
        type=int,
        help="Maximum upstream jobs to submit after downstream/repair attempts in submit-balanced mode.",
    )
    parser.add_argument(
        "--balanced-downstream-jobs",
        default=8,
        type=int,
        help="Maximum downstream chain jobs to submit before upstream attempts in submit-balanced mode.",
    )
    parser.add_argument(
        "--balanced-repair-jobs",
        default=4,
        type=int,
        help="Maximum repair jobs to submit before upstream attempts in submit-balanced mode.",
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
        help="Number of upstream segments per 2-GPU job.",
    )
    parser.add_argument("--multi-worker-workers", default=4, type=int)
    parser.add_argument(
        "--multi-worker-stage-plan",
        default="upstream_chain,downstream_chain,upstream_chain,downstream_chain",
        help="Comma-separated worker stage plan for submit-4x2-balanced mode.",
    )
    parser.add_argument(
        "--multi-worker-limit-per-worker",
        default="",
        help="Optional per-worker segment limit. Empty means each worker scans all remaining resumable items.",
    )
    parser.add_argument(
        "--allow-small-job-fallback-on-submit-limit",
        action="store_true",
        help="When submit-balanced cannot submit the 4x2 job due to the Slurm submit limit, allow fallback submission of small 2-GPU jobs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Ignore existing submission markers."
    )
    parser.add_argument("--qwen-max-tokens", default="16384")
    parser.add_argument("--qwen-text-merge-max-tokens", default="32768")
    parser.add_argument("--qwen-video-fps", default="1.0")
    parser.add_argument("--qwen-video-max-frames", default="48")
    parser.add_argument("--qwen-video-max-pixels", default="401408")
    parser.add_argument("--omni-http-timeout-sec", default="900")
    return parser.parse_args()


def load_segments(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_json(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for item in path.glob("*.json"):
        try:
            text = item.read_text(encoding="utf-8")
            if not text.strip():
                continue
            json.loads(text)
        except Exception:
            continue
        count += 1
    return count


def load_candidate_counts(annotation_root: Path) -> dict[str, int]:
    path = annotation_root / "candidate_trials" / "g001_candidate_trials.jsonl"
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        segment_id = payload.get("segment_id")
        if segment_id:
            counts[segment_id] = counts.get(segment_id, 0) + 1
    return counts


def segment_status(
    annotation_root: Path, segment_id: str, candidate_counts: dict[str, int]
) -> SegmentStatus:
    game_root = annotation_root
    return SegmentStatus(
        segment_id=segment_id,
        pov_events=count_json(game_root / "pov_events" / "g001" / segment_id),
        utterances=count_json(game_root / "utterances" / "g001" / segment_id),
        phase_events=valid_json_file(
            game_root / "phase_events" / "g001" / f"{segment_id}.json"
        ),
        global_events=valid_json_file(
            game_root / "global_events" / "g001" / f"{segment_id}.json"
        ),
        information_states=count_json(
            game_root / "information_states" / "g001" / segment_id
        ),
        memory_states=count_json(game_root / "memory_states" / "g001" / segment_id),
        belief_states=count_json(game_root / "belief_states" / "g001" / segment_id),
        candidate_trials=candidate_counts.get(segment_id, 0),
    )


def valid_json_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        return bool(text.strip()) and isinstance(json.loads(text), dict)
    except Exception:
        return False


def selected_segments(args: argparse.Namespace) -> list[tuple[int, str]]:
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    rows = load_segments(segments_path)
    end = args.end_index or len(rows)
    selected = []
    for index, row in enumerate(rows, start=1):
        if index < args.start_index or index > end:
            continue
        if row.get("game_id") != args.game_id:
            continue
        selected.append((index, row["segment_id"]))
    return selected


def stage_complete(status: SegmentStatus, stage: str) -> bool:
    if stage == "pov_events":
        return status.pov_events == len(PLAYERS)
    if stage == "utterances":
        return status.utterances == len(PLAYERS)
    if stage == "phase_events":
        return status.phase_events
    raise ValueError(f"unsupported upstream stage: {stage}")


def run_command(
    cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None
) -> str:
    env_prefix = ""
    if env:
        env_prefix = (
            " ".join(f"{key}={value}" for key, value in sorted(env.items())) + " "
        )
    print(env_prefix + " ".join(cmd))
    if dry_run:
        return "DRY_RUN"
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, env=run_env
    )
    if result.returncode == 0:
        return result.stdout.strip()
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"command failed with returncode={result.returncode}"
    if "AssocGrpSubmitJobsLimit" in message or "job submit limit" in message:
        print(f"submit_limit_reached: {message}")
        return SUBMIT_LIMIT
    raise subprocess.CalledProcessError(
        result.returncode, cmd, output=result.stdout, stderr=result.stderr
    )


def sbatch_env(
    args: argparse.Namespace,
    stage: str,
    segment_id: str,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> dict[str, str]:
    env = {
        "STAGE": stage,
        "SEGMENT_ID": segment_id,
        "LIMIT": "1",
        "RESUME": "1",
        "OVERWRITE": "0",
        "ANNOTATION_ROOT": args.annotation_root.as_posix(),
        "QWEN3_OMNI_MAX_TOKENS": args.qwen_max_tokens,
        "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": args.qwen_text_merge_max_tokens,
        "QWEN3_OMNI_VIDEO_FPS": args.qwen_video_fps,
        "QWEN3_OMNI_VIDEO_MAX_FRAMES": args.qwen_video_max_frames,
        "QWEN3_OMNI_VIDEO_MAX_PIXELS": args.qwen_video_max_pixels,
        "OMNI_HTTP_TIMEOUT_SEC": args.omni_http_timeout_sec,
    }
    if segment_ids:
        env["SEGMENT_IDS"] = ",".join(segment_ids)
    if target_player is not None:
        env["TARGET_PLAYER"] = target_player
    return env


def submit_stage(
    args: argparse.Namespace,
    stage: str,
    segment_id: str,
    dependency: str | None = None,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> str:
    env = sbatch_env(args, stage, segment_id, target_player, segment_ids)
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name(stage, segment_id, target_player, segment_ids)}",
    ]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(args.slurm_script.as_posix())
    return run_command(cmd, dry_run=args.dry_run, env=env)


def segment_number(segment_id: str) -> str:
    match = re.search(r"_seg_(\d{4})_", segment_id)
    return match.group(1) if match else segment_id[-8:]


def job_name(
    stage: str,
    segment_id: str,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> str:
    if segment_ids and len(segment_ids) > 1:
        first = segment_number(segment_ids[0])
        last = segment_number(segment_ids[-1])
        segment_index = f"{first}-{last}"
    else:
        segment_index = segment_number(segment_id)
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    suffix = f"-{target_player[:3].lower()}" if target_player else ""
    return f"og-{stage_name}-{segment_index}{suffix}"


def queue_counts() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%T"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"skip submit queue_cap reason=squeue_failed error={exc}")
        return None
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "total": len(states),
        "pending": sum(1 for state in states if state == "PENDING"),
    }


def queue_submit_allowed(args: argparse.Namespace) -> bool:
    if args.force:
        return True
    counts = queue_counts()
    if counts is None:
        return False
    total_cap_hit = args.max_total_jobs > 0 and counts["total"] >= args.max_total_jobs
    pending_cap_hit = (
        args.max_pending_jobs > 0 and counts["pending"] >= args.max_pending_jobs
    )
    if total_cap_hit or pending_cap_hit:
        print(
            "skip submit queue_cap "
            f"total={counts['total']} pending={counts['pending']} "
            f"max_total={args.max_total_jobs} max_pending={args.max_pending_jobs}"
        )
        return False
    print(
        "queue submit allowed "
        f"total={counts['total']} pending={counts['pending']} "
        f"max_total={args.max_total_jobs} max_pending={args.max_pending_jobs}"
    )
    return True
