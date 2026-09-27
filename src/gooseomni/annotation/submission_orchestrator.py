from .submission_stages import *  # noqa: F401,F403

def submit_balanced(
    args: argparse.Namespace,
    rows: list[tuple[int, str]],
    statuses: dict[str, SegmentStatus],
) -> None:
    if not queue_submit_allowed(args):
        print("submitted_balanced_total=0 downstream=0 repair=0 upstream=0 queue_cap=1")
        return
    multi_worker_status = submit_4x2_balanced(args)
    if multi_worker_status in {"submitted", "already_active"}:
        print(
            "submitted_balanced_total=0 "
            f"downstream=0 repair=0 upstream=0 multi_worker={multi_worker_status}"
        )
        return
    if multi_worker_status == "submit_limit" and not args.allow_small_job_fallback_on_submit_limit:
        print(
            "submitted_balanced_total=0 "
            "downstream=0 repair=0 upstream=0 "
            "multi_worker=submit_limit small_job_fallback=disabled"
        )
        return

    downstream_jobs = promote_downstream(
        with_max_jobs(args, args.balanced_downstream_jobs),
        statuses,
    )
    statuses = refresh_statuses(args, rows)
    repair_jobs = repair_downstream(
        with_max_jobs(args, args.balanced_repair_jobs),
        statuses,
    )
    upstream_args = upstream_range_args(args)
    upstream_rows = selected_segments(upstream_args)
    upstream_jobs = submit_upstream(
        with_max_jobs(upstream_args, args.balanced_upstream_jobs),
        refresh_statuses(upstream_args, upstream_rows),
    )
    print(
        "submitted_balanced_total="
        f"{downstream_jobs + repair_jobs + upstream_jobs} "
        f"downstream={downstream_jobs} repair={repair_jobs} upstream={upstream_jobs}"
    )



def effective_multi_worker_stage_plan(args: argparse.Namespace) -> str:
    rows = selected_segments(args)
    statuses = refresh_statuses(args, rows)
    if statuses and all(status.upstream_complete for status in statuses.values()):
        return ",".join([DOWNSTREAM_CHAIN_STAGE] * args.multi_worker_workers)
    if statuses and all(status.downstream_complete for status in statuses.values()):
        return ",".join([UPSTREAM_CHAIN_STAGE] * args.multi_worker_workers)
    return args.multi_worker_stage_plan


def submit_4x2_balanced(args: argparse.Namespace) -> str:
    if not queue_submit_allowed(args):
        print("submitted_4x2_balanced=0 reason=queue_cap")
        return "queue_cap"
    active_jobs = active_job_names()
    if "og-4x2-balanced" in active_jobs and not args.force:
        print("submitted_4x2_balanced=0 reason=already_active")
        return "already_active"
    stage_plan = effective_multi_worker_stage_plan(args)
    env = {
        "DATASET_ROOT": args.dataset_root.as_posix(),
        "SEGMENTS_JSONL": (args.segments_jsonl or (args.dataset_root / "segments.jsonl")).as_posix(),
        "ANNOTATION_ROOT": args.annotation_root.as_posix(),
        "GAME_ID": args.game_id,
        "WORKERS": str(args.multi_worker_workers),
        "STAGE_SET": "balanced",
        "STAGE_PLAN": stage_plan,
        "LIMIT_PER_WORKER": args.multi_worker_limit_per_worker,
        "RESUME": "1",
        "OVERWRITE": "0",
        "QWEN3_OMNI_MAX_TOKENS": args.qwen_max_tokens,
        "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": args.qwen_text_merge_max_tokens,
        "QWEN3_OMNI_VIDEO_FPS": args.qwen_video_fps,
        "QWEN3_OMNI_VIDEO_MAX_FRAMES": args.qwen_video_max_frames,
        "QWEN3_OMNI_VIDEO_MAX_PIXELS": args.qwen_video_max_pixels,
        "OMNI_HTTP_TIMEOUT_SEC": args.omni_http_timeout_sec,
    }
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=og-4x2-balanced",
        args.multi_worker_slurm_script.as_posix(),
    ]
    job_id = run_command(cmd, dry_run=args.dry_run, env=env)
    if job_id == SUBMIT_LIMIT:
        print("submitted_4x2_balanced=0 reason=submit_limit")
        return "submit_limit"
    else:
        print(f"submitted_4x2_balanced=1 job={job_id} stage_plan={stage_plan}")
        return "submitted"


def export_benchmark(args: argparse.Namespace) -> None:
    benchmark_root = args.annotation_root.parent / "benchmark"
    run_command(
        [
            ".venv/bin/python",
            "-u",
            "tools/package/export_tom_benchmark.py",
            "--dataset-root",
            args.dataset_root.as_posix(),
            "--annotation-root",
            args.annotation_root.as_posix(),
            "--benchmark-root",
            benchmark_root.as_posix(),
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            ".venv/bin/python",
            "-u",
            "tools/report/report_benchmark_quality.py",
            "--dataset-root",
            args.dataset_root.as_posix(),
            "--benchmark-root",
            benchmark_root.as_posix(),
        ],
        dry_run=args.dry_run,
    )


def print_status(rows: list[tuple[int, str]], statuses: dict[str, SegmentStatus]) -> None:
    totals = {
        "upstream_complete": 0,
        "global_complete": 0,
        "downstream_complete": 0,
        "ready_for_global": 0,
    }
    for _, segment_id in rows:
        status = statuses[segment_id]
        totals["upstream_complete"] += int(status.upstream_complete)
        totals["global_complete"] += int(status.global_events)
        totals["downstream_complete"] += int(status.downstream_complete)
        totals["ready_for_global"] += int(status.ready_for_global)
    print(json.dumps(totals, ensure_ascii=False, sort_keys=True))
    for index, segment_id in rows:
        status = statuses[segment_id]
        print(
            f"{index:03d} {segment_id} "
            f"pov={status.pov_events}/6 utt={status.utterances}/6 phase={int(status.phase_events)} "
            f"global={int(status.global_events)} info={status.information_states}/6 "
            f"memory={status.memory_states}/6 belief={status.belief_states}/6 trials={status.candidate_trials}"
        )


def main() -> None:
    args = parse_args()
    rows = selected_segments(args)
    candidate_counts = load_candidate_counts(args.annotation_root)
    statuses = {
        segment_id: segment_status(args.annotation_root, segment_id, candidate_counts)
        for _, segment_id in rows
    }
    if args.mode == "status":
        print_status(rows, statuses)
    elif args.mode == "submit-upstream":
        if not queue_submit_allowed(args):
            print("submitted_upstream_total=0 queue_cap=1")
            return
        submit_upstream(args, statuses)
    elif args.mode == "promote-downstream":
        if not queue_submit_allowed(args):
            print("submitted_downstream_total=0 queue_cap=1")
            return
        promote_downstream(args, statuses)
    elif args.mode == "repair-downstream":
        if not queue_submit_allowed(args):
            print("submitted_repair_total=0 queue_cap=1")
            return
        repair_downstream(args, statuses)
    elif args.mode == "submit-balanced":
        submit_balanced(args, rows, statuses)
    elif args.mode == "submit-4x2-balanced":
        submit_4x2_balanced(args)
    elif args.mode == "export":
        export_benchmark(args)


if __name__ == "__main__":
    main()
