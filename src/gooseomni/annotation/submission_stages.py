import argparse
import json
import time
from pathlib import Path

from gooseomni.annotation.submission_config import (
    DOWNSTREAM_ACTIVE_STAGES,
    DOWNSTREAM_CHAIN_STAGE,
    PLAYERS,
    SUBMIT_LIMIT,
    UPSTREAM_ACTIVE_STAGES,
    UPSTREAM_CHAIN_STAGE,
    UPSTREAM_STAGES,
    SegmentStatus,
    job_name,
    load_candidate_counts,
    segment_number,
    segment_status,
    selected_segments,
    stage_complete,
    submit_stage,
)
from gooseomni.annotation.submission_queue import (
    active_job_ids_by_name,
    active_job_names,
    active_segment_numbers_for_stage,
    chunked,
    downstream_marker_active,
    has_unsubmitted_ready_downstream,
    partial_downstream_batches,
    submit_partial_downstream_batch,
)


def submit_upstream(
    args: argparse.Namespace, statuses: dict[str, SegmentStatus]
) -> int:
    if (
        not args.force
        and args.stage is None
        and has_unsubmitted_ready_downstream(args, statuses)
    ):
        print("defer upstream: unsubmitted ready downstream segments exist")
        print("submitted_upstream_total=0")
        return 0
    active_jobs = active_job_names()
    submitted = 0
    rows = selected_segments(args)
    if args.stage is None:
        active_segments_by_stage = {
            stage: active_segment_numbers_for_stage(stage, active_jobs)
            for stage in UPSTREAM_ACTIVE_STAGES
        }
        missing_chain_segments = []
        for _, segment_id in rows:
            status = statuses[segment_id]
            if status.upstream_complete:
                continue
            segment_index = int(segment_number(segment_id))
            if (
                any(
                    segment_index in active_segments_by_stage[stage]
                    for stage in UPSTREAM_ACTIVE_STAGES
                )
                and not args.force
            ):
                print(
                    f"skip upstream_chain already_active segment_id={segment_id} segment_index={segment_index:04d}"
                )
                continue
            missing_chain_segments.append(segment_id)
        for segment_ids in chunked(missing_chain_segments, args.batch_size):
            segment_id = segment_ids[0]
            name = job_name(UPSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids)
            if name in active_jobs and not args.force:
                print(
                    "skip upstream_chain batch already_active "
                    f"segment_ids={','.join(segment_ids)} job_name={name}"
                )
                continue
            job_id = submit_stage(
                args, UPSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids
            )
            if job_id == SUBMIT_LIMIT:
                print(
                    f"stop upstream submit_limit stage={UPSTREAM_CHAIN_STAGE} segment_ids={','.join(segment_ids)}"
                )
                print(f"submitted_upstream_total={submitted}")
                return submitted
            print(
                f"submitted upstream_chain segment_ids={','.join(segment_ids)} job={job_id}"
            )
            submitted += 1
            if submitted >= args.max_jobs:
                print(f"max_jobs_reached={args.max_jobs}")
                return submitted
        print(f"submitted_upstream_total={submitted}")
        return submitted

    stage_batches: list[tuple[str, list[str]]] = []
    for stage in [args.stage] if args.stage else UPSTREAM_STAGES:
        active_segments = active_segment_numbers_for_stage(stage, active_jobs)
        missing = []
        for _, segment_id in rows:
            status = statuses[segment_id]
            if stage_complete(status, stage):
                continue
            segment_index = int(segment_number(segment_id))
            if segment_index in active_segments and not args.force:
                print(
                    f"skip upstream already_active stage={stage} segment_id={segment_id} segment_index={segment_index:04d}"
                )
                continue
            name = job_name(stage, segment_id)
            if name in active_jobs and not args.force:
                print(
                    f"skip upstream already_active stage={stage} segment_id={segment_id} job_name={name}"
                )
                continue
            missing.append(segment_id)
        stage_batches.extend(
            (stage, segment_ids) for segment_ids in chunked(missing, args.batch_size)
        )
    if args.stage is None:
        stage_order = {stage: index for index, stage in enumerate(UPSTREAM_STAGES)}
        stage_batches.sort(
            key=lambda item: (
                int(segment_number(item[1][0])),
                stage_order.get(item[0], len(stage_order)),
            )
        )
    for stage, segment_ids in stage_batches:
        segment_id = segment_ids[0]
        name = job_name(stage, segment_id, segment_ids=segment_ids)
        if name in active_jobs and not args.force:
            print(
                f"skip upstream batch already_active stage={stage} segment_ids={','.join(segment_ids)} job_name={name}"
            )
            continue
        job_id = submit_stage(args, stage, segment_id, segment_ids=segment_ids)
        if job_id == SUBMIT_LIMIT:
            print(
                f"stop upstream submit_limit stage={stage} segment_ids={','.join(segment_ids)}"
            )
            print(f"submitted_upstream_total={submitted}")
            return submitted
        print(
            f"submitted upstream stage={stage} segment_ids={','.join(segment_ids)} job={job_id}"
        )
        submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_upstream_total={submitted}")
    return submitted


def promote_downstream(
    args: argparse.Namespace, statuses: dict[str, SegmentStatus]
) -> int:
    active_jobs = active_job_names()
    active_ids = active_job_ids_by_name()
    submitted = 0
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    ready_segments = []
    for _, segment_id in selected_segments(args):
        status = statuses[segment_id]
        if not status.ready_for_global:
            continue
        segment_index = int(segment_number(segment_id))
        active_downstream = [
            job_name(stage, segment_id)
            for stage in DOWNSTREAM_ACTIVE_STAGES
            if segment_index in active_segments_by_stage[stage]
            or job_name(stage, segment_id) in active_jobs
        ]
        if active_downstream and not args.force:
            print(
                f"skip downstream already_active segment_id={segment_id} jobs={','.join(active_downstream)}"
            )
            continue
        marker_path = downstream_marker_path(args.annotation_root, segment_id)
        if marker_path.exists() and not args.force:
            if downstream_marker_active(marker_path, active_jobs, active_ids):
                print(
                    f"skip downstream already_submitted segment_id={segment_id} marker={marker_path.as_posix()}"
                )
                continue
            print(
                f"ignore stale downstream marker segment_id={segment_id} marker={marker_path.as_posix()}"
            )
        ready_segments.append(segment_id)

    ready_segment_ids = [
        segment_id
        for _, segment_id in selected_segments(args)
        if statuses[segment_id].ready_for_global
    ]
    for segment_ids in partial_downstream_batches(
        ready_segment_ids, active_jobs, args.batch_size
    ):
        added = submit_partial_downstream_batch(args, segment_ids, active_ids)
        if added == SUBMIT_LIMIT:
            print(f"submitted_downstream_total={submitted}")
            return submitted
        submitted += added
        if added and submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted

    for segment_ids in chunked(ready_segments, args.batch_size):
        segment_id = segment_ids[0]
        chain_job = submit_stage(
            args, DOWNSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids
        )
        if chain_job == SUBMIT_LIMIT:
            print(
                f"stop downstream submit_limit stage={DOWNSTREAM_CHAIN_STAGE} segment_ids={','.join(segment_ids)}"
            )
            print(f"submitted_downstream_total={submitted}")
            return submitted
        jobs = {DOWNSTREAM_CHAIN_STAGE: chain_job}
        for marker_segment_id in segment_ids:
            write_downstream_marker(
                downstream_marker_path(args.annotation_root, marker_segment_id),
                {
                    "segment_id": marker_segment_id,
                    "batch_segment_ids": segment_ids,
                    "submitted_at_unix": time.time(),
                    "jobs": jobs,
                },
                dry_run=args.dry_run,
            )
        print(
            f"submitted downstream_chain segment_ids={','.join(segment_ids)} job={chain_job}"
        )
        submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_downstream_total={submitted}")
    return submitted


def repair_downstream(
    args: argparse.Namespace, statuses: dict[str, SegmentStatus]
) -> int:
    active_jobs = active_job_names()
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    submitted = 0
    for _, segment_id in selected_segments(args):
        segment_index = int(segment_number(segment_id))
        status = statuses[segment_id]
        if not status.global_events:
            continue
        if status.information_states < len(PLAYERS):
            name = job_name("information_states", segment_id)
            if (
                segment_index in active_segments_by_stage["information_states"]
                or name in active_jobs
            ) and not args.force:
                print(
                    f"skip repair already_active stage=information_states segment_id={segment_id}"
                )
            else:
                job_id = submit_stage(args, "information_states", segment_id)
                if job_id == SUBMIT_LIMIT:
                    print(
                        f"stop repair submit_limit stage=information_states segment_id={segment_id}"
                    )
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(
                    f"submitted repair stage=information_states segment_id={segment_id} job={job_id}"
                )
                submitted += 1
        if status.memory_states < len(PLAYERS) and status.information_states == len(
            PLAYERS
        ):
            stage_active = segment_index in active_segments_by_stage["memory_states"]
            if stage_active and not args.force:
                print(
                    f"skip repair already_active stage=memory_states segment_id={segment_id}"
                )
                continue
            for player in missing_players(
                args.annotation_root, "memory_states", segment_id
            ):
                name = job_name("memory_states", segment_id, player)
                if name in active_jobs and not args.force:
                    print(
                        f"skip repair already_active stage=memory_states segment_id={segment_id} target={player}"
                    )
                    continue
                job_id = submit_stage(
                    args, "memory_states", segment_id, target_player=player
                )
                if job_id == SUBMIT_LIMIT:
                    print(
                        f"stop repair submit_limit stage=memory_states segment_id={segment_id} target={player}"
                    )
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(
                    f"submitted repair stage=memory_states segment_id={segment_id} target={player} job={job_id}"
                )
                submitted += 1
                if submitted >= args.max_jobs:
                    print(f"max_jobs_reached={args.max_jobs}")
                    return submitted
        if status.belief_states < len(PLAYERS) and status.memory_states == len(PLAYERS):
            stage_active = segment_index in active_segments_by_stage["belief_states"]
            if stage_active and not args.force:
                print(
                    f"skip repair already_active stage=belief_states segment_id={segment_id}"
                )
                continue
            for player in missing_players(
                args.annotation_root, "belief_states", segment_id
            ):
                name = job_name("belief_states", segment_id, player)
                if name in active_jobs and not args.force:
                    print(
                        f"skip repair already_active stage=belief_states segment_id={segment_id} target={player}"
                    )
                    continue
                job_id = submit_stage(
                    args, "belief_states", segment_id, target_player=player
                )
                if job_id == SUBMIT_LIMIT:
                    print(
                        f"stop repair submit_limit stage=belief_states segment_id={segment_id} target={player}"
                    )
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(
                    f"submitted repair stage=belief_states segment_id={segment_id} target={player} job={job_id}"
                )
                submitted += 1
                if submitted >= args.max_jobs:
                    print(f"max_jobs_reached={args.max_jobs}")
                    return submitted
        if status.candidate_trials == 0 and status.information_states == len(PLAYERS):
            name = job_name("candidate_trials", segment_id)
            if (
                segment_index in active_segments_by_stage["candidate_trials"]
                or name in active_jobs
            ) and not args.force:
                print(
                    f"skip repair already_active stage=candidate_trials segment_id={segment_id}"
                )
            else:
                job_id = submit_stage(args, "candidate_trials", segment_id)
                if job_id == SUBMIT_LIMIT:
                    print(
                        f"stop repair submit_limit stage=candidate_trials segment_id={segment_id}"
                    )
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(
                    f"submitted repair stage=candidate_trials segment_id={segment_id} job={job_id}"
                )
                submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_repair_total={submitted}")
    return submitted


def missing_players(annotation_root: Path, stage: str, segment_id: str) -> list[str]:
    path = annotation_root / stage / "g001" / segment_id
    existing = {item.stem for item in path.glob("*.json")} if path.exists() else set()
    return [player for player in PLAYERS if player not in existing]


def downstream_marker_path(annotation_root: Path, segment_id: str) -> Path:
    return annotation_root / "job_markers" / "downstream" / f"{segment_id}.json"


def write_downstream_marker(
    path: Path, payload: dict[str, object], *, dry_run: bool
) -> None:
    print(f"write_marker {path.as_posix()}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def refresh_statuses(
    args: argparse.Namespace, rows: list[tuple[int, str]]
) -> dict[str, SegmentStatus]:
    candidate_counts = load_candidate_counts(args.annotation_root)
    return {
        segment_id: segment_status(args.annotation_root, segment_id, candidate_counts)
        for _, segment_id in rows
    }


def with_max_jobs(args: argparse.Namespace, max_jobs: int) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.max_jobs = max_jobs
    return copied


def upstream_range_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    if args.upstream_start_index is not None:
        copied.start_index = args.upstream_start_index
    if args.upstream_end_index is not None:
        copied.end_index = args.upstream_end_index
    return copied
