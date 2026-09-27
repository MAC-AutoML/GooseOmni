from .submission_config import *  # noqa: F401,F403

def active_job_names() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def active_job_ids_by_name() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i %j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}
    jobs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            jobs[parts[1]] = parts[0]
    return jobs


def active_job_ids() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def downstream_marker_active(marker_path: Path, active_jobs: set[str], active_ids: dict[str, str]) -> bool:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    active_id_values = set(active_ids.values()) | active_job_ids()
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        return False
    for stage, value in jobs.items():
        job_id = str(value)
        if job_id in active_id_values:
            return True
        segment_id = str(payload.get("segment_id", ""))
        if segment_id and job_name(str(stage), segment_id) in active_jobs:
            return True
    return False


def active_segment_numbers_for_stage(stage: str, active_jobs: set[str]) -> set[int]:
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    prefix = f"og-{stage_name}-"
    active: set[int] = set()
    for name in active_jobs:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        match = re.match(r"(\d{4})(?:-(\d{4}))?", suffix)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        active.update(range(start, end + 1))
    return active


def active_segment_ranges_for_stage(stage: str, active_jobs: set[str]) -> list[tuple[int, int]]:
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    prefix = f"og-{stage_name}-"
    ranges: list[tuple[int, int]] = []
    for name in active_jobs:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        match = re.match(r"(\d{4})(?:-(\d{4}))?", suffix)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        ranges.append((start, end))
    return sorted(set(ranges))


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 1:
        return [[item] for item in items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def partial_downstream_batches(
    ready_segment_ids: list[str],
    active_jobs: set[str],
    batch_size: int,
) -> list[list[str]]:
    by_index = {int(segment_number(segment_id)): segment_id for segment_id in ready_segment_ids}
    batches: list[list[str]] = []
    seen_batches: set[str] = set()
    covered_segments: set[str] = set()
    for stage in ("global_events", "information_states", "memory_states"):
        for start, end in active_segment_ranges_for_stage(stage, active_jobs):
            segment_ids = [
                by_index[index]
                for index in range(start, end + 1)
                if index in by_index
            ]
            if not segment_ids:
                continue
            key = ",".join(segment_ids)
            if key not in seen_batches:
                batches.append(segment_ids)
                seen_batches.add(key)
                covered_segments.update(segment_ids)
    for segment_ids in chunked([item for item in ready_segment_ids if item not in covered_segments], batch_size):
        key = ",".join(segment_ids)
        if key not in seen_batches:
            batches.append(segment_ids)
            seen_batches.add(key)
    return batches


def active_batch_dependency(
    stage: str,
    segment_ids: list[str],
    active_ids: dict[str, str],
) -> str | None:
    name = job_name(stage, segment_ids[0], segment_ids=segment_ids)
    if name in active_ids:
        return active_ids[name]
    if len(segment_ids) > 1:
        return None
    single_name = job_name(stage, segment_ids[0])
    return active_ids.get(single_name)


def submit_partial_downstream_batch(
    args: argparse.Namespace,
    segment_ids: list[str],
    active_ids: dict[str, str],
) -> int | str:
    global_dependency = active_batch_dependency("global_events", segment_ids, active_ids)
    info_dependency = active_batch_dependency("information_states", segment_ids, active_ids)
    memory_dependency = active_batch_dependency("memory_states", segment_ids, active_ids)
    if not global_dependency and not info_dependency and not memory_dependency:
        return 0
    submitted = 0
    segment_id = segment_ids[0]
    info_name = job_name("information_states", segment_id, segment_ids=segment_ids)
    memory_name = job_name("memory_states", segment_id, segment_ids=segment_ids)
    belief_name = job_name("belief_states", segment_id, segment_ids=segment_ids)
    trial_name = job_name("candidate_trials", segment_id, segment_ids=segment_ids)
    active_jobs = set(active_ids)
    active_info_segments = active_segment_numbers_for_stage("information_states", active_jobs)
    active_memory_segments = active_segment_numbers_for_stage("memory_states", active_jobs)
    active_belief_segments = active_segment_numbers_for_stage("belief_states", active_jobs)
    active_trial_segments = active_segment_numbers_for_stage("candidate_trials", active_jobs)
    segment_numbers = {int(segment_number(item)) for item in segment_ids}
    if global_dependency and not segment_numbers.issubset(active_info_segments) and info_name not in active_ids:
        info_job = submit_stage(args, "information_states", segment_id, global_dependency, segment_ids=segment_ids)
        if info_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=information_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[info_name] = info_job
        info_dependency = info_job
        submitted += 1
        print(f"submitted downstream completion stage=information_states segment_ids={','.join(segment_ids)} job={info_job}")
    if info_dependency and not segment_numbers.issubset(active_memory_segments) and memory_name not in active_ids:
        memory_job = submit_stage(args, "memory_states", segment_id, info_dependency, segment_ids=segment_ids)
        if memory_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=memory_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[memory_name] = memory_job
        memory_dependency = memory_job
        submitted += 1
        print(f"submitted downstream completion stage=memory_states segment_ids={','.join(segment_ids)} job={memory_job}")
    if memory_dependency and not segment_numbers.issubset(active_belief_segments) and belief_name not in active_ids:
        belief_job = submit_stage(args, "belief_states", segment_id, memory_dependency, segment_ids=segment_ids)
        if belief_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=belief_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[belief_name] = belief_job
        submitted += 1
        print(f"submitted downstream completion stage=belief_states segment_ids={','.join(segment_ids)} job={belief_job}")
    if info_dependency and not segment_numbers.issubset(active_trial_segments) and trial_name not in active_ids:
        trial_job = submit_stage(args, "candidate_trials", segment_id, info_dependency, segment_ids=segment_ids)
        if trial_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=candidate_trials segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[trial_name] = trial_job
        submitted += 1
        print(f"submitted downstream completion stage=candidate_trials segment_ids={','.join(segment_ids)} job={trial_job}")
    return submitted


def has_unsubmitted_ready_downstream(args: argparse.Namespace, statuses: dict[str, SegmentStatus]) -> bool:
    active_jobs = active_job_names()
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    all_args = argparse.Namespace(**vars(args))
    all_args.start_index = 1
    all_args.end_index = None
    candidate_counts = load_candidate_counts(args.annotation_root)
    for _, segment_id in selected_segments(all_args):
        status = segment_status(args.annotation_root, segment_id, candidate_counts)
        if not status.ready_for_global:
            continue
        segment_index = int(segment_number(segment_id))
        active_stages = {
            stage
            for stage in DOWNSTREAM_STAGES
            if segment_index in active_segments_by_stage[stage] or job_name(stage, segment_id) in active_jobs
        }
        if active_stages and active_stages != set(DOWNSTREAM_STAGES):
            return True
        if active_stages == set(DOWNSTREAM_STAGES):
            continue
        if downstream_marker_path(args.annotation_root, segment_id).exists() and not args.force:
            continue
        return True
    return False

