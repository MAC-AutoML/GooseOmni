from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


QUALITY_PROFILE = {
    "QWEN3_OMNI_MAX_TOKENS": 16384,
    "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": 32768,
    "QWEN3_OMNI_VIDEO_FPS": 1.0,
    "QWEN3_OMNI_VIDEO_MAX_FRAMES": 24,
    "QWEN3_OMNI_VIDEO_MAX_PIXELS": 401408,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def cut_clip(source: Path, start: float, duration: float, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            source.as_posix(),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0 and output.exists() and output.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Split failed meeting-claim windows into shorter retry subwindows.")
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--source-results-root", type=Path, required=True)
    parser.add_argument("--source-prompt-template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subwindow-sec", type=float, default=10.0)
    args = parser.parse_args()

    tasks_by_id = {task["review_task_id"]: task for task in read_jsonl(args.source_queue)}
    retry_tasks: list[dict[str, Any]] = []
    clip_root = args.output_root / "clips"
    for error_path in sorted(args.source_results_root.glob("*.error.json")):
        error = read_json(error_path)
        source_id = str(error.get("review_task_id") or error_path.name.removesuffix(".error.json"))
        source_task = tasks_by_id.get(source_id)
        if not source_task:
            continue
        source_video = Path(source_task["primary_video_file"])
        source_duration = float(source_task["window_local_end_sec"]) - float(source_task["window_local_start_sec"])
        parts = max(1, int((source_duration + args.subwindow_sec - 0.001) // args.subwindow_sec))
        for part_index in range(parts):
            sub_start = part_index * args.subwindow_sec
            sub_end = min(source_duration, sub_start + args.subwindow_sec)
            if sub_end - sub_start < 2.0:
                continue
            task_id = f"{source_id}_retry{part_index:02d}_{int(args.subwindow_sec)}s"
            clip = clip_root / f"{task_id}.mp4"
            if not cut_clip(source_video, sub_start, sub_end - sub_start, clip):
                continue
            phase_local_start = float(source_task["window_local_start_sec"]) + sub_start
            phase_local_end = float(source_task["window_local_start_sec"]) + sub_end
            aligned_start = float(source_task["time_fields"]["aligned_start_sec"])
            task = dict(source_task)
            task.update(
                {
                    "review_task_id": task_id,
                    "task_type": "meeting_claim_grounding_retry_subwindow",
                    "priority": "high",
                    "qwen3_omni_quality_profile": QUALITY_PROFILE,
                    "primary_video_file": clip.as_posix(),
                    "source_failed_task_id": source_id,
                    "source_failed_video_file": source_video.as_posix(),
                    "retry_subwindow_index": part_index,
                    "window_local_start_sec": phase_local_start,
                    "window_local_end_sec": phase_local_end,
                    "window_abs_start_sec": aligned_start + phase_local_start,
                    "window_abs_end_sec": aligned_start + phase_local_end,
                    "acceptance_policy": {
                        "gold_source_after_qwen": "qwen_checked_only",
                        "human_verified_requires_codex_visual_gate": True,
                        "do_not_merge_directly_into_benchmark": True,
                        "reject_if_speaker_or_claim_text_not_grounded": True,
                        "retry_subwindow_requires_extra_review": True,
                    },
                }
            )
            retry_tasks.append(task)

    queue = args.output_root / "meeting_claim_grounding_retry_queue.jsonl"
    prompt = args.output_root / "meeting_claim_grounding_retry_prompt.md"
    write_jsonl(queue, retry_tasks)
    prompt.write_text(args.source_prompt_template.read_text(encoding="utf-8"), encoding="utf-8")
    summary = {
        "ok": True,
        "source_queue": args.source_queue.as_posix(),
        "source_results_root": args.source_results_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "retry_tasks": len(retry_tasks),
        "subwindow_sec": args.subwindow_sec,
        "quality_profile": QUALITY_PROFILE,
        "promotion_allowed_without_codex_human_review": False,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
