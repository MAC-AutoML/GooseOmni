from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
QUALITY_PROFILE = {
    "QWEN3_OMNI_MAX_TOKENS": 16384,
    "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": 32768,
    "QWEN3_OMNI_VIDEO_FPS": 1.0,
    "QWEN3_OMNI_VIDEO_MAX_FRAMES": 128,
    "QWEN3_OMNI_VIDEO_MAX_PIXELS": 401408,
}

PROMPT_TEMPLATE = """TASK: decrypto_clip_human_gold_review

You are reviewing short aligned Goose Goose Duck POV clips for a GooseOmni Theory-of-Mind benchmark.
The player names are exactly: Gemini, baile, beigang, mojiang, saoyi, xiaolu.

Review standard:
- Promote only when the clip evidence is strong enough for paper/leaderboard human-gold use.
- Treat TASK_JSON.candidate as a hypothesis to verify, not as evidence. If the video contradicts the candidate text, reject.
- Do not infer a hidden fact from another POV when judging the target player's perspective.
- Reject if speaker identity, spoken claim text, event grounding, local visibility, or behavior outcome is not directly supported.
- Reject UI-only, lobby, role/result, task-overlay, blank, or wrong-phase evidence.
- The primary video may be a review_sequence made by concatenating task clips. Use TASK_JSON.clip_evidence to interpret the order and player/phase for each segment.
- For kill/death/body/report events, explicitly read visible animation labels, nameplates, report banners, vote UI, and subtitles. Reject if those labels do not match the candidate actor/victim/claim identity.
- For short or vague claims such as "killed fast", "someone died", or "who killed X", reject unless the same clip directly grounds the exact actor, victim, and event being claimed.
- In evidence_summary, mention the exact visual/audio labels that support acceptance; if labels are unreadable, put the uncertainty in remaining_uncertainties and do not accept.
- Return strict JSON only.

Required JSON fields:
{
  "review_task_id": "string",
  "decision": "accept|reject|human_review_required",
  "gold_source_after_review": "human_verified|qwen_checked|qwen_weak",
  "speaker_verified": true,
  "claim_text_verified": true,
  "anchor_event_verified": true,
  "target_visibility_verified": true,
  "outcome_verified": true,
  "same_topic_chain_verified": true,
  "perspective_leakage_safe": true,
  "corrected_claim_text": "string|null",
  "corrected_speaker": "string|null",
  "corrected_outcome": "string|null",
  "acceptance_scope": "claim_truth|behavior_outcome_d|both|none",
  "evidence_summary": ["short direct evidence notes"],
  "reject_reasons": ["short reasons if rejected"],
  "remaining_uncertainties": ["uncertainties"],
  "needs_human_review": true
}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["rows", "items", "candidates", "manifest"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported manifest shape: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def phase_end(phase_id: str) -> float | None:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(2))) if match else None


def abs_mid(bounds: Any) -> float:
    if isinstance(bounds, list | tuple) and len(bounds) >= 2:
        return (float(bounds[0]) + float(bounds[1])) / 2.0
    return float(bounds)


def release_video(video_root: Path, phase_id: str, player: str) -> Path:
    return video_root / phase_id / f"{player}.mp4"


def clip_window(phase_id: str, abs_sec: float, pre_sec: float, post_sec: float) -> tuple[float, float]:
    start_abs = max(phase_start(phase_id), abs_sec - pre_sec)
    end_limit = phase_end(phase_id)
    end_abs = abs_sec + post_sec
    if end_limit is not None:
        end_abs = min(end_abs, end_limit)
    local_start = max(0.0, start_abs - phase_start(phase_id))
    duration = max(0.5, end_abs - start_abs)
    return local_start, duration


def extract_clip(
    video_root: Path,
    phase_id: str | None,
    player: str | None,
    abs_sec: float | None,
    output: Path,
    pre_sec: float,
    post_sec: float,
) -> dict[str, Any]:
    if not phase_id or player not in PLAYERS or abs_sec is None:
        return {"ok": False, "reason": "missing_phase_player_or_time", "phase_id": phase_id, "player": player}
    video = release_video(video_root, phase_id, player)
    if not video.exists():
        return {"ok": False, "reason": "video_missing", "video": video.as_posix(), "phase_id": phase_id, "player": player}
    local_start, duration = clip_window(phase_id, abs_sec, pre_sec, post_sec)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{local_start:.3f}",
        "-i",
        video.as_posix(),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output.as_posix(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0 and output.exists() and output.stat().st_size > 0,
        "video": video.as_posix(),
        "phase_id": phase_id,
        "player": player,
        "abs_sec": abs_sec,
        "local_start_sec": local_start,
        "duration_sec": duration,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def first_ok_clip(clips: list[dict[str, Any]], fallback: Path | None = None) -> str | None:
    for clip in clips:
        if clip.get("ok") and clip.get("output"):
            return str(clip["output"])
    return fallback.as_posix() if fallback else None


def context_files(clips: list[dict[str, Any]], primary: str | None) -> list[str]:
    values = []
    for clip in clips:
        output = clip.get("output")
        if clip.get("ok") and output and output != primary:
            values.append(str(output))
    return list(dict.fromkeys(values))


def concat_review_sequence(clips: list[dict[str, Any]], output: Path, *, reencode: bool = False) -> dict[str, Any]:
    inputs = [Path(str(clip["output"])) for clip in clips if clip.get("ok") and clip.get("output")]
    if not inputs:
        return {"ok": False, "reason": "no_input_clips"}
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.with_suffix(".txt")
    list_path.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in inputs), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path.as_posix()]
    if reencode:
        cmd.extend(["-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-movflags", "+faststart", output.as_posix()])
    else:
        cmd.extend(["-c", "copy", "-movflags", "+faststart", output.as_posix()])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0 and output.exists() and output.stat().st_size > 0,
        "inputs": [path.as_posix() for path in inputs],
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-8:],
    }


def build_claim_truth_task(
    row: dict[str, Any],
    idx: int,
    video_root: Path,
    clip_root: Path,
    pre_sec: float,
    post_sec: float,
    reencode_sequence: bool,
) -> dict[str, Any]:
    task_id = f"clip_claim_truth_{idx:04d}_{row['event_id']}_{row['claim_id']}_{row['target']}"
    prefix = clip_root / task_id
    event_abs = abs_mid(row["event_abs"])
    claim_abs = abs_mid(row["claim_abs"])
    clips = [
        extract_clip(video_root, row.get("event_phase"), row.get("source"), event_abs, prefix / "anchor_source.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("event_phase"), row.get("target"), event_abs, prefix / "anchor_target.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("claim_phase"), row.get("speaker"), claim_abs, prefix / "claim_speaker.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("claim_phase"), row.get("target"), claim_abs, prefix / "claim_target.mp4", pre_sec, post_sec),
    ]
    sequence = concat_review_sequence(clips, prefix / "review_sequence.mp4", reencode=reencode_sequence)
    primary = str(sequence["output"]) if sequence.get("ok") else first_ok_clip(clips)
    return {
        "review_task_id": task_id,
        "task_type": "claim_truth_clip_review",
        "priority": "high",
        "qwen3_omni_quality_profile": QUALITY_PROFILE,
        "primary_video_file": primary,
        "context_video_files": context_files(clips, primary),
        "clip_evidence": {
            "anchor_source_clip": clips[0],
            "anchor_target_clip": clips[1],
            "claim_speaker_clip": clips[2],
            "claim_target_clip": clips[3],
            "review_sequence_clip": sequence,
        },
        "candidate": row,
        "review_questions": [
            "Does the source clip directly ground the anchor event?",
            "Does the target clip show the target did not see/hear/publicly know the anchor event at that time?",
            "Does the speaker clip verify the speaker and claim text?",
            "Is the global claim truth label supported by clip evidence without leaking hidden POV information into the target perspective?",
        ],
        "required_acceptance": {
            "speaker_verified": True,
            "claim_text_verified": True,
            "anchor_event_verified": True,
            "target_visibility_verified": True,
            "perspective_leakage_safe": True,
        },
    }


def build_behavior_task(
    row: dict[str, Any],
    idx: int,
    video_root: Path,
    clip_root: Path,
    pre_sec: float,
    post_sec: float,
    reencode_sequence: bool,
) -> dict[str, Any]:
    task_id = f"clip_behavior_d_{idx:04d}_{row['claim_id']}_{row['vote_event_id']}_{row['listener']}"
    prefix = clip_root / task_id
    claim_abs = abs_mid([row["claim_abs_start_sec"], row["claim_abs_end_sec"]])
    vote_abs = abs_mid([row["vote_abs_start_sec"], row["vote_abs_end_sec"]])
    vote_source = row.get("vote_frame_player") if row.get("vote_frame_player") in PLAYERS else row.get("listener")
    clips = [
        extract_clip(video_root, row.get("claim_phase_id"), row.get("speaker"), claim_abs, prefix / "claim_speaker.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("claim_phase_id"), row.get("listener"), claim_abs, prefix / "claim_listener.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("vote_phase_id"), row.get("listener"), vote_abs, prefix / "vote_listener.mp4", pre_sec, post_sec),
        extract_clip(video_root, row.get("vote_phase_id"), vote_source, vote_abs, prefix / "vote_source.mp4", pre_sec, post_sec),
    ]
    sequence = concat_review_sequence(clips, prefix / "review_sequence.mp4", reencode=reencode_sequence)
    primary = str(sequence["output"]) if sequence.get("ok") else first_ok_clip(clips)
    return {
        "review_task_id": task_id,
        "task_type": "behavior_outcome_d_clip_review",
        "priority": "high",
        "qwen3_omni_quality_profile": QUALITY_PROFILE,
        "primary_video_file": primary,
        "context_video_files": context_files(clips, primary),
        "clip_evidence": {
            "claim_speaker_clip": clips[0],
            "claim_listener_clip": clips[1],
            "vote_listener_clip": clips[2],
            "vote_source_clip": clips[3],
            "review_sequence_clip": sequence,
        },
        "candidate": row,
        "review_questions": [
            "Does the claim speaker clip verify speaker identity and strategic claim text?",
            "Does the listener clip verify the listener could hear or observe the claim?",
            "Does the vote/outcome clip verify the later listener action?",
            "Is the claim-to-outcome chain same-topic and strong enough for a D perspective-taking probe?",
        ],
        "required_acceptance": {
            "speaker_verified": True,
            "claim_text_verified": True,
            "outcome_verified": True,
            "same_topic_chain_verified": True,
            "perspective_leakage_safe": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short clip-level Qwen3-Omni review packs for thin Decrypto benchmark categories.")
    parser.add_argument("--mode", choices=["claim_truth", "behavior_outcome_d"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pre-sec", type=float, default=5.0)
    parser.add_argument("--post-sec", type=float, default=7.0)
    parser.add_argument("--reencode-sequence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)[: args.limit]
    clip_root = args.output_root / "clips"
    tasks = []
    for idx, row in enumerate(rows, start=1):
        if args.mode == "claim_truth":
            tasks.append(
                build_claim_truth_task(row, idx, args.video_root, clip_root, args.pre_sec, args.post_sec, args.reencode_sequence)
            )
        else:
            tasks.append(
                build_behavior_task(row, idx, args.video_root, clip_root, args.pre_sec, args.post_sec, args.reencode_sequence)
            )
    queue = args.output_root / "clip_review_queue.jsonl"
    write_jsonl(queue, tasks)
    (args.output_root / "qwen3_omni_clip_review_prompt.md").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    summary = {
        "ok": True,
        "mode": args.mode,
        "input": args.input.as_posix(),
        "output_root": args.output_root.as_posix(),
        "queue": queue.as_posix(),
        "tasks": len(tasks),
        "tasks_with_primary_video": sum(1 for task in tasks if task.get("primary_video_file")),
        "clip_count": sum(len(task.get("context_video_files", [])) + int(bool(task.get("primary_video_file"))) for task in tasks),
        "quality_profile": QUALITY_PROFILE,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
