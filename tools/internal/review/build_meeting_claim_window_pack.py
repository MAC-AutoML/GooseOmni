from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
QUALITY_PROFILE = {
    "QWEN3_OMNI_MAX_TOKENS": 16384,
    "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": 32768,
    "QWEN3_OMNI_VIDEO_FPS": 1.0,
    "QWEN3_OMNI_VIDEO_MAX_FRAMES": 96,
    "QWEN3_OMNI_VIDEO_MAX_PIXELS": 401408,
}

PROMPT_TEMPLATE = """TASK: gooseomni_meeting_claim_grounding_window_v1

You are re-annotating a short aligned Goose Goose Duck meeting/vote window for a paper-grade GooseOmni Theory-of-Mind benchmark.
Canonical player IDs: Gemini, baile, beigang, mojiang, saoyi, xiaolu.

Goal:
- Extract only speech/chat/vote facts directly grounded in this short video window.
- Produce high-precision speaker/display-name/claim/reaction candidates for later Codex-human verification.
- Prefer "unknown" over guessing.

Rules:
- TASK_JSON is metadata, not gold.
- The primary video is a short window cut from a larger phase.
- Use local_window_sec for times within this short clip.
- Use phase_local_sec for the original phase-local time.
- Use abs_sec = aligned_start_sec + phase_local_sec.
- The six canonical player IDs are benchmark POV players, not necessarily every in-game display name visible in the lobby.
- If a spoken or visible display name is not one of the six canonical IDs or verified aliases, keep it as display text and set the canonical field to "unknown".
- Do not map 海螺 or 小杨鸭 to saoyi unless this exact clip proves it.
- Do not map non-POV display names such as 紫林, 猫手, 牛六, 温天, 雪豹 to any canonical player unless this exact clip proves the mapping.
- claim_target_players must contain only canonical benchmark IDs from the six-player list. If the target is not one of the six, leave claim_target_players empty and put the raw name in claim_target_display_names.
- Never put "unknown" inside claim_target_players; use an empty list plus claim_target_display_names instead.
- Reject or mark low confidence when chat text, audio, speaker, listener, or vote target is unclear.
- Discussion/roster/speaker-order panels are not vote events.
- Add vote_events only when the actual vote UI or ballot/result display visibly shows a cast vote, skipped vote, or vote result.
- Add reaction_links only when the listener's response is visible/heard in this clip; do not infer a reaction from a low-confidence or non-vote UI.
- For public meeting speech, heard_by should include all six benchmark POV players unless the clip directly proves a player could not hear it.
- You are an automatic assistant, not the final human verifier. Every extracted utterance, vote_event, and reaction_link must set needs_human_review to true.
- Return strict JSON only.

Alias prior, still verify per clip:
- Gemini: GEMINI, Gemini.
- baile: 白乐, 白樂.
- beigang: 北港, 北刚.
- mojiang: 末将, 末將, 墨将, 魔将, 摸奖.
- xiaolu: 小鹿, 小路, 路姐. Medium confidence.
- saoyi: 扫一, 扫姨, 骚艺. Medium confidence.

Required JSON:
{
  "review_task_id": "string",
  "phase_id": "string",
  "window_index": 0,
  "clip_evidence_quality": "high|medium|low|unusable",
  "global_uncertainties": ["string"],
  "utterances": [
    {
      "utterance_id": "string",
      "local_window_start_sec": 0.0,
      "local_window_end_sec": 0.0,
      "phase_local_start_sec": 0.0,
      "phase_local_end_sec": 0.0,
      "abs_start_sec": 0.0,
      "abs_end_sec": 0.0,
      "speaker_display_name": "string|unknown",
      "canonical_speaker": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "speaker_confidence": "high|medium|low|unknown",
      "speaker_evidence_types": ["visible_chat_row", "voice", "nameplate", "vote_ui", "prior_alias", "unclear"],
      "utterance_text": "string",
      "text_confidence": "high|medium|low|unknown",
      "heard_by": ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"],
      "claim_type": "accusation|defense|location|route|sighting|role_claim|vote_suggestion|question|agreement|contradiction|other|none",
      "claim_target_players": ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"],
      "claim_target_display_names": ["string"],
      "claim_time_referred": "current_meeting|previous_round|recent_before_report|unspecified|other",
      "strategic_function": "accuse|defend_self|defend_other|coordinate_vote|ask_for_information|answer_question|social_chatter|unknown",
      "tom_relevance": "high|medium|low",
      "evidence_text": "string",
      "needs_human_review": true
    }
  ],
  "vote_events": [
    {
      "local_window_sec": 0.0,
      "phase_local_sec": 0.0,
      "abs_sec": 0.0,
      "canonical_voter": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "vote_target_display_name": "string|skip|unknown",
      "canonical_vote_target": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|skip|unknown",
      "grounding_confidence": "high|medium|low|unknown",
      "evidence_text": "string",
      "needs_human_review": true
    }
  ],
  "reaction_links": [
    {
      "source_utterance_id": "string",
      "listener": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "reaction_utterance_id": "string|null",
      "reaction_vote_event_index": 0,
      "relation": "responds_to|contradicts|defends_against|follows_vote_suggestion|ignores_or_no_visible_response|uncertain",
      "confidence": "high|medium|low|unknown",
      "d_probe_candidate": true,
      "evidence_text": "string",
      "needs_human_review": true
    }
  ]
}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def phase_type_from_id(phase_id: str) -> str:
    if "final" in phase_id:
        return "vote_result"
    if "meeting" in phase_id:
        return "meeting"
    return "unknown"


def pick_primary_player(available: set[str], preferred: list[str]) -> str:
    for player in preferred:
        if player in available:
            return player
    return sorted(available)[0]


def load_meeting_phases(release_root: Path, primary_players: list[str]) -> list[dict[str, Any]]:
    manifest = read_jsonl(release_root / "inputs" / "manifest.jsonl")
    by_phase: dict[str, dict[str, Any]] = {}
    for row in manifest:
        phase_id = str(row.get("phase_id"))
        phase_type = str(row.get("phase_type") or phase_type_from_id(phase_id))
        if phase_type not in {"meeting", "final", "vote_result"} and "meeting" not in phase_id:
            continue
        player = str(row.get("player_id"))
        if player not in PLAYERS:
            continue
        phase = by_phase.setdefault(phase_id, {"phase_id": phase_id, "players": {}, "metadata": {}})
        phase["players"][player] = True
        phase["metadata"][player] = row

    phases = []
    for phase_id in sorted(by_phase, key=lambda value: int(value.split("_")[2])):
        phase = by_phase[phase_id]
        available = set(phase["players"])
        primary = pick_primary_player(available, primary_players)
        meta = phase["metadata"].get(primary) or next(iter(phase["metadata"].values()))
        video = release_root / "inputs" / "videos" / "g001" / phase_id / f"{primary}.mp4"
        if not video.exists():
            continue
        phases.append(
            {
                "phase_id": phase_id,
                "phase_type": phase_type_from_id(phase_id),
                "primary_player": primary,
                "source_video": video,
                "metadata": meta,
            }
        )
    return phases


def cut_window(source: Path, local_start: float, duration: float, output: Path, reencode: bool) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-i", source.as_posix(), "-t", f"{duration:.3f}"]
    if reencode:
        cmd.extend(["-c:v", "mpeg4", "-q:v", "4", "-c:a", "aac", "-movflags", "+faststart", output.as_posix()])
    else:
        cmd.extend(["-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-movflags", "+faststart", output.as_posix()])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "ok": proc.returncode == 0 and output.exists() and output.stat().st_size > 0,
        "source": source.as_posix(),
        "local_start_sec": local_start,
        "duration_sec": duration,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def build_tasks(
    release_root: Path,
    output_root: Path,
    window_sec: float,
    overlap_sec: float,
    limit: int | None,
    primary_players: list[str],
    reencode: bool,
) -> list[dict[str, Any]]:
    phases = load_meeting_phases(release_root, primary_players)
    tasks: list[dict[str, Any]] = []
    clip_root = output_root / "clips"
    step = max(1.0, window_sec - overlap_sec)
    for phase in phases:
        meta = phase["metadata"]
        duration = float(meta.get("duration_sec") or 0.0)
        aligned_start = float(meta.get("aligned_start_sec") or 0.0)
        windows = max(1, math.ceil(max(0.0, duration - overlap_sec) / step))
        for window_index in range(windows):
            local_start = min(window_index * step, max(0.0, duration - 1.0))
            local_end = min(duration, local_start + window_sec)
            if local_end - local_start < 2.0:
                continue
            task_id = f"mcgw_{len(tasks) + 1:05d}_{phase['phase_id']}_w{window_index:03d}_{phase['primary_player']}"
            clip = cut_window(
                phase["source_video"],
                local_start,
                local_end - local_start,
                clip_root / f"{task_id}.mp4",
                reencode,
            )
            if not clip["ok"]:
                continue
            tasks.append(
                {
                    "review_task_id": task_id,
                    "task_type": "meeting_claim_grounding_window_reannotation",
                    "priority": "high" if phase["phase_type"] == "meeting" else "medium",
                    "qwen3_omni_quality_profile": QUALITY_PROFILE,
                    "primary_video_file": clip["output"],
                    "context_video_files": [],
                    "game_id": "g001",
                    "phase_id": phase["phase_id"],
                    "phase_type": phase["phase_type"],
                    "primary_player": phase["primary_player"],
                    "window_index": window_index,
                    "window_local_start_sec": local_start,
                    "window_local_end_sec": local_end,
                    "window_abs_start_sec": aligned_start + local_start,
                    "window_abs_end_sec": aligned_start + local_end,
                    "source_phase_video": phase["source_video"].as_posix(),
                    "time_fields": {
                        "aligned_start_sec": aligned_start,
                        "aligned_end_sec": meta.get("aligned_end_sec"),
                        "phase_duration_sec": duration,
                        "abs_sec_formula": "abs_sec = aligned_start_sec + phase_local_sec",
                    },
                    "acceptance_policy": {
                        "gold_source_after_qwen": "qwen_checked_only",
                        "human_verified_requires_codex_visual_gate": True,
                        "do_not_merge_directly_into_benchmark": True,
                        "reject_if_speaker_or_claim_text_not_grounded": True,
                    },
                }
            )
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build short-window Qwen3-Omni meeting claim grounding pack.")
    parser.add_argument("--release-root", type=Path, default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--overlap-sec", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--primary-player", action="append", default=["Gemini", "baile", "beigang"])
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        args.release_root,
        args.output_root,
        args.window_sec,
        args.overlap_sec,
        args.limit,
        args.primary_player,
        args.reencode,
    )
    queue = args.output_root / "meeting_claim_grounding_window_queue.jsonl"
    prompt = args.output_root / "meeting_claim_grounding_window_prompt.md"
    write_jsonl(queue, tasks)
    prompt.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    summary = {
        "ok": True,
        "release_root": args.release_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "queue": queue.as_posix(),
        "prompt_template": prompt.as_posix(),
        "tasks": len(tasks),
        "window_sec": args.window_sec,
        "overlap_sec": args.overlap_sec,
        "tasks_with_primary_video": sum(1 for task in tasks if Path(task["primary_video_file"]).exists()),
        "quality_profile": QUALITY_PROFILE,
        "note": "Window-level Qwen outputs are qwen_checked candidates only; Codex-human gate is required before human_verified benchmark merge.",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
