from __future__ import annotations

import argparse
import json
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

PROMPT_TEMPLATE = """TASK: gooseomni_meeting_claim_grounding_v1

You are re-annotating aligned Goose Goose Duck meeting/vote clips for a paper-grade GooseOmni Theory-of-Mind benchmark.
The canonical player IDs are exactly: Gemini, baile, beigang, mojiang, saoyi, xiaolu.

Purpose:
- Build a reliable meeting/vote utterance and claim ledger for later Decrypto-style ToM probe generation.
- The downstream benchmark needs speaker identity, display-name grounding, claim content, listener availability, reactions, and vote/behavior outcomes.
- Do not optimize for quantity. Prefer fewer high-confidence rows with explicit evidence.

Critical rules:
- Treat TASK_JSON as metadata, not as gold.
- The video is the evidence. If speaker, visible chat text, audio, or vote target is unclear, mark it uncertain.
- Do not invent speech. Use concise paraphrase only when exact wording is not visually/audibly recoverable.
- Distinguish display name from canonical player ID. Only map to a canonical player when the alias/display evidence is strong.
- Do not map 海螺 or 小杨鸭 to saoyi unless this exact clip proves it.
- If a visible chat row does not match the hypothesized speaker, keep canonical_speaker as unknown.
- For every utterance/claim, include local and absolute time estimates. Time rule: abs_sec = aligned_start_sec + local_sec.
- For perspective-taking D probes, the key evidence is: speaker's claim, who could hear it, listener's information state if visible, and later listener response/vote.
- Return strict JSON only.

Use this alias prior cautiously:
- Gemini: GEMINI, Gemini. High confidence.
- baile: 白乐, 白樂. Medium-high confidence; still verify each candidate.
- beigang: 北港, 北刚. High confidence.
- mojiang: 末将, 末將, 墨将, 魔将, 摸奖. High confidence.
- xiaolu: 小鹿, 小路, 路姐. Medium confidence; verify visually per clip.
- saoyi: 扫一, 扫姨, 骚艺. Medium confidence; do not infer 海螺/小杨鸭 automatically.

Required JSON schema:
{
  "review_task_id": "string",
  "phase_id": "string",
  "phase_type": "meeting|voting|meeting_discussion|vote_result|unknown",
  "primary_player": "Gemini|baile|beigang|mojiang|saoyi|xiaolu",
  "clip_evidence_quality": "high|medium|low|unusable",
  "global_uncertainties": ["string"],
  "utterances": [
    {
      "utterance_id": "string",
      "local_start_sec": 0.0,
      "local_end_sec": 0.0,
      "abs_start_sec": 0.0,
      "abs_end_sec": 0.0,
      "speaker_display_name": "string|unknown",
      "canonical_speaker": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "speaker_grounding": {
        "confidence": "high|medium|low|unknown",
        "evidence_types": ["visible_chat_row", "voice", "nameplate", "vote_ui", "prior_alias", "unclear"],
        "evidence_text": "string"
      },
      "utterance_text": "string",
      "exact_text_visible_or_audible": true,
      "heard_by": ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"],
      "claim_type": "accusation|defense|location|route|sighting|role_claim|vote_suggestion|question|agreement|contradiction|other|none",
      "claim_target_players": ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"],
      "claim_time_referred": "current_meeting|previous_round|recent_before_report|unspecified|other",
      "strategic_function": "accuse|defend_self|defend_other|coordinate_vote|withhold_information|ask_for_information|answer_question|social_chatter|unknown",
      "tom_relevance": "high|medium|low",
      "needs_human_review": true,
      "review_reasons": ["string"]
    }
  ],
  "vote_events": [
    {
      "local_sec": 0.0,
      "abs_sec": 0.0,
      "voter_display_name": "string|unknown",
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
      "evidence_text": "string",
      "d_probe_candidate": true,
      "needs_human_review": true
    }
  ],
  "candidate_probe_notes": [
    {
      "query_variable": "claim_truth_vs_claim_awareness|trust_update|next_action_prediction|perspective_taking|none",
      "speaker": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "listener": "Gemini|baile|beigang|mojiang|saoyi|xiaolu|unknown",
      "why_useful": "string",
      "risk": "low|medium|high"
    }
  ]
}
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_manifest(release_root: Path) -> list[dict[str, Any]]:
    manifest = release_root / "inputs" / "manifest.jsonl"
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def phase_sort_key(phase_id: str) -> tuple[int, str]:
    parts = phase_id.split("_")
    for part in parts:
        if part.isdigit():
            return int(part), phase_id
    return 999999, phase_id


def phase_type_from_id(phase_id: str) -> str:
    if "meeting" in phase_id:
        return "meeting"
    if "vote" in phase_id:
        return "voting"
    if "final" in phase_id:
        return "vote_result"
    return "unknown"


def normalize_phase_type(phase_id: str, phase_type: str) -> str:
    if phase_type == "final" or "final" in phase_id:
        return "vote_result"
    if phase_type == "meeting" or "meeting" in phase_id:
        return "meeting"
    if phase_type == "vote" or phase_type == "voting" or "vote" in phase_id:
        return "voting"
    return phase_type or "unknown"


def pick_primary_player(available: set[str], preferred: list[str]) -> str:
    for player in preferred:
        if player in available:
            return player
    return sorted(available)[0]


def build_tasks(
    release_root: Path, limit: int | None, primary_players: list[str]
) -> list[dict[str, Any]]:
    rows = load_manifest(release_root)
    by_phase: dict[str, dict[str, Any]] = {}
    for row in rows:
        phase_id = str(row.get("phase_id"))
        phase_type = normalize_phase_type(
            phase_id, str(row.get("phase_type") or phase_type_from_id(phase_id))
        )
        if (
            phase_type not in {"meeting", "voting", "vote_result"}
            and "meeting" not in phase_id
        ):
            continue
        player_id = str(row.get("player_id"))
        if player_id not in PLAYERS:
            continue
        entry = by_phase.setdefault(
            phase_id,
            {
                "phase_id": phase_id,
                "phase_type": phase_type,
                "players": {},
                "metadata": {},
            },
        )
        entry["players"][player_id] = row.get("video_file") or row.get("video_path")
        entry["metadata"][player_id] = row

    tasks: list[dict[str, Any]] = []
    for phase_id in sorted(by_phase, key=phase_sort_key):
        entry = by_phase[phase_id]
        available = {player for player, path in entry["players"].items() if path}
        if not available:
            continue
        primary = pick_primary_player(available, primary_players)
        primary_video = (
            release_root / "inputs" / "videos" / "g001" / phase_id / f"{primary}.mp4"
        )
        if not primary_video.exists():
            candidate = entry["players"].get(primary)
            primary_video = (
                release_root / str(candidate) if candidate else primary_video
            )
        meta = entry["metadata"].get(primary) or next(iter(entry["metadata"].values()))
        context_videos = []
        for player in PLAYERS:
            if player == primary or player not in available:
                continue
            path = (
                release_root / "inputs" / "videos" / "g001" / phase_id / f"{player}.mp4"
            )
            if path.exists():
                context_videos.append(path.as_posix())
        tasks.append(
            {
                "review_task_id": f"mcg_{len(tasks) + 1:04d}_{phase_id}_{primary}",
                "task_type": "meeting_claim_grounding_reannotation",
                "priority": "high",
                "qwen3_omni_quality_profile": QUALITY_PROFILE,
                "primary_video_file": primary_video.as_posix(),
                "context_video_files": context_videos,
                "game_id": "g001",
                "phase_id": phase_id,
                "phase_type": entry["phase_type"],
                "primary_player": primary,
                "players": PLAYERS,
                "time_fields": {
                    "aligned_start_sec": meta.get("aligned_start_sec"),
                    "aligned_end_sec": meta.get("aligned_end_sec"),
                    "duration_sec": meta.get("duration_sec"),
                    "abs_sec_formula": meta.get(
                        "abs_sec_formula", "abs_sec = aligned_start_sec + local_sec"
                    ),
                },
                "phase_order": {
                    "episode_id": meta.get("episode_id"),
                    "episode_index": meta.get("episode_index"),
                    "phase_index_global": meta.get("phase_index_global"),
                    "phase_index_in_episode": meta.get("phase_index_in_episode"),
                    "phase_order_label_zh": meta.get("phase_order_label_zh"),
                    "meeting_round_index": meta.get("meeting_round_index"),
                    "gameplay_round_index": meta.get("gameplay_round_index"),
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
            break
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen3-Omni meeting/vote claim-grounding reannotation pack."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--primary-player", action="append", default=["Gemini", "baile", "beigang"]
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.release_root, args.limit, args.primary_player)
    queue = args.output_root / "meeting_claim_grounding_queue.jsonl"
    prompt_path = args.output_root / "meeting_claim_grounding_prompt.md"
    write_jsonl(queue, tasks)
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    summary = {
        "ok": True,
        "release_root": args.release_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "queue": queue.as_posix(),
        "prompt_template": prompt_path.as_posix(),
        "tasks": len(tasks),
        "tasks_with_primary_video": sum(
            1
            for task in tasks
            if task.get("primary_video_file")
            and Path(task["primary_video_file"]).exists()
        ),
        "total_context_videos": sum(
            len(task.get("context_video_files", [])) for task in tasks
        ),
        "quality_profile": QUALITY_PROFILE,
        "note": "Qwen outputs are qwen_checked candidates only; Codex-human visual gate is required before human_verified benchmark merge.",
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
