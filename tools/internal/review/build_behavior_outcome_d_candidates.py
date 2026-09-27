from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
STRATEGIC_CLAIM_TYPES = {
    "accusation",
    "defense",
    "location",
    "sighting",
    "vote_suggestion",
    "other",
}
MIN_CLAIM_LEN = 8
MIN_GAP_SEC = 1.0
MAX_GAP_SEC = 240.0


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def vote_target_from_description(description: str) -> tuple[str | None, str | None]:
    if "投票" not in description or "选择了" not in description:
        return None, None
    voter = next(
        (player for player in PLAYERS if f"{player} 在投票" in description), None
    )
    if not voter:
        return None, None
    target = description.split("选择了", 1)[1].strip().strip("。").strip()
    target = target.strip("\"'“”‘’ ")
    return voter, target or None


def clip_bounds(path: Path, player: str) -> tuple[float, float] | None:
    match = re.search(rf"g001_{re.escape(player)}_(\d+)_(\d+)\.mp4$", path.name)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def find_clip(clip_root: Path, player: str, abs_sec: float) -> Path | None:
    player_dir = clip_root / player
    if not player_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in player_dir.glob(f"g001_{player}_*.mp4"):
        bounds = clip_bounds(path, player)
        if not bounds:
            continue
        start, end = bounds
        if start <= abs_sec <= end:
            candidates.append((end - start, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def extract_frame(video: Path, abs_sec: float, output: Path) -> dict[str, Any]:
    bounds = clip_bounds(video, video.parent.name)
    if not bounds:
        return {
            "ok": False,
            "reason": "cannot_parse_clip_bounds",
            "video": video.as_posix(),
        }
    start, _ = bounds
    local_sec = max(0.0, abs_sec - start)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{local_sec:.3f}",
        "-i",
        video.as_posix(),
        "-frames:v",
        "1",
        output.as_posix(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0 and output.exists(),
        "video": video.as_posix(),
        "abs_sec": abs_sec,
        "local_sec": local_sec,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def release_video_path(release_video_root: Path, phase_id: str, player: str) -> Path:
    return release_video_root / phase_id / f"{player}.mp4"


def extract_release_frame(
    release_video_root: Path, phase_id: str, player: str, abs_sec: float, output: Path
) -> dict[str, Any]:
    video = release_video_path(release_video_root, phase_id, player)
    if not video.exists():
        return {
            "ok": False,
            "reason": "release_video_missing",
            "video": video.as_posix(),
            "phase_id": phase_id,
            "player": player,
            "abs_sec": abs_sec,
        }
    local_sec = max(0.0, abs_sec - phase_start(phase_id))
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{local_sec:.3f}",
        "-i",
        video.as_posix(),
        "-frames:v",
        "1",
        output.as_posix(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0 and output.exists(),
        "video": video.as_posix(),
        "phase_id": phase_id,
        "player": player,
        "abs_sec": abs_sec,
        "local_sec": local_sec,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def candidate_key(row: dict[str, Any]) -> tuple[float, str, str, str]:
    return (
        float(row["gap_sec"]),
        row["claim_id"],
        row["vote_event_id"],
        row["listener"],
    )


def first_phase_id(row: dict[str, Any]) -> str | None:
    phase_ids = row.get("source_segment_ids") or []
    return str(phase_ids[0]) if phase_ids else None


def build_candidates(
    ledger_root: Path,
    clip_root: Path | None,
    release_video_root: Path | None,
    output_root: Path,
    limit: int,
) -> list[dict[str, Any]]:
    claims = read_jsonl(ledger_root / "claims.jsonl")
    events = read_jsonl(ledger_root / "world_events.jsonl")
    candidates: list[dict[str, Any]] = []
    for event in events:
        voter, vote_target_text = vote_target_from_description(
            str(event.get("description", ""))
        )
        if not voter or not vote_target_text:
            continue
        vote_abs_start = float(event.get("abs_start_sec", 0.0))
        vote_abs_end = float(event.get("abs_end_sec", vote_abs_start))
        for claim in claims:
            speaker = claim.get("speaker")
            if speaker not in PLAYERS or speaker == voter:
                continue
            if voter not in claim.get("heard_by", []):
                continue
            if claim.get("claim_type") not in STRATEGIC_CLAIM_TYPES:
                continue
            claim_text = str(claim.get("content", ""))
            if len(claim_text) < MIN_CLAIM_LEN:
                continue
            claim_end = float(claim.get("abs_end_sec", 0.0))
            gap = vote_abs_start - claim_end
            if not (MIN_GAP_SEC <= gap <= MAX_GAP_SEC):
                continue
            vote_source = next(
                (p for p in event.get("source_povs", []) if p in PLAYERS), voter
            )
            claim_frame_player = voter
            vote_phase_id = first_phase_id(event)
            claim_phase_id = first_phase_id(claim)
            vote_clip = (
                find_clip(clip_root, vote_source, vote_abs_end) if clip_root else None
            )
            claim_clip = (
                find_clip(clip_root, claim_frame_player, claim_end)
                if clip_root
                else None
            )
            row_id = f"dbo_{len(candidates) + 1:06d}_{claim['claim_id']}_{event['world_event_id']}_{voter}"
            row = {
                "candidate_id": row_id,
                "game_id": event.get("game_id", "g001"),
                "template": "behavior_outcome_vote_influence",
                "speaker": speaker,
                "listener": voter,
                "claim_id": claim["claim_id"],
                "claim_type": claim.get("claim_type"),
                "claim_abs_start_sec": claim.get("abs_start_sec"),
                "claim_abs_end_sec": claim.get("abs_end_sec"),
                "claim_text": claim_text,
                "vote_event_id": event["world_event_id"],
                "vote_abs_start_sec": vote_abs_start,
                "vote_abs_end_sec": vote_abs_end,
                "vote_description": event.get("description", ""),
                "vote_target_text": vote_target_text,
                "canonical_vote_target_player": None,
                "gap_sec": gap,
                "selection_reason": "Strategic claim is heard by listener before an explicit vote-selection event.",
                "freeze_policy": {
                    "expected_listener_next_action": "vote",
                    "freeze_vote_target_text": True,
                    "freeze_canonical_target_player": False,
                },
                "review_status": "needs_human_review",
                "review_reasons": [],
                "claim_frame_player": claim_frame_player,
                "vote_frame_player": vote_source,
                "claim_phase_id": claim_phase_id,
                "vote_phase_id": vote_phase_id,
                "claim_clip": claim_clip.as_posix() if claim_clip else None,
                "vote_clip": vote_clip.as_posix() if vote_clip else None,
            }
            candidates.append(row)
    selected = sorted(candidates, key=candidate_key)[:limit]
    assets = output_root / "review_assets"
    for row in selected:
        if release_video_root and row.get("claim_phase_id"):
            result = extract_release_frame(
                release_video_root,
                str(row["claim_phase_id"]),
                str(row["claim_frame_player"]),
                float(row["claim_abs_end_sec"]),
                assets / f"{row['candidate_id']}_claim_frame.jpg",
            )
            row["claim_frame"] = result
            row["frame_source"] = "release_aligned_phase_video"
        elif row.get("claim_clip"):
            result = extract_frame(
                Path(row["claim_clip"]),
                float(row["claim_abs_end_sec"]),
                assets / f"{row['candidate_id']}_claim_frame.jpg",
            )
            row["claim_frame"] = result
            row["frame_source"] = "archive_abs_clip"
        if release_video_root and row.get("vote_phase_id"):
            result = extract_release_frame(
                release_video_root,
                str(row["vote_phase_id"]),
                str(row["vote_frame_player"]),
                float(row["vote_abs_end_sec"]),
                assets / f"{row['candidate_id']}_vote_frame.jpg",
            )
            row["vote_frame"] = result
            row["frame_source"] = "release_aligned_phase_video"
        elif row.get("vote_clip"):
            result = extract_frame(
                Path(row["vote_clip"]),
                float(row["vote_abs_end_sec"]),
                assets / f"{row['candidate_id']}_vote_frame.jpg",
            )
            row["vote_frame"] = result
            row["frame_source"] = "archive_abs_clip"
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict D behavior-outcome candidates from explicit vote evidence."
    )
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, default=None)
    parser.add_argument("--release-video-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clip_root and not args.release_video_root:
        raise SystemExit("one of --clip-root or --release-video-root is required")
    rows = build_candidates(
        args.ledger_root,
        args.clip_root,
        args.release_video_root,
        args.output_root,
        args.limit,
    )
    write_jsonl(args.output_root / "behavior_outcome_d_candidates.jsonl", rows)
    summary = {
        "ok": True,
        "output_root": args.output_root.as_posix(),
        "candidates": len(rows),
        "with_claim_frame": sum(
            1 for row in rows if row.get("claim_frame", {}).get("ok")
        ),
        "with_vote_frame": sum(
            1 for row in rows if row.get("vote_frame", {}).get("ok")
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
