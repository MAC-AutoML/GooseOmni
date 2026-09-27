from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
PUBLIC_MIN_HEARD = 4
CLAIM_TYPES = {"accusation", "defense", "location", "sighting"}
REACTION_TYPES = {"accusation", "defense", "location", "sighting", "role", "other"}
MIN_GAP_SEC = 8.0
MAX_GAP_SEC = 120.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def clip_bounds(path: Path, player: str) -> tuple[float, float] | None:
    match = re.search(rf"g001_{re.escape(player)}_(\d+)_(\d+)\.mp4$", path.name)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def find_clip(clip_root: Path, player: str, abs_sec: float) -> Path | None:
    player_dir = clip_root / player
    if not player_dir.exists():
        return None
    matches: list[tuple[float, Path]] = []
    for path in player_dir.glob(f"g001_{player}_*.mp4"):
        bounds = clip_bounds(path, player)
        if not bounds:
            continue
        start, end = bounds
        if start <= abs_sec <= end:
            matches.append((end - start, path))
    return sorted(matches, key=lambda item: item[0])[0][1] if matches else None


def extract_frame(video: Path, player: str, abs_sec: float, output: Path) -> dict[str, Any]:
    bounds = clip_bounds(video, player)
    if not bounds:
        return {"ok": False, "reason": "cannot_parse_clip_bounds", "video": video.as_posix()}
    local_sec = max(0.0, abs_sec - bounds[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{local_sec:.3f}", "-i", video.as_posix(), "-frames:v", "1", output.as_posix()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "ok": proc.returncode == 0 and output.exists(),
        "video": video.as_posix(),
        "abs_sec": abs_sec,
        "local_sec": local_sec,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def is_public_claim(claim: dict[str, Any]) -> bool:
    return claim.get("speaker") in PLAYERS and len(claim.get("heard_by", [])) >= PUBLIC_MIN_HEARD


def text_score(text: str) -> int:
    score = 0
    for token in ["不是", "为什么", "你", "他", "她", "谁", "投", "票", "刀", "杀", "看到", "看见", "没有", "没", "好人", "坏", "鸭", "鹅", "骗", "怀疑", "在哪"]:
        if token in text:
            score += 1
    return score


def reaction_action(reaction: dict[str, Any]) -> str:
    claim_type = reaction.get("claim_type")
    text = str(reaction.get("content", ""))
    if claim_type == "defense" or any(token in text for token in ["不是我", "我没", "我没有", "我一直", "我也在"]):
        return "defend"
    if claim_type == "accusation" or any(token in text for token in ["刀", "杀", "投", "票", "怀疑", "骗"]):
        return "accuse"
    return "respond"


def build_candidates(ledger_root: Path, clip_root: Path, output_root: Path, limit: int) -> list[dict[str, Any]]:
    claims = read_jsonl(ledger_root / "claims.jsonl")
    candidates: list[dict[str, Any]] = []
    for claim in claims:
        speaker = claim.get("speaker")
        if not is_public_claim(claim) or claim.get("claim_type") not in CLAIM_TYPES:
            continue
        claim_text = str(claim.get("content", ""))
        if len(claim_text) < 10 or text_score(claim_text) < 2:
            continue
        claim_end = float(claim.get("abs_end_sec", 0.0))
        for reaction in claims:
            listener = reaction.get("speaker")
            if listener not in PLAYERS or listener == speaker:
                continue
            if listener not in claim.get("heard_by", []):
                continue
            if reaction.get("claim_type") not in REACTION_TYPES:
                continue
            reaction_text = str(reaction.get("content", ""))
            if len(reaction_text) < 8 or text_score(reaction_text) < 3:
                continue
            reaction_start = float(reaction.get("abs_start_sec", 0.0))
            gap = reaction_start - claim_end
            if not (MIN_GAP_SEC <= gap <= MAX_GAP_SEC):
                continue
            same_phase = bool(set(claim.get("source_segment_ids", [])) & set(reaction.get("source_segment_ids", [])))
            if not same_phase and gap > 80:
                continue
            candidate_id = f"dur_{len(candidates) + 1:06d}_{claim['claim_id']}_{reaction['claim_id']}_{listener}"
            claim_frame_player = listener
            reaction_frame_player = next((p for p in reaction.get("heard_by", []) if p in PLAYERS and p != listener), listener)
            claim_clip = find_clip(clip_root, claim_frame_player, claim_end)
            reaction_clip = find_clip(clip_root, reaction_frame_player, float(reaction.get("abs_end_sec", reaction_start)))
            row = {
                "candidate_id": candidate_id,
                "game_id": claim.get("game_id", "g001"),
                "template": "utterance_reaction_perspective_taking",
                "speaker": speaker,
                "listener": listener,
                "claim_id": claim["claim_id"],
                "claim_type": claim.get("claim_type"),
                "claim_abs_start_sec": claim.get("abs_start_sec"),
                "claim_abs_end_sec": claim.get("abs_end_sec"),
                "claim_text": claim_text,
                "reaction_claim_id": reaction["claim_id"],
                "reaction_type": reaction.get("claim_type"),
                "reaction_abs_start_sec": reaction.get("abs_start_sec"),
                "reaction_abs_end_sec": reaction.get("abs_end_sec"),
                "reaction_text": reaction_text,
                "expected_listener_next_action": reaction_action(reaction),
                "behavior_outcome_evidence_type": "later_utterance",
                "gap_sec": gap,
                "same_source_segment": same_phase,
                "cue_score": text_score(claim_text) + text_score(reaction_text),
                "gold_freeze_policy": {
                    "freeze_next_action": True,
                    "freeze_trust_update": False,
                    "freeze_reaction_text": True,
                },
                "review_status": "needs_human_review",
                "claim_frame_player": claim_frame_player,
                "reaction_frame_player": reaction_frame_player,
                "claim_clip": claim_clip.as_posix() if claim_clip else None,
                "reaction_clip": reaction_clip.as_posix() if reaction_clip else None,
            }
            candidates.append(row)
    selected = sorted(candidates, key=lambda row: (-int(row["same_source_segment"]), row["gap_sec"], -row["cue_score"]))[:limit]
    assets = output_root / "review_assets"
    for row in selected:
        if row.get("claim_clip"):
            row["claim_frame"] = extract_frame(
                Path(row["claim_clip"]),
                row["claim_frame_player"],
                float(row["claim_abs_end_sec"]),
                assets / f"{row['candidate_id']}_claim_frame.jpg",
            )
        if row.get("reaction_clip"):
            row["reaction_frame"] = extract_frame(
                Path(row["reaction_clip"]),
                row["reaction_frame_player"],
                float(row["reaction_abs_end_sec"]),
                assets / f"{row['candidate_id']}_reaction_frame.jpg",
            )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D perspective-taking behavior candidates from later listener utterances.")
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_candidates(args.ledger_root, args.clip_root, args.output_root, args.limit)
    write_jsonl(args.output_root / "utterance_reaction_d_candidates.jsonl", rows)
    summary = {
        "ok": True,
        "output_root": args.output_root.as_posix(),
        "candidates": len(rows),
        "with_claim_frame": sum(1 for row in rows if row.get("claim_frame", {}).get("ok")),
        "with_reaction_frame": sum(1 for row in rows if row.get("reaction_frame", {}).get("ok")),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
