"""Checkpoint before pack/align: catch incomplete episodes and 漏局."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data_preprocess.common.json_utils import write_json

EP_RE = re.compile(r"^episode_(\d+)_(.+)\.mp4$")
GAME_RE = re.compile(r"^game_(\d+)_")
STICKY_SEC = 35 * 60


def expected_players(date: str, workspace: Path) -> list[str]:
    """Optional roster: workspace/sessions/<date>/players.txt (one id per line)."""
    path = workspace / "sessions" / date / "players.txt"
    if path.is_file():
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    episodes = workspace / "episodes" / date
    if episodes.is_dir():
        return sorted(
            p.name for p in episodes.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    return []


def list_episode_clips(date: str, workspace: Path) -> list[dict[str, Any]]:
    folder = workspace / "episodes" / date
    clips: list[dict[str, Any]] = []
    if not folder.is_dir():
        return clips
    for pov_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for path in sorted(pov_dir.glob("episode_*.mp4")):
            match = EP_RE.match(path.name)
            if not match:
                continue
            clips.append(
                {
                    "player": pov_dir.name,
                    "idx": int(match.group(1)),
                    "name": path.name,
                    "path": path,
                    "mtime": path.stat().st_mtime,
                }
            )
    return clips


def load_pack_manifest(date: str, workspace: Path) -> dict[str, Any] | None:
    path = workspace / "games" / date / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def evaluate_date(date: str, workspace: Path) -> dict[str, Any]:
    blockers: list[str] = []
    hints: list[str] = []
    expected = expected_players(date, workspace)
    clips = list_episode_clips(date, workspace)
    by_player: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        by_player.setdefault(str(clip["player"]), []).append(clip)
    present = sorted(by_player)
    missing = [p for p in expected if p not in by_player]
    if missing:
        blockers.append("episodes_incomplete")
        hints.append("missing POVs: " + ", ".join(missing))

    index_gaps: dict[str, list[int]] = {}
    for player, seq in by_player.items():
        idxs = sorted(int(c["idx"]) for c in seq)
        if not idxs:
            continue
        expect = list(range(idxs[0], idxs[-1] + 1))
        hole = [i for i in expect if i not in set(idxs)]
        if idxs[0] != 0:
            hole = list(range(idxs[0])) + hole
        if hole:
            index_gaps[player] = hole
    if index_gaps:
        blockers.append("episode_index_gaps")
        hints.append("episode index gaps: " + json.dumps(index_gaps, ensure_ascii=False))

    packed = load_pack_manifest(date, workspace)
    pending_dir = workspace / "games" / date / "pending"
    pending_files = (
        sorted(p.name for p in pending_dir.glob("*.mp4")) if pending_dir.is_dir() else []
    )
    pending_meta = list(packed.get("pending") or []) if packed else []
    if pending_files or pending_meta:
        blockers.append("pending_clips")
        hints.append("manual-cut pending clips before align")

    unused: list[str] = []
    game_holes: list[int] = []
    sticky_packed: list[str] = []
    stale = False
    if packed:
        used: set[tuple[str, str]] = set()
        for game in packed.get("games") or []:
            for pov in game.get("povs") or []:
                used.add((str(pov.get("player")), str(pov.get("source_episode"))))
                dur = float(pov.get("duration_sec") or 0)
                if dur >= STICKY_SEC:
                    sticky_packed.append(
                        f"{game.get('folder')}/{pov.get('player')} {dur:.0f}s"
                    )
        unused = [
            f"{c['player']}/{c['name']}"
            for c in clips
            if (c["player"], c["name"]) not in used
        ]
        if unused and "pending_clips" not in blockers:
            blockers.append("unused_episodes")
            hints.append("unused episodes: " + ", ".join(unused[:12]))
        indices: list[int] = []
        for game in packed.get("games") or []:
            folder = str(game.get("folder") or "")
            match = GAME_RE.match(folder)
            if match:
                indices.append(int(match.group(1)))
        if indices:
            span = list(range(min(indices), max(indices) + 1))
            have = set(indices)
            game_holes = [i for i in span if i not in have]
            if min(indices) != 0:
                game_holes = list(range(min(indices))) + game_holes
        if game_holes:
            blockers.append("game_index_holes")
            hints.append(
                "game index holes: " + ",".join(f"{i:02d}" for i in game_holes)
            )
        manifest_path = workspace / "games" / date / "manifest.json"
        pack_mtime = manifest_path.stat().st_mtime
        latest_ep = max((float(c["mtime"]) for c in clips), default=0.0)
        if clips and latest_ep > pack_mtime + 1:
            stale = True
            blockers.append("stale_pack")
            hints.append("episodes newer than pack; re-pack")
    elif "episodes_incomplete" not in blockers:
        blockers.append("not_packed")
        hints.append("pack episodes into games/ first")

    if sticky_packed:
        blockers.append("sticky_long")
        hints.append(">=35min clip may contain multiple games")

    unique: list[str] = []
    for item in blockers:
        if item not in unique:
            unique.append(item)
    blockers = unique

    ready_to_pack = not any(
        b in blockers for b in ("episodes_incomplete", "episode_index_gaps")
    )
    ready_to_align = ready_to_pack and packed is not None and not blockers
    if ready_to_pack and packed and "pending_clips" in blockers:
        next_action = "manual_pending_then_repack"
    elif "episodes_incomplete" in blockers:
        next_action = "wait_split"
    elif "episode_index_gaps" in blockers:
        next_action = "fix_gaps_then_repack"
    elif "not_packed" in blockers:
        next_action = "pack"
    elif any(
        b in blockers
        for b in (
            "game_index_holes",
            "unused_episodes",
            "stale_pack",
            "sticky_long",
        )
    ):
        next_action = "fix_then_repack"
    elif ready_to_align:
        next_action = "align"
    else:
        next_action = "inspect"

    return {
        "date": date,
        "expected": expected,
        "present_povs": present,
        "missing_povs": missing,
        "episode_count": len(clips),
        "packed": packed is not None,
        "pending": pending_files or [p.get("source_episode") for p in pending_meta],
        "unused_episodes": unused,
        "episode_index_gaps": index_gaps,
        "game_index_holes": game_holes,
        "sticky": sticky_packed,
        "stale_pack": stale,
        "blockers": blockers,
        "hints": hints,
        "ready_to_pack": ready_to_pack,
        "ready_to_align": ready_to_align,
        "next_action": next_action,
    }


def write_gate(report: dict[str, Any], workspace: Path) -> Path:
    date = str(report["date"])
    dest = workspace / "games" / date / "pipeline_gate.json"
    if dest.parent.is_dir():
        write_json(dest, report)
    cfg = workspace / "pipeline_gates"
    cfg.mkdir(parents=True, exist_ok=True)
    alt = cfg / f"{date}.json"
    write_json(alt, report)
    return alt if not dest.parent.is_dir() else dest


def assert_ready_to_align(date: str, workspace: Path) -> dict[str, Any]:
    report = evaluate_date(date, workspace)
    write_gate(report, workspace)
    if report["ready_to_align"]:
        return report
    raise RuntimeError(
        f"{date} align gate failed next={report['next_action']} "
        f"blockers={report['blockers']}"
    )
