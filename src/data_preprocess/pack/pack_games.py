"""Pack per-POV episodes into multi-POV game_* folders."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from data_preprocess.common.json_utils import write_json
from data_preprocess.common.media import probe_duration
from data_preprocess.pack.gate import evaluate_date, write_gate
from data_preprocess.split.evidence import canon_result_label

TOL_SEC = 15.0
CLUSTER_TOL = 15.0
MIN_POVS = 2
EP_RE = re.compile(r"^episode_(\d+)_(.+)\.mp4$")


def wipe_dir(path: Path, attempts: int = 2) -> None:
    if not path.exists():
        return
    last_err: OSError | None = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_err = exc
            for p in sorted(path.rglob("*"), reverse=True):
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        p.rmdir()
                except OSError:
                    continue
            time.sleep(0.2 * (i + 1))
    stale = path.with_name(f"{path.name}.stale_{os.getpid()}")
    try:
        path.rename(stale)
    except OSError:
        if last_err:
            raise last_err
        raise
    shutil.rmtree(stale, ignore_errors=True)


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_clips(date: str, workspace: Path) -> dict[str, list[dict]]:
    folder = workspace / "episodes" / date
    out: dict[str, list[dict]] = {}
    if not folder.is_dir():
        return out
    for pov_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        clips = []
        for path in sorted(pov_dir.glob("episode_*.mp4")):
            match = EP_RE.match(path.name)
            if not match:
                continue
            raw = match.group(2)
            clips.append(
                {
                    "player": pov_dir.name,
                    "idx": int(match.group(1)),
                    "raw_label": raw,
                    "label": canon_result_label(raw),
                    "path": path,
                    "duration": probe_duration(path),
                    "used": False,
                }
            )
        if clips:
            out[pov_dir.name] = clips
    return out


def pick_anchor(clips: dict[str, list[dict]]) -> str:
    return max(sorted(clips), key=lambda player: len(clips[player]))


def assign_to_anchor(clips: dict[str, list[dict]], anchor: str) -> list[dict]:
    games: list[dict] = []
    for clip in clips[anchor]:
        games.append(
            {
                "label": clip["label"],
                "raw_labels": {clip["raw_label"]},
                "members": [clip],
                "anchor_player": anchor,
                "anchor_idx": clip["idx"],
                "anchor_duration": clip["duration"],
            }
        )
        clip["used"] = True
    for player, seq in clips.items():
        if player == anchor:
            continue
        pairs: list[tuple[float, int, int, int, dict]] = []
        for clip in seq:
            for gi, game in enumerate(games):
                if clip["label"] != game["label"]:
                    continue
                ddur = abs(clip["duration"] - game["anchor_duration"])
                if ddur > TOL_SEC:
                    continue
                dord = abs(clip["idx"] - game["anchor_idx"])
                pairs.append((ddur, dord, clip["idx"], gi, clip))
        pairs.sort()
        taken: set[int] = set()
        for _ddur, _dord, _cidx, gi, clip in pairs:
            if clip["used"] or gi in taken:
                continue
            game = games[gi]
            game["members"].append(clip)
            game["raw_labels"].add(clip["raw_label"])
            clip["used"] = True
            taken.add(gi)
    return games


def cluster_leftovers(clips: dict[str, list[dict]]) -> list[dict]:
    leftover = []
    for seq in clips.values():
        leftover.extend(c for c in seq if not c["used"])
    leftover.sort(key=lambda c: (c["player"], c["idx"]))
    extra = []
    for clip in leftover:
        if clip["used"]:
            continue
        group = [clip]
        for other in leftover:
            if other is clip or other["used"]:
                continue
            if other["player"] == clip["player"]:
                continue
            if (
                other["label"] == clip["label"]
                and abs(other["duration"] - clip["duration"]) <= CLUSTER_TOL
            ):
                group.append(other)
        if len(group) >= MIN_POVS:
            for item in group:
                item["used"] = True
            extra.append(
                {
                    "label": clip["label"],
                    "raw_labels": {x["raw_label"] for x in group},
                    "members": group,
                    "anchor_player": None,
                    "anchor_idx": min(x["idx"] for x in group),
                    "anchor_duration": clip["duration"],
                }
            )
    extra.sort(key=lambda g: min(m["idx"] for m in g["members"]))
    return extra


def drop_singleton_games(games: list[dict]) -> list[dict]:
    kept = []
    for game in games:
        if len(game["members"]) >= MIN_POVS:
            kept.append(game)
            continue
        for member in game["members"]:
            member["used"] = False
    return kept


def assign_timeline(anchor_games: list[dict], extra: list[dict]) -> list[dict]:
    for game in anchor_games:
        game["timeline_index"] = int(game["anchor_idx"])
    taken = {int(g["timeline_index"]) for g in anchor_games}
    nxt = (max(taken) + 1) if taken else 0
    extra_sorted = sorted(
        extra, key=lambda g: min(int(m["idx"]) for m in g["members"])
    )
    for game in extra_sorted:
        while nxt in taken:
            nxt += 1
        game["timeline_index"] = nxt
        taken.add(nxt)
        nxt += 1
    out = list(anchor_games) + extra_sorted
    out.sort(key=lambda g: int(g["timeline_index"]))
    return out


def pack_date(
    date: str,
    workspace: Path,
    *,
    overwrite: bool = False,
    require_gate: bool = True,
) -> dict:
    workspace = workspace.resolve()
    out_root = workspace / "games" / date
    if require_gate:
        gate = evaluate_date(date, workspace)
        if not gate.get("ready_to_pack"):
            write_gate(gate, workspace)
            return {
                "date": date,
                "skipped": True,
                "reason": "gate_blocked",
                "next_action": gate.get("next_action"),
                "blockers": gate.get("blockers"),
            }
    if out_root.exists() and not overwrite:
        return {
            "date": date,
            "skipped": True,
            "reason": "already_packed",
            "path": str(out_root),
        }
    clips = load_clips(date, workspace)
    if not clips:
        return {"date": date, "skipped": True, "reason": "no_episodes"}

    anchor = pick_anchor(clips)
    games = drop_singleton_games(assign_to_anchor(clips, anchor))
    games = assign_timeline(games, cluster_leftovers(clips))
    games = [g for g in games if len(g["members"]) >= MIN_POVS]

    if out_root.exists():
        wipe_dir(out_root)
    pending_dir = out_root / "pending"

    manifest_games = []
    for game in games:
        i = int(game["timeline_index"])
        durs = [m["duration"] for m in game["members"]]
        mean = sum(durs) / len(durs)
        name_dur = game.get("anchor_duration", mean)
        folder_name = f"game_{i:02d}_{game['label']}_{round(name_dur)}s"
        folder = out_root / folder_name
        povs = []
        for member in sorted(game["members"], key=lambda x: x["player"]):
            place(member["path"], folder / f"{member['player']}.mp4")
            povs.append(
                {
                    "player": member["player"],
                    "source_episode": member["path"].name,
                    "duration_sec": round(member["duration"], 3),
                    "outcome_label_raw": member["raw_label"],
                    "outcome_label": member["label"],
                }
            )
        manifest_games.append(
            {
                "folder": folder_name,
                "duration_sec_mean": round(mean, 3),
                "result_label": game["label"],
                "pov_count": len(povs),
                "anchor": game.get("anchor_player"),
                "timeline_index": i,
                "povs": povs,
            }
        )

    pending = []
    for seq in clips.values():
        for clip in seq:
            if clip["used"]:
                continue
            dest = pending_dir / f"{clip['player']}_{clip['path'].name}"
            place(clip["path"], dest)
            pending.append(
                {
                    "player": clip["player"],
                    "source_episode": clip["path"].name,
                    "duration_sec": round(clip["duration"], 3),
                    "outcome_label": clip["label"],
                }
            )

    manifest = {
        "session": date,
        "source": f"episodes/{date}",
        "anchor": anchor,
        "games": manifest_games,
        "pending": pending,
    }
    write_json(out_root / "manifest.json", manifest)
    gate_report = evaluate_date(date, workspace)
    gate_path = write_gate(gate_report, workspace)
    return {
        "date": date,
        "anchor": anchor,
        "games": len(games),
        "pending": len(pending),
        "gate": str(gate_path),
        "ready_to_align": bool(gate_report.get("ready_to_align")),
        "next_action": gate_report.get("next_action"),
    }
