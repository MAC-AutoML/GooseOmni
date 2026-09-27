from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
GOOD_EVENT_TYPES = {"movement", "player_movement", "interaction", "player_interaction", "combat", "death", "player_death"}
CRITICAL_TYPES = {"interaction", "player_interaction", "combat", "death", "player_death"}
BAD_TEXT = {"会议", "讨论", "投票", "界面", "结果", "切换", "任务", "等待", "观战", "meeting", "voting", "result", "task"}
MIN_PHASE_LOCAL_SEC = 15.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def phase_id(row: dict[str, Any]) -> str | None:
    phase_ids = row.get("source_segment_ids") or []
    return str(phase_ids[0]) if phase_ids else None


def mid_time(row: dict[str, Any]) -> float:
    return (float(row["abs_start_sec"]) + float(row["abs_end_sec"])) / 2.0


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {(str(edge.get("event_id")), str(edge.get("player_id"))): str(edge.get("visibility")) for edge in edges}


def used_anchor_events(main_pass_root: Path) -> set[str]:
    groups = read_jsonl(main_pass_root / "annotations" / "diagnostics" / "probe_groups.jsonl")
    used: set[str] = set()
    for group in groups:
        used.update(str(event_id) for event_id in group.get("anchor_event_ids", []))
    return used


def load_skipped_events(paths: list[Path]) -> set[str]:
    skipped: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in ["event_id", "anchor_event_id"]:
                if row.get(key):
                    skipped.add(str(row[key]))
            if row.get("cluster_key"):
                skipped.add(str(row["cluster_key"]).split(":", 1)[0])
    return skipped


def clean_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "")
    if event.get("phase_type") != "gameplay":
        return False
    if event_type not in GOOD_EVENT_TYPES:
        return False
    pid = phase_id(event)
    if not pid:
        return False
    if float(event.get("abs_start_sec", 0.0) or 0.0) - phase_start(pid) < MIN_PHASE_LOCAL_SEC:
        return False
    if float(event.get("certainty", 0.0) or 0.0) < 0.8:
        return False
    text = str(event.get("description") or "").lower()
    if any(token in text for token in BAD_TEXT):
        return False
    return bool(event.get("source_povs"))


def direct_players(event: dict[str, Any], edges: dict[tuple[str, str], str]) -> list[str]:
    players = []
    for player in PLAYERS:
        visibility = edges.get((str(event["world_event_id"]), player))
        if visibility in {"direct_visual", "direct_audio", "public_ui"}:
            players.append(player)
    return players


def hidden_players(event: dict[str, Any], edges: dict[tuple[str, str], str]) -> list[str]:
    text = str(event.get("description") or "")
    players = []
    for player in PLAYERS:
        if player in text:
            continue
        if edges.get((str(event["world_event_id"]), player)) == "not_visible":
            players.append(player)
    return players


def event_score(event: dict[str, Any], source_count: int, hidden_count: int) -> tuple[float, str]:
    score = 0.0
    event_type = str(event.get("event_type") or "")
    if event_type in CRITICAL_TYPES:
        score -= 8.0
    if event_type in {"movement", "player_movement"}:
        score -= 2.0
    score -= min(hidden_count, 5)
    score += abs(source_count - 1) * 3.0
    score += float(event.get("abs_start_sec", 0.0) or 0.0) / 100000.0
    return score, str(event["world_event_id"])


def extract_frame(video_root: Path, pid: str, player: str, abs_sec: float, output: Path) -> dict[str, Any]:
    video = video_root / pid / f"{player}.mp4"
    if not video.exists():
        return {"ok": False, "reason": "video_missing", "video": video.as_posix(), "phase_id": pid, "player": player}
    local_sec = max(0.0, abs_sec - phase_start(pid))
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
        "phase_id": pid,
        "player": player,
        "abs_sec": abs_sec,
        "local_sec": local_sec,
        "output": output.as_posix(),
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def compose_pair(row: dict[str, Any], output: Path) -> bool:
    left = row.get("anchor_source_frame", {}).get("output")
    right = row.get("anchor_target_frame", {}).get("output")
    if not left or not right or not Path(left).exists() or not Path(right).exists():
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack=inputs=2[out]",
        "-map",
        "[out]",
        output.as_posix(),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return output.exists() and output.stat().st_size > 0


def make_contact(pair_paths: list[Path], output: Path) -> bool:
    if not pair_paths:
        return False
    list_path = output.parent / "hidden_event_pair_list.txt"
    list_path.write_text("".join(f"file {path.name}\n" for path in pair_paths), encoding="utf-8")
    rows = max(1, (len(pair_paths) + 1) // 2)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path.name,
        "-vf",
        f"scale=640:180,tile=2x{rows}",
        "-frames:v",
        "1",
        output.name,
    ]
    subprocess.run(cmd, cwd=output.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return output.exists() and output.stat().st_size > 0


def build_rows(
    main_pass_root: Path,
    video_root: Path,
    output_dir: Path,
    limit: int,
    max_targets_per_event: int,
    include_used_anchors: bool,
    skipped_events: set[str],
) -> list[dict[str, Any]]:
    ledger_root = main_pass_root / "annotations" / "oracle_ledger"
    events = read_jsonl(ledger_root / "world_events.jsonl")
    edges = edge_lookup(read_jsonl(ledger_root / "visibility_edges.jsonl"))
    used = set() if include_used_anchors else used_anchor_events(main_pass_root)
    candidates: list[tuple[tuple[float, str], dict[str, Any]]] = []
    for event in events:
        event_id = str(event.get("world_event_id"))
        if event_id in used or event_id in skipped_events:
            continue
        if not clean_event(event):
            continue
        direct = direct_players(event, edges)
        hidden = hidden_players(event, edges)
        if not direct or not hidden:
            continue
        for target in hidden[:max_targets_per_event]:
            row = {
                "event_id": event_id,
                "target": target,
                "source": direct[0],
                "event_phase": phase_id(event),
                "event_abs": [event["abs_start_sec"], event["abs_end_sec"]],
                "event_type": event.get("event_type"),
                "event_description": event.get("description"),
                "source_povs": event.get("source_povs", []),
                "direct_players": direct,
                "hidden_players": hidden,
                "certainty": event.get("certainty"),
                "template_suggestion": "critical_hidden_event" if str(event.get("event_type")) in CRITICAL_TYPES else "route_hidden_event",
            }
            candidates.append((event_score(event, len(direct), len(hidden)), row))
    selected = [row for _score, row in sorted(candidates, key=lambda item: item[0])[:limit]]
    pair_paths: list[Path] = []
    for idx, row in enumerate(selected, start=1):
        row["idx"] = idx
        prefix = f"{idx:03d}_{row['event_id']}_{row['target']}"
        event_abs = sum(float(v) for v in row["event_abs"]) / 2.0
        row["anchor_source_frame"] = extract_frame(video_root, row["event_phase"], row["source"], event_abs, output_dir / f"{prefix}_source_{row['source']}.jpg")
        row["anchor_target_frame"] = extract_frame(video_root, row["event_phase"], row["target"], event_abs, output_dir / f"{prefix}_target_{row['target']}.jpg")
        pair = output_dir / f"{prefix}_PAIR.jpg"
        if compose_pair(row, pair):
            row["pair_frame"] = pair.as_posix()
            pair_paths.append(pair)
    make_contact(pair_paths, output_dir / "hidden_event_visual_precheck_contact.jpg")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual precheck pack for new hidden/route event candidates.")
    parser.add_argument("--main-pass-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-targets-per-event", type=int, default=1)
    parser.add_argument("--include-used-anchors", action="store_true")
    parser.add_argument("--skip-reviewed-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        args.main_pass_root,
        args.video_root,
        args.output_dir,
        args.limit,
        args.max_targets_per_event,
        args.include_used_anchors,
        load_skipped_events(args.skip_reviewed_jsonl),
    )
    write_json(args.output_dir / "manifest.json", rows)
    summary = {
        "ok": True,
        "output_dir": args.output_dir.as_posix(),
        "specs": len(rows),
        "with_pair": sum(1 for row in rows if row.get("pair_frame")),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
