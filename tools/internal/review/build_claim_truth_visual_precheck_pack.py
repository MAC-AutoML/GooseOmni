from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
GOOD_CLAIM_TYPES = {"location", "defense", "accusation", "sighting"}
GOOD_EVENT_TYPES = {
    "movement",
    "player_movement",
    "interaction",
    "player_interaction",
    "combat",
    "death",
    "player_death",
}
BAD_EVENT_TEXT = {
    "会议",
    "讨论",
    "投票",
    "界面",
    "结果",
    "切换",
    "任务",
    "meeting",
    "voting",
    "result",
    "task",
}
MIN_PHASE_LOCAL_SEC = 15.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def phase_id(row: dict[str, Any]) -> str | None:
    phase_ids = row.get("source_segment_ids") or []
    return str(phase_ids[0]) if phase_ids else None


def mid_time(row: dict[str, Any]) -> float:
    return (float(row["abs_start_sec"]) + float(row["abs_end_sec"])) / 2.0


def by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(edge.get("event_id")), str(edge.get("player_id"))): edge for edge in edges
    }


def load_skipped_clusters(paths: list[Path]) -> set[tuple[str, str]]:
    skipped: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("cluster_key") and ":" in str(row["cluster_key"]):
                event_id, claim_id = str(row["cluster_key"]).split(":", 1)
                skipped.add((event_id, claim_id))
            elif row.get("event_id") and row.get("claim_id"):
                skipped.add((str(row["event_id"]), str(row["claim_id"])))
            elif row.get("anchor_event_id") and row.get("claim_id"):
                skipped.add((str(row["anchor_event_id"]), str(row["claim_id"])))
    return skipped


def has_bad_event_text(event: dict[str, Any]) -> bool:
    text = str(event.get("description") or "").lower()
    return any(token in text for token in BAD_EVENT_TEXT)


def clean_event(event: dict[str, Any]) -> bool:
    if event.get("phase_type") != "gameplay":
        return False
    event_type = str(event.get("event_type") or "")
    if event_type not in GOOD_EVENT_TYPES:
        return False
    pid = phase_id(event)
    if not pid:
        return False
    if (
        float(event.get("abs_start_sec", 0.0) or 0.0) - phase_start(pid)
        < MIN_PHASE_LOCAL_SEC
    ):
        return False
    if has_bad_event_text(event):
        return False
    if float(event.get("certainty", 0.0) or 0.0) < 0.75:
        return False
    return bool(event.get("source_povs"))


def usable_claim(claim: dict[str, Any]) -> bool:
    if claim.get("claim_type") not in GOOD_CLAIM_TYPES:
        return False
    if claim.get("speaker") not in PLAYERS:
        return False
    if (
        len(str(claim.get("content") or claim.get("normalized_content") or "").strip())
        < 8
    ):
        return False
    if float(claim.get("certainty", 0.0) or 0.0) < 0.7:
        return False
    return phase_id(claim) is not None


def speaker_matches_event(event: dict[str, Any], claim: dict[str, Any]) -> bool:
    speaker = str(claim.get("speaker") or "")
    text = str(event.get("description") or "")
    actors = {str(actor) for actor in event.get("actors", [])}
    return speaker in actors or speaker in text


def direct_players(
    event: dict[str, Any], edges: dict[tuple[str, str], dict[str, Any]]
) -> list[str]:
    values = []
    for player in PLAYERS:
        visibility = edges.get((str(event["world_event_id"]), player), {}).get(
            "visibility"
        )
        if visibility in {"direct_visual", "direct_audio", "public_ui"}:
            values.append(player)
    return values


def hidden_players(
    event: dict[str, Any],
    claim: dict[str, Any],
    edges: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    heard = set(claim.get("heard_by", []))
    values = []
    for player in PLAYERS:
        visibility = edges.get((str(event["world_event_id"]), player), {}).get(
            "visibility"
        )
        if visibility == "not_visible" and player in heard:
            values.append(player)
    return values


def claim_event_score(
    event: dict[str, Any], claim: dict[str, Any], link: dict[str, Any], target: str
) -> tuple[float, str]:
    gap = float(claim["abs_end_sec"]) - float(event["abs_end_sec"])
    score = 0.0
    score += abs(gap - 20.0) / 10.0
    score -= float(link.get("confidence", 0.0) or 0.0) * 5.0
    if link.get("truth_status_global") == "supported":
        score += 1.0
    if claim.get("claim_type") in {"location", "defense"}:
        score -= 4.0
    if event.get("event_type") in {"movement", "player_movement"}:
        score -= 2.0
    if target in {"Gemini", "saoyi", "baile"}:
        score -= 0.5
    return score, str(event["world_event_id"])


def extract_frame(
    video_root: Path, pid: str, player: str, abs_sec: float, output: Path
) -> dict[str, Any]:
    video = video_root / pid / f"{player}.mp4"
    if not video.exists():
        return {
            "ok": False,
            "reason": "video_missing",
            "video": video.as_posix(),
            "phase_id": pid,
            "player": player,
        }
    local_sec = max(0.0, abs_sec - phase_start(pid))
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{local_sec:.3f}",
            "-i",
            video.as_posix(),
            "-frames:v",
            "1",
            output.as_posix(),
        ],
        capture_output=True,
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


def compose_quad(row: dict[str, Any], output: Path) -> bool:
    inputs = [
        row.get("anchor_source_frame", {}).get("output"),
        row.get("anchor_target_frame", {}).get("output"),
        row.get("claim_speaker_frame", {}).get("output"),
        row.get("claim_target_frame", {}).get("output"),
    ]
    if any(not path or not Path(path).exists() for path in inputs):
        return False
    cmd = [
        "ffmpeg",
        "-y",
        *sum((["-i", str(path)] for path in inputs), []),
        "-filter_complex",
        "[0:v]scale=480:270[a];[1:v]scale=480:270[b];[2:v]scale=480:270[c];[3:v]scale=480:270[d];"
        "[a][b][c][d]xstack=inputs=4:layout=0_0|480_0|0_270|480_270[out]",
        "-map",
        "[out]",
        output.as_posix(),
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return output.exists() and output.stat().st_size > 0


def make_contact(quad_paths: list[Path], output: Path) -> bool:
    if not quad_paths:
        return False
    list_path = output.parent / "claim_truth_quad_list.txt"
    list_path.write_text(
        "".join(f"file {path.name}\n" for path in quad_paths), encoding="utf-8"
    )
    rows = max(1, (len(quad_paths) + 1) // 2)
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
        f"scale=480:270,tile=2x{rows}",
        "-frames:v",
        "1",
        output.name,
    ]
    subprocess.run(
        cmd,
        cwd=output.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return output.exists() and output.stat().st_size > 0


def build_rows(
    ledger_root: Path,
    video_root: Path,
    output_dir: Path,
    limit: int,
    skipped: set[tuple[str, str]],
    max_targets_per_cluster: int,
    allow_speaker_mismatch: bool,
    truth_status: str,
) -> list[dict[str, Any]]:
    events = by_id(read_jsonl(ledger_root / "world_events.jsonl"), "world_event_id")
    claims = by_id(read_jsonl(ledger_root / "claims.jsonl"), "claim_id")
    links = read_jsonl(ledger_root / "claim_truth_links.jsonl")
    edges = edge_lookup(read_jsonl(ledger_root / "visibility_edges.jsonl"))
    candidates: list[tuple[tuple[float, str], dict[str, Any]]] = []
    cluster_counts: dict[tuple[str, str], int] = {}
    for link in links:
        if truth_status != "any" and link.get("truth_status_global") != truth_status:
            continue
        claim = claims.get(str(link.get("claim_id")))
        if not claim or not usable_claim(claim):
            continue
        for event_id in link.get("world_event_ids", []):
            event = events.get(str(event_id))
            if not event or not clean_event(event):
                continue
            cluster_key = (str(event["world_event_id"]), str(claim["claim_id"]))
            if cluster_key in skipped:
                continue
            if float(claim["abs_end_sec"]) <= float(event["abs_end_sec"]) + 1.0:
                continue
            if float(claim["abs_end_sec"]) - float(event["abs_end_sec"]) > 240.0:
                continue
            if not allow_speaker_mismatch and not speaker_matches_event(event, claim):
                continue
            if truth_status == "supported" and str(
                claim.get("speaker")
            ) not in direct_players(event, edges):
                continue
            direct = direct_players(event, edges)
            if not direct:
                continue
            for target in hidden_players(event, claim, edges):
                if cluster_counts.get(cluster_key, 0) >= max_targets_per_cluster:
                    continue
                row = {
                    "event_id": event["world_event_id"],
                    "claim_id": claim["claim_id"],
                    "target": target,
                    "source": direct[0],
                    "speaker": claim["speaker"],
                    "truth_status_global": link.get("truth_status_global"),
                    "event_phase": phase_id(event),
                    "claim_phase": phase_id(claim),
                    "event_abs": [event["abs_start_sec"], event["abs_end_sec"]],
                    "claim_abs": [claim["abs_start_sec"], claim["abs_end_sec"]],
                    "gap_sec": float(claim["abs_end_sec"])
                    - float(event["abs_end_sec"]),
                    "event_type": event.get("event_type"),
                    "event_description": event.get("description"),
                    "claim_type": claim.get("claim_type"),
                    "claim_text": claim.get("content")
                    or claim.get("normalized_content"),
                    "heard_by": claim.get("heard_by", []),
                    "source_povs": event.get("source_povs", []),
                    "link_id": link.get("claim_truth_link_id"),
                    "link_confidence": link.get("confidence"),
                }
                candidates.append((claim_event_score(event, claim, link, target), row))
                cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
    selected = [
        row for _score, row in sorted(candidates, key=lambda item: item[0])[:limit]
    ]
    quad_paths: list[Path] = []
    for idx, row in enumerate(selected, start=1):
        prefix = f"{idx:03d}_{row['event_id']}_{row['claim_id']}_{row['target']}"
        event_abs = sum(float(v) for v in row["event_abs"]) / 2.0
        claim_abs = sum(float(v) for v in row["claim_abs"]) / 2.0
        row["idx"] = idx
        row["anchor_source_frame"] = extract_frame(
            video_root,
            row["event_phase"],
            row["source"],
            event_abs,
            output_dir / f"{prefix}_anchor_source_{row['source']}.jpg",
        )
        row["anchor_target_frame"] = extract_frame(
            video_root,
            row["event_phase"],
            row["target"],
            event_abs,
            output_dir / f"{prefix}_anchor_target_{row['target']}.jpg",
        )
        row["claim_speaker_frame"] = extract_frame(
            video_root,
            row["claim_phase"],
            row["speaker"],
            claim_abs,
            output_dir / f"{prefix}_claim_speaker_{row['speaker']}.jpg",
        )
        row["claim_target_frame"] = extract_frame(
            video_root,
            row["claim_phase"],
            row["target"],
            claim_abs,
            output_dir / f"{prefix}_claim_target_{row['target']}.jpg",
        )
        quad = output_dir / f"{prefix}_quad.jpg"
        if compose_quad(row, quad):
            row["quad_frame"] = quad.as_posix()
            quad_paths.append(quad)
    make_contact(quad_paths, output_dir / "claim_truth_visual_precheck_contact.jpg")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visual precheck pack for contradicted claim-truth candidates."
    )
    parser.add_argument("--main-pass-root", type=Path, required=True)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(
            "runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--truth-status",
        choices=["contradicted", "supported", "any"],
        default="contradicted",
    )
    parser.add_argument("--max-targets-per-cluster", type=int, default=1)
    parser.add_argument("--skip-reviewed-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--allow-speaker-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skipped = load_skipped_clusters(args.skip_reviewed_jsonl)
    rows = build_rows(
        args.main_pass_root / "annotations" / "oracle_ledger",
        args.video_root,
        args.output_dir,
        args.limit,
        skipped,
        args.max_targets_per_cluster,
        args.allow_speaker_mismatch,
        args.truth_status,
    )
    write_json(args.output_dir / "manifest.json", rows)
    summary = {
        "ok": True,
        "output_dir": args.output_dir.as_posix(),
        "specs": len(rows),
        "with_quad": sum(1 for row in rows if row.get("quad_frame")),
        "skipped_clusters": len(skipped),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
