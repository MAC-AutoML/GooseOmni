from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut variable-length gameplay-aware aligned phases from raw Omni Goose videos.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--phase-jsonl", default="runs/gooseomni_gameplay_pass1/phase_dataset/phases.jsonl", type=Path)
    parser.add_argument("--output-root", default="runs/gooseomni_gameplay_pass1/phase_dataset", type=Path)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-copy", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def has_encoder(name: str) -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], check=True, capture_output=True, text=True)
    except Exception:
        return False
    return name in result.stdout


def video_encode_args() -> list[str]:
    if has_encoder("libx264"):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    return ["-c:v", "mpeg4", "-q:v", "4"]


def cut_video(source: Path, output: Path, start: float, duration: float, overwrite: bool, reencode: bool) -> dict[str, float]:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_duration = probe_duration(source)
    end = start + duration
    actual_start = max(0.0, start)
    actual_end = min(raw_duration, end)
    actual_duration = actual_end - actual_start
    if actual_duration <= 0:
        raise ValueError(f"no source media overlaps requested window: {source} start={start} duration={duration}")
    pad_start = max(0.0, -start)
    pad_end = max(0.0, end - raw_duration)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    cmd += ["-ss", f"{actual_start:.3f}", "-i", str(source), "-t", f"{actual_duration:.3f}"]
    if reencode or pad_start or pad_end:
        vf = []
        af = []
        if pad_start or pad_end:
            vf.append(f"tpad=start_duration={pad_start:.3f}:stop_duration={pad_end:.3f}:color=black")
            af.append(f"adelay={int(round(pad_start * 1000))}:all=1")
            if pad_end:
                af.append(f"apad=pad_dur={pad_end:.3f}")
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-t", f"{duration:.3f}"] + video_encode_args() + ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-avoid_negative_ts", "make_zero", str(output)]
    subprocess.run(cmd, check=True)
    return {"padding_start_sec": pad_start, "padding_end_sec": pad_end, "actual_raw_start_sec": actual_start, "actual_raw_end_sec": actual_end}


def main() -> None:
    args = parse_args()
    old_segments = load_jsonl(args.dataset_root / "segments.jsonl")
    sync = {}
    raw = {}
    for seg in old_segments:
        if seg["game_id"] != args.game_id:
            continue
        for pov in seg.get("povs", []):
            player = pov["player_id"]
            sync.setdefault(player, float(pov["corrected_sync_raw_start_sec"]))
            raw.setdefault(player, Path(pov["source_raw_video"]))
    phases = load_jsonl(args.phase_jsonl)
    exported = []
    skipped = []
    videos = 0
    for phase in phases:
        phase_record = dict(phase)
        phase_record["povs"] = []
        for player in PLAYERS:
            if player not in sync or player not in raw:
                skipped.append({"phase_id": phase["phase_id"], "player_id": player, "reason": "missing_sync_or_raw"})
                continue
            source = raw[player]
            raw_start = sync[player] + float(phase["aligned_start_sec"])
            raw_end = sync[player] + float(phase["aligned_end_sec"])
            if raw_end <= raw_start:
                skipped.append({"phase_id": phase["phase_id"], "player_id": player, "reason": "invalid_raw_window", "raw_start_sec": raw_start, "raw_end_sec": raw_end})
                continue
            rel = Path("videos") / args.game_id / phase["phase_id"] / f"{player}.mp4"
            out = args.output_root / rel
            if args.dry_run:
                print(f"{source} -> {out} raw=[{raw_start:.3f},{raw_end:.3f}]")
            else:
                cut_meta = cut_video(source, out, raw_start, raw_end - raw_start, args.overwrite, reencode=not args.stream_copy)
            if args.dry_run:
                cut_meta = {"padding_start_sec": max(0.0, -raw_start), "padding_end_sec": 0.0, "actual_raw_start_sec": max(0.0, raw_start), "actual_raw_end_sec": raw_end}
            phase_record["povs"].append({
                "player_id": player,
                "video_file": rel.as_posix(),
                "source_raw_video": source.as_posix(),
                "corrected_sync_raw_start_sec": sync[player],
                "raw_start_sec": raw_start,
                "raw_end_sec": raw_end,
                "aligned_start_sec": phase["aligned_start_sec"],
                "aligned_end_sec": phase["aligned_end_sec"],
                **cut_meta,
            })
            videos += 1
        phase_record["pov_count"] = len(phase_record["povs"])
        exported.append(phase_record)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "phase_segments.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output_root / "phase_segments.jsonl").open("w", encoding="utf-8") as f:
            for row in exported:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        (args.output_root / "cut_skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_root / "manifest.json").write_text(json.dumps({
            "dataset": "gooseomni",
            "format": "gooseomni_gameplay_phase_dataset_v1",
            "game_id": args.game_id,
            "phase_count": len(exported),
            "video_count": videos,
            "skipped_count": len(skipped),
            "time_rule": "raw_start = corrected_sync_raw_start_sec + aligned_start_sec; abs_sec = aligned_start_sec + local_sec",
            "source": "raw long videos recut by Qwen3-Omni consensus gameplay/meeting phases",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phases": len(exported), "videos": videos, "skipped": len(skipped)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
