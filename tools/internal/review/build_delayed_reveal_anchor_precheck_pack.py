from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from tools.internal.review.build_delayed_reveal_from_verified_anchors import validate_delayed_reveal_spec


PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
VIDEO_ROOT = Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001")
CRITICAL_TOKENS = {"死亡", "尸体", "击杀", "倒地", "被杀", "血迹", "killed", "death", "body", "blood"}
MIN_LOCAL_SEC = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual anchor precheck pack for delayed-public-reveal candidates.")
    parser.add_argument("--main-pass-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def local_start(row: dict[str, Any]) -> float | None:
    phase_ids = row.get("source_segment_ids") or []
    if not phase_ids:
        return None
    match = re.search(r"_(\d{6})_(\d{6})$", str(phase_ids[0]))
    if not match:
        return None
    return float(row.get("abs_start_sec", 0.0) or 0.0) - float(int(match.group(1)))


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(edge.get("event_id")), str(edge.get("player_id"))): edge for edge in edges}


def is_candidate_anchor(event: dict[str, Any]) -> bool:
    if event.get("phase_type") != "gameplay":
        return False
    if local_start(event) is not None and local_start(event) < MIN_LOCAL_SEC:
        return False
    if not event.get("source_povs") or not event.get("source_segment_ids"):
        return False
    if float(event.get("certainty", 0.0) or 0.0) < 0.75:
        return False
    description = str(event.get("description") or "")
    return event.get("event_type") in {"death", "player_death", "combat"} or any(token in description for token in CRITICAL_TOKENS)


def snap(video: Path, local_sec: float, out: Path) -> bool:
    if not video.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{max(local_sec, 0.0):.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def truncate(value: str, max_len: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def make_contact(rows: list[dict[str, Any]], output: Path, columns: int = 4) -> None:
    thumb_w, thumb_h = 360, 203
    label_h = 72
    pad = 8
    font = ImageFont.load_default()
    cells: list[Image.Image] = []
    for row in rows:
        img_path = Path(row["frame_path"])
        cell = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        draw = ImageDraw.Draw(cell)
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            cell.paste(img, ((thumb_w - img.width) // 2, 0))
        label = f"{row['idx']:03d} {row['event_id']} {row['source_pov']} local={row['local_sec']:.1f}s"
        desc = truncate(row["description"], 54)
        draw.text((pad, thumb_h + 5), label, fill="black", font=font)
        draw.text((pad, thumb_h + 25), desc, fill="black", font=font)
        draw.text((pad, thumb_h + 45), f"valid_specs={row['valid_spec_count']}", fill="black", font=font)
        cells.append(cell)
    if not cells:
        return
    rows_count = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * (thumb_h + label_h)), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % columns) * thumb_w, (i // columns) * (thumb_h + label_h)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        for path in sorted(args.output_dir.glob("*")):
            if path.is_file():
                path.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_root = args.main_pass_root / "annotations" / "oracle_ledger"
    events = read_jsonl(ledger_root / "world_events.jsonl")
    claims = read_jsonl(ledger_root / "claims.jsonl")
    edges = edge_lookup(read_jsonl(ledger_root / "visibility_edges.jsonl"))

    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: (float(row.get("abs_start_sec", 0.0) or 0.0), str(row.get("world_event_id")))):
        if not is_candidate_anchor(event):
            continue
        hidden_targets = [
            player
            for player in PLAYERS
            if edges.get((str(event.get("world_event_id")), player), {}).get("visibility") == "not_visible"
        ]
        valid_specs = []
        for target in hidden_targets:
            for claim in claims:
                if target not in claim.get("heard_by", []):
                    continue
                if local_start(claim) is not None and local_start(claim) < MIN_LOCAL_SEC:
                    continue
                if not validate_delayed_reveal_spec(event, target, claim):
                    valid_specs.append({"target": target, "claim_id": claim.get("claim_id")})
        if not valid_specs:
            continue
        phase_id = str(event["source_segment_ids"][0])
        source_pov = str(event["source_povs"][0])
        local_sec = ((float(event["abs_start_sec"]) + float(event["abs_end_sec"])) / 2.0) - phase_start(phase_id)
        frame_path = args.output_dir / f"{len(rows)+1:03d}_{event['world_event_id']}_{source_pov}_anchor.jpg"
        snap(VIDEO_ROOT / phase_id / f"{source_pov}.mp4", local_sec, frame_path)
        rows.append(
            {
                "idx": len(rows) + 1,
                "event_id": event["world_event_id"],
                "source_pov": source_pov,
                "phase_id": phase_id,
                "local_sec": local_sec,
                "abs_start_sec": event.get("abs_start_sec"),
                "abs_end_sec": event.get("abs_end_sec"),
                "event_type": event.get("event_type"),
                "description": event.get("description", ""),
                "hidden_targets": hidden_targets,
                "valid_spec_count": len(valid_specs),
                "example_specs": valid_specs[:8],
                "frame_path": frame_path.as_posix(),
            }
        )
        if len(rows) >= args.limit:
            break

    make_contact(rows, args.output_dir / "anchor_precheck_contact.jpg")
    (args.output_dir / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output_dir": args.output_dir.as_posix(), "anchors": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
