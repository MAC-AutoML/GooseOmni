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

from tools.internal.review.build_delayed_reveal_from_verified_anchors import (
    validate_delayed_reveal_spec,
)

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
VIDEO_ROOT = Path(
    "runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"
)
CRITICAL_TOKENS = {
    "死亡",
    "尸体",
    "击杀",
    "倒地",
    "被杀",
    "血迹",
    "killed",
    "death",
    "body",
    "blood",
}
MIN_LOCAL_SEC = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visual precheck pack for delayed-public-reveal specs."
    )
    parser.add_argument("--main-pass-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-targets-per-cluster", type=int, default=1)
    parser.add_argument("--skip-reviewed-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def local_start(row: dict[str, Any]) -> float | None:
    phase_ids = row.get("source_segment_ids") or []
    if not phase_ids:
        return None
    match = re.search(r"_(\d{6})_(\d{6})$", str(phase_ids[0]))
    if not match:
        return None
    return float(row.get("abs_start_sec", 0.0) or 0.0) - float(int(match.group(1)))


def midpoint_local(row: dict[str, Any]) -> float:
    phase_id = str(row["source_segment_ids"][0])
    return (
        (float(row["abs_start_sec"]) + float(row["abs_end_sec"])) / 2.0
    ) - phase_start(phase_id)


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
            event_id = str(row.get("event_id") or "")
            claim_id = str(row.get("claim_id") or "")
            if event_id and claim_id:
                skipped.add((event_id, claim_id))
                continue
            cluster = str(row.get("cluster_key") or "")
            if ":" in cluster:
                event_id, claim_id = cluster.split(":", 1)
                skipped.add((event_id, claim_id))
    return skipped


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
    return event.get("event_type") in {"death", "player_death", "combat"} or any(
        token in description for token in CRITICAL_TOKENS
    )


def snap(video: Path, local_sec: float, out: Path) -> bool:
    if not video.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(local_sec, 0.0):.2f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def text(row: dict[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def truncate(value: str, max_len: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def spec_score(event: dict[str, Any], claim: dict[str, Any]) -> tuple[float, str, str]:
    delay = float(claim["abs_start_sec"]) - float(event["abs_end_sec"])
    content = text(claim, "content") + " " + text(claim, "normalized_content")
    score = 0.0
    if claim.get("speaker") in PLAYERS:
        score += 5.0
    if claim.get("claim_type") in {"accusation", "sighting"}:
        score += 3.0
    if any(name in content for name in PLAYERS):
        score += 2.0
    if any(token in content for token in ["杀", "刀", "死", "尸体", "击杀"]):
        score += 2.0
    if 45.0 <= delay <= 240.0:
        score += 2.0
    return (-score, str(event.get("world_event_id")), str(claim.get("claim_id")))


def compose_quad(row: dict[str, Any], out: Path) -> bool:
    image_paths = [
        Path(row[key])
        for key in [
            "anchor_source_frame",
            "anchor_target_frame",
            "reveal_speaker_frame",
            "reveal_listener_frame",
        ]
    ]
    if any(not path.exists() for path in image_paths):
        return False
    thumb_w, thumb_h = 360, 203
    label_h = 94
    cell_w, cell_h = thumb_w, thumb_h + label_h
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 2), "white")
    labels = [
        ("ANCHOR SOURCE", row["anchor_source_label"]),
        ("ANCHOR TARGET", row["anchor_target_label"]),
        ("REVEAL SPEAKER", row["reveal_speaker_label"]),
        ("REVEAL LISTENER", row["reveal_listener_label"]),
    ]
    for i, img_path in enumerate(image_paths):
        cell = Image.new("RGB", (cell_w, cell_h), "white")
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))
        cell.paste(img, ((thumb_w - img.width) // 2, 0))
        draw = ImageDraw.Draw(cell)
        draw.text((6, thumb_h + 4), labels[i][0], fill="black", font=font)
        draw.text(
            (6, thumb_h + 22), truncate(labels[i][1], 58), fill="black", font=font
        )
        draw.text(
            (6, thumb_h + 40),
            truncate(row["event_description"] if i < 2 else row["claim_text"], 58),
            fill="black",
            font=font,
        )
        draw.text(
            (6, thumb_h + 58),
            f"{row['idx']:03d} {row['event_id']} + {row['claim_id']}",
            fill="black",
            font=font,
        )
        sheet.paste(cell, ((i % 2) * cell_w, (i // 2) * cell_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return out.exists()


def make_contact(rows: list[dict[str, Any]], output: Path, columns: int = 2) -> None:
    if not rows:
        return
    thumbs: list[Image.Image] = []
    font = ImageFont.load_default()
    for row in rows:
        path = Path(row["quad_frame"])
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((500, 360))
        cell = Image.new("RGB", (500, 400), "white")
        cell.paste(img, ((500 - img.width) // 2, 0))
        draw = ImageDraw.Draw(cell)
        draw.text(
            (6, 364),
            f"{row['idx']:03d} {row['event_id']} {row['claim_id']} target={row['target']}",
            fill="black",
            font=font,
        )
        draw.text((6, 382), truncate(row["claim_text"], 76), fill="black", font=font)
        thumbs.append(cell)
    if not thumbs:
        return
    row_count = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 500, row_count * 400), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * 500, (i // columns) * 400))
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
    skipped_clusters = load_skipped_clusters(args.skip_reviewed_jsonl)

    specs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for event in events:
        if not is_candidate_anchor(event):
            continue
        hidden_targets = [
            player
            for player in PLAYERS
            if edges.get((str(event.get("world_event_id")), player), {}).get(
                "visibility"
            )
            == "not_visible"
        ]
        for target in hidden_targets:
            for claim in claims:
                if target not in claim.get("heard_by", []):
                    continue
                if (
                    local_start(claim) is not None
                    and local_start(claim) < MIN_LOCAL_SEC
                ):
                    continue
                if not validate_delayed_reveal_spec(event, target, claim):
                    if (
                        str(event.get("world_event_id")),
                        str(claim.get("claim_id")),
                    ) in skipped_clusters:
                        continue
                    specs.append((event, claim, target))

    specs = sorted(specs, key=lambda item: spec_score(item[0], item[1]))
    rows: list[dict[str, Any]] = []
    cluster_counts: dict[tuple[str, str], int] = {}
    for event, claim, target in specs:
        event_id = str(event["world_event_id"])
        claim_id = str(claim["claim_id"])
        cluster_key = (event_id, claim_id)
        if cluster_counts.get(cluster_key, 0) >= args.max_targets_per_cluster:
            continue
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
        source = str(event["source_povs"][0])
        event_phase = str(event["source_segment_ids"][0])
        claim_phase = str(claim["source_segment_ids"][0])
        speaker = str(claim.get("speaker") or "")
        if speaker not in PLAYERS:
            continue
        listener = target
        idx = len(rows) + 1
        row_dir = args.output_dir
        anchor_source = (
            row_dir / f"{idx:03d}_{event_id}_{claim_id}_anchor_source_{source}.jpg"
        )
        anchor_target = (
            row_dir / f"{idx:03d}_{event_id}_{claim_id}_anchor_target_{target}.jpg"
        )
        reveal_speaker = (
            row_dir / f"{idx:03d}_{event_id}_{claim_id}_reveal_speaker_{speaker}.jpg"
        )
        reveal_listener = (
            row_dir / f"{idx:03d}_{event_id}_{claim_id}_reveal_listener_{listener}.jpg"
        )
        if not snap(
            VIDEO_ROOT / event_phase / f"{source}.mp4",
            midpoint_local(event),
            anchor_source,
        ):
            continue
        if not snap(
            VIDEO_ROOT / event_phase / f"{target}.mp4",
            midpoint_local(event),
            anchor_target,
        ):
            continue
        if not snap(
            VIDEO_ROOT / claim_phase / f"{speaker}.mp4",
            midpoint_local(claim),
            reveal_speaker,
        ):
            continue
        if not snap(
            VIDEO_ROOT / claim_phase / f"{listener}.mp4",
            midpoint_local(claim),
            reveal_listener,
        ):
            continue
        row = {
            "idx": idx,
            "event_id": event_id,
            "claim_id": claim_id,
            "target": target,
            "source": source,
            "speaker": speaker,
            "event_phase": event_phase,
            "claim_phase": claim_phase,
            "event_abs": [event.get("abs_start_sec"), event.get("abs_end_sec")],
            "claim_abs": [claim.get("abs_start_sec"), claim.get("abs_end_sec")],
            "delay_sec": float(claim["abs_start_sec"]) - float(event["abs_end_sec"]),
            "event_description": text(event, "description"),
            "claim_text": text(claim, "content") or text(claim, "normalized_content"),
            "anchor_source_frame": anchor_source.as_posix(),
            "anchor_target_frame": anchor_target.as_posix(),
            "reveal_speaker_frame": reveal_speaker.as_posix(),
            "reveal_listener_frame": reveal_listener.as_posix(),
            "anchor_source_label": f"{source} sees event at local {midpoint_local(event):.1f}s",
            "anchor_target_label": f"{target} at same event time",
            "reveal_speaker_label": f"{speaker} claim local {midpoint_local(claim):.1f}s",
            "reveal_listener_label": f"{listener} hears claim local {midpoint_local(claim):.1f}s",
        }
        quad = row_dir / f"{idx:03d}_{event_id}_{claim_id}_{target}_quad.jpg"
        row["quad_frame"] = quad.as_posix()
        compose_quad(row, quad)
        rows.append(row)
        if len(rows) >= args.limit:
            break

    make_contact(rows, args.output_dir / "visual_precheck_contact.jpg")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"ok": True, "output_dir": args.output_dir.as_posix(), "specs": len(rows)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
