from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

VIDEO_ROOT = Path(
    "runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def video_path(phase_id: str, player_id: str) -> Path:
    return VIDEO_ROOT / phase_id / f"{player_id}.mp4"


def snap(video: Path, sec: float, out: Path) -> bool:
    if not video.exists():
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(sec, 0):.2f}",
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


def hstack(paths: list[Path], out: Path) -> bool:
    if not paths or any(not path.exists() for path in paths):
        return False
    inputs: list[str] = []
    filters: list[str] = []
    for i, path in enumerate(paths):
        inputs.extend(["-i", str(path)])
        filters.append(f"[{i}:v]scale=426:240[v{i}]")
    filter_complex = (
        ";".join(filters)
        + ";"
        + "".join(f"[v{i}]" for i in range(len(paths)))
        + f"hstack=inputs={len(paths)}[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def contact_sheet(pair_paths: list[Path], out: Path) -> bool:
    if not pair_paths:
        return False
    inputs: list[str] = []
    filters: list[str] = []
    layout: list[str] = []
    for i, path in enumerate(pair_paths):
        inputs.extend(["-i", str(path)])
        filters.append(f"[{i}:v]scale=426:120[v{i}]")
        layout.append(f"{(i % 2) * 426}_{(i // 2) * 120}")
    filter_complex = (
        ";".join(filters)
        + ";"
        + "".join(f"[v{i}]" for i in range(len(pair_paths)))
        + f"xstack=inputs={len(pair_paths)}:layout={'|'.join(layout)}[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate review assets for delayed-public-reveal candidates."
    )
    parser.add_argument("--pass-root", type=Path, required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pass_root = args.pass_root
    out_dir = pass_root / "review/codex_human_review_assets" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        row["probe_group_id"]: row
        for row in read_jsonl(pass_root / "annotations/diagnostics/probe_groups.jsonl")
    }
    events = {
        row["world_event_id"]: row
        for row in read_jsonl(
            pass_root / "annotations/oracle_ledger/world_events.jsonl"
        )
    }
    claims = {
        row["claim_id"]: row
        for row in read_jsonl(pass_root / "annotations/oracle_ledger/claims.jsonl")
    }
    hidden_gold = {
        row["probe_group_id"]: row
        for row in read_jsonl(pass_root / "annotations/diagnostics/hidden_gold.jsonl")
    }

    manifest: list[dict[str, Any]] = []
    pairs: list[Path] = []
    for idx, group_id in enumerate(args.ids, 1):
        group = groups[group_id]
        event = events[group["anchor_event_ids"][0]]
        target = group["target_player"]
        source = event["source_povs"][0]
        event_phase = event["source_segment_ids"][0]
        event_abs = (float(event["abs_start_sec"]) + float(event["abs_end_sec"])) / 2
        event_local = event_abs - phase_start(event_phase)

        source_img = out_dir / f"{idx:03d}_{group_id}_anchor_source_{source}.jpg"
        target_img = out_dir / f"{idx:03d}_{group_id}_anchor_target_{target}.jpg"
        snap(video_path(event_phase, source), event_local, source_img)
        snap(video_path(event_phase, target), event_local, target_img)

        reveal_infos: list[dict[str, Any]] = []
        reveal_images: list[Path] = []
        for claim_id in group.get("later_public_claim_ids", []):
            claim = claims[claim_id]
            claim_phase = claim["source_segment_ids"][0]
            claim_abs = (
                float(claim["abs_start_sec"]) + float(claim["abs_end_sec"])
            ) / 2
            claim_local = claim_abs - phase_start(claim_phase)
            speaker = claim.get("speaker")
            listener = (
                target
                if target in claim.get("heard_by", [])
                else (claim.get("heard_by") or [target])[0]
            )
            speaker_img = (
                out_dir
                / f"{idx:03d}_{group_id}_reveal_{claim_id}_speaker_{speaker}.jpg"
            )
            listener_img = (
                out_dir
                / f"{idx:03d}_{group_id}_reveal_{claim_id}_listener_{listener}.jpg"
            )
            snap(video_path(claim_phase, speaker), claim_local, speaker_img)
            snap(video_path(claim_phase, listener), claim_local, listener_img)
            reveal_images.extend([speaker_img, listener_img])
            reveal_infos.append(
                {
                    "claim_id": claim_id,
                    "speaker": speaker,
                    "listener": listener,
                    "abs_sec": claim_abs,
                    "claim_type": claim.get("claim_type"),
                    "content": claim.get("content"),
                    "normalized_content": claim.get("normalized_content"),
                    "heard_by": claim.get("heard_by", []),
                    "source_segment_ids": claim.get("source_segment_ids", []),
                }
            )

        pair = out_dir / f"{idx:03d}_{group_id}_DELAYED_REVEAL_PAIR.jpg"
        pair_ok = hstack([source_img, target_img, *reveal_images[:2]], pair)
        if pair_ok:
            pairs.append(pair)

        manifest.append(
            {
                "idx": idx,
                "probe_group_id": group_id,
                "target_player": target,
                "anchor_event": event,
                "later_public_claims": reveal_infos,
                "hidden_gold": hidden_gold.get(group_id),
                "pair_image": str(pair),
                "pair_ok": pair_ok,
            }
        )

    contact_sheet(pairs, out_dir / "contact_sheet.jpg")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_dir)
    print(
        "groups",
        len(manifest),
        "pairs",
        len(pairs),
        "contact",
        (out_dir / "contact_sheet.jpg").exists(),
    )
    for row in manifest:
        print(
            json.dumps(
                {
                    "idx": row["idx"],
                    "probe_group_id": row["probe_group_id"],
                    "pair_ok": row["pair_ok"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
