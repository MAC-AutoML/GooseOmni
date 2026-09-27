from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DEFAULT_PASS = "runs/gooseomni_decrypto_diagnostic_pass16_codex_human_verified_cumulative_0001_0004"
VIDEO_ROOT = Path(
    "runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001"
)


def read_jsonl(path: Path) -> list[dict]:
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


def hstack(left: Path, right: Path, out: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack=inputs=2",
            str(out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def make_contact(pair_paths: list[Path], out: Path) -> bool:
    if not pair_paths:
        return False
    inputs: list[str] = []
    filter_parts: list[str] = []
    layout: list[str] = []
    for i, path in enumerate(pair_paths):
        inputs.extend(["-i", str(path)])
        filter_parts.append(f"[{i}:v]scale=426:120[v{i}]")
        layout.append(f"{(i % 3) * 426}_{(i // 3) * 120}")
    filter_complex = (
        ";".join(filter_parts)
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


def compact_prompt_fields(prompts: list[dict]) -> list[dict]:
    return [
        {
            "probe_id": row.get("probe_id"),
            "probe_type": row.get("probe_type"),
            "input_condition": row.get("input_condition"),
            "prompt_len": len(row.get("prompt") or ""),
            "leak_markers_present": any(
                s in (row.get("prompt") or "")
                for s in ["hidden_gold", "forbidden_event_ids"]
            ),
        }
        for row in prompts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-root", default=DEFAULT_PASS)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--max-claims", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pass_root = Path(args.pass_root)
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
    prompts_by_group: dict[str, list[dict]] = {}
    for row in read_jsonl(
        pass_root / "benchmark/gooseomni_v1/interactive_diagnostics/prompts.jsonl"
    ):
        prompts_by_group.setdefault(row["probe_group_id"], []).append(row)

    manifest = []
    all_pairs: list[Path] = []
    for idx, gid in enumerate(args.ids, 1):
        group = groups[gid]
        event = events[group["anchor_event_ids"][0]]
        target = group["target_player"]
        source = event["source_povs"][0]
        event_phase = event["source_segment_ids"][0]
        event_abs = (float(event["abs_start_sec"]) + float(event["abs_end_sec"])) / 2
        event_local = event_abs - phase_start(event_phase)

        event_source_img = out_dir / f"{idx:03d}_{gid}_anchor_source_{source}.jpg"
        event_target_img = out_dir / f"{idx:03d}_{gid}_anchor_target_{target}.jpg"
        event_pair = out_dir / f"{idx:03d}_{gid}_ANCHOR_PAIR.jpg"
        snap(video_path(event_phase, source), event_local, event_source_img)
        snap(video_path(event_phase, target), event_local, event_target_img)
        event_pair_ok = hstack(event_source_img, event_target_img, event_pair)
        if event_pair_ok:
            all_pairs.append(event_pair)

        claim_infos = []
        for claim_idx, claim_id in enumerate(
            group.get("related_claim_ids", [])[: args.max_claims], 1
        ):
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
                / f"{idx:03d}_{gid}_claim{claim_idx}_{claim_id}_speaker_{speaker}.jpg"
            )
            listener_img = (
                out_dir
                / f"{idx:03d}_{gid}_claim{claim_idx}_{claim_id}_listener_{listener}.jpg"
            )
            claim_pair = (
                out_dir / f"{idx:03d}_{gid}_CLAIM{claim_idx}_{claim_id}_PAIR.jpg"
            )
            speaker_ok = snap(
                video_path(claim_phase, speaker), claim_local, speaker_img
            )
            listener_ok = snap(
                video_path(claim_phase, listener), claim_local, listener_img
            )
            pair_ok = hstack(speaker_img, listener_img, claim_pair)
            if pair_ok:
                all_pairs.append(claim_pair)
            claim_infos.append(
                {
                    "claim_id": claim_id,
                    "speaker": speaker,
                    "listener": listener,
                    "abs_sec": claim_abs,
                    "claim_type": claim.get("claim_type"),
                    "strategic_role": claim.get("strategic_role"),
                    "content": claim.get("content"),
                    "normalized_content": claim.get("normalized_content"),
                    "speaker_image_ok": speaker_ok,
                    "listener_image_ok": listener_ok,
                    "pair_image": str(claim_pair),
                    "pair_ok": pair_ok,
                }
            )

        manifest.append(
            {
                "idx": idx,
                "probe_group_id": gid,
                "template": group.get("template"),
                "target_player": target,
                "cutoff_abs_sec": group.get("cutoff_abs_sec"),
                "diagnostic_families": group.get("diagnostic_families"),
                "anchor_event": event,
                "anchor_pair_image": str(event_pair),
                "anchor_pair_ok": event_pair_ok,
                "claims": claim_infos,
                "hidden_gold": hidden_gold.get(gid),
                "prompt_summary": compact_prompt_fields(prompts_by_group.get(gid, [])),
            }
        )

    make_contact(all_pairs, out_dir / "contact_sheet.jpg")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_dir)
    print(
        "groups",
        len(manifest),
        "pairs",
        len(all_pairs),
        "contact",
        (out_dir / "contact_sheet.jpg").exists(),
    )
    for row in manifest:
        print(
            json.dumps(
                {
                    "idx": row["idx"],
                    "probe_group_id": row["probe_group_id"],
                    "anchor_pair_ok": row["anchor_pair_ok"],
                    "claim_pairs": [c["pair_ok"] for c in row["claims"]],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
