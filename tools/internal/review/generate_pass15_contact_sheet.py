from __future__ import annotations

import json
import re
import subprocess
import argparse
from pathlib import Path


PASS = Path("runs/gooseomni_decrypto_diagnostic_pass14_codex_human_verified_cumulative_0001_0002")
RELEASE = Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2/inputs/videos/g001")
DEFAULT_OUT_NAME = "pass15_candidates_410_430_contact"
DEFAULT_IDS = [
    "g001_pg_000104_Gemini",
    "g001_pg_000105_baile",
    "g001_pg_000106_beigang",
    "g001_pg_000195_Gemini",
    "g001_pg_000196_baile",
    "g001_pg_000197_beigang",
    "g001_pg_000198_Gemini",
    "g001_pg_000199_baile",
    "g001_pg_000200_beigang",
    "g001_pg_000204_Gemini",
    "g001_pg_000205_baile",
    "g001_pg_000206_beigang",
    "g001_pg_000207_Gemini",
    "g001_pg_000208_baile",
    "g001_pg_000209_beigang",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phase_start(phase_id: str) -> float:
    match = re.search(r"_(\d{6})_(\d{6})$", phase_id)
    return float(int(match.group(1))) if match else 0.0


def snap(video: Path, sec: float, out: Path) -> bool:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{max(sec, 0):.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return out.exists() and out.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-root", default=str(PASS))
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--ids-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pass_root = Path(args.pass_root)
    ids = list(args.ids)
    if args.ids_file:
        ids = [line.strip() for line in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = pass_root / "review/codex_human_review_assets" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    groups = {row["probe_group_id"]: row for row in read_jsonl(pass_root / "annotations/diagnostics/probe_groups.jsonl")}
    events = {row["world_event_id"]: row for row in read_jsonl(pass_root / "annotations/oracle_ledger/world_events.jsonl")}
    manifest = []
    for idx, gid in enumerate(ids, 1):
        group = groups[gid]
        event = events[group["anchor_event_ids"][0]]
        source = event["source_povs"][0]
        target = group["target_player"]
        phase = event["source_segment_ids"][0]
        abs_sec = (float(event["abs_start_sec"]) + float(event["abs_end_sec"])) / 2
        local_sec = abs_sec - phase_start(phase)
        src = out / f"{idx:03d}_{gid}_source_{source}.jpg"
        tgt = out / f"{idx:03d}_{gid}_target_{target}.jpg"
        pair = out / f"{idx:03d}_{gid}_PAIR.jpg"
        ok1 = snap(RELEASE / phase / f"{source}.mp4", local_sec, src)
        ok2 = snap(RELEASE / phase / f"{target}.mp4", local_sec, tgt)
        label = f"{idx} {gid} {event['world_event_id']} src:{source} tgt:{target} abs:{abs_sec:.1f} {event.get('event_type')} {event.get('location')}"
        if ok1 and ok2:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src),
                    "-i",
                    str(tgt),
                    "-filter_complex",
                    "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack=inputs=2",
                    str(pair),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        manifest.append(
            {
                "idx": idx,
                "gid": gid,
                "event_id": event["world_event_id"],
                "source": source,
                "target": target,
                "abs_sec": abs_sec,
                "event_type": event.get("event_type"),
                "location": event.get("location"),
                "desc": event.get("description"),
                "source_image": str(src),
                "target_image": str(tgt),
                "pair_image": str(pair),
                "ok": pair.exists() and pair.stat().st_size > 0,
            }
        )

    ok_rows = [row for row in manifest if row["ok"]]
    inputs: list[str] = []
    filter_parts: list[str] = []
    layout: list[str] = []
    for i, row in enumerate(ok_rows):
        inputs.extend(["-i", row["pair_image"]])
        filter_parts.append(f"[{i}:v]scale=426:120[v{i}]")
        layout.append(f"{(i % 3) * 426}_{(i // 3) * 120}")
    if ok_rows:
        filter_complex = (
            ";".join(filter_parts)
            + ";"
            + "".join(f"[v{i}]" for i in range(len(ok_rows)))
            + f"xstack=inputs={len(ok_rows)}:layout={'|'.join(layout)}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[out]", str(out / "contact_sheet.jpg")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print("pairs", sum(row["ok"] for row in manifest), "manifest", len(manifest), "contact", (out / "contact_sheet.jpg").exists())
    for row in manifest:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
