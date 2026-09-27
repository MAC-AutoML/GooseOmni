from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
MEETING_TYPES = {"meeting", "discussion", "voting", "vote_result", "exile"}
GAMEPLAY_TYPES = {"gameplay", "body_report", "transition", "dead_spectating"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Qwen phase-window annotations into gameplay-aware variable-length phases."
    )
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument(
        "--window-root",
        default="runs/gooseomni_gameplay_pass1/phase_window_annotations",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default="runs/gooseomni_gameplay_pass1/phase_dataset",
        type=Path,
    )
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--tick-sec", default=10.0, type=float)
    parser.add_argument("--min-phase-sec", default=30.0, type=float)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def coarse_type(phase_type: str) -> str:
    if phase_type in MEETING_TYPES:
        return "meeting"
    if phase_type == "result":
        return "final"
    if phase_type in GAMEPLAY_TYPES:
        return "gameplay"
    return "unknown"


def load_windows(root: Path, game_id: str) -> list[dict[str, Any]]:
    rows = []
    for p in sorted((root / "phase_windows" / game_id).glob("*/*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for item in obj.get("phase_windows", []):
            rows.append(item)
    return rows


def classify_tick(
    windows: list[dict[str, Any]], t: float
) -> tuple[str, dict[str, Any]]:
    votes = Counter()
    sources = []
    evidence = []
    for item in windows:
        if item.get("abs_start_sec", 10**9) <= t < item.get("abs_end_sec", -1):
            c = coarse_type(str(item.get("phase_type", "unknown")))
            conf = float(item.get("confidence", 0.4) or 0.4)
            votes[c] += max(0.1, conf)
            sources.append(item.get("player_id"))
            if item.get("evidence") and len(evidence) < 6:
                evidence.append(
                    {
                        "player_id": item.get("player_id"),
                        "phase_type": item.get("phase_type"),
                        "evidence": item.get("evidence"),
                        "confidence": conf,
                    }
                )
    if not votes:
        return "unknown", {"source_povs": [], "evidence": [], "confidence": 0.0}
    label, score = votes.most_common(1)[0]
    total = sum(votes.values())
    return label, {
        "source_povs": sorted({s for s in sources if s}),
        "evidence": evidence,
        "confidence": round(score / total, 3) if total else 0.0,
    }


def smooth(
    labels: list[tuple[float, str, dict[str, Any]]],
    min_phase_sec: float,
    tick_sec: float,
) -> list[tuple[float, str, dict[str, Any]]]:
    if not labels:
        return []
    out = labels[:]
    min_ticks = max(1, int(round(min_phase_sec / tick_sec)))
    changed = True
    while changed:
        changed = False
        runs = []
        start = 0
        for i in range(1, len(out) + 1):
            if i == len(out) or out[i][1] != out[start][1]:
                runs.append((start, i, out[start][1]))
                start = i
        for idx, (s, e, lab) in enumerate(runs):
            if e - s >= min_ticks or lab == "unknown":
                continue
            prev_lab = runs[idx - 1][2] if idx > 0 else None
            next_lab = runs[idx + 1][2] if idx + 1 < len(runs) else None
            fill = (
                prev_lab
                if prev_lab == next_lab and prev_lab is not None
                else (prev_lab or next_lab)
            )
            if fill:
                for j in range(s, e):
                    out[j] = (out[j][0], fill, out[j][2])
                changed = True
                break
    return out


def main() -> None:
    args = parse_args()
    segments = load_jsonl(args.dataset_root / "segments.jsonl")
    game_segments = [s for s in segments if s["game_id"] == args.game_id]
    if not game_segments:
        raise SystemExit("no segments")
    windows = load_windows(args.window_root, args.game_id)
    max_end = max(float(s["aligned_end_sec"]) for s in game_segments)
    first_window_start = min(
        (float(w.get("abs_start_sec", max_end)) for w in windows),
        default=min(float(s["aligned_start_sec"]) for s in game_segments),
    )
    ticks = []
    t = 0.0
    while t < max_end:
        label, meta = classify_tick(windows, t + args.tick_sec / 2)
        ticks.append((t, label, meta))
        t += args.tick_sec
    ticks = smooth(ticks, args.min_phase_sec, args.tick_sec)

    phases = []
    start_idx = 0
    for i in range(1, len(ticks) + 1):
        if i == len(ticks) or ticks[i][1] != ticks[start_idx][1]:
            start = ticks[start_idx][0]
            end = min(max_end, ticks[i - 1][0] + args.tick_sec)
            label = ticks[start_idx][1]
            metas = [m for _, _, m in ticks[start_idx:i]]
            source_povs = sorted({p for m in metas for p in m.get("source_povs", [])})
            if label == "unknown" and not source_povs:
                label = "unknown_gap"
            evidence = []
            seen_evidence = set()
            for m in metas:
                for e in m.get("evidence", []):
                    key = (e.get("player_id"), e.get("phase_type"), e.get("evidence"))
                    if key in seen_evidence:
                        continue
                    seen_evidence.add(key)
                    if len(evidence) < 12:
                        evidence.append(e)
            if end > start:
                idx = len(phases)
                phase_id = f"{args.game_id}_phase_{idx:03d}_{label}_{int(round(start)):06d}_{int(round(end)):06d}"
                phases.append(
                    {
                        "phase_id": phase_id,
                        "game_id": args.game_id,
                        "phase_index": idx,
                        "phase_type": label,
                        "aligned_start_sec": start,
                        "aligned_end_sec": end,
                        "duration_sec": end - start,
                        "source_povs": source_povs,
                        "evidence": evidence,
                        "confidence": round(
                            sum(float(m.get("confidence", 0)) for m in metas)
                            / max(1, len(metas)),
                            3,
                        ),
                        "needs_human_review": label in {"unknown", "unknown_gap"}
                        or len(source_povs) < 3,
                    }
                )
            start_idx = i

    # Ensure the first gameplay interval is present even if Qwen review windows start later than game time 0.
    # The legacy aligned windows begin at 80s, but the release must include game_start -> first meeting.
    pre_window_end = max(0.0, first_window_start)
    if phases and pre_window_end > 0:
        if (
            phases[0]["aligned_start_sec"] == 0
            and phases[0]["phase_type"] in {"unknown", "unknown_gap"}
            and phases[0]["aligned_end_sec"] <= pre_window_end + args.tick_sec
        ):
            phases[0].update(
                {
                    "phase_type": "gameplay",
                    "source_povs": PLAYERS,
                    "evidence": [
                        {
                            "evidence": "补回游戏开始到第一个 Qwen 审查窗口前的 gameplay 区间；边界来自已对齐旧窗口起点。"
                        }
                    ],
                    "confidence": 0.7,
                    "needs_human_review": True,
                }
            )
        elif phases[0]["aligned_start_sec"] > 0:
            first = phases[0]
            phases.insert(
                0,
                {
                    "phase_id": f"{args.game_id}_phase_000_gameplay_000000_{int(round(first['aligned_start_sec'])):06d}",
                    "game_id": args.game_id,
                    "phase_index": 0,
                    "phase_type": "gameplay",
                    "aligned_start_sec": 0.0,
                    "aligned_end_sec": first["aligned_start_sec"],
                    "duration_sec": first["aligned_start_sec"],
                    "source_povs": PLAYERS,
                    "evidence": [
                        {
                            "evidence": "补回第一局开始到第一次会议前的 gameplay 区间；边界来自后续会议开始绝对时间。"
                        }
                    ],
                    "confidence": 0.7,
                    "needs_human_review": True,
                },
            )
    # Reindex phase ids after insertion.
    for idx, phase in enumerate(phases):
        phase["phase_index"] = idx
        phase["phase_id"] = (
            f"{args.game_id}_phase_{idx:03d}_{phase['phase_type']}_{int(round(phase['aligned_start_sec'])):06d}_{int(round(phase['aligned_end_sec'])):06d}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "phases.json").write_text(
        json.dumps(phases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "phases.jsonl").open("w", encoding="utf-8") as f:
        for row in phases:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {"phases": len(phases), "windows": len(windows), "max_end": max_end},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
