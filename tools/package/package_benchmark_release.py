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
import os
import shutil
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
VERSION = "gooseomni_benchmark_release_v2"
ABS_FORMULA = "abs_sec = aligned_start_sec + local_sec"
SOURCE_KEYS = [
    "direct_visual_observation",
    "speech_claim",
    "public_result",
    "inferred_belief",
    "hidden_or_not_visible_information",
]
WORKFLOW_KEYS = {
    "raw_response",
    "normalization",
    "risk_of_perspective_leakage_original",
    "confidence",
    "time",
    "evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Omni Goose gameplay as a benchmark release.")
    parser.add_argument("--source-release", default=Path("runs/gooseomni_gameplay_pass1/release_gameplay_aligned_v1"), type=Path)
    parser.add_argument("--output-dir", default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"), type=Path)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--update-existing", action="store_true", help="Rewrite JSON/README in an existing benchmark release without deleting videos.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def phase_sort_key(phase_id: str, ann: dict[str, Any]) -> tuple[float, int, str]:
    return (float(ann["aligned_start_sec"]), int(phase_id.split("_phase_")[1].split("_")[0]), phase_id)


def build_phase_index(source_root: Path, game_id: str) -> dict[str, dict[str, Any]]:
    phase_rows: dict[str, dict[str, Any]] = {}
    for ann_path in sorted((source_root / "annotations" / game_id).glob("*/*.json")):
        ann = load_json(ann_path)
        phase_id = ann["phase_id"]
        if phase_id not in phase_rows:
            phase_rows[phase_id] = {
                "phase_id": phase_id,
                "phase_type": ann["phase_type"],
                "episode_id": ann.get("game_episode_id") or f"{game_id}_episode_000",
                "episode_index": int(ann.get("game_episode_index", 0)),
                "aligned_start_sec": float(ann["aligned_start_sec"]),
                "aligned_end_sec": float(ann["aligned_end_sec"]),
                "duration_sec": float(ann["duration_sec"]),
            }

    ordered = sorted(phase_rows.values(), key=lambda row: (row["episode_index"], row["aligned_start_sec"]))
    global_order = sorted(phase_rows.values(), key=lambda row: phase_sort_key(row["phase_id"], row))
    global_index = {row["phase_id"]: idx for idx, row in enumerate(global_order)}

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        by_episode.setdefault(row["episode_id"], []).append(row)

    enriched: dict[str, dict[str, Any]] = {}
    for _episode_id, rows in by_episode.items():
        gameplay_round = 0
        meeting_round = 0
        for idx, row in enumerate(rows):
            row = dict(row)
            phase_type = row["phase_type"]
            if phase_type == "gameplay":
                gameplay_round += 1
                row["gameplay_round_index"] = gameplay_round
                row["meeting_round_index"] = None
                row["phase_order_label_zh"] = f"第{row['episode_index'] + 1}局第{gameplay_round}次跑动过程"
            elif phase_type == "meeting":
                meeting_round += 1
                row["gameplay_round_index"] = None
                row["meeting_round_index"] = meeting_round
                row["phase_order_label_zh"] = f"第{row['episode_index'] + 1}局第{meeting_round}次会议"
            else:
                row["gameplay_round_index"] = None
                row["meeting_round_index"] = None
                row["phase_order_label_zh"] = f"第{row['episode_index'] + 1}局最终结果"
            row["phase_index_global"] = global_index[row["phase_id"]]
            row["phase_index_in_episode"] = idx
            row["previous_phase_id"] = rows[idx - 1]["phase_id"] if idx > 0 else None
            row["next_phase_id"] = rows[idx + 1]["phase_id"] if idx + 1 < len(rows) else None
            enriched[row["phase_id"]] = row
    return enriched


def risk(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text in {"high", "高"} or "高" in text:
        return "high"
    if text in {"medium", "中", "中等"} or "中" in text:
        return "medium"
    if text in {"low", "低", "无", "none"} or "低" in text:
        return "low"
    if text in {"unknown", "未知", ""} or "未知" in text:
        return "unknown"
    return "unknown"


def meaningful(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "unknown", "未知", "无", "none", "None"}
    if isinstance(value, list):
        return bool(value)
    return True


def source_types(event: dict[str, Any]) -> list[str]:
    values = [key for key in SOURCE_KEYS if meaningful(event.get(key))]
    return values or ["direct_visual_observation"]


def visibility(types: list[str]) -> str:
    if "hidden_or_not_visible_information" in types:
        return "hidden"
    if "inferred_belief" in types:
        return "inferred"
    if "speech_claim" in types:
        return "heard_speech"
    if "public_result" in types:
        return "public"
    return "pov_visible"


def timed_base(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_start_sec": item.get("local_start_sec"),
        "local_end_sec": item.get("local_end_sec"),
        "abs_start_sec": item.get("abs_start_sec"),
        "abs_end_sec": item.get("abs_end_sec"),
    }


def clean_observation(event: dict[str, Any]) -> dict[str, Any]:
    types = source_types(event)
    out = {
        **timed_base(event),
        "event_type": event.get("event_type", "unknown"),
        "location": event.get("location", "unknown"),
        "visible_players": event.get("visible_players", []),
        "actor": event.get("actor", "unknown"),
        "source_types": types,
        "visibility": visibility(types),
        "description": event.get("description", ""),
        "evidence": event.get("evidence", ""),
        "certainty": event.get("certainty", "unknown"),
        "needs_human_review": bool(event.get("needs_human_review", False)),
    }
    if meaningful(event.get("speech_claim")):
        out["claim_text"] = event["speech_claim"]
    if meaningful(event.get("public_result")):
        out["public_result_text"] = event["public_result"]
    if meaningful(event.get("inferred_belief")):
        out["inferred_belief_text"] = event["inferred_belief"]
    if meaningful(event.get("hidden_or_not_visible_information")):
        out["hidden_information"] = event["hidden_or_not_visible_information"]
    return out


def clean_utterance(item: dict[str, Any]) -> dict[str, Any]:
    out = {
        **timed_base(item),
        "speaker": item.get("speaker", "unknown"),
        "transcript": item.get("transcript", ""),
        "claim_text": item.get("speech_claim", ""),
        "claims": item.get("claims", []),
        "source_types": ["speech_claim"],
        "visibility": "heard_speech",
        "evidence": item.get("evidence", ""),
        "certainty": item.get("certainty", "unknown"),
        "needs_human_review": bool(item.get("needs_human_review", False)),
    }
    return out


def clean_private_memory(items: Any) -> list[Any]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            cleaned.append({k: v for k, v in item.items() if k not in WORKFLOW_KEYS and k not in SOURCE_KEYS})
        else:
            cleaned.append(item)
    return cleaned


def required_sources_for_question(question: dict[str, Any]) -> list[str]:
    text = " ".join(str(question.get(key, "")) for key in ["question_type", "question", "available_information", "evidence"])
    sources: list[str] = []
    if any(token in text for token in ["看见", "视觉", "视频", "visible", "movement", "task", "location", "尸体", "击杀"]):
        sources.append("direct_visual_observation")
    if any(token in text for token in ["发言", "声称", "claim", "transcript", "说"]):
        sources.append("speech_claim")
    if any(token in text for token in ["投票", "结果", "public", "放逐", "胜利"]):
        sources.append("public_result")
    if any(token in text for token in ["认为", "怀疑", "belief", "推断"]):
        sources.append("inferred_belief")
    if not sources:
        sources.append("direct_visual_observation")
    return list(dict.fromkeys(sources))


def answerable_from_input(sources: list[str]) -> str:
    has_video = any(src in sources for src in ["direct_visual_observation", "public_result"])
    has_audio = "speech_claim" in sources
    if has_video and has_audio:
        return "video_audio"
    if has_video:
        return "video_only"
    if has_audio:
        return "audio_only"
    return "not_answerable_without_gold"


def clean_tom_question(question: dict[str, Any], player_id: str) -> dict[str, Any]:
    q_risk = risk(question.get("risk_of_perspective_leakage"))
    required_sources = required_sources_for_question(question)
    needs_review = bool(question.get("needs_human_review", False)) or q_risk in {"medium", "high"}
    return {
        "question_type": question.get("question_type", "unknown"),
        "question": question.get("question", ""),
        "answer": question.get("answer", "unknown"),
        "available_information": question.get("available_information", ""),
        "hidden_information": question.get("hidden_information", ""),
        "evidence": question.get("evidence", ""),
        "risk_of_perspective_leakage": q_risk,
        "certainty": question.get("certainty", "unknown"),
        "needs_human_review": needs_review,
        "requires_video_evidence": "direct_visual_observation" in required_sources,
        "required_source_types": required_sources,
        "target_player_perspective": player_id,
        "answerable_from_input": answerable_from_input(required_sources),
    }


def clean_review_reasons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    replacements = {
        "utterances_restored_from_raw_response": "utterances_restored_after_time_cleanup",
    }
    for item in value:
        text = str(item)
        text = replacements.get(text, text)
        text = text.replace("raw_response", "model_output")
        text = text.replace("normalization", "time_cleanup")
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def base_fields(ann: dict[str, Any], phase: dict[str, Any], *, video_file: str, metadata_file: str | None = None) -> dict[str, Any]:
    out = {
        "dataset": "gooseomni",
        "version": VERSION,
        "game_id": ann["game_id"],
        "episode_id": phase["episode_id"],
        "episode_index": phase["episode_index"],
        "phase_id": phase["phase_id"],
        "phase_type": phase["phase_type"],
        "phase_index_global": phase["phase_index_global"],
        "phase_index_in_episode": phase["phase_index_in_episode"],
        "phase_order_label_zh": phase["phase_order_label_zh"],
        "gameplay_round_index": phase["gameplay_round_index"],
        "meeting_round_index": phase["meeting_round_index"],
        "previous_phase_id": phase["previous_phase_id"],
        "next_phase_id": phase["next_phase_id"],
        "player_id": ann["player_id"],
        "video_file": video_file,
        "aligned_start_sec": phase["aligned_start_sec"],
        "aligned_end_sec": phase["aligned_end_sec"],
        "duration_sec": phase["duration_sec"],
        "abs_sec_formula": ABS_FORMULA,
    }
    if metadata_file is not None:
        out["input_metadata_file"] = metadata_file
    if ann.get("phase_subtype"):
        out["phase_subtype"] = ann["phase_subtype"]
    return out


def metadata_row(ann: dict[str, Any], phase: dict[str, Any], video_file: str) -> dict[str, Any]:
    row = base_fields(ann, phase, video_file=video_file)
    row["roster"] = PLAYERS
    return row


def gold_row(ann: dict[str, Any], phase: dict[str, Any], video_file: str, metadata_file: str, annotation_file: str) -> dict[str, Any]:
    review_reasons = clean_review_reasons(ann.get("review_reasons"))
    row = base_fields(ann, phase, video_file=video_file, metadata_file=metadata_file)
    row.update(
        {
            "annotation_file": annotation_file,
            "player_status": ann.get("player_status", {}),
            "role_and_goal": ann.get("role_and_goal", {}),
            "observations": [clean_observation(item) for item in ann.get("gameplay_trace", []) if isinstance(item, dict)],
            "utterances": [clean_utterance(item) for item in ann.get("utterances", []) if isinstance(item, dict)],
            "private_memory": clean_private_memory(ann.get("private_memory", [])),
            "belief_state": ann.get("belief_state", {}) if isinstance(ann.get("belief_state"), dict) else {},
            "tom_questions": [clean_tom_question(item, ann["player_id"]) for item in ann.get("tom_questions", []) if isinstance(item, dict)],
            "needs_human_review": bool(ann.get("needs_human_review", False)) or bool(review_reasons),
            "review_reasons": review_reasons,
        }
    )
    if phase["phase_type"] == "gameplay" and not row["observations"]:
        row["needs_human_review"] = True
        row["review_reasons"] = list(dict.fromkeys(row["review_reasons"] + ["empty_observations"]))
    if phase["phase_type"] == "meeting" and not row["utterances"]:
        row["needs_human_review"] = True
        row["review_reasons"] = list(dict.fromkeys(row["review_reasons"] + ["empty_utterances"]))
    return row


def readme() -> str:
    return """# Omni Goose Benchmark Release v2

This directory is a benchmark-facing export of the gameplay-aligned Omni Goose data.

## Layout

```text
release_benchmark_v2/
├── README.md
├── inputs/
│   ├── videos/g001/{phase_id}/{player_id}.mp4
│   ├── metadata/g001/{phase_id}/{player_id}.json
│   └── manifest.jsonl
└── gold_annotations/
    └── g001/{phase_id}/{player_id}.json
```

`inputs/` is the model-facing side. It contains videos and minimal metadata only.
`gold_annotations/` is the evaluator-facing side. It contains weak-gold labels, private memory, belief state, and ToM question answers.

## Time

Each video starts at local time `0`.

```text
abs_sec = aligned_start_sec + local_sec
```

Events in gold annotations include both local and absolute aligned game time.

## Phase Order

Every JSON includes explicit round fields:

```text
episode_id
episode_index
phase_index_global
phase_index_in_episode
phase_order_label_zh
gameplay_round_index
meeting_round_index
previous_phase_id
next_phase_id
```

Examples:

```text
第1局第1次跑动过程
第1局第1次会议
第1局最终结果
第2局第1次跑动过程
```

## Label Status

Gold annotations are automatic weak-gold labels and should be reviewed before being treated as final human labels.
Use `needs_human_review`, `review_reasons`, per-event `certainty`, and ToM `risk_of_perspective_leakage` to prioritize manual review.

Leakage risk values are normalized to:

```text
low
medium
high
unknown
```
"""


def main() -> None:
    args = parse_args()
    if not args.source_release.exists():
        raise FileNotFoundError(args.source_release)
    if args.output_dir.exists():
        if args.update_existing:
            pass
        elif not args.overwrite:
            raise SystemExit(f"Output directory exists: {args.output_dir}")
        else:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_index = build_phase_index(args.source_release, args.game_id)
    manifest_rows: list[dict[str, Any]] = []
    link_counts = {"hardlink": 0, "copy": 0, "exists": 0}

    for ann_path in sorted((args.source_release / "annotations" / args.game_id).glob("*/*.json")):
        ann = load_json(ann_path)
        phase = phase_index[ann["phase_id"]]
        player = ann["player_id"]
        src_video = args.source_release / ann["video_file"]
        video_file = f"inputs/videos/{args.game_id}/{phase['phase_id']}/{player}.mp4"
        metadata_file = f"inputs/metadata/{args.game_id}/{phase['phase_id']}/{player}.json"
        annotation_file = f"gold_annotations/{args.game_id}/{phase['phase_id']}/{player}.json"

        method = link_or_copy(src_video, args.output_dir / video_file)
        link_counts[method] += 1

        metadata = metadata_row(ann, phase, video_file)
        gold = gold_row(ann, phase, video_file, metadata_file, annotation_file)
        write_json(args.output_dir / metadata_file, metadata)
        write_json(args.output_dir / annotation_file, gold)
        manifest_rows.append(metadata)

    manifest_rows.sort(key=lambda row: (row["episode_index"], row["phase_index_in_episode"], row["player_id"]))
    write_jsonl(args.output_dir / "inputs" / "manifest.jsonl", manifest_rows)
    (args.output_dir / "README.md").write_text(readme(), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "videos": len(list((args.output_dir / "inputs" / "videos" / args.game_id).glob("*/*.mp4"))),
                "metadata": len(list((args.output_dir / "inputs" / "metadata" / args.game_id).glob("*/*.json"))),
                "gold_annotations": len(list((args.output_dir / "gold_annotations" / args.game_id).glob("*/*.json"))),
                "link_counts": link_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
