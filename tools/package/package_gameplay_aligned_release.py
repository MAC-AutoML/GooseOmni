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
import shutil
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package gameplay-aware per-video Omni Goose release.")
    parser.add_argument("--phase-root", default="runs/gooseomni_gameplay_pass1/phase_dataset", type=Path)
    parser.add_argument("--annotation-root", default="runs/gooseomni_gameplay_pass1/phase_annotations", type=Path)
    parser.add_argument("--output-dir", default="runs/gooseomni_gameplay_pass1/release_gameplay_aligned_v1", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def strip_raw(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_raw(v) for k, v in obj.items() if k != "raw_response"}
    if isinstance(obj, list):
        return [strip_raw(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    shutil.copytree(args.phase_root / "videos", args.output_dir / "videos")
    phases = load_jsonl(args.phase_root / "phase_segments.jsonl")
    annotations_written = 0
    missing_annotations = []
    for phase in phases:
        for pov in phase.get("povs", []):
            src = args.annotation_root / "annotations" / phase["game_id"] / phase["phase_id"] / f"{pov['player_id']}.json"
            ann = strip_raw(load_json(src))
            if not ann:
                missing_annotations.append({"phase_id": phase["phase_id"], "player_id": pov["player_id"]})
                ann = {
                    "dataset": "gooseomni",
                    "format": "gooseomni_gameplay_aligned_per_video_v1",
                    "game_id": phase["game_id"],
                    "phase_id": phase["phase_id"],
                    "phase_type": phase["phase_type"],
                    "player_id": pov["player_id"],
                    "video_file": pov["video_file"],
                    "aligned_start_sec": phase["aligned_start_sec"],
                    "aligned_end_sec": phase["aligned_end_sec"],
                    "duration_sec": phase["duration_sec"],
                    "evidence": phase.get("evidence", []),
                    "confidence": phase.get("confidence", 0.0),
                    "time": {
                        "aligned_start_sec": phase["aligned_start_sec"],
                        "aligned_end_sec": phase["aligned_end_sec"],
                        "duration_sec": phase["duration_sec"],
                        "abs_sec_formula": "abs_sec = aligned_start_sec + local_sec",
                    },
                    "player_status": {"alive_state": "unknown", "death_abs_sec": None, "death_evidence": "missing qwen annotation"},
                    "role_and_goal": {"role": "unknown", "faction": "unknown", "visible_goal": "unknown", "evidence": "missing qwen annotation"},
                    "gameplay_trace": [],
                    "utterances": [],
                    "private_memory": [],
                    "belief_state": {},
                    "tom_questions": [],
                    "needs_human_review": True,
                }
            ann["format"] = "gooseomni_gameplay_aligned_per_video_v1"
            ann.setdefault("aligned_start_sec", phase["aligned_start_sec"])
            ann.setdefault("aligned_end_sec", phase["aligned_end_sec"])
            ann.setdefault("duration_sec", phase["duration_sec"])
            ann.setdefault("evidence", phase.get("evidence", []))
            ann.setdefault("confidence", phase.get("confidence", 0.0))
            ann["annotation_file"] = f"annotations/{phase['game_id']}/{phase['phase_id']}/{pov['player_id']}.json"
            out = args.output_dir / ann["annotation_file"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(ann, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            annotations_written += 1
    readme = f"""# Omni Goose Gameplay-Aligned Release v1\n\nTop-level structure:\n\n```text\nrelease_gameplay_aligned_v1/\n├── README.md\n├── videos/\n└── annotations/\n```\n\nEach video has exactly one matching annotation JSON:\n\n```text\nvideos/g001/{{phase_id}}/{{player_id}}.mp4\nannotations/g001/{{phase_id}}/{{player_id}}.json\n```\n\nThis release is organized by real gameplay/meeting phases, not fixed 90-second windows. The old 90-second aligned clips were used only as Qwen3-Omni review windows for discovering semantic phase boundaries.\n\nTime rule:\n\n```text\nabs_sec = aligned_start_sec + local_sec\n```\n\nCounts:\n\n- phases: {len(phases)}\n- videos: {sum(len(p.get('povs', [])) for p in phases)}\n- annotations: {annotations_written}\n- missing_qwen_annotations_filled_for_review: {len(missing_annotations)}\n\nAnnotation fields include player_status, role_and_goal, gameplay_trace, utterances, private_memory, belief_state, and tom_questions. Labels are Qwen3-Omni weak annotations and require human verification before being claimed as human-verified.\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    # Keep release top-level limited to README.md, videos/, and annotations/.
    # Detailed packaging counts stay in README.md to avoid extra public entrypoints.
    print(json.dumps({"phases": len(phases), "annotations": annotations_written, "missing": len(missing_annotations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
