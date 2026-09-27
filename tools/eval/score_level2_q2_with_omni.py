#!/usr/bin/env python3
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
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.models.utils.omni_http_client import OmniHttpClient

DEFAULT_INPUT = ROOT / "results" / "results_qwen3_omni_level2_extended_audio-video_offline-q2.json"
DEFAULT_OUTPUT = ROOT / "results" / "results_qwen3_omni_level2_extended_audio-video_qwen3-judge.json"
DEFAULT_DUMMY_VIDEO = ROOT / "data" / "level_2_extended" / "videos" / "yang_gen_020.mp4"


def _score_from_text(text: str) -> int:
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not match:
        return 0
    value = max(0.0, min(100.0, float(match.group(0))))
    return min([0, 25, 50, 75, 100], key=lambda item: abs(item - value))


def _ensure_dummy_video(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    fallback = ROOT / "data" / "level_2_extended" / "videos" / "luo23_001.mp4"
    if fallback.exists() and fallback.stat().st_size > 0:
        return fallback
    raise FileNotFoundError(f"Missing dummy video: {path}")


def _build_prompt(reference: str, candidate: str) -> str:
    return (
        "You are a strict multilingual evaluator for dialog continuation.\n"
        "Compare the candidate answer with the reference answer.\n"
        "Score semantic match, intent correctness, and key information completeness.\n"
        "Use exactly one of these scores: 0, 25, 50, 75, 100.\n"
        "Output ONLY the number.\n\n"
        f"[Reference]\n{reference}\n\n"
        f"[Candidate]\n{candidate}\n"
    )


def score_file(input_path: Path, output_path: Path, server_url: str, dummy_video: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    client = OmniHttpClient(server_url)
    dummy_video = _ensure_dummy_video(dummy_video)

    for row in rows:
        reference = str(row.get("q2_reference") or "").strip()
        candidate = str(row.get("q2_response") or "").strip()
        if not bool(row.get("q1_correct")) or not reference:
            row["q2_score"] = None if bool(row.get("q1_correct")) else 0
            continue
        if not candidate:
            row["q2_score"] = 0
            row["q2_judge_response"] = ""
            continue
        raw = client.call_api(
            str(dummy_video),
            _build_prompt(reference, candidate),
            use_video=False,
            use_audio=False,
            max_retries=3,
            retry_delay=1.0,
        )
        row["q2_judge_response"] = raw or ""
        row["q2_score"] = _score_from_text(raw or "")

    scores = [
        float(row["q2_score"])
        for row in rows
        if isinstance(row.get("q2_score"), (int, float)) and bool(row.get("q1_correct")) and row.get("q2_reference")
    ]
    payload["q2_avg_score"] = sum(scores) / len(scores) if scores else 0.0
    payload["q2_count"] = len(scores)
    payload["q2_judge_model"] = "qwen3_omni"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Level 2 Q2 responses with a local Omni server")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--dummy-video", type=Path, default=DEFAULT_DUMMY_VIDEO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_file(args.input, args.output, args.server_url.rstrip("/"), args.dummy_video)
    print(args.output)


if __name__ == "__main__":
    main()
