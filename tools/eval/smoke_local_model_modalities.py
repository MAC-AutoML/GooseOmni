#!/usr/bin/env python3
"""Run one real local-model request for every supported media ablation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

MODES = {
    "qwen2_5_omni": ("av", "video", "audio", "text"),
    "qwen3_omni": ("av", "video", "audio", "text"),
    "qwen3_omni_thinking": ("av", "video", "audio", "text"),
    "miniomni_2": ("av", "video", "audio", "null_media"),
    "omnivinci": ("av", "video", "audio", "text"),
    "vita_1_5": ("av", "video", "audio", "text"),
    "baichuan_omni_1_5": ("av", "video", "audio", "text"),
    "ming": ("av", "video", "audio", "text"),
}

MODE_FLAGS = {
    "av": (True, True),
    "video": (True, False),
    "audio": (False, True),
    "text": (False, False),
    "null_media": (False, False),
}

MODE_PROMPTS = {
    "av": "请用一句中文概括视频中可见且可听的内容。",
    "video": "请只根据画面，用一句中文描述主要视觉内容。",
    "audio": "请只根据声音，用一句中文概括听到的内容。",
    "text": "不使用媒体内容，仅回答：2+2等于多少？",
    "null_media": "媒体已消融为静音和空白画面，仅回答：2+2等于多少？",
}


def run_mode(
    server_url: str, sample: Path, mode: str, timeout_sec: float
) -> dict[str, object]:
    use_video, use_audio = MODE_FLAGS[mode]
    started = time.monotonic()
    with sample.open("rb") as handle:
        response = requests.post(
            f"{server_url.rstrip('/')}/v1/infer",
            files={"video": (sample.name, handle, "video/mp4")},
            data={
                "prompt": MODE_PROMPTS[mode],
                "use_video": str(use_video).lower(),
                "use_audio": str(use_audio).lower(),
            },
            timeout=timeout_sec,
        )
    elapsed = time.monotonic() - started
    try:
        payload = response.json()
    except ValueError:
        payload = {"body": response.text[:1000]}
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    return {
        "mode": mode,
        "use_video": use_video,
        "use_audio": use_audio,
        "status_code": response.status_code,
        "latency_sec": round(elapsed, 3),
        "answer": str(answer)[:1000],
        "error": payload.get("error") if isinstance(payload, dict) else None,
        "ok": response.status_code == 200 and bool(str(answer).strip()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODES), required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=600)
    args = parser.parse_args()

    if not args.sample.is_file():
        raise FileNotFoundError(args.sample)
    rows = [
        run_mode(args.server_url, args.sample, mode, args.timeout_sec)
        for mode in MODES[args.model]
    ]
    report = {"model": args.model, "sample": str(args.sample), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
