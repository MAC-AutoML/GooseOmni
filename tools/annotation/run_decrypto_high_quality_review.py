from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.decrypto_diagnostics import PLAYERS, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Omni high-quality review for Decrypto diagnostic queue items."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("runs/gooseomni_gameplay_pass1/release_benchmark_v2"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["mock", "local", "qwen", "openai"], default="mock"
    )
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def try_parse_json(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"value": obj}, True
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else {"value": obj}, True
            except Exception:
                return {}, False
    return {}, False


def phase_ids_for_task(task: dict[str, Any]) -> list[str]:
    phase_ids: list[str] = []
    group = task.get("probe_group") if isinstance(task.get("probe_group"), dict) else {}
    claim = task.get("claim") if isinstance(task.get("claim"), dict) else {}
    for row in [
        group,
        claim,
        *task.get("anchor_events", []),
        *task.get("related_claims", []),
    ]:
        if isinstance(row, dict):
            phase_ids.extend(
                str(item) for item in row.get("source_segment_ids", []) if item
            )
    return list(dict.fromkeys(phase_ids))


def preferred_players_for_task(task: dict[str, Any]) -> list[str]:
    players: list[str] = []
    for event in task.get("anchor_events", []):
        if isinstance(event, dict):
            players.extend(
                player for player in event.get("source_povs", []) if player in PLAYERS
            )
    group = task.get("probe_group") if isinstance(task.get("probe_group"), dict) else {}
    target = group.get("target_player")
    if target in PLAYERS:
        players.append(target)
    claim = task.get("claim") if isinstance(task.get("claim"), dict) else {}
    speaker = claim.get("speaker")
    if speaker in PLAYERS:
        players.append(speaker)
    for related in task.get("related_claims", []):
        if isinstance(related, dict) and related.get("speaker") in PLAYERS:
            players.append(related["speaker"])
    players.extend(PLAYERS)
    return list(dict.fromkeys(players))


def resolve_video(task: dict[str, Any], release_root: Path) -> Path:
    explicit = task.get("primary_video_file")
    if explicit:
        path = Path(str(explicit))
        if path.exists():
            return path
    for phase_id in phase_ids_for_task(task):
        for player in preferred_players_for_task(task):
            path = (
                release_root / "inputs" / "videos" / "g001" / phase_id / f"{player}.mp4"
            )
            if path.exists():
                return path
    raise FileNotFoundError(
        f"Could not resolve review video for task {task.get('review_task_id')}"
    )


def build_prompt(template: str, task: dict[str, Any], video_path: Path) -> str:
    payload = {
        "review_task": task,
        "primary_video_file": video_path.as_posix(),
        "context_video_files": task.get("context_video_files", []),
        "review_instruction": (
            "Use the primary video as direct evidence. Context video paths identify additional same-task clips for "
            "a human or a multi-video-capable reviewer; if your backend only receives the primary video, do not "
            "pretend context-only evidence is visible. Mark remaining_uncertainties or human_review_required. "
            "Return strict JSON only."
        ),
    }
    return (
        template
        + "\n\nTASK_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.queue)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    template = args.prompt_template.read_text(encoding="utf-8")
    backend = create_backend(
        args.backend,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        server_url=args.server_url,
    )
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for task in tasks:
        task_id = task["review_task_id"]
        out = args.output_root / f"{task_id}.json"
        if out.exists() and args.resume and not args.overwrite:
            stats["skipped"] += 1
            continue
        try:
            video_path = resolve_video(task, args.release_root)
            prompt = build_prompt(template, task, video_path)
            if args.backend == "mock":
                raw = json.dumps(
                    {
                        "review_task_id": task_id,
                        "decision": "human_review_required",
                        "gold_source_after_review": "qwen_weak",
                        "corrected_event": {},
                        "corrected_claim": {},
                        "corrected_probe_group": task.get("probe_group", {}),
                        "corrected_prompts": [],
                        "review_reasons": ["mock backend placeholder"],
                        "remaining_uncertainties": [
                            "mock backend does not inspect video"
                        ],
                        "needs_human_review": True,
                    },
                    ensure_ascii=False,
                )
            else:
                raw = backend.annotate_video(video_path, prompt)
            parsed, parse_ok = try_parse_json(raw)
            write_json(
                out,
                {
                    "review_task_id": task_id,
                    "task_type": task.get("task_type"),
                    "priority": task.get("priority"),
                    "video_file": video_path.as_posix(),
                    "parse_ok": parse_ok,
                    "parsed": parsed,
                    "raw_response": raw,
                    "source_task": task,
                },
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            write_json(
                args.output_root / f"{task_id}.error.json",
                {
                    "review_task_id": task_id,
                    "error": str(exc),
                    "source_task": task,
                },
            )
            stats["error"] += 1
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
