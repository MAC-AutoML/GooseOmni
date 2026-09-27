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
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import filter_povs, filter_segments, parse_json_array, parse_partial_json_array_objects, retry_prompt_for_compact_json, save_error, video_path_for

VALID_PHASE_TYPES = {
    "gameplay",
    "meeting",
    "discussion",
    "voting",
    "vote_result",
    "exile",
    "body_report",
    "transition",
    "result",
    "dead_spectating",
    "lobby_or_loading",
    "unknown",
}

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni phase/window classifier for gameplay-aware Omni Goose.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--output-root", default="runs/gooseomni_gameplay_pass1/phase_window_annotations", type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prompt_for(segment: Any, pov: Any) -> str:
    context = {
        "dataset": segment.dataset,
        "game_id": segment.game_id,
        "segment_id": segment.segment_id,
        "player_id": pov.player_id,
        "video_file": pov.video_file,
        "aligned_start_sec": segment.aligned_start_sec,
        "aligned_end_sec": segment.aligned_end_sec,
        "duration_sec": segment.duration_sec,
        "players": PLAYERS,
        "existing_boundary_candidates_for_reference_only": pov.qwen_round_boundary_candidates,
    }
    return f"""TASK: gameplay_phase_window
你是《鹅鸭杀》多视角视频 benchmark 的真实阶段定位员。这个视频是旧的重叠审查窗口，不是最终切分。你的任务是看懂当前 POV 视频中哪些 local 时间属于真实游戏跑动/任务/目击，哪些属于会议/投票/放逐/结果。

严格规则：
- 只看 current_pov 视频，不使用其他 POV 或后续信息。
- 不允许把整个窗口机械标成 meeting；只有画面确实是讨论/投票/玩家列表/放逐界面时才标 meeting/discussion/voting/vote_result/exile。
- 如果画面是地图中角色移动、做任务、跟随、同屏相遇、看到尸体、看到击杀、报案前后行动，必须标 gameplay/body_report 等。
- 如果玩家死亡或进入观战/幽灵视角，标 dead_spectating，并写证据。
- existing_boundary_candidates 只能作为参考证据，不能直接照抄。
- 输出 strict JSON 数组，不要 Markdown。
- local_start_sec/local_end_sec 必须在 0 到 duration_sec 之间；瞬时事件也给至少 1 秒时间窗。
- 绝对禁止输出 aligned_start_sec/aligned_end_sec。例如 aligned_start_sec=80, aligned_end_sec=170, duration_sec=90 时，视频第一帧必须写 local_start_sec=0，最后一帧必须写 local_end_sec=90，不能写 80 或 170。
- 不确定写 unknown 并降低 confidence，needs_human_review=true。

current_pov:
{json.dumps(context, ensure_ascii=False, indent=2)}

输出数组。每个对象必须包含：
local_start_sec, local_end_sec, phase_type, boundary_type, player_alive_state,
location, visible_players, key_visual_facts, key_audio_claims,
direct_visual_observation, speech_claim, public_result, inferred_belief,
hidden_or_not_visible_information, role_or_goal_clue, death_or_body_clue,
confidence, evidence, needs_human_review。

phase_type 只能是 gameplay, meeting, discussion, voting, vote_result, exile, body_report, transition, result, dead_spectating, lobby_or_loading, unknown。
最多输出 8 个按时间排序的区间。"""


def normalize_item(item: dict[str, Any], segment: Any, pov: Any, index: int) -> dict[str, Any]:
    payload = dict(item)
    def as_float(key: str, default: float) -> float:
        try:
            return float(payload.get(key, default))
        except (TypeError, ValueError):
            return default
    raw_start = as_float("local_start_sec", 0.0)
    raw_end = as_float("local_end_sec", min(segment.duration_sec, raw_start + 1.0))
    # Qwen sometimes returns absolute aligned seconds despite the prompt. Convert them.
    if (
        segment.aligned_start_sec <= raw_start <= segment.aligned_end_sec
        and segment.aligned_start_sec <= raw_end <= segment.aligned_end_sec
        and raw_end > segment.duration_sec
    ):
        raw_start -= segment.aligned_start_sec
        raw_end -= segment.aligned_start_sec
    start = max(0.0, min(segment.duration_sec, raw_start))
    end = max(0.0, min(segment.duration_sec, raw_end))
    if end <= start:
        end = min(segment.duration_sec, start + 1.0)
        if end <= start:
            start = max(0.0, end - 1.0)
    phase_type = str(payload.get("phase_type") or "unknown")
    if phase_type not in VALID_PHASE_TYPES:
        phase_type = "unknown"
        payload["needs_human_review"] = True
    visible_players = payload.get("visible_players", [])
    if not isinstance(visible_players, list):
        visible_players = []
    visible_players = [p for p in visible_players if p in PLAYERS]
    out = {
        "window_event_id": f"{segment.segment_id}_{pov.player_id}_phasewin_{index:03d}",
        "dataset": segment.dataset,
        "game_id": segment.game_id,
        "segment_id": segment.segment_id,
        "player_id": pov.player_id,
        "video_file": pov.video_file,
        "local_start_sec": start,
        "local_end_sec": end,
        "abs_start_sec": segment.aligned_start_sec + start,
        "abs_end_sec": segment.aligned_start_sec + end,
        "phase_type": phase_type,
        "boundary_type": str(payload.get("boundary_type") or "none"),
        "player_alive_state": str(payload.get("player_alive_state") or "unknown"),
        "location": str(payload.get("location") or "unknown")[:120],
        "visible_players": visible_players,
        "key_visual_facts": payload.get("key_visual_facts") if isinstance(payload.get("key_visual_facts"), list) else [],
        "key_audio_claims": payload.get("key_audio_claims") if isinstance(payload.get("key_audio_claims"), list) else [],
        "direct_visual_observation": bool(payload.get("direct_visual_observation", False)),
        "speech_claim": bool(payload.get("speech_claim", False)),
        "public_result": bool(payload.get("public_result", False)),
        "inferred_belief": bool(payload.get("inferred_belief", False)),
        "hidden_or_not_visible_information": payload.get("hidden_or_not_visible_information") if isinstance(payload.get("hidden_or_not_visible_information"), list) else [],
        "role_or_goal_clue": str(payload.get("role_or_goal_clue") or "unknown")[:160],
        "death_or_body_clue": str(payload.get("death_or_body_clue") or "unknown")[:160],
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.5) or 0.5))),
        "evidence": str(payload.get("evidence") or "unknown")[:240],
        "needs_human_review": bool(payload.get("needs_human_review", False)),
    }
    if out["phase_type"] == "unknown" or out["confidence"] < 0.55:
        out["needs_human_review"] = True
    return out


def mock_response(segment: Any) -> str:
    return json.dumps([
        {
            "local_start_sec": 0,
            "local_end_sec": segment.duration_sec,
            "phase_type": "unknown",
            "boundary_type": "none",
            "player_alive_state": "unknown",
            "location": "unknown",
            "visible_players": [],
            "key_visual_facts": [],
            "key_audio_claims": [],
            "direct_visual_observation": False,
            "speech_claim": False,
            "public_result": False,
            "inferred_belief": False,
            "hidden_or_not_visible_information": [],
            "role_or_goal_clue": "unknown",
            "death_or_body_clue": "unknown",
            "confidence": 0.4,
            "evidence": "mock",
            "needs_human_review": True,
        }
    ], ensure_ascii=False)


def main() -> None:
    args = parse_args()
    backend = None if args.backend == "mock" else create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    stats = {"ok": 0, "skipped": 0, "error": 0}
    for segment in filter_segments(load_segments_jsonl(segments_path), game_id=args.game_id, segment_id=args.segment_id, limit=args.limit, skip=args.skip, stride=args.stride):
        for pov in filter_povs(segment, args.player_id):
            out = args.output_root / "phase_windows" / segment.game_id / segment.segment_id / f"{pov.player_id}.json"
            if out.exists() and args.resume and not args.overwrite:
                stats["skipped"] += 1
                continue
            prompt = prompt_for(segment, pov)
            raw = ""
            try:
                if args.backend == "mock":
                    raw = mock_response(segment)
                else:
                    raw = backend.annotate_video(video_path_for(args.dataset_root, pov), prompt)
                try:
                    items = parse_json_array(raw)
                except Exception:
                    if args.backend == "mock":
                        raise
                    retry = retry_prompt_for_compact_json(prompt, max_items=5)
                    raw = backend.annotate_video(video_path_for(args.dataset_root, pov), retry)
                    try:
                        items = parse_json_array(raw)
                    except Exception:
                        items = parse_partial_json_array_objects(raw, max_items=5)
                        if not items:
                            raise
                    prompt = retry
                normalized = [normalize_item(item, segment, pov, idx) for idx, item in enumerate(items, start=1)]
                write_json(out, {
                    "dataset": segment.dataset,
                    "game_id": segment.game_id,
                    "segment_id": segment.segment_id,
                    "player_id": pov.player_id,
                    "video_file": pov.video_file,
                    "aligned_start_sec": segment.aligned_start_sec,
                    "aligned_end_sec": segment.aligned_end_sec,
                    "duration_sec": segment.duration_sec,
                    "phase_windows": normalized,
                    "raw_response": raw,
                })
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                save_error(dataset_root=args.dataset_root, annotation_root=args.output_root, stage="phase_window_annotations", segment=segment, pov=pov, prompt=prompt, raw_response=raw, error=exc)
                stats["error"] += 1
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
