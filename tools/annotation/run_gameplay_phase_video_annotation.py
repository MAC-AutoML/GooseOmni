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
from gooseomni.benchmark.io import write_json
from gooseomni.benchmark.pipeline import parse_json_object, parse_partial_json_object, retry_prompt_for_compact_object

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate final variable-length gameplay phases with Qwen3-Omni.")
    parser.add_argument("--phase-root", default="runs/gooseomni_gameplay_pass1/phase_dataset", type=Path)
    parser.add_argument("--output-root", default="runs/gooseomni_gameplay_pass1/phase_annotations", type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--game-id", default="g001")
    parser.add_argument("--phase-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_for(phase: dict[str, Any], pov: dict[str, Any], prior_state: dict[str, Any]) -> str:
    payload = {
        "game_id": phase["game_id"],
        "phase_id": phase["phase_id"],
        "phase_type": phase["phase_type"],
        "aligned_start_sec": phase["aligned_start_sec"],
        "aligned_end_sec": phase["aligned_end_sec"],
        "duration_sec": phase["duration_sec"],
        "player_id": pov["player_id"],
        "video_file": pov["video_file"],
        "phase_boundary_evidence": phase.get("evidence", [])[:8],
        "prior_same_player_context": prior_state,
    }
    if phase["phase_type"] == "gameplay":
        phase_focus = "跑动、任务、同屏玩家、尸体/击杀/报案、身份/技能 UI、玩家目标和私有视觉记忆"
    elif phase["phase_type"] == "meeting":
        phase_focus = "会议发言、claim、投票、放逐、公有结果，以及这些发言如何引用前面看到的事实"
    else:
        phase_focus = "不确定或缺口阶段的可见事实核验：只记录当前视频能确认的画面/语音，并将边界和内容标为 needs_human_review"
    return f"""TASK: gameplay_phase_video_annotation
你是 GooseOmni / Theory-of-Mind benchmark 的 Qwen3-Omni 视频标注员。当前视频已经按真实游戏阶段裁剪，不是固定 90 秒窗口。

阶段重点：{phase_focus}

严格规则：
- 只根据当前 player_id 这个 POV 视频标注，不得使用其他 POV 私有信息。
- prior_same_player_context 只来自同一玩家更早 phase 的已标注私有记忆，可用于维持该玩家自己的 rolling memory；它不是全局真相，也不能引入其他 POV 私有知识。
- 必须区分 direct_visual_observation、speech_claim、public_result、inferred_belief、hidden_or_not_visible_information。
- gameplay 阶段必须优先标真实视频过程：位置、移动、任务、技能/身份 UI、同屏玩家、谁直接看见谁、尸体、击杀、报案、追逐、死亡/观战状态。
- meeting 阶段必须标发言、声称、投票、公有结果，并说明哪些内容只是 speech claim。
- role_and_goal 只能来自当前视频可见 UI 或当前玩家明确说法；不确定写 unknown。
- player_status 必须给 alive/dead/unknown；死亡或观战必须写 evidence。
- private_memory 只能写当前 POV 自己看见/听见且会影响后续怀疑、投票或行动的信息，并结合 prior_same_player_context 更新，不得泄漏其他玩家视角。
- belief_state 应体现该玩家在当前 phase 结束时基于自己视觉、听到的发言和公开结果形成的怀疑/未知/确定信息。
- tom_questions 必须至少尝试生成与视频视觉事实有关的问题；不要只生成纯会议文本题。
- 输出 strict JSON 对象，不要 Markdown。

context:
{json.dumps(payload, ensure_ascii=False, indent=2)}

输出对象必须包含：
player_status, role_and_goal, gameplay_trace, utterances, private_memory, belief_state, tom_questions, needs_human_review。

gameplay_trace 每项包含 local_start_sec, local_end_sec, event_type, location, visible_players, actor, direct_visual_observation, speech_claim, public_result, inferred_belief, hidden_or_not_visible_information, description, evidence, certainty, needs_human_review。
utterances 每项包含 local_start_sec, local_end_sec, speaker, transcript, speech_claim, claims, evidence, certainty, needs_human_review。
tom_questions 每项包含 question_type, question, answer, available_information, hidden_information, evidence, risk_of_perspective_leakage, certainty, needs_human_review。
"""


def normalize(obj: dict[str, Any], phase: dict[str, Any], pov: dict[str, Any], raw: str) -> dict[str, Any]:
    def ensure_list(key: str) -> list[Any]:
        value = obj.get(key, [])
        return value if isinstance(value, list) else []
    status = obj.get("player_status") if isinstance(obj.get("player_status"), dict) else {}
    role = obj.get("role_and_goal") if isinstance(obj.get("role_and_goal"), dict) else {}
    return {
        "dataset": "gooseomni",
        "format": "gooseomni_gameplay_phase_annotation_v1",
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
        "player_status": {
            "alive_state": status.get("alive_state", "unknown"),
            "death_abs_sec": status.get("death_abs_sec"),
            "death_evidence": status.get("death_evidence", "unknown"),
        },
        "role_and_goal": {
            "role": role.get("role", "unknown"),
            "faction": role.get("faction", "unknown"),
            "visible_goal": role.get("visible_goal", "unknown"),
            "evidence": role.get("evidence", "unknown"),
        },
        "gameplay_trace": ensure_list("gameplay_trace"),
        "utterances": ensure_list("utterances"),
        "private_memory": ensure_list("private_memory"),
        "belief_state": obj.get("belief_state") if isinstance(obj.get("belief_state"), dict) else {},
        "tom_questions": ensure_list("tom_questions"),
        "needs_human_review": bool(obj.get("needs_human_review", False)),
        "raw_response": raw,
    }




def compact_player_memory(annotation: dict[str, Any]) -> dict[str, Any]:
    def tail_list(key: str, limit: int) -> list[Any]:
        value = annotation.get(key, [])
        if not isinstance(value, list):
            return []
        compact = []
        for item in value[-limit:]:
            if isinstance(item, dict):
                compact.append({k: item.get(k) for k in item.keys() if k in {"local_start_sec", "local_end_sec", "event_type", "location", "visible_players", "actor", "description", "evidence", "certainty", "speaker", "transcript", "claims", "question_type", "question", "answer"}})
            else:
                compact.append(item)
        return compact

    return {
        "last_phase_id": annotation.get("phase_id"),
        "last_phase_type": annotation.get("phase_type"),
        "last_aligned_end_sec": annotation.get("aligned_end_sec"),
        "player_status": annotation.get("player_status", {}),
        "role_and_goal": annotation.get("role_and_goal", {}),
        "recent_private_memory": tail_list("private_memory", 8),
        "recent_gameplay_trace": tail_list("gameplay_trace", 8),
        "recent_utterances": tail_list("utterances", 8),
        "belief_state": annotation.get("belief_state", {}) if isinstance(annotation.get("belief_state"), dict) else {},
    }


def load_existing_annotation(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def mock_obj(phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_status": {"alive_state": "unknown", "death_abs_sec": None, "death_evidence": "mock"},
        "role_and_goal": {"role": "unknown", "faction": "unknown", "visible_goal": "unknown", "evidence": "mock"},
        "gameplay_trace": [],
        "utterances": [],
        "private_memory": [],
        "belief_state": {},
        "tom_questions": [],
        "needs_human_review": True,
    }


def main() -> None:
    args = parse_args()
    phases = [p for p in load_jsonl(args.phase_root / "phase_segments.jsonl") if p["game_id"] == args.game_id]
    phases.sort(key=lambda row: (float(row.get("aligned_start_sec", 0.0)), int(row.get("phase_index", 0))))
    if args.phase_id:
        phases = [p for p in phases if p["phase_id"] == args.phase_id]
    phases = phases[args.skip::args.stride]
    if args.limit is not None:
        phases = phases[:args.limit]
    backend = None if args.backend == "mock" else create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    stats = {"ok": 0, "skipped": 0, "error": 0}
    memory_by_player: dict[str, dict[str, Any]] = {}
    for phase in phases:
        for pov in phase.get("povs", []):
            if args.player_id and pov["player_id"] != args.player_id:
                continue
            out = args.output_root / "annotations" / phase["game_id"] / phase["phase_id"] / f"{pov['player_id']}.json"
            if out.exists() and args.resume and not args.overwrite:
                existing = load_existing_annotation(out)
                if existing is not None:
                    memory_by_player[pov["player_id"]] = compact_player_memory(existing)
                stats["skipped"] += 1
                continue
            prior_state = memory_by_player.get(pov["player_id"], {"status": "no_prior_same_player_context"})
            prompt = prompt_for(phase, pov, prior_state)
            raw = ""
            try:
                if args.backend == "mock":
                    obj = mock_obj(phase)
                    raw = json.dumps(obj, ensure_ascii=False)
                else:
                    raw = backend.annotate_video(args.phase_root / pov["video_file"], prompt)
                    try:
                        obj = parse_json_object(raw)
                    except Exception:
                        prompt = retry_prompt_for_compact_object(prompt)
                        raw = backend.annotate_video(args.phase_root / pov["video_file"], prompt)
                        try:
                            obj = parse_json_object(raw)
                        except Exception:
                            obj = parse_partial_json_object(raw)
                            if obj is None:
                                raise
                normalized = normalize(obj, phase, pov, raw)
                write_json(out, normalized)
                memory_by_player[pov["player_id"]] = compact_player_memory(normalized)
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                err = args.output_root / "errors" / phase["game_id"] / phase["phase_id"] / f"{pov['player_id']}.json"
                write_json(err, {"phase_id": phase["phase_id"], "player_id": pov["player_id"], "video_file": pov["video_file"], "error_message": repr(exc), "prompt": prompt, "raw_response": raw})
                stats["error"] += 1
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
