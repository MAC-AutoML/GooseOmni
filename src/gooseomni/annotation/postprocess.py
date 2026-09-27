from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .json_utils import write_json
from .schemas import CandidateTrial, GlobalEvent, InformationState, Utterance


def load_pov_event_files(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("events", []))
    return records


def build_meeting_utterances(pov_events_dir: Path, output_path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    for event in load_pov_event_files(pov_events_dir):
        text = event.get("description", "")
        if "会议" not in text and "发言" not in text and "说" not in text:
            continue
        utterances.append(
            Utterance(
                game_id=event["game_id"],
                player_id=event["player_id"],
                clip_id=event["clip_id"],
                start_sec=event["start_sec"],
                end_sec=event["end_sec"],
                speaker_id=event["player_id"],
                text=text,
                mentioned_players=event.get("mentioned_players", []),
                speech_act=event.get("event_type"),
                confidence=event.get("confidence", 0.0),
            )
        )
    write_json(output_path, [item.model_dump() for item in utterances])
    return utterances


def build_information_states(pov_events_dir: Path, output_path: Path) -> list[InformationState]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in load_pov_event_files(pov_events_dir):
        grouped[(event["game_id"], event["player_id"])].append(event)

    states: list[InformationState] = []
    for (game_id, player_id), events in sorted(grouped.items()):
        cutoff_time = max(item["end_sec"] for item in events)
        states.append(
            InformationState(
                game_id=game_id,
                player_id=player_id,
                cutoff_time=cutoff_time,
                known_facts=[item["description"] for item in events],
                private_observations=[item["description"] for item in events],
                confidence=min(
                    1.0,
                    sum(item.get("confidence", 0.0) for item in events) / max(1, len(events)),
                ),
            )
        )
    write_json(output_path, [item.model_dump() for item in states])
    return states


def merge_global_events(pov_events_dir: Path, output_path: Path) -> list[GlobalEvent]:
    events = load_pov_event_files(pov_events_dir)
    global_events: list[GlobalEvent] = []
    for index, event in enumerate(events):
        global_events.append(
            GlobalEvent(
                game_id=event["game_id"],
                start_sec=event["start_sec"],
                end_sec=event["end_sec"],
                event_type=event["event_type"],
                description=event["description"],
                involved_players=sorted(
                    set(event.get("visible_players", []) + event.get("mentioned_players", []))
                ),
                source_clip_ids=[event["clip_id"]],
                source_player_ids=[event["player_id"]],
                confidence=event.get("confidence", 0.0),
            )
        )
    write_json(
        output_path,
        [
            {"global_event_id": f"ge_{idx:06d}", **item.model_dump()}
            for idx, item in enumerate(global_events or [], start=1)
        ],
    )
    return global_events


def build_candidate_trials(
    global_events_path: Path,
    information_states_path: Path,
    output_path: Path,
) -> list[CandidateTrial]:
    import json

    global_events = json.loads(global_events_path.read_text(encoding="utf-8"))
    information_states = json.loads(information_states_path.read_text(encoding="utf-8"))
    trials: list[CandidateTrial] = []

    for state in information_states:
        player_events = _events_for_player(global_events, state)
        hidden_events = _hidden_events_for_player(global_events, state)
        belief_events = _events_by_type(
            player_events,
            {"audio_cue", "meeting", "role_clue", "body_report", "kill", "encounter"},
        )
        intent_events = _events_by_type(player_events, {"accusation", "defense", "vote"})

        trials.append(
            _make_trial(
                index=len(trials) + 1,
                state=state,
                question_type="first_order_belief",
                question=(
                    f"在 {state['cutoff_time']} 秒时，{state['player_id']} "
                    "基于自己视角可以确认哪些信息？"
                ),
                answer="; ".join(
                    _unique_texts(
                        [event.get("description", "") for event in belief_events]
                        + state.get("known_facts", [])
                    )[:3]
                )
                or "无可靠信息",
                supporting_events=belief_events[:3] or player_events[:3],
                confidence=state.get("confidence", 0.0),
            )
        )

        if hidden_events:
            target = hidden_events[0]
            trials.append(
                _make_trial(
                    index=len(trials) + 1,
                    state=state,
                    question_type="hidden_information",
                    question=(
                        f"在 {state['cutoff_time']} 秒时，{state['player_id']} "
                        "可能还不知道哪条由其他视角观察到的信息？"
                    ),
                    answer=_short_text(target.get("description", "") or "无可靠信息"),
                    supporting_events=[target],
                    distractors=[
                        _short_text(item.get("description", ""))
                        for item in hidden_events[1:4]
                        if item.get("description")
                    ],
                    confidence=min(state.get("confidence", 0.0), target.get("confidence", 0.0)),
                )
            )

        if intent_events:
            target = intent_events[0]
            trials.append(
                _make_trial(
                    index=len(trials) + 1,
                    state=state,
                    question_type="intent_inference",
                    question=(
                        f"{state['player_id']} 的投票、怀疑或辩护主要依据是什么？"
                    ),
                    answer=_short_text(target.get("description", "") or "无可靠信息"),
                    supporting_events=[target],
                    distractors=[
                        _short_text(item.get("description", ""))
                        for item in intent_events[1:4]
                        if item.get("description")
                    ],
                    confidence=min(0.85, state.get("confidence", 0.0)),
                )
            )

    write_json(output_path, [item.model_dump() for item in trials])
    return trials


def _events_for_player(
    global_events: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        event
        for event in global_events
        if event.get("game_id") == state["game_id"]
        and event.get("end_sec", 0) <= state["cutoff_time"]
        and state["player_id"] in event.get("source_player_ids", [])
    ]


def _hidden_events_for_player(
    global_events: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    useful_types = {"kill", "body_report", "accusation", "defense", "vote", "role_clue"}
    return [
        event
        for event in global_events
        if event.get("game_id") == state["game_id"]
        and event.get("end_sec", 0) <= state["cutoff_time"]
        and state["player_id"] not in event.get("source_player_ids", [])
        and event.get("event_type") in useful_types
    ]


def _events_by_type(
    events: list[dict[str, Any]], event_types: set[str]
) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") in event_types]


def _make_trial(
    *,
    index: int,
    state: dict[str, Any],
    question_type: str,
    question: str,
    answer: str,
    supporting_events: list[dict[str, Any]],
    confidence: float,
    distractors: list[str] | None = None,
) -> CandidateTrial:
    return CandidateTrial(
        game_id=state["game_id"],
        trial_id=f"trial_{index:06d}",
        question_type=question_type,
        target_player_id=state["player_id"],
        cutoff_time=state["cutoff_time"],
        question=question,
        answer=_short_text(answer),
        distractors=_unique_texts(distractors or [])[:3],
        supporting_global_event_ids=[
            item.get("global_event_id", "") for item in supporting_events if item.get("global_event_id")
        ],
        supporting_information_state_ids=[state["player_id"]],
        difficulty="auto_initial",
        confidence=confidence,
    )


def _unique_texts(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for text in texts:
        normalized = " ".join(str(text).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _short_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"
