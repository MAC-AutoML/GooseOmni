from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_preprocess.align.prompts import frame_state_prompt
from data_preprocess.common.backend import ModelBackend
from data_preprocess.common.json_utils import parse_json
from data_preprocess.common.media import (
    ExtractFrame,
    extract_frame,
    frame_filename,
    probe_fps,
    sample_times,
)

VALID_STATES = {"start_screen", "in_game", "other"}
PROBLEM_NO_ENTER = "no_enter_after_role"
PROBLEM_ALREADY_IN_GAME = "already_in_game"
PROBLEM_NO_START = "no_start_screen"
PROBLEM_EXTRACT = "extract_failed"
PROBLEM_CLASSIFY = "classify_failed"


@dataclass
class FrameLabel:
    sec: float
    path: Path
    state: str
    reason: str
    raw_response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sec": self.sec,
            "path": str(self.path),
            "state": self.state,
            "reason": self.reason,
        }


@dataclass
class TransitionResult:
    enter_gameplay_sec: float | None
    before_sec: float | None
    after_sec: float | None
    before_path: Path | None
    after_path: Path | None
    role_sec: float | None = None
    role_path: Path | None = None
    coarse_window: str | None = None
    needs_review: bool = False
    problem: str | None = None
    frame_states: list[FrameLabel] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enter_gameplay_sec": self.enter_gameplay_sec,
            "role_sec": self.role_sec,
            "role_path": None if self.role_path is None else str(self.role_path),
            "before_sec": self.before_sec,
            "after_sec": self.after_sec,
            "before_path": None if self.before_path is None else str(self.before_path),
            "after_path": None if self.after_path is None else str(self.after_path),
            "coarse_window": self.coarse_window,
            "needs_review": self.needs_review,
            "problem": self.problem,
            "frame_states": [row.as_dict() for row in self.frame_states],
        }


def parse_frame_state(raw: str) -> tuple[str, str]:
    payload = parse_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("frame state response must be a JSON object")
    state = str(payload.get("state", "")).strip()
    if state not in VALID_STATES:
        compact = raw.lower().replace(" ", "")
        if "in_game" in compact or "ingame" in compact:
            state = "in_game"
        elif "start_screen" in compact or "startscreen" in compact:
            state = "start_screen"
        else:
            state = "other"
    reason = str(payload.get("reason") or "")[:40]
    return state, reason


def classify_frame(
    backend: ModelBackend,
    start_template: Path,
    frame_path: Path,
) -> tuple[str, str, str]:
    raw = backend.generate(
        frame_state_prompt(),
        None,
        image_paths=[str(start_template), str(frame_path)],
    )
    if not raw:
        raise ValueError("empty frame-state response")
    state, reason = parse_frame_state(raw)
    return state, reason, raw


def detect_transition(
    video_path: Path,
    *,
    backend: ModelBackend,
    start_template: Path,
    work_dir: Path,
    role_search_sec: float = 30.0,
    enter_search_sec: float = 10.0,
    coarse_fps: float = 4.0,
    refine_fps: float | None = None,
    refine: bool = True,
    max_refine_fps: float = 30.0,
    extract_fn: ExtractFrame | None = None,
    fps_fn: Any = None,
) -> TransitionResult:
    """Find role-reveal first, then enter-game shortly after."""
    extractor = extract_fn or extract_frame
    labels: list[FrameLabel] = []

    def failed(problem: str, *, window: str) -> TransitionResult:
        return TransitionResult(
            enter_gameplay_sec=None,
            before_sec=None,
            after_sec=None,
            before_path=None,
            after_path=None,
            coarse_window=window,
            needs_review=True,
            problem=problem,
            frame_states=labels,
        )

    def pull(sec: float) -> FrameLabel:
        existing = next((row for row in labels if abs(row.sec - sec) < 1e-6), None)
        if existing is not None:
            return existing
        output = work_dir / frame_filename(sec)
        path = extractor(video_path, sec, output)
        if path is None:
            raise RuntimeError(f"failed to extract frame at {sec:.3f}s")
        state, reason, raw = classify_frame(backend, start_template, path)
        label = FrameLabel(
            sec=sec, path=path, state=state, reason=reason, raw_response=raw
        )
        labels.append(label)
        return label

    try:
        role = _find_first_state(
            0.0, role_search_sec, coarse_fps, want="start_screen", pull=pull
        )
    except Exception as exc:  # noqa: BLE001
        problem = PROBLEM_EXTRACT if "extract" in str(exc).lower() else PROBLEM_CLASSIFY
        return failed(problem, window=f"role-search-0-{_fmt_window(role_search_sec)}")

    if role is None:
        return failed(
            PROBLEM_NO_START,
            window=f"role-search-0-{_fmt_window(role_search_sec)}",
        )

    enter_end = role.sec + enter_search_sec
    window = f"role@{_fmt_window(role.sec)}+{_fmt_window(enter_search_sec)}"
    try:
        before, after = _scan_enter(
            start=role.sec, end=enter_end, fps=coarse_fps, role=role, pull=pull
        )
    except Exception as exc:  # noqa: BLE001
        problem = PROBLEM_EXTRACT if "extract" in str(exc).lower() else PROBLEM_CLASSIFY
        result = failed(problem, window=window)
        result.role_sec = role.sec
        result.role_path = role.path
        return result

    if after is None:
        return TransitionResult(
            enter_gameplay_sec=None,
            role_sec=role.sec,
            role_path=role.path,
            before_sec=None if before is None else before.sec,
            after_sec=None,
            before_path=None if before is None else before.path,
            after_path=None,
            coarse_window=window,
            needs_review=True,
            problem=PROBLEM_NO_ENTER,
            frame_states=labels,
        )

    if refine:
        fps = refine_fps
        if fps is None:
            probe = fps_fn or probe_fps
            fps = min(float(probe(video_path)), max_refine_fps)
        before, after = _refine_bounds(
            before=before or role, after=after, fps=fps, pull=pull
        )

    return TransitionResult(
        enter_gameplay_sec=after.sec,
        role_sec=role.sec,
        role_path=role.path,
        before_sec=before.sec,
        after_sec=after.sec,
        before_path=before.path,
        after_path=after.path,
        coarse_window=window,
        needs_review=False,
        problem=None,
        frame_states=labels,
    )


def _fmt_window(sec: float) -> str:
    if abs(sec - int(sec)) < 1e-9:
        return str(int(sec))
    return f"{sec:g}"


def _find_first_state(
    start: float, end: float, fps: float, *, want: str, pull: Any
) -> FrameLabel | None:
    for sec in sample_times(start, end, fps):
        label = pull(sec)
        if label.state == want:
            return label
    return None


def _scan_enter(
    *,
    start: float,
    end: float,
    fps: float,
    role: FrameLabel,
    pull: Any,
) -> tuple[FrameLabel | None, FrameLabel | None]:
    last_start = role
    for sec in sample_times(start, end, fps):
        label = pull(sec)
        if label.state == "start_screen":
            last_start = label
        if label.state == "in_game":
            return last_start, label
    return last_start, None


def _refine_bounds(
    *,
    before: FrameLabel,
    after: FrameLabel,
    fps: float,
    pull: Any,
) -> tuple[FrameLabel, FrameLabel]:
    last_start = before
    first_in_game = after
    for sec in sample_times(before.sec, after.sec, fps):
        if abs(sec - before.sec) < 1e-6 or abs(sec - after.sec) < 1e-6:
            continue
        label = pull(sec)
        if label.state == "start_screen":
            last_start = label
        elif label.state == "in_game":
            first_in_game = label
            break
    return last_start, first_in_game
