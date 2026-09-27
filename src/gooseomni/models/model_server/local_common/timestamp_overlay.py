from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def create_timestamp_overlay_video(
    source: str | Path,
    output: str | Path,
    global_start_sec: float,
) -> Path:
    """Burn an auditable absolute timestamp into a visual-only inference copy."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video for timestamp overlay: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("timestamp overlay requires valid FPS and frame dimensions")
    target = Path(output)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create timestamp overlay video: {target}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            absolute_sec = global_start_sec + frame_index / fps
            label = f"ABS {absolute_sec:.2f}s"
            cv2.rectangle(frame, (8, 8), (250, 48), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index == 0:
        raise ValueError("timestamp overlay decoded no frames")
    return target


def create_timestamp_contact_sheet(
    source: str | Path,
    output: str | Path,
    global_start_sec: float,
    sample_fps: float = 1.0,
    columns: int = 5,
    frame_width: int = 384,
    source_start_sec: float = 0.0,
    source_end_sec: float | None = None,
) -> Path:
    """Create an ordered visual review sheet with absolute timestamps."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video for contact sheet: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0 or sample_fps <= 0:
        capture.release()
        raise ValueError("contact sheet requires valid FPS and frame count")
    source_duration = frame_count / source_fps
    range_start = max(0.0, source_start_sec)
    range_end = min(source_duration, source_end_sec or source_duration)
    if range_end <= range_start:
        capture.release()
        raise ValueError("contact sheet source range must be non-empty")
    duration = range_end - range_start
    frames: list[np.ndarray] = []
    sample_count = max(1, int(math.ceil(duration * sample_fps)))
    try:
        for index in range(sample_count):
            relative_sec = min((index + 0.5) / sample_fps, duration - 1 / source_fps)
            source_sec = range_start + relative_sec
            capture.set(cv2.CAP_PROP_POS_MSEC, source_sec * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            scale = frame_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (frame_width, max(1, int(frame.shape[0] * scale))),
            )
            label = f"ABS {global_start_sec + source_sec:.2f}s"
            cv2.rectangle(frame, (4, 4), (210, 38), (0, 0, 0), -1)
            cv2.putText(
                frame, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError("contact sheet decoded no frames")
    rows = math.ceil(len(frames) / columns)
    height = max(frame.shape[0] for frame in frames)
    canvas = np.zeros((rows * height, columns * frame_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        top = (index // columns) * height
        left = (index % columns) * frame_width
        canvas[top : top + frame.shape[0], left : left + frame_width] = frame
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise ValueError(f"cannot write contact sheet: {target}")
    return target


def create_affine_anchor_sheet(
    source: str | Path,
    output: str | Path,
    global_times: list[float],
    scale: float,
    offset: float,
    frame_width: int = 384,
) -> Path:
    """Render exact global anchors using raw_sec = scale * global_sec + offset."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video for affine anchors: {source}")
    frames: list[np.ndarray] = []
    try:
        for global_sec in global_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, (scale * global_sec + offset) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            resize_scale = frame_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (frame_width, max(1, int(frame.shape[0] * resize_scale))),
            )
            cv2.rectangle(frame, (4, 4), (210, 38), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"ABS {global_sec:.2f}s",
                (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != len(global_times):
        raise ValueError("affine anchor sheet could not decode every requested anchor")
    height = max(frame.shape[0] for frame in frames)
    canvas = np.zeros((height, len(frames) * frame_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        left = index * frame_width
        canvas[: frame.shape[0], left : left + frame_width] = frame
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise ValueError(f"cannot write affine anchor sheet: {target}")
    return target
