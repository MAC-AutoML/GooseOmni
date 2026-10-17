"""ffmpeg / ffprobe helpers."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

ExtractFrame = Callable[[Path, float, Path], Path | None]


def probe_duration(video_path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(video_path),
            ],
            text=True,
        ).strip()
    )


def probe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip().splitlines()[0]
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        num, den = float(num_s), float(den_s)
        if den == 0:
            raise ValueError(f"invalid frame rate: {text}")
        return num / den
    return float(text)


def sample_times(start_sec: float, end_sec: float, fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if end_sec + 1e-9 < start_sec:
        return []
    step = 1.0 / float(fps)
    times: list[float] = []
    index = 0
    while True:
        stamp = start_sec + index * step
        if stamp > end_sec + 1e-9:
            break
        times.append(round(stamp, 6))
        index += 1
        if index > 20_000:
            raise ValueError("sample_times produced too many stamps")
    if times and (end_sec - times[-1]) > 1e-4:
        times.append(round(end_sec, 6))
    if not times:
        times.append(round(start_sec, 6))
    return times


def frame_filename(sec: float) -> str:
    return f"t_{sec:.6f}.jpg"


def extract_frame(video_path: Path, sec: float, output_path: Path) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{float(sec):.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        return None
    return output_path


def cut_from_sec(
    *,
    source: Path,
    output: Path,
    enter_sec: float,
    overwrite: bool = False,
) -> Path:
    """Trim from enter_sec to end (align)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output
    if enter_sec <= 1e-6:
        if output.exists():
            output.unlink()
        try:
            output.hardlink_to(source)
        except OSError:
            shutil.copy2(source, output)
        return output
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-ss",
        f"{float(enter_sec):.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        message = (completed.stderr or completed.stdout or "ffmpeg cut failed").strip()
        raise RuntimeError(message[:400])
    return output


def cut_interval(
    *,
    source: Path,
    output: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool = True,
    reencode: bool = False,
) -> Path:
    """Cut [start_sec, end_sec) from source (split)."""
    duration = float(end_sec) - float(start_sec)
    if duration <= 0:
        raise ValueError(f"invalid interval: {start_sec} -> {end_sec}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.001, duration):.3f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
    ]
    if reencode:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"])
    else:
        command.extend(["-c", "copy"])
    command.append(str(output))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        if not reencode:
            if output.exists():
                output.unlink()
            return cut_interval(
                source=source,
                output=output,
                start_sec=start_sec,
                end_sec=end_sec,
                overwrite=True,
                reencode=True,
            )
        message = (completed.stderr or completed.stdout or "ffmpeg cut failed").strip()
        raise RuntimeError(message[:400])
    return output
