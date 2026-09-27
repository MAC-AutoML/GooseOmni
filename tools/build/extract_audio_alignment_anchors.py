from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists()
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def decode_envelope(path: Path, sample_rate: int = 8000, envelope_hz: int = 100) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    frame_size = sample_rate // envelope_hz
    samples = samples[: len(samples) // frame_size * frame_size]
    frames = samples.reshape(-1, frame_size)
    envelope = np.sqrt(np.mean(frames * frames, axis=1))
    envelope = np.diff(envelope, prepend=envelope[0])
    scale = np.std(envelope)
    return (envelope - np.mean(envelope)) / (scale if scale > 1e-8 else 1.0)


def correlate_start(search: np.ndarray, query: np.ndarray) -> tuple[int, float]:
    size = 1 << (len(search) + len(query) - 1).bit_length()
    spectrum = np.fft.rfft(search, size) * np.fft.rfft(query[::-1], size)
    correlation = np.fft.irfft(spectrum, size)[: len(search) + len(query) - 1]
    valid = correlation[len(query) - 1 : len(search)]
    index = int(np.argmax(valid))
    denominator = np.linalg.norm(search[index : index + len(query)]) * np.linalg.norm(query)
    score = float(valid[index] / denominator) if denominator > 1e-8 else 0.0
    return index, score


def local_anchors(
    reference: np.ndarray,
    target: np.ndarray,
    reference_zero_sec: float,
    envelope_hz: int = 100,
    spacing_sec: float = 600,
    query_sec: float = 40,
    search_radius_sec: float = 900,
    approximate_delta_sec: float | None = None,
) -> list[dict[str, object]]:
    if approximate_delta_sec is None:
        bootstrap_start = min(
            max(0, int((reference_zero_sec + 300) * envelope_hz)),
            max(0, len(reference) - int(300 * envelope_hz)),
        )
        bootstrap_end = min(len(reference), bootstrap_start + int(300 * envelope_hz))
        bootstrap_query = reference[bootstrap_start:bootstrap_end]
        bootstrap_index, _ = correlate_start(target, bootstrap_query)
        approximate_delta_sec = (bootstrap_index - bootstrap_start) / envelope_hz
    approximate_delta = approximate_delta_sec
    anchors = []
    half_query = int(query_sec * envelope_hz / 2)
    radius = int(search_radius_sec * envelope_hz)
    centers = np.arange(reference_zero_sec, len(reference) / envelope_hz, spacing_sec)
    for index, center_sec in enumerate(centers):
        center = int(center_sec * envelope_hz)
        left = center - half_query
        right = center + half_query
        if left < 0 or right > len(reference):
            continue
        expected = int((center_sec + approximate_delta) * envelope_hz)
        search_left = max(0, expected - radius - half_query)
        search_right = min(len(target), expected + radius + half_query)
        search = target[search_left:search_right]
        query = reference[left:right]
        if len(search) <= len(query):
            continue
        local, score = correlate_start(search, query)
        target_center = (search_left + local + half_query) / envelope_hz
        anchors.append(
            {
                "global_abs_sec": float(center_sec - reference_zero_sec),
                "raw_sec": target_center,
                "event_type": "shared_audio",
                "evidence_id": f"shared_audio_{index:03d}",
                "confidence": max(0.0, min(1.0, score)),
                "correlation_score": score,
                "reference_raw_sec": float(center_sec),
            }
        )
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cross-POV shared-audio anchors.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-zero-sec", type=float, required=True)
    parser.add_argument("--offset-hints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    players = payload["games"][0]["players"]
    paths = {
        str(row["player_id"]): (ROOT / str(row["video_path"])).resolve()
        for row in players
    }
    reference = decode_envelope(paths["Gemini"])
    hints: dict[str, float] = {}
    if args.offset_hints:
        hint_payload = json.loads(args.offset_hints.read_text(encoding="utf-8"))
        hints = {
            str(row["player_id"]): float(row["raw_start_sec"])
            for row in hint_payload.get("offsets", [])
        }
    anchors = {
        "Gemini": [
            {
                "global_abs_sec": float(center - args.reference_zero_sec),
                "raw_sec": float(center),
                "event_type": "reference_clock",
                "evidence_id": f"reference_clock_{index:03d}",
                "confidence": 1.0,
            }
            for index, center in enumerate(
                np.arange(args.reference_zero_sec, len(reference) / 100, 600)
            )
        ]
    }
    for player, path in paths.items():
        if player == "Gemini":
            continue
        anchors[player] = local_anchors(
            reference,
            decode_envelope(path),
            args.reference_zero_sec,
            search_radius_sec=60 if player in hints else 900,
            approximate_delta_sec=(
                hints[player] - args.reference_zero_sec if player in hints else None
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(anchors, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
