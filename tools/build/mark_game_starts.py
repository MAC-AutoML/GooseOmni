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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.annotation.schemas import SyncOffset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update first-round start offsets for raw POV videos."
    )
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument(
        "--output-path",
        default="data/processed/sync_offsets.json",
        type=Path,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing offsets with newly generated zero-offset templates.",
    )
    parser.add_argument(
        "--probe-duration",
        action="store_true",
        help="Include raw video duration metadata for manual review.",
    )
    return parser.parse_args()


def probe_duration(video_path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return float(result.stdout.strip())


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("offsets", payload)
    existing: dict[tuple[str, str], dict[str, object]] = {}
    for item in records:
        offset = SyncOffset.model_validate(item)
        existing[(offset.game_id, offset.player_id)] = dict(item)
    return existing


def build_offsets(
    raw_dir: Path,
    existing: dict[tuple[str, str], dict[str, object]],
    overwrite: bool,
    include_duration: bool,
) -> list[dict[str, object]]:
    offsets: list[dict[str, object]] = []
    for video_path in sorted(raw_dir.glob("*/*.mp4")):
        game_id = video_path.parent.name
        player_id = video_path.stem
        key = (game_id, player_id)
        if key in existing and not overwrite:
            record = existing[key]
        else:
            record = {
                "game_id": game_id,
                "player_id": player_id,
                "raw_start_sec": 0.0,
                "evidence": "TODO: mark first-round start time in this raw recording",
                "confidence": 0.0,
            }
        if include_duration:
            record["raw_duration_sec"] = probe_duration(video_path)
        offsets.append(record)
    return offsets


def main() -> None:
    args = parse_args()
    existing = load_existing(args.output_path)
    offsets = build_offsets(
        raw_dir=args.raw_dir,
        existing=existing,
        overwrite=args.overwrite,
        include_duration=args.probe_duration,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps({"offsets": offsets}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"offsets={len(offsets)} path={args.output_path}")


if __name__ == "__main__":
    main()
