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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.annotation.postprocess import build_candidate_trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial Theory-of-Mind candidate trials.")
    parser.add_argument("--global-events", default="data/processed/global_events.json", type=Path)
    parser.add_argument(
        "--information-states",
        default="data/processed/information_states.json",
        type=Path,
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/candidate_trials.json",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = build_candidate_trials(
        args.global_events,
        args.information_states,
        args.output_path,
    )
    print(f"candidate_trials={len(trials)}")


if __name__ == "__main__":
    main()
