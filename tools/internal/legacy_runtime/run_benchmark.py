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
from dataclasses import replace
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline import Level1Pipeline, default_level1_config
from gooseomni.models.model_server.clients import CLIENTS, create_client


def _resolve_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    model_name = CONFIG.benchmark("level1.model", "")
    if model_name:
        return model_name
    raise SystemExit("Missing model name. Use --model or set benchmark.level1.model in config.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GooseOmni Benchmark Runner")
    parser.add_argument("--model", choices=sorted(CLIENTS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    model_name = _resolve_model_name(args)
    test = create_client(model_name)

    config = default_level1_config(test.model_name)
    if config.output_path.exists():
        answer = input(f"Existing results found: {config.output_path}. Resume? (y/N) ").strip().lower()
        if answer in {"y", "yes"}:
            config = replace(config, resume=True)
    if args.max_samples is not None or args.start_index:
        config = replace(
            config,
            max_samples=args.max_samples,
            start_index=args.start_index,
        )

    pipeline = Level1Pipeline(test, config)
    pipeline.run()


if __name__ == "__main__":
    main()
