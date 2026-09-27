"""Unified GooseOmni command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from gooseomni.config import CONFIG, PATHS
from gooseomni.data_pipeline import DataPipelineRunner, load_data_config
from gooseomni.data_pipeline.local_review import (
    apply_local_review,
    prepare_local_review,
)
from gooseomni.data_pipeline.runner import load_run_manifest
from gooseomni.data_pipeline.v2 import validate_v2
from gooseomni.evaluation.reports import write_report
from gooseomni.evaluation.runner import (
    EvaluationConfig,
    plan_evaluation,
    run_evaluation,
)
from gooseomni.evaluation.scoring import score_evaluation
from gooseomni.models.registry import get_model_spec, list_model_specs

ANNOTATE_SCRIPTS = {
    "sync": ["tools/build/infer_sync_offsets.py"],
    "split": ["tools/build/split_raw_videos.py"],
    "run": ["tools/annotation/run_qwen_annotation.py"],
    "postprocess": [
        "tools/build/build_meeting_utterances.py",
        "tools/build/build_information_states.py",
        "tools/build/merge_global_events.py",
        "tools/build/build_candidate_trials.py",
    ],
}

BENCHMARK_SCRIPTS = {
    "build": ["tools/build/build_decrypto_style_diagnostics.py"],
    "validate": ["tools/validate/validate_benchmark_manifest.py"],
    "package": ["tools/package/package_gooseomni_benchmark_release.py"],
}

def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(PATHS.root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src, str(PATHS.root), env.get("PYTHONPATH", "")) if part
    )
    env["GOOSEOMNI_ROOT"] = str(PATHS.root)
    return env


def _run_scripts(scripts: Sequence[str], arguments: Sequence[str]) -> int:
    for script in scripts:
        command = [sys.executable, str(PATHS.root / script), *arguments]
        completed = subprocess.run(command, cwd=PATHS.root, env=_runtime_env(), check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def _run_internal_script(script: str, arguments: Sequence[str]) -> int:
    path = (PATHS.root / script).resolve()
    tools_root = (PATHS.root / "tools").resolve()
    if path.suffix != ".py" or tools_root not in path.parents or not path.is_file():
        raise ValueError(f"Internal script must be a Python file under tools/: {script}")
    return _run_scripts([str(path.relative_to(PATHS.root))], arguments)


def _models_list(as_json: bool) -> int:
    rows = [
        {
            "name": spec.name,
            "kind": spec.kind,
            "extra": spec.extra,
            "native_inputs": list(spec.native_inputs),
            "pipeline_inputs": list(spec.pipeline_inputs),
            "native_outputs": list(spec.native_outputs),
            "pipeline_outputs": list(spec.pipeline_outputs),
        }
        for spec in list_model_specs()
    ]
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        native = ",".join(row["native_inputs"])
        pipeline = ",".join(row["pipeline_inputs"])
        native_outputs = ",".join(row["native_outputs"])
        pipeline_outputs = ",".join(row["pipeline_outputs"])
        print(
            f"{row['name']:<28} {row['kind']:<6} extra={row['extra']:<10} "
            f"native={native:<24} pipeline={pipeline:<16} "
            f"native_output={native_outputs:<16} pipeline_output={pipeline_outputs}"
        )
    return 0


def _model_python(name: str) -> Path:
    configured = CONFIG.model(name).get("python_bin")
    if not configured:
        return Path(sys.executable)
    path = Path(str(configured))
    return path if path.is_absolute() else PATHS.root / path


def _models_check(name: str | None) -> int:
    specs = [get_model_spec(name)] if name else list_model_specs()
    failed = False
    for spec in specs:
        if spec.kind == "api":
            print(f"{spec.name}: api adapter configured")
            continue
        model_path = spec.model_path
        server_script = spec.resolved_server_script
        python_bin = _model_python(spec.name)
        ready = bool(
            model_path
            and model_path.exists()
            and server_script
            and server_script.is_file()
            and python_bin.is_file()
        )
        print(
            f"{spec.name}: {'ready' if ready else 'missing'} "
            f"model={model_path} server={server_script} python={python_bin}"
        )
        failed = failed or not ready
    return 1 if failed else 0


def _models_serve(name: str, host: str | None, port: int | None, extra: list[str]) -> int:
    spec = get_model_spec(name)
    if spec.kind != "local" or spec.resolved_server_script is None:
        raise ValueError(f"Model {name} does not provide a local server")
    model_config = CONFIG.model(name)
    python_bin = _model_python(name)
    if not python_bin.is_file():
        raise FileNotFoundError(f"Model {name} runtime Python is missing: {python_bin}")
    command = [
        str(python_bin),
        str(spec.resolved_server_script),
        "--host",
        host or str(model_config.get("host", "127.0.0.1")),
        "--port",
        str(port or int(model_config.get("port", 0))),
        *extra,
    ]
    return subprocess.run(command, cwd=PATHS.root, env=_runtime_env(), check=False).returncode


def _add_forwarding_domain(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    actions: Sequence[str],
) -> None:
    domain = subparsers.add_parser(name)
    action_parsers = domain.add_subparsers(dest="action", required=True)
    for action in actions:
        parser = action_parsers.add_parser(action)
        parser.add_argument("arguments", nargs=argparse.REMAINDER)


def _add_data_domain(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    data = subparsers.add_parser("data")
    actions = data.add_subparsers(dest="action", required=True)
    validate_input = actions.add_parser("validate-input")
    validate_input.add_argument("--config", required=True, type=Path)
    build = actions.add_parser("build")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--from-stage")
    build.add_argument("--to-stage")
    status = actions.add_parser("status")
    status.add_argument("--run", required=True, type=Path)
    review = actions.add_parser("review")
    review.add_argument("--run", required=True, type=Path)
    review.add_argument("--kind", required=True, choices=["trajectory", "tom"])
    review.add_argument("--mode", required=True, choices=["prepare", "apply"])
    review.add_argument("--response", type=Path)
    review.add_argument("--output", type=Path)
    review.add_argument("--force", action="store_true")
    for action in ("validate", "package"):
        stage = actions.add_parser(action)
        source = stage.add_mutually_exclusive_group(required=True)
        source.add_argument("--config", type=Path)
        source.add_argument("--run", type=Path)
        stage.add_argument("--resume", action="store_true")


def _add_eval_domain(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    evaluation = subparsers.add_parser("eval")
    actions = evaluation.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        parser = actions.add_parser(action)
        parser.add_argument("--benchmark", default="gooseomni_v2")
        parser.add_argument("--run", type=Path, default=Path("runs/eval/latest"))
        parser.add_argument("--models", nargs="+", default=["all"])
        parser.add_argument("--model", action="append", dest="model_aliases")
        parser.add_argument("--tracks", nargs="+", default=["leaderboard_core"])
        parser.add_argument("--track", action="append", dest="track_aliases")
        parser.add_argument("--modalities", nargs="+", choices=["text", "v", "a", "av"])
        parser.add_argument("--prompt-version", default="gooseomni_eval_v2")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--skip", type=int, default=0)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--resume", action="store_true")
    score = actions.add_parser("score")
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--benchmark", default="gooseomni_v2")
    report = actions.add_parser("report")
    report.add_argument("--run", required=True, type=Path)


def _evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    models = args.model_aliases or args.models
    tracks = args.track_aliases or args.tracks
    return EvaluationConfig(
        benchmark=args.benchmark,
        run_root=args.run,
        models=tuple(models),
        tracks=tuple(tracks),
        modalities=tuple(args.modalities) if args.modalities else None,
        prompt_version=args.prompt_version,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
        resume=args.resume,
        workers=args.workers,
    )


def _run_data_command(args: argparse.Namespace) -> int:
    if args.action == "status":
        print(json.dumps(load_run_manifest(args.run), ensure_ascii=False, indent=2))
        return 0
    if args.action == "review":
        if args.mode == "prepare":
            result = prepare_local_review(args.run, args.kind)
        else:
            if args.response is None:
                raise ValueError("--response is required for review apply")
            result = apply_local_review(
                args.run,
                args.kind,
                args.response,
                output_path=args.output,
                force=args.force,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.action == "validate" and args.run:
        result = validate_v2(args.run / "benchmark_staging", probe_media=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    config_path = getattr(args, "config", None)
    if config_path is None:
        manifest = load_run_manifest(args.run)
        config_path = manifest.get("config_path")
        if not config_path:
            raise ValueError("run manifest does not record config_path")
    config = load_data_config(config_path)
    if args.action == "validate-input":
        issues = config.validate_inputs(require_files=True)
        print(json.dumps({"ok": not issues, "issues": issues}, ensure_ascii=False, indent=2))
        return 1 if issues else 0
    runner = DataPipelineRunner(config)
    from_stage = getattr(args, "from_stage", None)
    to_stage = getattr(args, "to_stage", None)
    if args.action in {"validate", "package"}:
        from_stage = args.action
    result = runner.run(resume=args.resume, from_stage=from_stage, to_stage=to_stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_eval_command(args: argparse.Namespace) -> int:
    if args.action in {"plan", "run"}:
        config = _evaluation_config(args)
        result = plan_evaluation(config) if args.action == "plan" else run_evaluation(config)
    elif args.action == "score":
        result = score_evaluation(args.run, args.benchmark)
    else:
        path = write_report(args.run)
        result = {"report": str(path)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gooseomni")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    models = subparsers.add_parser("models")
    model_actions = models.add_subparsers(dest="action", required=True)
    list_parser = model_actions.add_parser("list")
    list_parser.add_argument("--json", action="store_true")
    check_parser = model_actions.add_parser("check")
    check_parser.add_argument("model", nargs="?")
    serve_parser = model_actions.add_parser("serve")
    serve_parser.add_argument("model")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    serve_parser.add_argument("arguments", nargs=argparse.REMAINDER)

    _add_forwarding_domain(subparsers, "annotate", ANNOTATE_SCRIPTS)
    _add_forwarding_domain(subparsers, "benchmark", BENCHMARK_SCRIPTS)
    _add_data_domain(subparsers)
    _add_eval_domain(subparsers)

    internal = subparsers.add_parser("internal", help=argparse.SUPPRESS)
    internal_actions = internal.add_subparsers(dest="action", required=True)
    internal_run = internal_actions.add_parser("run", help=argparse.SUPPRESS)
    internal_run.add_argument("script")
    internal_run.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.domain == "models":
        if args.action == "list":
            return _models_list(args.json)
        if args.action == "check":
            return _models_check(args.model)
        return _models_serve(args.model, args.host, args.port, args.arguments)

    if args.domain == "internal":
        forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        return _run_internal_script(args.script, forwarded)

    if args.domain == "data":
        return _run_data_command(args)
    if args.domain == "eval":
        return _run_eval_command(args)

    scripts_by_domain = {
        "annotate": ANNOTATE_SCRIPTS,
        "benchmark": BENCHMARK_SCRIPTS,
    }
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    return _run_scripts(scripts_by_domain[args.domain][args.action], forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
