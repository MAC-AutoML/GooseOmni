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
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gooseomni.benchmark.backends import create_backend
from gooseomni.benchmark.io import load_segments_jsonl, write_jsonl
from gooseomni.benchmark.pipeline import (
    annotation_path,
    annotate_text_with_segment_context,
    append_review_items,
    filter_segments,
    load_json_if_exists,
    normalize_candidate_trial_payload,
    output_root,
    parse_json_array,
    parse_partial_json_array_objects,
    retry_prompt_for_compact_json,
    save_error,
)
from gooseomni.benchmark.prompts import candidate_trial_prompt
from gooseomni.benchmark.schema import CandidateTrial, VALID_PLAYERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ToM candidate trials.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--backend", choices=["mock", "qwen", "local"], default="mock")
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-url", default=None)
    return parser.parse_args()


def _load_global_events(dataset_root: Path, output_root_path: Path | None, segment: object) -> list[dict]:
    payload = load_json_if_exists(
        annotation_path(dataset_root, "global_events", segment, annotation_root=output_root_path)
    )
    return [] if payload is None else payload.get("global_events", [])


def _load_information_states(dataset_root: Path, output_root_path: Path | None, segment: object) -> list[dict]:
    rows: list[dict] = []
    for player_id in VALID_PLAYERS:
        payload = load_json_if_exists(
            annotation_path(dataset_root, "information_states", segment, player_id, output_root_path)
        )
        if payload is not None:
            rows.append(payload)
    return rows



def _candidate_segment_exists(output_path: Path, segment_id: str) -> bool:
    if not output_path.exists():
        return False
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("segment_id") == segment_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _append_candidate_trials_once(output_path: Path, segment_id: str, trials: list[CandidateTrial]) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if _candidate_segment_exists(output_path, segment_id):
            return False
        with output_path.open("a", encoding="utf-8") as handle:
            for trial in trials:
                handle.write(json.dumps(trial.model_dump(), ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
    return True

def main() -> None:
    args = parse_args()
    backend = create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    root = args.output_root or output_root(args.dataset_root)
    output_path = root / "candidate_trials" / "g001_candidate_trials.jsonl"
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for segment in filter_segments(
        load_segments_jsonl(segments_path),
        game_id=args.game_id,
        segment_id=args.segment_id,
        limit=args.limit,
        skip=args.skip,
        stride=args.stride,
    ):
        if output_path.exists() and args.resume and not args.overwrite:
            text = output_path.read_text(encoding="utf-8")
            if segment.segment_id in text:
                stats["skipped"] += 1
                continue
        prompt = candidate_trial_prompt(
            segment,
            _load_global_events(args.dataset_root, args.output_root, segment),
            _load_information_states(args.dataset_root, args.output_root, segment),
        )
        raw_response = ""
        try:
            raw_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment)
            try:
                parsed_items = parse_json_array(raw_response)
            except Exception as first_error:  # noqa: BLE001
                prompt = retry_prompt_for_compact_json(prompt, max_items=5)
                retry_response = annotate_text_with_segment_context(backend, prompt, args.dataset_root, segment)
                try:
                    parsed_items = parse_json_array(retry_response)
                    raw_response = retry_response
                except Exception as second_error:  # noqa: BLE001
                    recovered = parse_partial_json_array_objects(retry_response, max_items=5)
                    if not recovered:
                        recovered = parse_partial_json_array_objects(raw_response, max_items=5)
                    if recovered:
                        parsed_items = recovered
                        raw_response = retry_response if "{" in retry_response else raw_response
                    else:
                        raw_response = (
                            "FIRST_ERROR:\n"
                            + repr(first_error)
                            + "\n\nFIRST_RESPONSE:\n"
                            + raw_response
                            + "\n\nRETRY_ERROR:\n"
                            + repr(second_error)
                            + "\n\nRETRY_RESPONSE:\n"
                            + retry_response
                        )
                        raise ValueError(raw_response) from second_error
            trials = [
                CandidateTrial.model_validate(normalize_candidate_trial_payload(item, segment, index))
                for index, item in enumerate(parsed_items, start=1)
            ]
            appended = _append_candidate_trials_once(output_path, segment.segment_id, trials)
            if not appended:
                stats["skipped"] += 1
                continue
            append_review_items(
                args.dataset_root,
                annotation_root=args.output_root,
                stage="candidate_trials",
                segment=segment,
                items=trials,
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            save_error(
                dataset_root=args.dataset_root,
                annotation_root=args.output_root,
                stage="candidate_trials",
                segment=segment,
                prompt=prompt,
                raw_response=raw_response,
                error=exc,
            )
            stats["error"] += 1
    print(stats)


if __name__ == "__main__":
    main()
