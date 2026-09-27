from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path("data/gooseomni")
DEFAULT_RUN_ROOT = Path("runs/gooseomni_oracle_pass1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an Omni Goose ToM benchmark release directory."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, type=Path)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, type=Path)
    parser.add_argument("--annotation-root", default=None, type=Path)
    parser.add_argument("--benchmark-root", default=None, type=Path)
    parser.add_argument("--validation-path", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--release-name", default="GooseOmni-ToM")
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Copy aligned videos into the release.",
    )
    parser.add_argument(
        "--include-annotations",
        action="store_true",
        help="Copy structured Qwen annotation JSON files.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow packaging when completion validation is not complete.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def should_ignore_release_path(rel: Path, ignore_names: set[str]) -> bool:
    for part in rel.parts:
        if part in ignore_names:
            return True
        if part.startswith("."):
            return True
        if ".bak_" in part or part.endswith(".bak") or part.endswith(".lock"):
            return True
    return False


def copy_tree(src: Path, dst: Path, *, ignore_names: set[str] | None = None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ignore_names = ignore_names or set()
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if should_ignore_release_path(rel, ignore_names):
            continue
        copy_file(path, dst / rel)


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    lines = []
    for path in iter_files(root):
        if path.name == "SHA256SUMS":
            continue
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_readme(
    output_dir: Path,
    *,
    release_name: str,
    version: str,
    include_videos: bool,
    include_annotations: bool,
    counts: dict[str, Any],
) -> None:
    video_note = (
        "included under data/gooseomni/videos/"
        if include_videos
        else "not included; paths are preserved in metadata"
    )
    annotation_note = (
        "included under annotations/"
        if include_annotations
        else "not included except candidate_trials.jsonl"
    )
    text = f"""# {release_name} {version}

This release contains the Omni Goose GooseOmni / Theory-of-Mind benchmark export.

## Contents

- `data/gooseomni/manifest.json`: dataset-level metadata.
- `data/gooseomni/segments.jsonl`: strict aligned 6-POV segment metadata.
- `benchmark/weak/trials.jsonl`: benchmark trials across input conditions.
- `benchmark/weak/gold_weak.jsonl`: weak Qwen3-Omni gold labels and reasoning constraints.
- `benchmark/weak/metadata.json`: benchmark export metadata.
- `benchmark/human_review_queue.jsonl`: items recommended for human review.
- `benchmark/reports/`: dataset, benchmark, and annotation quality cards.
- `annotations/candidate_trials/g001_candidate_trials.jsonl`: source candidate ToM trials.
- `completion_validation.json`: completion validator output.
- `SHA256SUMS`: checksums for all packaged files.

## Counts

- segments: {counts["segments"]}
- videos: {counts["videos"]}
- candidate_trials: {counts["candidate_trials"]}
- benchmark_trials: {counts["benchmark_trials"]}
- review_queue: {counts["review_queue"]}
- validation_complete: {counts["validation_complete"]}

## Media And Annotation Scope

- aligned videos: {video_note}
- full intermediate annotations: {annotation_note}

## Time Semantics

Each segment has `aligned_start_sec` and `aligned_end_sec`. Local video time starts from zero inside each POV clip. Absolute game time is:

```text
abs_sec = aligned_start_sec + local_sec
```

The benchmark preserves this alignment so that all POV clips for a segment refer to the same absolute game interval.

## Label Status

The gold labels are weak labels generated by Qwen3-Omni and are intended for GooseOmni / Theory-of-Mind benchmark development. Rows in `human_review_queue.jsonl` should be prioritized for human verification before claiming a fully human-verified benchmark.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotation_root = args.annotation_root or args.run_root / "annotations"
    benchmark_root = args.benchmark_root or args.run_root / "benchmark"
    validation_path = (
        args.validation_path or args.run_root / "completion_validation.json"
    )

    validation = read_json(validation_path)
    if not validation.get("complete") and not args.allow_incomplete:
        raise SystemExit(f"Refusing to package incomplete benchmark: {validation_path}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    dataset_dst = args.output_dir / "data" / "gooseomni"
    for name in ["manifest.json", "segments.json", "segments.jsonl", "skipped.json"]:
        src = args.dataset_root / name
        if src.exists():
            copy_file(src, dataset_dst / name)
    if args.include_videos:
        copy_tree(args.dataset_root / "videos", dataset_dst / "videos")

    copy_tree(benchmark_root, args.output_dir / "benchmark")
    copy_file(validation_path, args.output_dir / "completion_validation.json")
    md_validation = validation_path.with_suffix(".md")
    if md_validation.exists():
        copy_file(md_validation, args.output_dir / "completion_validation.md")

    candidate_src = annotation_root / "candidate_trials" / "g001_candidate_trials.jsonl"
    copy_file(
        candidate_src,
        args.output_dir
        / "annotations"
        / "candidate_trials"
        / "g001_candidate_trials.jsonl",
    )
    if args.include_annotations:
        copy_tree(
            annotation_root,
            args.output_dir / "annotations",
            ignore_names={"errors", "raw", "logs", "__pycache__"},
        )

    manifest = read_json(args.dataset_root / "manifest.json")
    candidate_rows = read_jsonl(candidate_src)
    benchmark_rows = read_jsonl(benchmark_root / "weak" / "trials.jsonl")
    review_rows = read_jsonl(benchmark_root / "human_review_queue.jsonl")
    counts = {
        "segments": int(manifest.get("segment_count", 0)),
        "videos": int(manifest.get("video_count", 0)),
        "candidate_trials": len(candidate_rows),
        "benchmark_trials": len(benchmark_rows),
        "review_queue": len(review_rows),
        "validation_complete": bool(validation.get("complete")),
    }
    release_manifest = {
        "release_name": args.release_name,
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest.get("dataset"),
        "dataset_format": manifest.get("format"),
        "game_id": manifest.get("strict_game_id") or manifest.get("game_id"),
        "counts": counts,
        "include_videos": args.include_videos,
        "include_annotations": args.include_annotations,
        "source_paths": {
            "dataset_root": args.dataset_root.as_posix(),
            "annotation_root": annotation_root.as_posix(),
            "benchmark_root": benchmark_root.as_posix(),
            "validation_path": validation_path.as_posix(),
        },
    }
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_release_readme(
        args.output_dir,
        release_name=args.release_name,
        version=args.version,
        include_videos=args.include_videos,
        include_annotations=args.include_annotations,
        counts=counts,
    )
    write_checksums(args.output_dir)
    print(
        json.dumps(
            {"output_dir": args.output_dir.as_posix(), "counts": counts},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
