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

from gooseomni.benchmark.checkers import (
    run_claim_fact_checker,
    run_consistency_checker,
    run_evidence_checker,
    run_perspective_leakage_checker,
)
from gooseomni.benchmark.io import load_segments_jsonl, write_json
from gooseomni.benchmark.pipeline import annotation_path, filter_segments, load_json_if_exists, output_root
from gooseomni.benchmark.schema import CheckerReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Omni Goose annotation checkers.")
    parser.add_argument("--dataset-root", default="data/gooseomni", type=Path)
    parser.add_argument("--output-root", default=None, type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--checker", choices=["evidence", "claim_fact", "perspective_leakage", "consistency", "all"], default="all")
    return parser.parse_args()


def _write_report(root: Path, checker: str, segment: object, findings: list) -> None:
    report = CheckerReport(
        game_id=segment.game_id,
        segment_id=segment.segment_id,
        checker_name=checker,
        findings=findings,
        passed=not any(item.verdict == "fail" or item.leakage for item in findings),
        needs_human_review=any(item.verdict in {"unsupported", "uncertain", "fail"} or item.leakage for item in findings),
    )
    write_json(root / "checker_reports" / checker / segment.game_id / f"{segment.segment_id}.json", report)


def main() -> None:
    args = parse_args()
    root = args.output_root or output_root(args.dataset_root)
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    stats = {"reports": 0}
    for segment in filter_segments(load_segments_jsonl(segments_path), game_id=args.game_id, segment_id=args.segment_id, limit=args.limit, skip=args.skip, stride=args.stride):
        if args.checker in {"evidence", "all"}:
            findings = []
            for pov in segment.povs:
                ann = load_json_if_exists(annotation_path(args.dataset_root, "pov_events", segment, pov.player_id, root))
                if ann:
                    findings.extend(run_evidence_checker(ann))
            _write_report(root, "evidence", segment, findings)
            stats["reports"] += 1
        if args.checker in {"claim_fact", "all"}:
            findings = []
            for pov in segment.povs:
                ann = load_json_if_exists(annotation_path(args.dataset_root, "pov_events", segment, pov.player_id, root))
                if ann:
                    findings.extend(run_claim_fact_checker(ann))
            _write_report(root, "claim_fact", segment, findings)
            stats["reports"] += 1
        if args.checker in {"perspective_leakage", "all"}:
            findings = []
            for pov in segment.povs:
                belief = load_json_if_exists(annotation_path(args.dataset_root, "belief_states", segment, pov.player_id, root))
                if belief:
                    findings.extend(run_perspective_leakage_checker(belief))
            _write_report(root, "perspective_leakage", segment, findings)
            stats["reports"] += 1
        if args.checker in {"consistency", "all"}:
            ann = load_json_if_exists(annotation_path(args.dataset_root, "global_events", segment, annotation_root=root))
            findings = run_consistency_checker(ann or {})
            _write_report(root, "consistency", segment, findings)
            stats["reports"] += 1
    print(stats)


if __name__ == "__main__":
    main()

