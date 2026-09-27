import json
from pathlib import Path

from tools.validate.rebuild_benchmark_manifest import build_manifest
from tools.validate.validate_benchmark_manifest import validate_manifest


def test_manifest_counts_and_public_leak_scan(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    paths = {
        "public/leaderboard_core/trials.jsonl": 889,
        "private/leaderboard_core/hidden_gold.jsonl": 889,
        "public/raw_video_smoke/raw_video_smoke.jsonl": 48,
        "public/leaderboard_core/probe_groups_public.jsonl": 276,
        "public/agentic_midgame_prediction/trials.jsonl": 160,
        "private/agentic_midgame_prediction/hidden_gold.jsonl": 160,
    }
    for relpath, count in paths.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" * count, encoding="utf-8")

    manifest = build_manifest(root)
    assert manifest["validation"]["line_counts"] == paths
    assert manifest["validation"]["public_file_leak_hits"] == []


def test_manifest_detects_private_keys_in_public_json(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    for relpath in (
        "public/leaderboard_core/trials.jsonl",
        "private/leaderboard_core/hidden_gold.jsonl",
        "public/raw_video_smoke/raw_video_smoke.jsonl",
        "public/leaderboard_core/probe_groups_public.jsonl",
        "public/agentic_midgame_prediction/trials.jsonl",
        "private/agentic_midgame_prediction/hidden_gold.jsonl",
    ):
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    leak = root / "public/leak.json"
    leak.write_text(json.dumps({"hidden_gold": {"answer": "secret"}}), encoding="utf-8")

    manifest = build_manifest(root)
    assert manifest["validation"]["public_file_leak_hits"]


def test_published_manifest_validation(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    counts = (889, 889, 48, 276, 160, 160)
    paths = (
        "public/leaderboard_core/trials.jsonl",
        "private/leaderboard_core/hidden_gold.jsonl",
        "public/raw_video_smoke/raw_video_smoke.jsonl",
        "public/leaderboard_core/probe_groups_public.jsonl",
        "public/agentic_midgame_prediction/trials.jsonl",
        "private/agentic_midgame_prediction/hidden_gold.jsonl",
    )
    for relpath, count in zip(paths, counts, strict=True):
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" * count, encoding="utf-8")
    manifest = build_manifest(root)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_manifest(root)["ok"] is True

    (root / paths[0]).write_text("{}\n", encoding="utf-8")
    report = validate_manifest(root)
    assert report["ok"] is False
    assert "benchmark line counts do not match the release contract" in report["issues"]
