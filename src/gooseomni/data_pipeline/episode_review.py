from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import sha256_tree, write_json


def validate_audited_episode_payload(
    payload: dict[str, Any],
    players: list[str],
    minimum_anchors: int,
    median_limit: float,
    p95_limit: float,
) -> list[str]:
    issues: list[str] = []
    episodes = payload.get("episodes", [])
    episode_ids = {str(row.get("episode_id")) for row in episodes}
    assets = {
        str(row.get("evidence_id")): row for row in payload.get("evidence_assets", [])
    }
    for asset_id, asset in assets.items():
        if str(asset.get("player_id")) not in players:
            issues.append(f"episode evidence has non-canonical player: {asset_id}")
        if not Path(str(asset.get("frame_pack_path", ""))).is_file():
            issues.append(f"missing episode evidence asset: {asset_id}")
    reports = payload.get("alignment_report", {}).get("reports", [])
    report_by_pair = {
        (str(row.get("episode_id")), str(row.get("player_id"))): row for row in reports
    }
    for episode in episodes:
        episode_id = str(episode.get("episode_id"))
        start = float(episode.get("abs_start_sec", -1))
        end = float(episode.get("abs_end_sec", -1))
        if start < 0 or end <= start:
            issues.append(f"invalid audited episode range: {episode_id}")
        phases = episode.get("phases", [])
        if not phases or float(phases[0].get("abs_start_sec", -1)) != start:
            issues.append(
                f"audited episode phases do not start at episode start: {episode_id}"
            )
        if not phases or float(phases[-1].get("abs_end_sec", -1)) != end:
            issues.append(
                f"audited episode phases do not end at episode end: {episode_id}"
            )
        for left, right in zip(phases, phases[1:], strict=False):
            if float(left["abs_end_sec"]) != float(right["abs_start_sec"]):
                issues.append(
                    f"audited episode phases are not contiguous: {episode_id}"
                )
    for episode_id in episode_ids:
        for player in players:
            report = report_by_pair.get((episode_id, player))
            if report is None:
                issues.append(f"missing audited episode report: {episode_id}:{player}")
                continue
            evidence_ids = {str(row) for row in report.get("evidence_ids", [])}
            if len(evidence_ids) < minimum_anchors:
                issues.append(
                    f"audited episode has too few anchors: {episode_id}:{player}"
                )
            if evidence_ids - assets.keys():
                issues.append(
                    f"audited episode cites missing evidence: {episode_id}:{player}"
                )
            if float(report.get("median_residual_sec", 999)) > median_limit:
                issues.append(
                    f"audited episode median residual exceeds limit: {episode_id}:{player}"
                )
            if float(report.get("p95_residual_sec", 999)) > p95_limit:
                issues.append(
                    f"audited episode p95 residual exceeds limit: {episode_id}:{player}"
                )
    return issues


def audited_episode_stage(context: Any) -> dict[str, Any] | None:
    source = (context.config.stage_cache or {}).get("episodes")
    if source is None:
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    players = [
        player.player_id for game in context.config.games for player in game.players
    ]
    issues = validate_audited_episode_payload(
        payload,
        players,
        context.config.minimum_alignment_anchors,
        context.config.median_residual_limit_sec,
        context.config.p95_residual_limit_sec,
    )
    if issues:
        write_json(
            context.run_root / "quarantine/audited_episodes.json", {"issues": issues}
        )
        raise RuntimeError(f"audited episodes failed closed: {issues}")
    report = payload["alignment_report"]
    report["ok"] = True
    write_json(context.run_root / "artifacts/episode_alignment_report.json", report)
    write_json(context.run_root / "artifacts/episodes.json", payload["episodes"])
    return {
        "mode": "audited_cache",
        "episodes": len(payload["episodes"]),
        "output": str(context.run_root / "artifacts/episodes.json"),
        "input_hash": sha256_tree(source),
        "alignment_report": report,
    }
