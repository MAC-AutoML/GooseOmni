from __future__ import annotations

from statistics import median
from typing import Any

START_TYPES = {"game_start"}
END_TYPES = {"game_end", "result", "victory", "defeat"}


def consensus_boundaries(
    rows: list[dict[str, Any]], tolerance_sec: float = 3.0, minimum_povs: int = 2
) -> list[dict[str, Any]]:
    """Cluster semantic boundaries without trusting model-provided round indexes."""
    ordered = sorted(rows, key=lambda row: float(row["aligned_start_sec"]))
    clusters: list[list[dict[str, Any]]] = []
    for row in ordered:
        timestamp = float(row["aligned_start_sec"])
        matching = next(
            (
                group
                for group in reversed(clusters)
                if group[0]["boundary_type"] == row["boundary_type"]
                and abs(float(group[-1]["aligned_start_sec"]) - timestamp)
                <= tolerance_sec
            ),
            None,
        )
        if matching is None:
            matching = []
            clusters.append(matching)
        matching.append(row)
    consensus = []
    for index, group in enumerate(clusters):
        povs = sorted({str(row["player_id"]) for row in group})
        public_ui = any(bool(row.get("public_ui")) for row in group)
        if len(povs) < minimum_povs and not public_ui:
            continue
        consensus.append(
            {
                "boundary_id": f"boundary_{index:04d}",
                "boundary_type": group[0]["boundary_type"],
                "abs_sec": median(float(row["aligned_start_sec"]) for row in group),
                "source_povs": povs,
                "evidence_ids": sorted(
                    {str(row.get("evidence_id", row["clip_id"])) for row in group}
                ),
            }
        )
    return consensus


def build_episodes(
    game_id: str, boundaries: list[dict[str, Any]], session_end_sec: float
) -> list[dict[str, Any]]:
    starts = [row for row in boundaries if row["boundary_type"] in START_TYPES]
    if not starts:
        raise ValueError("no consensus game_start boundary")
    episodes = []
    for index, start in enumerate(starts):
        next_start = starts[index + 1]["abs_sec"] if index + 1 < len(starts) else None
        candidates = [
            row
            for row in boundaries
            if row["boundary_type"] in END_TYPES
            and row["abs_sec"] > start["abs_sec"]
            and (next_start is None or row["abs_sec"] < next_start)
        ]
        end_sec = candidates[0]["abs_sec"] if candidates else next_start or session_end_sec
        if end_sec <= start["abs_sec"]:
            continue
        phases = _phases(start["abs_sec"], end_sec, boundaries)
        episodes.append(
            {
                "episode_id": f"{game_id}_episode_{index:03d}",
                "game_id": game_id,
                "abs_start_sec": start["abs_sec"],
                "abs_end_sec": end_sec,
                "start_evidence_ids": start["evidence_ids"],
                "phases": phases,
            }
        )
    return episodes


def validate_episode_anchors(
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    players: list[str],
    minimum_anchors: int = 3,
    median_limit: float = 1.0,
    p95_limit: float = 2.0,
) -> dict[str, Any]:
    """Measure per-POV semantic boundary residuals against cross-POV consensus."""
    issues: list[str] = []
    reports = []
    consensus = consensus_boundaries(rows)
    for episode in episodes:
        start = float(episode["abs_start_sec"])
        end = float(episode["abs_end_sec"])
        anchors = [row for row in consensus if start <= float(row["abs_sec"]) <= end]
        for player in players:
            residuals = []
            evidence_ids = []
            local = [
                row
                for row in rows
                if str(row.get("player_id")) == player
                and start <= float(row.get("aligned_start_sec", -1)) <= end
            ]
            for anchor in anchors:
                matches = [
                    row
                    for row in local
                    if row.get("boundary_type") == anchor["boundary_type"]
                    and abs(float(row["aligned_start_sec"]) - float(anchor["abs_sec"]))
                    <= 3.0
                ]
                if matches:
                    chosen = min(
                        matches,
                        key=lambda row: abs(
                            float(row["aligned_start_sec"]) - float(anchor["abs_sec"])
                        ),
                    )
                    residuals.append(
                        abs(float(chosen["aligned_start_sec"]) - float(anchor["abs_sec"]))
                    )
                    evidence_ids.append(str(chosen.get("evidence_id", chosen["clip_id"])))
            ordered = sorted(residuals)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
            med = median(ordered) if ordered else None
            report = {
                "episode_id": episode["episode_id"],
                "player_id": player,
                "anchor_count": len(ordered),
                "median_residual_sec": med,
                "p95_residual_sec": p95,
                "evidence_ids": evidence_ids,
            }
            reports.append(report)
            if len(ordered) < minimum_anchors:
                issues.append(
                    f"{episode['episode_id']}:{player} has fewer than {minimum_anchors} anchors"
                )
            elif med is not None and med > median_limit:
                issues.append(f"{episode['episode_id']}:{player} median residual exceeds limit")
            elif p95 is not None and p95 > p95_limit:
                issues.append(f"{episode['episode_id']}:{player} p95 residual exceeds limit")
    return {"ok": not issues, "issues": issues, "reports": reports}


def _phases(
    start_sec: float, end_sec: float, boundaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    relevant = [
        row
        for row in boundaries
        if start_sec < float(row["abs_sec"]) < end_sec
        and row["boundary_type"]
        in {"meeting_start", "meeting_or_alert", "return_to_game", "vote_result"}
    ]
    points = [(start_sec, "gameplay")]
    for row in sorted(relevant, key=lambda item: item["abs_sec"]):
        phase = "gameplay" if row["boundary_type"] == "return_to_game" else "meeting"
        points.append((float(row["abs_sec"]), phase))
    points.append((end_sec, "end"))
    return [
        {
            "phase_index": index,
            "phase_type": points[index][1],
            "abs_start_sec": points[index][0],
            "abs_end_sec": points[index + 1][0],
        }
        for index in range(len(points) - 1)
        if points[index + 1][0] > points[index][0]
    ]
