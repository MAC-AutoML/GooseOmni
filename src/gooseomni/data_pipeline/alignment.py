from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class Anchor:
    global_abs_sec: float
    raw_sec: float
    event_type: str
    evidence_id: str
    confidence: float = 1.0


def _fit_pair(first: Anchor, second: Anchor) -> tuple[float, float] | None:
    delta = second.global_abs_sec - first.global_abs_sec
    if abs(delta) < 1e-9:
        return None
    scale = (second.raw_sec - first.raw_sec) / delta
    return scale, first.raw_sec - scale * first.global_abs_sec


def _least_squares(anchors: list[Anchor]) -> tuple[float, float]:
    mean_x = sum(row.global_abs_sec for row in anchors) / len(anchors)
    mean_y = sum(row.raw_sec for row in anchors) / len(anchors)
    variance = sum((row.global_abs_sec - mean_x) ** 2 for row in anchors)
    if variance <= 1e-9:
        raise ValueError("alignment anchors do not span time")
    covariance = sum(
        (row.global_abs_sec - mean_x) * (row.raw_sec - mean_y) for row in anchors
    )
    scale = covariance / variance
    return scale, mean_y - scale * mean_x


def fit_affine_ransac(
    anchors: list[Anchor],
    residual_threshold_sec: float = 2.0,
    iterations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit raw_sec = scale * global_abs_sec + offset and reject outliers."""
    if len(anchors) < 3:
        raise ValueError("at least three alignment anchors are required")
    rng = random.Random(seed)
    best: list[Anchor] = []
    for _ in range(iterations):
        pair = rng.sample(anchors, 2)
        fitted = _fit_pair(pair[0], pair[1])
        if fitted is None:
            continue
        scale, offset = fitted
        if not 0.95 <= scale <= 1.05:
            continue
        inliers = [
            row
            for row in anchors
            if abs(row.raw_sec - (scale * row.global_abs_sec + offset))
            <= residual_threshold_sec
        ]
        if len(inliers) > len(best):
            best = inliers
    if len(best) < 3:
        raise ValueError("fewer than three alignment anchors survived RANSAC")
    scale, offset = _least_squares(best)
    residuals = sorted(
        abs(row.raw_sec - (scale * row.global_abs_sec + offset)) for row in best
    )
    p95_index = min(len(residuals) - 1, max(0, int(0.95 * len(residuals))))
    return {
        "scale": scale,
        "offset": offset,
        "anchor_count": len(anchors),
        "inlier_count": len(best),
        "median_residual_sec": median(residuals),
        "p95_residual_sec": residuals[p95_index],
        "inlier_evidence_ids": [row.evidence_id for row in best],
    }


def raw_interval(
    mapping: dict[str, Any], start_sec: float, end_sec: float, duration_sec: float
) -> tuple[float, float]:
    scale = float(mapping["scale"])
    offset = float(mapping["offset"])
    raw_start = scale * start_sec + offset
    raw_end = scale * end_sec + offset
    if raw_start < 0 or raw_end <= raw_start or raw_end > duration_sec + 1e-6:
        raise ValueError("aligned interval maps outside the source video")
    return raw_start, raw_end


def validate_alignment(
    payload: dict[str, Any],
    players: list[str],
    median_limit: float = 1.0,
    p95_limit: float = 2.0,
    minimum_anchors: int = 3,
) -> list[str]:
    issues: list[str] = []
    if payload.get("reference_player") != "Gemini":
        issues.append("reference_player must be Gemini")
    mappings = payload.get("mappings", {})
    required_coverage_start = 0.0
    for player in players:
        row = mappings.get(player)
        if not isinstance(row, dict):
            issues.append(f"missing alignment mapping: {player}")
            continue
        if int(row.get("inlier_count", 0)) < minimum_anchors:
            issues.append(f"insufficient alignment anchors: {player}")
        if float(row.get("median_residual_sec", 999)) > median_limit:
            issues.append(f"median residual exceeds limit: {player}")
        if float(row.get("p95_residual_sec", 999)) > p95_limit:
            issues.append(f"p95 residual exceeds limit: {player}")
        if float(row.get("scale", 0)) <= 0:
            issues.append(f"invalid alignment scale: {player}")
        else:
            required_coverage_start = max(
                required_coverage_start,
                -float(row.get("offset", 0)) / float(row["scale"]),
            )
        if "hardcoded_correction" in row:
            issues.append(f"hardcoded correction is forbidden: {player}")
    recorded_coverage = float(payload.get("common_coverage_start_sec", -1.0))
    if recorded_coverage + 1e-6 < required_coverage_start:
        issues.append("common coverage start does not prevent negative raw time")
    return issues
