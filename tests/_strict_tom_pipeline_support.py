from __future__ import annotations

import json

import pytest

from gooseomni.data_pipeline.aligned_clips import valid_clip
from gooseomni.data_pipeline.alignment import (
    Anchor,
    fit_affine_ransac,
    raw_interval,
    validate_alignment,
)
from gooseomni.data_pipeline.episode_review import validate_audited_episode_payload
from gooseomni.data_pipeline.episodes import (
    build_episodes,
    consensus_boundaries,
    validate_episode_anchors,
)
from gooseomni.data_pipeline.information_state import (
    build_information_state,
    observed_cutoffs,
    validate_claim_hearing,
)
from gooseomni.data_pipeline.pilot_stages import _read_records
from gooseomni.data_pipeline.pilot_validation import validate_pilot
from gooseomni.data_pipeline.reviewed_trials import accepted_trials
from gooseomni.data_pipeline.tom_trials import (
    TOM_LAYERS,
    build_trial_group,
    evidence_for_layer,
    structured_gold,
    subject_only_evidence,
    validate_trials,
)
from gooseomni.data_pipeline.trajectory import (
    fuse_trajectory,
    normalize_audio_event,
    normalize_visual_event,
)
from gooseomni.data_pipeline.trajectory_review import (
    _apply_decisions,
    load_visual_error_candidates,
    recoverable_visual_errors,
    review_and_write_trajectory,
)

# ruff: noqa: F401


__all__ = [name for name in globals() if not name.startswith("__")]
