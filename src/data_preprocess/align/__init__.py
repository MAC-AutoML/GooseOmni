"""Enter-game alignment: trim each POV to first in_game frame after role reveal."""

from data_preprocess.align.detect import TransitionResult, detect_transition
from data_preprocess.align.runner import AlignConfig, run_session_align

__all__ = [
    "AlignConfig",
    "TransitionResult",
    "detect_transition",
    "run_session_align",
]
