from __future__ import annotations

import numpy as np

from tools.build.extract_audio_alignment_anchors import correlate_start


def test_audio_correlation_recovers_query_position() -> None:
    rng = np.random.default_rng(7)
    query = rng.normal(size=200)
    search = np.concatenate([rng.normal(size=73), query, rng.normal(size=91)])
    index, score = correlate_start(search, query)
    assert index == 73
    assert score > 0.99
