from __future__ import annotations

from gooseomni.data_pipeline.perception_windows import phase_windows

EPISODES = [
    {
        "episode_id": "g001_episode_001",
        "phases": [
            {
                "phase_index": 1,
                "phase_type": "meeting",
                "abs_start_sec": 1782.5,
                "abs_end_sec": 2162.5,
            },
            {
                "phase_index": 2,
                "phase_type": "gameplay",
                "abs_start_sec": 2162.5,
                "abs_end_sec": 2202.5,
            },
            {
                "phase_index": 3,
                "phase_type": "meeting",
                "abs_start_sec": 2202.5,
                "abs_end_sec": 2482.5,
            },
        ],
    }
]


def test_v14_phase_windows_are_exact_and_never_cross_boundaries() -> None:
    rows = phase_windows(EPISODES, (2140, 2230), (), 30)
    assert [
        (row["start_sec"], row["end_sec"], row["phase_type"])
        for row in rows
    ] == [
        (2140, 2162.5, "meeting"),
        (2162.5, 2192.5, "gameplay"),
        (2192.5, 2202.5, "gameplay"),
        (2202.5, 2230, "meeting"),
    ]


def test_phase_windows_subtract_excluded_ranges_before_chunking() -> None:
    rows = phase_windows(EPISODES, (2160, 2210), ((2168, 2172),), 30)
    assert all(not (row["start_sec"] < 2172 and row["end_sec"] > 2168) for row in rows)
    assert all(row["end_sec"] - row["start_sec"] <= 30 for row in rows)
