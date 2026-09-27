from __future__ import annotations

import json
from pathlib import Path

from gooseomni.annotation.backends.qwen_omni import (
    MockQwenOmniBackend,
    _aligned_window,
    _audio_windows,
)
from gooseomni.annotation.json_utils import parse_json
from gooseomni.annotation.postprocess import build_candidate_trials
from gooseomni.annotation.runner import (
    annotate_pov_events,
    filter_clips,
    normalize_pov_event_payload,
)
from gooseomni.annotation.schemas import Clip
from gooseomni.annotation.splitting import (
    SplitConfig,
    build_clip_plan,
    load_sync_offsets,
    run_split,
)
from gooseomni.annotation.sync import (
    RawVideo,
    SyncConfig,
    infer_sync_offsets,
    normalize_sync_payload,
)
from tools.build.mark_game_starts import build_offsets


def test_build_clip_plan_uses_aligned_time(tmp_path: Path) -> None:
    source = tmp_path / "data" / "raw" / "g001" / "P1.mp4"
    output = tmp_path / "data" / "processed" / "clips"
    clips = build_clip_plan(
        video_path=source,
        raw_dir=tmp_path / "data" / "raw",
        output_root=output,
        duration=200,
        offset_sec=20,
        segment_sec=90,
        overlap_sec=10,
    )

    assert [clip.clip_id for clip in clips] == [
        "g001_P1_0_90",
        "g001_P1_80_170",
        "g001_P1_160_180",
    ]
    assert clips[0].clip_path.endswith("g001/P1/g001_P1_0_90.mp4")


def test_load_sync_offsets(tmp_path: Path) -> None:
    path = tmp_path / "sync_offsets.json"
    path.write_text(
        json.dumps(
            {
                "offsets": [
                    {
                        "game_id": "g001",
                        "player_id": "P1",
                        "raw_start_sec": 12.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_sync_offsets(path) == {("g001", "P1"): 12.5}


def test_annotate_pov_events_with_mock_backend(tmp_path: Path) -> None:
    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_0_90",
        clip_path=str(tmp_path / "g001_P1_0_90.mp4"),
        start_sec=0,
        end_sec=90,
    )
    stats = annotate_pov_events(
        clips=[clip],
        backend=MockQwenOmniBackend(),
        output_dir=tmp_path / "out",
        error_dir=tmp_path / "errors",
        resume=False,
    )

    assert stats == {"ok": 1, "error": 0, "skipped": 0}
    payload = json.loads((tmp_path / "out" / "g001_P1_0_90.json").read_text())
    assert payload["events"][0]["clip_id"] == "g001_P1_0_90"


def test_filter_clips() -> None:
    clips = [
        Clip(
            game_id="g001",
            player_id="P1",
            clip_id="a",
            clip_path="a.mp4",
            start_sec=0,
            end_sec=1,
        ),
        Clip(
            game_id="g001",
            player_id="P2",
            clip_id="b",
            clip_path="b.mp4",
            start_sec=0,
            end_sec=1,
        ),
    ]

    selected = filter_clips(clips, game_id="g001", player_id="P2", limit=1)
    assert [clip.clip_id for clip in selected] == ["b"]


def test_audio_windows_cover_aligned_range_without_overlap() -> None:
    assert _audio_windows(2140.0, 2230.0, 30.0) == [
        (2140.0, 2170.0),
        (2170.0, 2200.0),
        (2200.0, 2230.0),
    ]


def test_aligned_window_reads_audio_prompt_context() -> None:
    assert _aligned_window("x aligned_time=[2140.0, 2230.0] y") == (
        2140.0,
        2230.0,
    )


def test_build_offsets_preserves_existing_value(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "g001"
    raw.mkdir(parents=True)
    (raw / "P1.mp4").write_bytes(b"")

    offsets = build_offsets(
        raw_dir=tmp_path / "data" / "raw",
        existing={
            ("g001", "P1"): {
                "game_id": "g001",
                "player_id": "P1",
                "raw_start_sec": 31.0,
            }
        },
        overwrite=False,
        include_duration=False,
    )

    assert offsets[0]["raw_start_sec"] == 31.0


def test_run_split_can_limit_manifest(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    game_dir = raw_dir / "g001"
    game_dir.mkdir(parents=True)
    (game_dir / "P1.mp4").write_bytes(b"")
    (game_dir / "P2.mp4").write_bytes(b"")

    monkeypatch.setattr("gooseomni.annotation.splitting.probe_duration", lambda path: 200.0)
    monkeypatch.setattr("gooseomni.annotation.splitting.split_clip", lambda *args, **kwargs: None)

    clips = run_split(
        SplitConfig(
            raw_dir=raw_dir,
            output_dir=tmp_path / "clips",
            manifest_path=tmp_path / "manifest.jsonl",
            game_id="g001",
            player_id="P1",
            limit_clips=1,
        )
    )

    assert len(clips) == 1
    assert clips[0].player_id == "P1"


def test_run_split_writes_time_sorted_manifest(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    game_dir = raw_dir / "g001"
    game_dir.mkdir(parents=True)
    (game_dir / "P1.mp4").write_bytes(b"")
    (game_dir / "P2.mp4").write_bytes(b"")

    monkeypatch.setattr("gooseomni.annotation.splitting.probe_duration", lambda path: 100.0)
    monkeypatch.setattr("gooseomni.annotation.splitting.split_clip", lambda *args, **kwargs: None)
    manifest = tmp_path / "manifest.jsonl"

    run_split(
        SplitConfig(
            raw_dir=raw_dir,
            output_dir=tmp_path / "clips",
            manifest_path=manifest,
            segment_sec=60,
            overlap_sec=10,
        )
    )

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert '"clip_id":"g001_P1_0_60"' in lines[0]
    assert '"clip_id":"g001_P2_0_60"' in lines[1]


def test_runner_normalizes_common_qwen_schema_drift(tmp_path: Path) -> None:
    class DriftBackend:
        def generate(self, prompt: str, video_path: str | None = None) -> str:
            return '[{"event_type":"role_clue","description":"x","confidence":"高"}]'

    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_0_90",
        clip_path=str(tmp_path / "clip.mp4"),
        start_sec=0,
        end_sec=90,
    )
    stats = annotate_pov_events(
        clips=[clip],
        backend=DriftBackend(),
        output_dir=tmp_path / "out",
        error_dir=tmp_path / "errors",
        resume=False,
    )

    assert stats["ok"] == 1


def test_runner_expands_zero_length_event_time() -> None:
    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_160_250",
        clip_path="clip.mp4",
        start_sec=160,
        end_sec=250,
    )

    normalized = normalize_pov_event_payload(
        {
            "start_sec": 160,
            "end_sec": 160,
            "event_type": "audio_cue",
            "description": "玩家开始发言。",
        },
        clip,
    )

    assert normalized["start_sec"] == 160
    assert normalized["end_sec"] == 161


def test_runner_clamps_event_time_to_clip_window() -> None:
    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_160_250",
        clip_path="clip.mp4",
        start_sec=160,
        end_sec=250,
    )

    normalized = normalize_pov_event_payload(
        {
            "start_sec": 120,
            "end_sec": 120,
            "event_type": "audio_cue",
            "description": "玩家开始发言。",
        },
        clip,
    )

    assert normalized["start_sec"] == 160
    assert normalized["end_sec"] == 161


def test_runner_joins_location_list() -> None:
    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_0_90",
        clip_path="clip.mp4",
        start_sec=0,
        end_sec=90,
    )

    normalized = normalize_pov_event_payload(
        {
            "event_type": "movement",
            "description": "玩家经过多个区域。",
            "location": ["下水道", "街道"],
        },
        clip,
    )

    assert normalized["location"] == "下水道、街道"


def test_runner_uses_clip_identity_over_model_identity() -> None:
    clip = Clip(
        game_id="g001",
        player_id="P1",
        clip_id="g001_P1_0_90",
        clip_path="clip.mp4",
        start_sec=0,
        end_sec=90,
    )

    normalized = normalize_pov_event_payload(
        {
            "clip_id": "wrong",
            "game_id": "wrong",
            "player_id": "other",
            "event_type": "audio_cue",
            "description": "其他玩家发言。",
        },
        clip,
    )

    assert normalized["clip_id"] == "g001_P1_0_90"
    assert normalized["game_id"] == "g001"
    assert normalized["player_id"] == "P1"


def test_parse_json_recovers_complete_items_from_truncated_array() -> None:
    payload = parse_json('[{"a": 1}, {"b": 2}, {"c":')
    assert payload == [{"a": 1}, {"b": 2}]


def test_parse_json_recovers_unclosed_fenced_array() -> None:
    payload = parse_json('```json\n[{"a": 1}, {"b":')
    assert payload == [{"a": 1}]


def test_build_candidate_trials_reads_json_arrays(tmp_path: Path) -> None:
    global_events = tmp_path / "global_events.json"
    states = tmp_path / "states.json"
    output = tmp_path / "trials.json"
    global_events.write_text(
        json.dumps(
            [
                {
                    "global_event_id": "ge_000001",
                    "game_id": "g001",
                    "start_sec": 0,
                    "end_sec": 1,
                    "event_type": "audio_cue",
                    "description": "P1 听到会议发言。",
                    "source_player_ids": ["P1"],
                },
                {
                    "global_event_id": "ge_000002",
                    "game_id": "g001",
                    "start_sec": 0,
                    "end_sec": 1,
                    "event_type": "kill",
                    "description": "P2 视角看到击杀。",
                    "source_player_ids": ["P2"],
                }
            ]
        ),
        encoding="utf-8",
    )
    states.write_text(
        json.dumps(
            [
                {
                    "game_id": "g001",
                    "player_id": "P1",
                    "cutoff_time": 1,
                    "known_facts": ["P1 听到会议发言。"],
                    "confidence": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )

    trials = build_candidate_trials(global_events, states, output)

    assert len(trials) == 2
    assert trials[0].supporting_global_event_ids == ["ge_000001"]
    assert trials[1].question_type == "hidden_information"
    assert trials[1].supporting_global_event_ids == ["ge_000002"]


def test_normalize_sync_payload_uses_video_identity() -> None:
    video = RawVideo(
        game_id="g001",
        player_id="P1",
        path=Path("data/raw/g001/P1.mp4"),
        duration_sec=100,
    )

    payload = normalize_sync_payload(
        {
            "game_id": "wrong",
            "player_id": "wrong",
            "raw_start_sec": 42,
            "confidence": "高",
        },
        video,
    )

    assert payload["game_id"] == "g001"
    assert payload["player_id"] == "P1"
    assert payload["confidence"] == 0.9


def test_infer_sync_offsets_writes_valid_offsets(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "g001"
    raw_dir.mkdir(parents=True)
    (raw_dir / "P1.mp4").write_bytes(b"")

    class SyncBackend:
        def generate(self, prompt: str, video_path: str | None = None) -> str:
            return (
                '{"game_id":"x","player_id":"y","raw_start_sec":12.5,'
                '"evidence":"进入第一局地图","confidence":0.8}'
            )

    monkeypatch.setattr("gooseomni.annotation.sync.probe_duration", lambda path: 90.0)
    monkeypatch.setattr(
        "gooseomni.annotation.sync.make_review_clip",
        lambda *args, **kwargs: tmp_path / "review.mp4",
    )

    stats = infer_sync_offsets(
        SyncConfig(
            raw_dir=tmp_path / "data" / "raw",
            output_path=tmp_path / "data" / "processed" / "sync_offsets.json",
            review_dir=tmp_path / "data" / "processed" / "sync_review",
            error_dir=tmp_path / "annotations" / "errors",
            backend=SyncBackend(),
            resume=False,
        )
    )

    payload = json.loads(
        (tmp_path / "data" / "processed" / "sync_offsets.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats == {"ok": 1, "error": 0, "skipped": 0}
    assert payload["offsets"][0]["game_id"] == "g001"
    assert payload["offsets"][0]["player_id"] == "P1"
    assert payload["offsets"][0]["raw_start_sec"] == 12.5
