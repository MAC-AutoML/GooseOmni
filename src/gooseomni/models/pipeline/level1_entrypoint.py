from __future__ import annotations

import os
from pathlib import Path

from gooseomni.config.paths import PATHS
from gooseomni.config.settings import CONFIG
from gooseomni.models.pipeline.level1_pipeline import Level1Config, Level1Pipeline
from gooseomni.models.pipeline.modality import output_path_for
from gooseomni.models.pipeline.model_client import ModelClient
from gooseomni.models.utils.dataset_downloader import ensure_default_dataset_available


def default_level1_config(model_name: str) -> Level1Config:
    dataset_path_raw = os.getenv("GOOSEOMNI_LEVEL1_DATASET") or CONFIG.benchmark(
        "level1.dataset_path", ""
    )
    video_dir_raw = os.getenv("GOOSEOMNI_LEVEL1_VIDEO_DIR") or CONFIG.benchmark(
        "level1.video_dir", ""
    )
    log_dir = os.getenv("GOOSEOMNI_LEVEL1_LOG_DIR") or CONFIG.benchmark(
        "level1.log_dir", ""
    )

    dataset_path = (
        Path(dataset_path_raw)
        if dataset_path_raw
        else PATHS.data_level_1 / "dataset.json"
    )
    video_dir = Path(video_dir_raw) if video_dir_raw else PATHS.data_level_1 / "videos"
    output_path = output_path_for(1, model_name)
    log_dir = Path(log_dir) if log_dir else PATHS.results_logs

    if not dataset_path_raw and not video_dir_raw:
        ensure_default_dataset_available("level1", dataset_path, video_dir)

    return Level1Config(
        dataset_path=dataset_path,
        video_dir=video_dir,
        output_path=output_path,
        log_dir=log_dir,
        resume=False,
    )


def run_level1(omni_test: ModelClient) -> dict:
    pipeline = Level1Pipeline(omni_test, default_level1_config(omni_test.model_name))
    return pipeline.run()
