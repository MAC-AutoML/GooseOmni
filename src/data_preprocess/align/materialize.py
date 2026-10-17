from __future__ import annotations

import shutil
from pathlib import Path

from data_preprocess.align.detect import (
    PROBLEM_ALREADY_IN_GAME,
    PROBLEM_NO_ENTER,
    TransitionResult,
)
from data_preprocess.common.json_utils import write_json
from data_preprocess.common.media import cut_from_sec


def write_review_artifacts(*, review_dir: Path, result: TransitionResult) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    if result.role_path is not None and Path(result.role_path).is_file():
        shutil.copy2(result.role_path, review_dir / "role.jpg")
    if result.before_path is not None and Path(result.before_path).is_file():
        shutil.copy2(result.before_path, review_dir / "before.jpg")
    enter_src = result.after_path
    if enter_src is not None and Path(enter_src).is_file():
        shutil.copy2(enter_src, review_dir / "enter.jpg")
        shutil.copy2(enter_src, review_dir / "after.jpg")
    if result.needs_review:
        write_json(
            review_dir / "problem.json",
            {
                "problem": result.problem or PROBLEM_NO_ENTER,
                "needs_review": True,
                **result.as_dict(),
            },
        )


def materialize(
    *,
    source: Path,
    output_video: Path,
    review_dir: Path,
    result: TransitionResult,
    overwrite: bool = False,
) -> Path | None:
    write_review_artifacts(review_dir=review_dir, result=result)
    if result.enter_gameplay_sec is None or (
        result.needs_review and result.problem not in {PROBLEM_ALREADY_IN_GAME}
    ):
        return None
    return cut_from_sec(
        source=source,
        output=output_video,
        enter_sec=float(result.enter_gameplay_sec),
        overwrite=overwrite,
    )
