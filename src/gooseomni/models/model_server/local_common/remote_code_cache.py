from __future__ import annotations

import shutil
import os
from pathlib import Path


def sync_remote_code_cache(model_dir: str, package_name: str) -> None:
    source_root = Path(model_dir)
    if not source_root.exists():
        return

    from transformers.utils import HF_MODULES_CACHE

    cache_base = Path(os.getenv("HF_MODULES_CACHE", HF_MODULES_CACHE))
    cache_root = cache_base / "transformers_modules" / package_name
    if not cache_root.exists():
        return

    source_files = [p for p in source_root.glob("*.py") if p.is_file()]
    if not source_files:
        return

    target_dirs = [cache_root, *[p for p in cache_root.iterdir() if p.is_dir()]]
    for target_dir in target_dirs:
        for source_file in source_files:
            target_file = target_dir / source_file.name
            if not target_file.exists() or target_file.stat().st_size != source_file.stat().st_size:
                shutil.copy2(source_file, target_file)
