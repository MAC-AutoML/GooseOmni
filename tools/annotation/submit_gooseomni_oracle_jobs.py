#!/usr/bin/env python3
"""CLI wrapper for GooseOmni Slurm annotation orchestration."""

from gooseomni.annotation.submission_config import parse_args
from gooseomni.annotation.submission_orchestrator import main

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
