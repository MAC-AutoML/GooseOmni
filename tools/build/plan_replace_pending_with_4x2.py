from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPLACEABLE_PREFIXES = (
    "og-pov-",
    "og-utt-",
    "og-phase-",
    "og-up-",
    "og-glob-",
    "og-info-",
    "og-mem-",
    "og-belief-",
    "og-trial-",
    "og-chain-",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan replacement of many pending Omni Goose jobs with one 4x2 worker job."
    )
    parser.add_argument(
        "--output",
        default=Path("runs/gooseomni_oracle_pass1/pending_replace_4x2_plan.json"),
        type=Path,
    )
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def squeue_rows() -> list[dict[str, str]]:
    user = os.environ.get("USER", "")
    if not user:
        raise RuntimeError("USER environment variable is not set")
    result = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%i|%j|%T|%M|%R"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["bash", "-lc", 'squeue -u "$USER" -h -o "%i|%j|%T|%M|%R"'],
            check=True,
            capture_output=True,
            text=True,
        )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 4)
        if len(parts) == 5:
            rows.append(
                {
                    "job_id": parts[0],
                    "job_name": parts[1],
                    "state": parts[2],
                    "elapsed": parts[3],
                    "reason": parts[4],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = squeue_rows()
    replaceable = [
        row
        for row in rows
        if row["state"] == "PENDING"
        and row["job_name"].startswith(REPLACEABLE_PREFIXES)
    ]
    if args.limit is not None:
        replaceable = replaceable[: args.limit]
    payload = {
        "replaceable_pending_count": len(replaceable),
        "replaceable_job_ids": [row["job_id"] for row in replaceable],
        "replaceable_jobs": replaceable,
        "scancel_command": "scancel " + " ".join(row["job_id"] for row in replaceable)
        if replaceable
        else "",
        "submit_4x2_command": (
            ".venv/bin/python tools/annotation/submit_gooseomni_oracle_jobs.py "
            "--dataset-root data/gooseomni "
            "--annotation-root runs/gooseomni_oracle_pass1/annotations "
            "--mode submit-4x2-balanced "
            "--multi-worker-stage-plan upstream_chain,downstream_chain,upstream_chain,downstream_chain"
        ),
        "note": "This script only writes a plan. It does not cancel or submit jobs.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = args.output.with_suffix(".md")
    lines = [
        "# Pending Replacement Plan",
        "",
        f"- replaceable_pending_count: {len(replaceable)}",
        "",
        "## scancel command",
        "",
        "```bash",
        payload["scancel_command"],
        "```",
        "",
        "## submit 4x2 command",
        "",
        "```bash",
        payload["submit_4x2_command"],
        "```",
        "",
        "## first jobs",
        "",
    ]
    for row in replaceable[:80]:
        lines.append(
            f"- {row['job_id']} {row['job_name']} {row['state']} {row['reason']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "replaceable_pending_count": len(replaceable),
                "output": args.output.as_posix(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
