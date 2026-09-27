from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gooseomni.benchmark.decrypto_diagnostics import (  # noqa: E402
    write_json,
    write_jsonl,
)

PROBE_FILES = [
    "probes_A_pre_reveal.jsonl",
    "probes_B_reconstruct.jsonl",
    "probes_C_false_belief.jsonl",
    "probes_D_perspective_taking.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge human_verified Decrypto probe groups from multiple passes."
    )
    parser.add_argument(
        "--base-pass-root",
        type=Path,
        required=True,
        help="Pass whose oracle_ledger will be copied.",
    )
    parser.add_argument("--input-pass-root", type=Path, action="append", required=True)
    parser.add_argument("--output-pass-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def is_human_verified(row: dict[str, Any]) -> bool:
    return (
        row.get("gold_source") == "human_verified"
        or row.get("recommended_gold_source") == "human_verified"
    )


def dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for row in rows:
        value = str(row.get(key))
        if value in seen:
            continue
        seen.add(value)
        out.append(row)
    return out


def semantic_key(
    row: dict[str, Any],
) -> tuple[str, tuple[str, ...], str, str, tuple[str, ...]]:
    query_variable = row.get("query_variable") or {}
    return (
        str(row.get("target_player")),
        tuple(str(item) for item in row.get("anchor_event_ids", [])),
        str(row.get("template")),
        str(query_variable.get("type")),
        tuple(str(item) for item in row.get("related_claim_ids", [])),
    )


def safe_id_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "x"


def collision_safe_group_id(
    original_id: str, row: dict[str, Any], used_ids: set[str]
) -> str:
    if original_id not in used_ids:
        return original_id
    anchors = (
        "_".join(str(item) for item in row.get("anchor_event_ids", [])) or "no_anchor"
    )
    template = safe_id_part(str(row.get("template", "template")))
    query_type = safe_id_part(
        str((row.get("query_variable") or {}).get("type", "query"))
    )
    candidate_base = f"{original_id}__{template}_{query_type}_{safe_id_part(anchors)}"
    candidate = candidate_base
    index = 2
    while candidate in used_ids:
        candidate = f"{candidate_base}_{index}"
        index += 1
    return candidate


def rewrite_probe_id(row: dict[str, Any], old_group_id: str, new_group_id: str) -> None:
    probe_id = row.get("probe_id")
    if not probe_id or old_group_id == new_group_id:
        return
    probe_id_text = str(probe_id)
    if probe_id_text.startswith(old_group_id):
        row["probe_id"] = f"{new_group_id}{probe_id_text[len(old_group_id) :]}"
    else:
        row["probe_id"] = f"{new_group_id}_{safe_id_part(probe_id_text)}"


def remap_group_rows(
    rows: list[dict[str, Any]], group_id_map: dict[str, str | None]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        old_group_id = str(row.get("probe_group_id"))
        new_group_id = group_id_map.get(old_group_id)
        if not new_group_id:
            continue
        next_row = dict(row)
        next_row["probe_group_id"] = new_group_id
        rewrite_probe_id(next_row, old_group_id, new_group_id)
        out.append(next_row)
    return out


def merge_human_verified_passes(
    base_pass_root: Path,
    input_pass_roots: list[Path],
    output_pass_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_pass_root.exists():
        if not overwrite:
            raise SystemExit(f"output exists: {output_pass_root}")
        shutil.rmtree(output_pass_root)

    annotation_root = output_pass_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        base_pass_root / "annotations" / "oracle_ledger",
        annotation_root / "oracle_ledger",
    )

    groups: list[dict[str, Any]] = []
    hidden_gold: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    probes_by_file: dict[str, list[dict[str, Any]]] = {
        filename: [] for filename in PROBE_FILES
    }
    review_records: list[dict[str, Any]] = []
    semantic_to_group_id: dict[
        tuple[str, tuple[str, ...], str, str, tuple[str, ...]], str
    ] = {}
    used_group_ids: set[str] = set()

    for pass_root in input_pass_roots:
        diag = pass_root / "annotations" / "diagnostics"
        pass_groups = [
            row
            for row in read_jsonl(diag / "probe_groups.jsonl")
            if is_human_verified(row)
        ]
        group_id_map: dict[str, str | None] = {}
        for row in pass_groups:
            old_group_id = str(row["probe_group_id"])
            key = semantic_key(row)
            if key in semantic_to_group_id:
                group_id_map[old_group_id] = None
                continue
            new_group_id = collision_safe_group_id(old_group_id, row, used_group_ids)
            next_row = dict(row)
            next_row["probe_group_id"] = new_group_id
            groups.append(next_row)
            group_id_map[old_group_id] = new_group_id
            semantic_to_group_id[key] = new_group_id
            used_group_ids.add(new_group_id)

        hidden_gold.extend(
            remap_group_rows(
                [
                    row
                    for row in read_jsonl(diag / "hidden_gold.jsonl")
                    if is_human_verified(row)
                ],
                group_id_map,
            )
        )
        quality.extend(
            remap_group_rows(
                [
                    row
                    for row in read_jsonl(diag / "diagnostic_quality.jsonl")
                    if row.get("probe_group_id") in group_id_map
                    and (
                        is_human_verified(row)
                        or group_id_map.get(str(row.get("probe_group_id")))
                    )
                ],
                group_id_map,
            )
        )
        for filename in PROBE_FILES:
            probes_by_file[filename].extend(
                remap_group_rows(
                    [
                        row
                        for row in read_jsonl(diag / filename)
                        if is_human_verified(row)
                    ],
                    group_id_map,
                )
            )
        review_records.extend(
            read_jsonl(pass_root / "review" / "codex_human_review_records.jsonl")
        )

    hidden_gold = dedupe(hidden_gold, "probe_group_id")
    quality = dedupe(quality, "probe_group_id")
    for filename in PROBE_FILES:
        probes_by_file[filename] = dedupe(probes_by_file[filename], "probe_id")

    diag_out = annotation_root / "diagnostics"
    write_jsonl(diag_out / "probe_groups.jsonl", groups)
    write_jsonl(diag_out / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diag_out / "diagnostic_quality.jsonl", quality)
    for filename in PROBE_FILES:
        write_jsonl(diag_out / filename, probes_by_file[filename])

    review_dir = output_pass_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(review_dir / "source_review_records.jsonl", review_records)

    counts = {
        "probe_groups": len(groups),
        "hidden_gold": len(hidden_gold),
        "quality": len(quality),
        "probes": sum(len(rows) for rows in probes_by_file.values()),
        "A": len(probes_by_file["probes_A_pre_reveal.jsonl"]),
        "B": len(probes_by_file["probes_B_reconstruct.jsonl"]),
        "C": len(probes_by_file["probes_C_false_belief.jsonl"]),
        "D": len(probes_by_file["probes_D_perspective_taking.jsonl"]),
    }
    summary = {
        "base_pass_root": base_pass_root.as_posix(),
        "input_pass_roots": [path.as_posix() for path in input_pass_roots],
        "output_pass_root": output_pass_root.as_posix(),
        "selection": "gold_source == human_verified",
        "dedupe_policy": "dedupe exact semantic key; rename colliding probe_group_id when semantics differ",
        "counts": counts,
        "probe_group_ids": [row["probe_group_id"] for row in groups],
    }
    write_json(output_pass_root / "README.json", summary)
    (output_pass_root / "README.md").write_text(
        "# GooseOmni Human-Verified Decrypto Subset\n\n"
        "This pass contains only Decrypto-style probe groups that passed the Codex-human verification gate.\n\n"
        f"- probe_groups: {counts['probe_groups']}\n"
        f"- probes: {counts['probes']}\n"
        f"- A/B/C/D: {counts['A']}/{counts['B']}/{counts['C']}/{counts['D']}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = merge_human_verified_passes(
        base_pass_root=args.base_pass_root,
        input_pass_roots=args.input_pass_root,
        output_pass_root=args.output_pass_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
