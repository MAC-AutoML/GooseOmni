from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .provenance import sha256_file, stable_hash, write_json
from .v2 import read_jsonl, write_jsonl

REVIEW_KINDS = {"trajectory", "tom"}


def _static_candidate_root(root: Path) -> Path:
    nested = root / "static_trials"
    return nested if nested.is_dir() else root


def _review_transport_schema(id_key: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        id_key: {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["accept", "repair", "reject"],
                        },
                        "reason": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "final_value": {"type": ["string", "null"]},
                    },
                    "required": [
                        id_key,
                        "decision",
                        "reason",
                        "evidence_ids",
                        "final_value",
                    ],
                },
            }
        },
        "required": ["decisions"],
    }


def _static_review_rows(candidate_root: Path) -> list[dict[str, Any]]:
    root = _static_candidate_root(candidate_root)
    trials = {str(row["trial_id"]): row for row in read_jsonl(root / "trials.jsonl")}
    gold = {str(row["trial_id"]): row for row in read_jsonl(root / "gold.jsonl")}
    hidden = {
        str(row["trial_id"]): row for row in read_jsonl(root / "hidden_gold.jsonl")
    }
    if set(trials) != set(gold) or set(trials) != set(hidden):
        raise ValueError("static review inputs must have identical trial IDs")
    rows = []
    for trial_id in sorted(trials):
        evidence = sorted(
            {
                str(value)
                for source in (gold[trial_id], hidden[trial_id])
                for value in source.get("acceptable_evidence_ids", [])
            }
        )
        rows.append(
            {
                "trial_id": trial_id,
                "public_trial": trials[trial_id],
                "gold": gold[trial_id],
                "hidden_gold": hidden[trial_id],
                "available_evidence_ids": evidence,
            }
        )
    return rows


def prepare_review_shards(
    candidate_root: Path,
    review_root: Path,
    shard_size: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    rows = _static_review_rows(candidate_root)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must not be negative")
        rows = rows[:limit]
    review_root.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        shard_root = review_root / f"shard_{index:05d}"
        shard_root.mkdir(parents=True, exist_ok=True)
        input_path = shard_root / "input.jsonl"
        schema_path = shard_root / "response_schema.json"
        prompt_path = shard_root / "prompt.txt"
        output_path = shard_root / "response.json"
        write_jsonl(input_path, shard_rows)
        write_json(schema_path, _review_transport_schema("trial_id"))
        prompt_path.write_text(
            "你是 GooseOmni 严格静态 trial 审核器。为每个 trial_id 恰好输出一个决定。"
            "evidence_ids 只能来自 available_evidence_ids；accept/reject 的 final_value "
            "为 null；repair 的 final_value 为完整修复对象的 JSON 字符串。证据不足必须 "
            "reject。\n\n审核队列 JSONL：\n"
            + "\n".join(json.dumps(row, ensure_ascii=False) for row in shard_rows)
            + "\n",
            encoding="utf-8",
        )
        shards.append(
            {
                "index": index,
                "count": len(shard_rows),
                "input": str(input_path),
                "input_sha256": sha256_file(input_path),
                "prompt": str(prompt_path),
                "schema": str(schema_path),
                "output": str(output_path),
            }
        )
    manifest = {
        "candidate_root": str(candidate_root),
        "candidate_count": len(rows),
        "shard_size": shard_size,
        "shards": shards,
    }
    write_json(review_root / "manifest.json", manifest)
    return manifest


def _model_verified(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            **{key: _model_verified(item) for key, item in value.items()},
            "gold_source": "model_verified",
        }
    if isinstance(value, list):
        return [_model_verified(item) for item in value]
    return value


def merge_review_shards(review_root: Path, output_path: Path) -> dict[str, Any]:
    manifest_path = review_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [
        row["output"] for row in manifest["shards"] if not Path(row["output"]).is_file()
    ]
    if missing:
        return {"complete": False, "validated": 0, "missing_outputs": missing}
    merged = []
    for shard in manifest["shards"]:
        candidates = {
            str(row["trial_id"]): row for row in read_jsonl(Path(shard["input"]))
        }
        payload = json.loads(Path(shard["output"]).read_text(encoding="utf-8"))
        decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(decisions, list):
            raise ValueError("review shard response must contain decisions array")
        ids = [str(row.get("trial_id")) for row in decisions]
        if len(ids) != len(set(ids)) or set(ids) != set(candidates):
            raise ValueError("review shard must cover every trial exactly once")
        for decision in decisions:
            trial_id = str(decision["trial_id"])
            action = str(decision.get("decision"))
            if action not in {"accept", "repair", "reject"}:
                raise ValueError(f"invalid review decision: {trial_id}")
            evidence = [str(value) for value in decision.get("evidence_ids", [])]
            allowed = set(candidates[trial_id]["available_evidence_ids"])
            if set(evidence) - allowed:
                raise ValueError(
                    f"review cites evidence outside allow-list: {trial_id}"
                )
            raw_final = decision.get("final_value")
            final = None
            if action == "repair":
                if not isinstance(raw_final, str):
                    raise ValueError(
                        f"repair requires JSON-string final_value: {trial_id}"
                    )
                try:
                    final = json.loads(raw_final)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid repair JSON: {trial_id}") from exc
                if not isinstance(final, dict):
                    raise ValueError(f"repair must decode to object: {trial_id}")
                final = _model_verified(final)
                for key in ("public_trial", "gold", "hidden_gold"):
                    row = final.get(key)
                    if isinstance(row, dict) and str(row.get("trial_id")) != trial_id:
                        raise ValueError(f"repair changed trial_id: {trial_id}")
            elif raw_final is not None:
                raise ValueError(f"non-repair final_value must be null: {trial_id}")
            merged.append(
                {
                    "trial_id": trial_id,
                    "decision": action,
                    "reason": str(decision.get("reason", "")),
                    "evidence_ids": evidence,
                    "final_value": final,
                }
            )
    write_jsonl(output_path, merged)
    summary = {
        "complete": True,
        "validated": len(merged),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json(review_root / "merge_summary.json", summary)
    return summary


def _review_paths(run_root: Path, kind: str) -> tuple[Path, Path, str]:
    if kind == "trajectory":
        return (
            run_root / "local_codex/trajectory_review_queue.jsonl",
            run_root / "local_codex/trajectory_decisions.jsonl",
            "trajectory_node_id",
        )
    if kind == "tom":
        return (
            run_root / "local_codex/review_queue.jsonl",
            run_root / "local_codex/decisions.jsonl",
            "probe_group_id",
        )
    raise ValueError(f"unknown local review kind: {kind}")


def _image_paths(rows: list[dict[str, Any]], kind: str) -> list[str]:
    values: set[str] = set()
    for row in rows:
        candidates = (
            [row.get("frame_pack_path")]
            if kind == "trajectory"
            else row.get("frame_pack_paths", [])
        )
        values.update(str(path) for path in candidates if path)
    missing = [path for path in values if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing review frame packs: {missing[:3]}")
    return sorted(values)


def _response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["accept", "repair", "reject"],
                        },
                        "reason": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "final_value": {
                            "type": ["string", "null"],
                        },
                    },
                    "required": [
                        "id",
                        "decision",
                        "reason",
                        "evidence_ids",
                        "final_value",
                    ],
                },
            }
        },
        "required": ["decisions"],
    }


def _prompt(rows: list[dict[str, Any]], kind: str) -> str:
    identity = "trajectory_node_id" if kind == "trajectory" else "probe_group_id"
    specific = (
        "逐项查看附带关键帧。私有 POV 事实不得变成 global fact；任务 UI 不得修成 "
        "movement；不确定的玩家、地点和任务写 unknown 或 reject。"
        if kind == "trajectory"
        else "按 probe group 一次裁决并原子应用 A/B/C/D。review-only hidden evidence "
        "只用于核验 gold；certainty=unknown、未来证据不足或语义含糊时 reject。"
    )
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    return f"""你是 GooseOmni 严格数据审核器。只输出符合 schema 的 JSON。
必须为下面每个 {identity} 恰好输出一个决定，id 使用原值，不得遗漏、重复或新增。
decision 只能是 accept、repair、reject；evidence_ids 只能来自该项 available_evidence_ids。
accept/reject 的 final_value 必须为 null；repair 的 final_value 必须是包含完整且最小
修复对象的 JSON 字符串（不是直接嵌套对象）。
不得把模型推测写成事实，证据不足必须 reject。{specific}

审核队列 JSONL：
{payload}
"""


def prepare_local_review(
    run_root: Path, kind: str, bundle_root: Path | None = None
) -> dict[str, Any]:
    queue_path, _, id_key = _review_paths(run_root, kind)
    rows = read_jsonl(queue_path)
    if not rows:
        raise ValueError(f"empty local review queue: {queue_path}")
    identifiers = [str(row.get(id_key)) for row in rows]
    if len(identifiers) != len(set(identifiers)) or "None" in identifiers:
        raise ValueError("review queue contains duplicate or missing IDs")
    target = bundle_root or run_root / f"local_codex/automation/{kind}"
    target.mkdir(parents=True, exist_ok=True)
    prompt_path = target / "prompt.txt"
    schema_path = target / "response_schema.json"
    images_path = target / "image_paths.txt"
    prompt_path.write_text(_prompt(rows, kind), encoding="utf-8")
    write_json(schema_path, _response_schema())
    images = _image_paths(rows, kind)
    images_path.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")
    manifest = {
        "kind": kind,
        "queue_path": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "queue_items": len(rows),
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "image_paths": str(images_path),
        "image_count": len(images),
    }
    write_json(target / "bundle_manifest.json", manifest)
    return manifest


def apply_local_review(
    run_root: Path,
    kind: str,
    response_path: Path,
    output_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    queue_path, default_output, id_key = _review_paths(run_root, kind)
    target = output_path or default_output
    if target.exists() and not force:
        raise FileExistsError(f"refusing to overwrite local decisions: {target}")
    queue = read_jsonl(queue_path)
    candidates = {str(row[id_key]): row for row in queue}
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("Codex response must contain a decisions array")
    response_ids = [str(row.get("id")) for row in decisions]
    if len(response_ids) != len(set(response_ids)) or set(response_ids) != set(
        candidates
    ):
        raise ValueError("Codex response must cover every queue item exactly once")
    validated = []
    for row in decisions:
        identifier = str(row["id"])
        decision = str(row.get("decision"))
        if decision not in {"accept", "repair", "reject"}:
            raise ValueError(f"invalid decision: {identifier}")
        evidence = [str(value) for value in row.get("evidence_ids", [])]
        allowed = {
            str(value)
            for value in candidates[identifier].get("available_evidence_ids", [])
        }
        if set(evidence) - allowed:
            raise ValueError(
                f"decision cites evidence outside allow-list: {identifier}"
            )
        final_value_raw = row.get("final_value")
        final_value = None
        if decision == "repair":
            if not isinstance(final_value_raw, str):
                raise ValueError(
                    f"repair requires JSON-string final_value: {identifier}"
                )
            try:
                final_value = json.loads(final_value_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"repair final_value is not valid JSON: {identifier}"
                ) from exc
            if not isinstance(final_value, dict):
                raise ValueError(
                    f"repair final_value must decode to object: {identifier}"
                )
        if decision != "repair" and final_value_raw is not None:
            raise ValueError(f"non-repair final_value must be null: {identifier}")
        validated.append(
            {
                id_key: identifier,
                "decision": decision,
                "reason": str(row.get("reason", "")),
                "evidence_ids": evidence,
                "final_value": final_value,
            }
        )
    write_jsonl(target, validated)
    audit = {
        "kind": kind,
        "model": "local-codex",
        "queue_sha256": sha256_file(queue_path),
        "response_sha256": sha256_file(response_path),
        "decisions_sha256": sha256_file(target),
        "decision_count": len(validated),
        "decision_counts": {
            action: sum(row["decision"] == action for row in validated)
            for action in ("accept", "repair", "reject")
        },
        "completed_unix": time.time(),
        "content_hash": stable_hash(validated),
    }
    write_json(target.with_suffix(".audit.json"), audit)
    return {**audit, "output": str(target)}
