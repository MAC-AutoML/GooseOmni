from .pipeline_normalize import *  # noqa: F401,F403
from .pipeline_normalize import _clamp_certainty

def parse_json_array(raw_response: str) -> list[dict[str, Any]]:
    payload = safe_json_loads(raw_response)
    if not isinstance(payload, list):
        raise ValueError("model response must be a JSON array")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("model response array items must be JSON objects")
    return payload


def parse_partial_json_array_objects(raw_response: str, *, max_items: int = 4) -> list[dict[str, Any]]:
    """Recover complete objects from a truncated JSON array response."""
    start = raw_response.find("[")
    if start < 0:
        return []
    items: list[dict[str, Any]] = []
    depth = 0
    object_start: int | None = None
    in_string = False
    escape = False
    for index in range(start + 1, len(raw_response)):
        char = raw_response[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and object_start is not None:
                    try:
                        parsed = json.loads(raw_response[object_start : index + 1])
                    except json.JSONDecodeError:
                        object_start = None
                        continue
                    if isinstance(parsed, dict):
                        parsed["needs_human_review"] = True
                        parsed["certainty"] = min(_clamp_certainty(parsed.get("certainty"), 0.5), 0.5)
                        items.append(parsed)
                        if len(items) >= max_items:
                            break
                    object_start = None
    return items


def retry_prompt_for_compact_json(prompt: str, *, max_items: int = 4) -> str:
    return (
        prompt
        + "\n\n重要：上一次输出可能过长或 JSON 不完整。请重新输出完整 strict JSON 数组。"
        + f"最多 {max_items} 个对象，只保留最高价值证据。"
        + "description/evidence/transcript 必须极短；supporting ids 各最多 6 个；不要 Markdown；不要解释；无法确定时输出 []。"
    )


def retry_prompt_for_compact_object(prompt: str) -> str:
    return (
        prompt
        + "\n\n重要：上一次输出可能过长或 JSON 不完整。请重新输出一个完整 strict JSON 对象。"
        + "所有列表最多 4 项；每项 content/evidence/reason 不超过 50 个中文字符；"
        + "不要 Markdown；不要解释；缺失或不确定内容写 unknown，并设置 needs_human_review=true。"
    )


def parse_partial_json_object(raw_response: str) -> dict[str, Any] | None:
    start = raw_response.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw_response)):
        char = raw_response[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw_response[start : index + 1])
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, dict):
                    parsed["needs_human_review"] = True
                    parsed["certainty"] = min(_clamp_certainty(parsed.get("certainty"), 0.5), 0.5)
                    return parsed
                return None
    return None


def parse_json_array_with_video_retry(
    *,
    backend: Any,
    video_path: Path,
    prompt: str,
    max_items: int = 4,
) -> tuple[str, list[dict[str, Any]], str]:
    raw_response = backend.annotate_video(video_path, prompt)
    try:
        return raw_response, parse_json_array(raw_response), prompt
    except Exception as first_error:  # noqa: BLE001
        retry_prompt = retry_prompt_for_compact_json(prompt, max_items=max_items)
        retry_response = backend.annotate_video(video_path, retry_prompt)
        try:
            return retry_response, parse_json_array(retry_response), retry_prompt
        except Exception as second_error:  # noqa: BLE001
            combined = (
                "FIRST_ERROR:\n"
                + repr(first_error)
                + "\n\nFIRST_RESPONSE:\n"
                + raw_response
                + "\n\nRETRY_ERROR:\n"
                + repr(second_error)
                + "\n\nRETRY_RESPONSE:\n"
                + retry_response
            )
            raise ValueError(combined) from second_error


def parse_json_object(raw_response: str) -> dict[str, Any]:
    payload = safe_json_loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def annotation_path(
    dataset_root: Path,
    stage: str,
    segment: Segment,
    player_id: str | None = None,
    annotation_root: Path | None = None,
) -> Path:
    root = (annotation_root or output_root(dataset_root)) / stage / segment.game_id
    if player_id is None:
        return root / f"{segment.segment_id}.json"
    return root / segment.segment_id / f"{player_id}.json"


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return safe_json_loads(text)
    except Exception:
        return None


def should_review_item(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    certainty = item.get("certainty")
    if isinstance(certainty, (int, float)) and certainty < 0.5:
        reasons.append("certainty_below_0.5")
    if item.get("speaker") == "unknown":
        reasons.append("speaker_unknown")
    if item.get("conflict") is True:
        reasons.append("conflict")
    if item.get("needs_human_review") is True:
        reasons.append("needs_human_review")
    if item.get("risk_of_perspective_leakage") in {"medium", "high"}:
        reasons.append("perspective_leakage_risk")
    return reasons


def append_review_items(
    dataset_root: Path,
    *,
    annotation_root: Path | None = None,
    stage: str,
    segment: Segment,
    items: Iterable[BaseModel | dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for item in items:
        payload = item.model_dump() if isinstance(item, BaseModel) else dict(item)
        reasons = should_review_item(payload)
        if reasons:
            rows.append(
                {
                    "stage": stage,
                    "game_id": segment.game_id,
                    "segment_id": segment.segment_id,
                    "reasons": reasons,
                    "item": payload,
                }
            )
    if rows:
        root = annotation_root or output_root(dataset_root)
        write_jsonl(root / "review_queue" / "items.jsonl", rows, append=True)


def filter_segments(
    segments: list[Segment],
    *,
    game_id: str | None = None,
    segment_id: str | None = None,
    limit: int | None = None,
    skip: int = 0,
    stride: int = 1,
) -> list[Segment]:
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    selected = [
        segment
        for segment in segments
        if (game_id is None or segment.game_id == game_id)
        and (segment_id is None or segment.segment_id == segment_id)
    ]
    selected = selected[skip::stride]
    return selected[:limit] if limit is not None else selected


def filter_povs(segment: Segment, player_id: str | None = None) -> list[POVRef]:
    if player_id is not None:
        validate_player_id(player_id)
    return [pov for pov in segment.povs if player_id is None or pov.player_id == player_id]


def video_path_for(dataset_root: Path, pov: POVRef) -> Path:
    return resolve_video_path(dataset_root, pov.video_file)


def annotate_text_with_segment_context(
    backend: Any,
    prompt: str,
    dataset_root: Path,
    segment: Segment,
    player_id: str | None = None,
) -> str:
    if getattr(backend, "requires_video_for_text", False):
        context_pov = None
        if player_id is not None:
            context_pov = next((pov for pov in segment.povs if pov.player_id == player_id), None)
        if context_pov is None:
            context_pov = segment.povs[0]
        return backend.annotate_video(video_path_for(dataset_root, context_pov), prompt)
    return backend.annotate_text(prompt)
