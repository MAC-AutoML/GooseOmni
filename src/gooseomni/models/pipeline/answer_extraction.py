from __future__ import annotations

import re
from typing import Iterable


def extract_choice(text: str, choices: Iterable[str]) -> str:
    valid = {c.upper() for c in choices}
    raw = (text or "").strip()
    if not raw:
        return ""

    answer_text = _final_answer_region(raw)
    direct = _extract_from_region(answer_text, valid)
    if direct:
        return direct
    return _extract_from_region(raw[-1200:], valid)


def _final_answer_region(text: str) -> str:
    lower = text.lower()
    if "</think>" in lower:
        return text[lower.rfind("</think>") + len("</think>") :].strip()
    assistant_region = _last_assistant_region(text)
    if assistant_region:
        return assistant_region
    if "\nanswer" in lower:
        return text[lower.rfind("\nanswer") :].strip()
    return text[-1200:].strip()


def _last_assistant_region(text: str) -> str:
    patterns = [
        r"<\|im_start\|>\s*assistant\s*\n",
        r"(?:^|\n)assistant\s*\n",
        r"(?:^|\n)assistant\s*:\s*",
    ]
    last_match: re.Match[str] | None = None
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches and (last_match is None or matches[-1].start() > last_match.start()):
            last_match = matches[-1]
    if last_match is None:
        return ""
    region = text[last_match.end() :].strip()
    region = re.split(r"\n(?:user|system|assistant)\s*:?\s*(?:\n|$)", region, maxsplit=1, flags=re.IGNORECASE)[0]
    region = re.split(r"<\|im_end\|>|<\|im_start\|>", region, maxsplit=1, flags=re.IGNORECASE)[0]
    return region.strip()


def _extract_from_region(text: str, valid: set[str]) -> str:
    up = (text or "").upper()
    ordered = "".join(sorted(valid))
    first = re.match(rf"^\s*([{ordered}])\s*(?:[\.\):：,，。]|$)", up)
    if first:
        return first.group(1)

    patterns = [
        rf"(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+CHOICE|CORRECT\s+ANSWER)\s*(?:IS|:)?\s*([{ordered}])\b",
        rf"(?:CHOICE|OPTION)\s*([{ordered}])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, up)
        if matches:
            return matches[-1]

    matches = re.findall(rf"\b([{ordered}])\b", up)
    return matches[-1] if matches else ""
