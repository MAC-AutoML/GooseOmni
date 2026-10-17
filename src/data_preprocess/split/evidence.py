"""Start/end OCR evidence gates and filename helpers."""

from __future__ import annotations

import re

START_TYPES = {"game_start"}
END_TYPES = {"game_end", "result", "victory", "defeat"}

_ROLE_BLURB_VOTE_OK = re.compile(r"(跳过投票|被投票出局|投票出局)")
_MEETING_VOTE_UI = re.compile(
    r"(投票界面|中央.*[「\"']?投票|大字[「\"']?投票|[「\"']投票[「\"']|第一行[「\"']?投票)"
)
_START_REJECT = re.compile(
    r"(准备房|开始游戏|角色选择|职责|的房间|说明\s*[:：]\s*本局角色|"
    r"角色名|角色说明|角色\s*[:：]|14\s*/\s*16|勾选|搜索|"
    r"伪造的|伪钞|任务列表|小地图)"
)
_END_REJECT = re.compile(
    r"(准备房|开始游戏|角色选择|职责|的房间|角色揭示|游戏模式)"
)
_HAS_GAME_MODE = re.compile(r"游戏模式\s*[:：]?\s*(经典版|自定义)")
_ROLE_NAMES = (
    "决斗呆呆鸟",
    "承办丧葬者",
    "生存主义者",
    "加拿大鹅",
    "星界行者",
    "正义使者",
    "专业杀手",
    "流浪儿童",
    "恶魔猎手",
    "观鸟者",
    "探测员",
    "跟踪者",
    "政治家",
    "通灵者",
    "模仿者",
    "殡仪员",
    "静音鸭",
    "告密者",
    "爆炸王",
    "派对鸭",
    "食人族",
    "掠夺者",
    "狙击手",
    "大白鹅",
    "工程师",
    "呆呆鸟",
    "连环杀手",
    "变形者",
    "鉴定师",
    "鸟类学家",
    "超能力者",
    "侦探",
    "警长",
    "网红",
    "肉汁",
    "保镖",
    "复仇者",
    "士兵",
    "法医",
    "杀手",
    "刺客",
    "忍者",
    "间谍",
    "巫医",
    "布谷鸟",
    "秃鹫",
    "鸽子",
    "鹈鹕",
    "猎鹰",
    "渡渡鸟",
    "恋人",
    "食鸟",
    "哑剧",
    "通灵",
    "海盗",
)
_HAS_ROLE_NAME = re.compile("|".join(_ROLE_NAMES))
_HAS_ROLE_BLURB = re.compile(
    r"(生存|获胜|胜利条件|假任务|任务是假的|跳过投票|被投票出局|投票出局|"
    r"职责说明|变身|爱人|活到最后|放置炸弹|摆脱鸭子)"
)
_HAS_CLASSIC_TEAM_TAG = re.compile(
    r"经典版[\s\S]{0,80}[\[【](?:鹅|鸭子)[\]】]"
)
_HAS_RESULT = re.compile(r"(胜利|失败)")


def _strip_role_blurb_vote(text: str) -> str:
    return _ROLE_BLURB_VOTE_OK.sub("", text)


def evidence_accepts_start(evidence: object) -> bool:
    text = str(evidence or "").strip()
    if not text:
        return False
    if _START_REJECT.search(text):
        return False
    if _MEETING_VOTE_UI.search(_strip_role_blurb_vote(text)):
        return False
    if not _HAS_GAME_MODE.search(text):
        return False
    if not (
        _HAS_ROLE_NAME.search(text)
        or _HAS_ROLE_BLURB.search(text)
        or _HAS_CLASSIC_TEAM_TAG.search(text)
    ):
        return False
    return True


def evidence_accepts_end(evidence: object) -> bool:
    text = str(evidence or "").strip()
    if not text:
        return False
    if _END_REJECT.search(text):
        return False
    if _MEETING_VOTE_UI.search(_strip_role_blurb_vote(text)):
        return False
    if text in ("胜利", "失败"):
        return False
    if not _HAS_RESULT.search(text):
        return False
    return bool(re.search(r".{2,}(胜利|失败)", text))


def boundary_evidence_ok(row: dict) -> bool:
    boundary_type = str(row.get("boundary_type") or "")
    evidence = row.get("evidence")
    if boundary_type in START_TYPES:
        return evidence_accepts_start(evidence)
    if boundary_type in END_TYPES:
        return evidence_accepts_end(evidence)
    return False


def canon_result_label(label: str) -> str:
    text = str(label or "")
    text = text.replace("鹅鸭杀", "\x00")
    for src in ("鹅鸭", "鹅鹕", "鹅鹏"):
        text = text.replace(src, "鹈鹕")
    return text.replace("\x00", "鹅鸭杀")


def end_result_label(evidence: object, *, end_type: str = "") -> str:
    text = str(evidence or "").replace("\n", " ").strip()
    match = re.search(r"([\w\u4e00-\u9fff]{1,12})\s*(胜利|失败)", text)
    if match:
        return canon_result_label(f"{match.group(1)}{match.group(2)}")
    kind = str(end_type or "").strip().lower()
    if kind == "defeat":
        return "失败"
    if kind in {"victory", "game_end", "result"}:
        return "胜利"
    return "结果"


def export_episode_filename(index: str | int, *, end_label: str) -> str:
    raw = canon_result_label(str(end_label or "结果"))
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "", raw) or "结果"
    idx = str(index).zfill(3) if str(index).isdigit() else str(index)
    return f"episode_{idx}_{safe}.mp4"
