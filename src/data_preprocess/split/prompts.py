from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowClip:
    game_id: str
    player_id: str
    clip_id: str
    clip_path: str
    start_sec: float
    end_sec: float


def _clip_context(clip: WindowClip) -> str:
    return (
        f"game_id={clip.game_id}, player_id={clip.player_id}, "
        f"clip_id={clip.clip_id}, aligned_time=[{clip.start_sec}, {clip.end_sec}]"
    )


def episode_boundary_start_prompt(clip: WindowClip) -> str:
    return f"""你是《鹅鸭杀》【单局开局】审查员。本阶段只标 game_start，禁止标任何结束。
只看本 POV 视频窗口；没有合格开局幕布就输出 []。

片段信息：{_clip_context(clip)}

【位置与大小】关键文字必须在视角正中央且超大；角落/侧栏/小条一律 pass。

【game_start = 开局角色揭示幕布】必须同时满足：
1. 全屏纯色/渐变幕布，几乎无地图、无准备房按钮
2. 正中央三排大字：游戏模式 + 角色名 + 角色说明（缺一不可）
3. 可有玩家名 + 单个本角色立绘

【硬否定】准备房/开始游戏/角色选择/地图/会议投票/结算「胜利」「失败」→ 不要标。
局内左侧「职责」任务列表、小地图、走动、带「[伪造的]」的假任务 → 一律不要标。
本阶段即使看见胜利幕布也不要输出。

evidence：必须 OCR 真实可见「游戏模式」+ 中央角色名；禁止套话。
【aligned 硬约束】aligned_start_sec/aligned_end_sec 必须落在幕布仍占满画面的那一秒；
禁止对齐到准备房/设置菜单/黑场过渡/已进地图游玩；禁止图省事贴窗首或窗尾。
输出：JSON 数组，最多 1 条 game_start；无则 []。不要 Markdown。
字段：game_id, player_id, clip_id, round_index, aligned_start_sec, aligned_end_sec,
boundary_type, evidence, confidence, source_povs, needs_review。
boundary_type 只能是 game_start；aligned_* 落在 aligned_time 内。
"""


def episode_boundary_end_prompt(clip: WindowClip) -> str:
    return f"""你是《鹅鸭杀》【单局结算】审查员。本阶段只标 game_end/result/victory/defeat，禁止标开局。
只看本 POV 视频窗口；没有合格结算幕布就输出 []。

片段信息：{_clip_context(clip)}

【位置与大小】「胜利/失败」必须在视角正中央且超大；角落/侧栏一律 pass。

【game_end = 结算胜利幕布】必须同时满足：
1. 全屏纯色/渐变幕布
2. 正中央超大号「xxx 胜利」或「xxx 失败」

【硬否定】准备房/开局角色揭示三排字/地图/会议「投票」「讨论」+ 玩家格子 → 不要标。
本阶段不要输出 game_start。

evidence：必须 OCR 中央「胜利/失败」原文；禁止套话。
【aligned 硬约束】aligned_* 必须对准结算幕布仍全屏可见的那一秒；
禁止对齐到会议/地图/准备房；禁止贴窗首或窗尾凑数。
输出：JSON 数组，最多 1 条结束边界；无则 []。不要 Markdown。
字段：game_id, player_id, clip_id, round_index, aligned_start_sec, aligned_end_sec,
boundary_type, evidence, confidence, source_povs, needs_review。
boundary_type 只能是 game_end, result, victory, defeat；aligned_* 落在 aligned_time 内。
"""
