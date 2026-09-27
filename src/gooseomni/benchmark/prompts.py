from __future__ import annotations

import json
from typing import Any

from .schema import POVRef, Segment, VALID_PLAYERS


COMMON_RULES = """共同标注规则：
- 这是严格 6-POV 对齐数据；同一 segment 内所有 POV 的 local time 0 对应同一个 aligned_start_sec。
- 只输出 strict JSON，不要 Markdown，不要解释性前后缀。
- 不确定时写 "unknown"，并降低 certainty；不能编造玩家、地点、角色或因果。
- 明确区分 direct observation（当前 POV 直接看到/听到/界面显示）和 speech claim（玩家口头声称）。
- direct observation 不能从玩家口头指认推断为事实；speech claim 只能作为声明。
- 除 global_merge 任务外，禁止使用其他 POV 或后续信息；只根据当前输入可见信息标注。
- 模型只输出 local_start_sec / local_end_sec；程序会补充 abs_start_sec / abs_end_sec。
- local_start_sec / local_end_sec 是当前视频片段内的本地秒数，必须从 0 开始计算；绝不能输出 aligned_start_sec/aligned_end_sec 这种绝对游戏秒。
- local_start_sec / local_end_sec 必须落在 0 到 segment.duration_sec 之间。
- 即使是瞬时事件，也必须给一个非零时间窗，例如 local_start_sec=t, local_end_sec=min(t+1, duration_sec)。
"""


COMPACT_OUTPUT_RULES = """输出规模约束：
- 只保留对 ToM / 阵营推理 / 会议投票 / 可见性判断有价值的关键事件，避免逐秒流水账。
- 单个 POV 事件任务最多输出 8 个对象；单个发言任务最多输出 8 个对象；公开阶段任务最多输出 5 个对象。
- description、evidence、transcript 保持短句；每个字段优先不超过 60 个中文字符。
- 如果细节很多，优先合并为一个较长时间窗事件，并在 evidence 中概括关键证据。
- 输出必须是完整可解析 JSON；宁可少输出，也不要输出会被截断的长 JSON。
"""


def _segment_context(segment: Segment) -> str:
    return json.dumps(
        {
            "dataset": segment.dataset,
            "game_id": segment.game_id,
            "segment_id": segment.segment_id,
            "aligned_start_sec": segment.aligned_start_sec,
            "aligned_end_sec": segment.aligned_end_sec,
            "duration_sec": segment.duration_sec,
            "players": list(VALID_PLAYERS),
        },
        ensure_ascii=False,
        indent=2,
    )


def _pov_context(pov: POVRef) -> str:
    return json.dumps(
        {
            "player_id": pov.player_id,
            "video_file": pov.video_file,
            "round_boundary_candidates": pov.qwen_round_boundary_candidates,
        },
        ensure_ascii=False,
        indent=2,
    )


def pov_event_prompt(segment: Segment, pov: POVRef) -> str:
    return f"""TASK: pov_event
你是《鹅鸭杀》Theory-of-Mind benchmark 的 POV 事件初标员。

{COMMON_RULES}

{COMPACT_OUTPUT_RULES}

segment:
{_segment_context(segment)}

current_pov:
{_pov_context(pov)}

只观看 current_pov 的视频，不得借用其他 POV。请抽取当前 POV 在该片段中能直接观察到或听到的关键事件。
输出 JSON 数组。每个对象必须包含：
dataset, game_id, segment_id, player_id, event_id, local_start_sec, local_end_sec,
event_type, location, actor, visible_players, mentioned_players, description,
is_direct_observation, is_speech_claim, speaker, utterance, claim_type,
certainty, evidence, source_pov, needs_human_review。

要求：
- player_id 必须是当前 POV 玩家。
- source_pov 必须是数组，至少包含当前 player_id。
- event_type 使用英文短标签，例如 movement, task, encounter, meeting, discussion,
  vote, ejection, body_report, kill, role_clue, task_ui, uncertainty。
- actor 不确定时写 null；speaker/utterance/claim_type 只在 speech claim 相关事件里填写。
- evidence 用中文，写清楚画面/声音/界面证据。
- 如果只是有人说了某事，is_speech_claim=true，is_direct_observation=false。
- 如果当前 POV 直接看到地图、玩家、尸体、投票 UI 或任务 UI，is_direct_observation=true。
- 最多输出 8 个最关键事件；普通移动、重复 UI、无法支撑 ToM 的闲置片段可以省略。
- 如果没有可靠事件，输出 []。
"""


def utterance_prompt(segment: Segment, pov: POVRef) -> str:
    return f"""TASK: utterance
你是《鹅鸭杀》会议/语音初标员。

{COMMON_RULES}

{COMPACT_OUTPUT_RULES}

segment:
{_segment_context(segment)}

current_pov:
{_pov_context(pov)}

只根据 current_pov 的音频和画面字幕/聊天框标注发言，不得使用其他 POV。
输出 JSON 数组。每个对象必须包含：
dataset, game_id, segment_id, player_id, utterance_id, local_start_sec,
local_end_sec, speaker, speaker_confidence, transcript, text, addressee,
mentioned_players, speech_act, claims, possible_intents, certainty, evidence,
source_pov, is_direct_observation, is_speech_claim, needs_human_review。

要求：
- speaker 不确定时写 "unknown"，certainty 降低。
- claims 是数组；每个 claim 包含 claim_id, speaker, claim_type, content,
  subject_players, object_players, mentioned_players, locations, time_referred,
  is_direct_observation_claim, certainty, evidence。
- possible_intents 是弱推测，只能用于候选筛选；必须给 confidence 和 reason。
- speech_act 可用 accuse, defend_self, defend_other, ask_question, answer_question,
  provide_alibi, challenge_alibi, vote_suggestion, joke, noise, other, unknown。
- 对无法听清或无法确定说话人的内容，needs_human_review=true。
- 最多输出 8 条关键发言；重复闲聊、噪声、无法听清且无 ToM 价值的内容可以省略。
- 如果没有可标注发言，输出 []。
"""


def global_merge_prompt(
    segment: Segment,
    pov_event_annotations: list[dict[str, Any]],
    utterance_annotations: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "pov_event_annotations": pov_event_annotations,
            "utterance_annotations": utterance_annotations,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""TASK: global_merge
你是多 POV 全局事件融合器。当前任务允许使用同一 segment 内全部 6 个 POV 的初标结果。

{COMMON_RULES}

segment:
{_segment_context(segment)}

annotations:
{payload}

请合并重复事件，识别多 POV 冲突，生成全局事件。输出 JSON 数组。
每个对象必须包含：
dataset, game_id, segment_id, target_player, global_event_id,
local_start_sec, local_end_sec, event_type, description, actors, involved_players,
visible_to, heard_by, supporting_pov_event_ids, supporting_utterance_ids, conflict,
certainty, evidence, source_pov, is_direct_observation, is_speech_claim,
needs_human_review。

要求：
- target_player 对全局事件写 "global"。
- source_pov 是支持该全局事件的 POV 列表。
- 最多输出 4 个全局事件；每个事件的 supporting_pov_event_ids 和 supporting_utterance_ids 各最多 6 个，禁止生成连续大列表。
- description/evidence 用短句；不要复述所有发言，只保留 ToM 有价值的核心事实。
- 如果不同 POV 对同一事实冲突，conflict=true 且 needs_human_review=true。
- 不要引入这个 segment 之外的信息。
"""


def phase_event_prompt(segment: Segment, phase_sources: list[dict[str, Any]]) -> str:
    payload = json.dumps(phase_sources, ensure_ascii=False, indent=2)
    return f"""TASK: phase_event
你是《鹅鸭杀》公开阶段事件检测器。

{COMMON_RULES}

{COMPACT_OUTPUT_RULES}

segment:
{_segment_context(segment)}

phase_sources:
{payload}

请根据 6 个 POV 的 round_boundary_candidates 或已有标注，检测公开阶段事件。
输出 JSON 数组。每个对象必须包含：
dataset, game_id, segment_id, phase_event_id, local_start_sec, local_end_sec,
phase_type, visible_to_all, evidence_povs, certainty, evidence, source_pov,
is_direct_observation, is_speech_claim, needs_human_review。

phase_type 只能是 discussion, voting, vote_result, exile, action,
body_report, meeting_start, unknown。
最多输出 5 个公开阶段事件；没有明确阶段切换时输出 []。
只标公开阶段，不要推断玩家身份或意图。
"""


def memory_state_prompt(
    segment: Segment,
    target_player: str,
    previous_memory_state: dict[str, Any] | None,
    pov_events: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    phase_events: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "target_player": target_player,
            "previous_memory_state": previous_memory_state,
            "pov_events": pov_events,
            "utterances": utterances,
            "phase_events": phase_events,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""TASK: memory_state
你是玩家主观记忆状态构造器。请维护 target_player 跨 segment 的 rolling memory。

{COMMON_RULES}

segment:
{_segment_context(segment)}

input_state_and_annotations:
{payload}

输出 strict JSON 对象，必须包含：
dataset, game_id, segment_id, target_player, cutoff_abs_sec,
memory_items, memory_delta, needs_human_review。

memory_items 每项必须包含 memory_id, first_observed_abs_sec,
last_referenced_abs_sec, memory_type, content, source_event_ids,
source_claim_ids, confidence, decay_status, visibility, needs_human_review。

强约束：
- memory_type 只能是 direct_visual, heard_claim, public_result, self_action, inferred, phase。
- decay_status 只能是 active, stale, contradicted；不要输出 stable。
- visibility 只能是 private 或 public。
- memory_delta 必须是数组；每项包含 operation, memory_id, reason。
- 最多输出 8 个 memory_items，只保留 ToM / 投票 / 怀疑 / 可见性相关记忆。
- memory_item 必须挂 source_event_ids 或 source_claim_ids。
- 没有证据 ID 的 memory_item 必须 needs_human_review=true。
- heard_claim 不能升级成真实事实。
- 不要把其他 POV 私有信息写入 target_player 的 private memory。
"""


def belief_state_prompt(
    segment: Segment,
    target_player: str,
    memory_state: dict[str, Any],
    pov_events: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    global_events: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "target_player": target_player,
            "memory_state": memory_state,
            "pov_events": pov_events,
            "utterances": utterances,
            "global_events": global_events,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""TASK: belief_state
你是玩家有限视角信念状态构造器。

{COMMON_RULES}

segment:
{_segment_context(segment)}

input_annotations:
{payload}

输出 strict JSON 对象，必须包含：
dataset, game_id, segment_id, target_player, cutoff_abs_sec,
knows, does_not_know, believes_or_suspects, trust_state,
forbidden_information, certainty, evidence, source_pov, needs_human_review。

强约束：
- belief_state 不等于 global truth。
- knows 和 does_not_know 必须是数组；不要输出嵌套对象。
- believes_or_suspects 必须是数组；每项包含 player, belief_type, content, confidence。
- forbidden_information 必须是对象数组；每项包含 hidden_event_id, reason。
- 每个数组最多 8 项；content/evidence 用短句。
- 必须列出 target_player 在 cutoff 前不可知的 forbidden_information。
- 不得用 forbidden_information 支撑 knows 或 believes_or_suspects。
- 不确定时降低 certainty 并设置 needs_human_review=true。
"""


def checker_prompt(
    checker_name: str,
    segment: Segment,
    payload: dict[str, Any],
) -> str:
    return f"""TASK: checker
你是 GooseOmni 标注自检器，检查已有标注，不重新生成标注。

checker_name={checker_name}

{COMMON_RULES}

segment:
{_segment_context(segment)}

payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

输出 strict JSON 数组。每个对象必须包含：
checker_name, annotation_id, verdict, reason, suggested_fix, confidence,
leakage, leaked_items。

verdict 只能是 supported, partially_supported, unsupported, uncertain, pass, fail。
重点检查：证据是否支持、claim/fact 是否混淆、是否使用 forbidden_information、时间和 6-POV 合并是否一致。
"""


def information_state_prompt(
    segment: Segment,
    target_player: str,
    global_events: list[dict[str, Any]],
    pov_events: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    *,
    hidden_global_event_refs: list[dict[str, Any]] | None = None,
    projection_rules: list[str] | None = None,
) -> str:
    payload = json.dumps(
        {
            "target_player": target_player,
            "visible_or_public_global_events": global_events,
            "target_pov_events": pov_events,
            "target_heard_utterances": utterances,
            "hidden_global_event_refs_for_unknowns_only": hidden_global_event_refs or [],
            "projection_rules": projection_rules or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""TASK: information_state
你是 Theory-of-Mind benchmark 的信息状态构建器。

{COMMON_RULES}

segment:
{_segment_context(segment)}

input_annotations:
{payload}

请推断 target_player 在 cutoff_abs_sec=segment.aligned_end_sec 时知道什么、相信什么、不知道什么。
输入已经由程序投影到 target_player 的有限视角：
- visible_or_public_global_events 是该玩家可见、可听或公开阶段的信息。
- target_pov_events 是该玩家自己的 POV 事件。
- target_heard_utterances 是该玩家自己的 POV 中听到的发言。
- hidden_global_event_refs_for_unknowns_only 只能用于 unknown_or_unseen_information / unknowns，不能用于 known_facts、beliefs、suspicion_state 或 trust_state。

强约束：
- 不得把 hidden_global_event_refs_for_unknowns_only 当成 target_player 已知信息。
- 不得用其他 POV 私有事实支撑 target_player 的 belief。
- 发言 claim 只能作为 heard claim，不能自动升级为全局事实。
- source_pov 必须包含 target_player；如果使用公开事件，也可以包含支持该公开事件的 POV。
- 输出必须短：每个列表最多 5 项；每项不超过 60 个中文字符。
- known_facts、beliefs、unknowns、private_observations、public_information、false_or_uncertain_beliefs 必须是字符串数组，不要嵌套对象。
- available_information、unknown_or_unseen_information 只输出短对象数组，每项最多包含 type、content、source_ids。
- suspicion_state、trust_state 只输出玩家名到短字符串或分数的映射。
- evidence 不超过 80 个中文字符。

输出 strict JSON 对象，必须包含：
dataset, game_id, segment_id, target_player, cutoff_abs_sec,
available_information, unknown_or_unseen_information, suspicion_state, trust_state,
known_facts, beliefs, unknowns, private_observations, public_information,
false_or_uncertain_beliefs, certainty, evidence, source_pov,
needs_human_review。

不确定时写入 unknowns，并降低 certainty。
"""


def candidate_trial_prompt(
    segment: Segment,
    global_events: list[dict[str, Any]],
    information_states: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "global_events": global_events,
            "information_states": information_states,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""TASK: candidate_trial
你是 Theory-of-Mind benchmark 候选题生成器。

{COMMON_RULES}

segment:
{_segment_context(segment)}

input_annotations:
{payload}

请生成能用于 ToM benchmark 的候选题。输出 JSON 数组。
每个对象必须包含：
dataset, game_id, segment_id, trial_id, question_type, trial_type, target_player,
cutoff_abs_sec, question, answer, distractors, available_information,
hidden_information, expected_answer_basis, supporting_global_event_ids,
supporting_information_state_ids, risk_of_perspective_leakage,
certainty, evidence, source_pov, needs_human_review。

要求：
- question_type 只能是 first_order_belief, second_order_belief,
  hidden_information, false_belief, intent_inference, knowledge_access, belief_state。
- risk_of_perspective_leakage 只能是 low, medium, high。
- 如果答案依赖 target_player 不可能知道的其他 POV 私有信息，风险标 medium/high。
- 不确定时降低 certainty，并设置 needs_human_review=true。
"""
