# GooseOmni Decrypto-style ToM 诊断层

本层目标不是把 segment 视频包装成孤立 QA，而是把严格 6-POV 对齐的 `g001` replay 转成 trajectory-derived Theory-of-Mind diagnostic benchmark：

```text
6-POV aligned replay trajectory
-> oracle trajectory ledger
-> visibility / belief / memory graph
-> Decrypto-style A/B/C/D ToM probe groups
-> interactive diagnostics + static benchmark trials
-> controlled evaluation
```

benchmark 单位不是 segment，而是：

```text
cutoff_abs_sec + target_player + query_variable + information_gap
```

## 核心研究问题

给定一个真实多人社交博弈轨迹，模型能否区分：

1. 全局真实发生了什么；
2. 某个玩家当时看见/听见了什么；
3. 某个玩家没看见但别人看见了什么；
4. 某个玩家听到的 claim 与真实事件是否冲突；
5. 某个玩家是否知道这个冲突；
6. 某个玩家在真相揭示前会形成什么错误信念；
7. 真相揭示后，模型能否重建揭示前的错误信念；
8. 一个玩家能否预测另一个玩家会如何理解自己的发言。

## 输出目录

```text
annotations/oracle_ledger/
  world_events.jsonl
  claims.jsonl
  phase_events.jsonl
  visibility_edges.jsonl
  belief_memory_snapshots.jsonl
  claim_truth_links.jsonl
  canonical_event_map.jsonl

annotations/diagnostics/
  probe_groups.jsonl
  probes_A_pre_reveal.jsonl
  probes_B_reconstruct.jsonl
  probes_C_false_belief.jsonl
  probes_D_perspective_taking.jsonl
  hidden_gold.jsonl
  diagnostic_quality.jsonl

benchmark/gooseomni_v1/
  interactive_diagnostics/
  static_trials/
  reports/
```

## Qwen3-Omni 质量默认值

Decrypto-style 诊断层不能继续使用 4096-token 粗标默认值。Slurm 模板 `configs/slurm/qwen3_omni_decrypto_diagnostics.slurm` 固定以下质量优先配置：

```text
QWEN3_OMNI_MAX_TOKENS=16384
QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS=32768
QWEN3_OMNI_VIDEO_FPS=1.0
QWEN3_OMNI_VIDEO_MAX_FRAMES=96
QWEN3_OMNI_VIDEO_MAX_PIXELS=401408
```

高价值会议、claim-heavy phase 或人工复核任务可以把 `QWEN3_OMNI_VIDEO_MAX_FRAMES` 提升到 `128` 或 `192`。

## 通用提示词前缀

所有 Qwen3-Omni oracle extraction / merge / probe synthesis prompt 都必须写入以下固定上下文：

```text
Dataset: GooseOmni
Game: Goose Goose Duck / 鹅鸭杀风格多人社交博弈
Game ID: g001
Players:
- Gemini
- baile
- beigang
- mojiang
- saoyi
- xiaolu

Important time rule:
- Each POV clip local time starts at 0.
- abs_sec = aligned_start_sec + local_sec.
- If you output local_start_sec/local_end_sec, they must be local to the clip.
- The program may supplement abs_start_sec/abs_end_sec, but your event ordering must be time-consistent.

Important epistemic rule:
- Distinguish global truth from what each player could know.
- Do not say a player knew an event unless that player directly saw it, heard it, or it was public before cutoff.
- If an event is only visible in another POV, mark it hidden for the target player.
- Speech claims are not truth. Claims must be linked to evidence and truth status separately.
- Unknown is acceptable. Do not invent exact roles, intentions, or locations without evidence.

Output:
- Strict JSON only.
- No markdown.
- No natural-language summary outside JSON.
- Include evidence and confidence for every event/claim/link.
```

## Oracle ledger extraction prompt

```text
SYSTEM:
You are building a trajectory-derived Theory-of-Mind benchmark from a 6-POV Goose Goose Duck replay.
Your task is not to write QA questions. Your task is to create an oracle trajectory ledger: what happened, who saw/heard it, who did not, and which speech claims were made.

You may inspect all provided POVs because this is oracle annotation.
However, when writing visibility and belief fields, strictly separate global truth from each player's local perspective.

PLAYERS:
Gemini, baile, beigang, mojiang, saoyi, xiaolu

TIME:
Each input video has local time starting at 0.
Segment aligned_start_sec = {aligned_start_sec}
Segment aligned_end_sec = {aligned_end_sec}
abs_sec = aligned_start_sec + local_sec

TASK:
Extract richly detailed candidate ledger entries:
1. world_events: visual or public events that occurred in the game world.
2. claims: speech acts that assert location, action, role, sighting, accusation, defense, vote suggestion, trust, suspicion, or intent.
3. phase_events: meeting start, discussion, vote, vote result, exile, death reveal, result screen, lobby transition.
4. visibility_edges: for each important event, record each player's relation to it.
5. claim_truth_candidates: possible links between a claim and one or more world events.

IMPORTANT:
- Do not collapse speech claim into truth.
- If a player says "I was in X", record this as a claim, not as a world event unless visually confirmed.
- If only one POV sees an event, it is private to that POV unless later publicly revealed.
- If multiple POVs see the same event due to overlap or shared view, list all source_povs.
- If the event is uncertain, mark needs_human_review=true.
- Prefer rich detail over short output. Do not omit borderline ToM-relevant events.

OUTPUT STRICT JSON.
```

## Probe generation prompt

```text
SYSTEM:
You are generating Decrypto-style Theory-of-Mind diagnostic probes.
Each probe group must test an information gap, not simple factual recall.

IMPORTANT:
- A prompt must not reveal oracle truth.
- B prompt may reveal oracle truth but must ask the model to reconstruct the target player's earlier belief.
- B prompt must not include any previous probe response.
- C prompt may reveal oracle truth but asks about another player's pre-reveal belief.
- D prompt asks a speaker to predict a listener's interpretation or action.
- Keep prompts concise enough for evaluation but include enough context to be answerable.
- All outputs must define strict JSON answer schema.

INPUT:
probe_group
target_player_memory_snapshot
other_player_memory_snapshots
oracle_truth_events
related_claims
public_history

OUTPUT:
A_pre_reveal_belief
B_post_reveal_reconstruct_previous_belief
C_other_agent_false_belief
D_perspective_taking_prediction when speaker/listener information asymmetry and later behavior exist
hidden_gold for scorer only
```

## 运行命令

```bash
.venv/bin/python tools/build/build_oracle_trajectory_ledger.py \
  --release-root runs/gooseomni_gameplay_pass1/release_benchmark_v2 \
  --output-root annotations \
  --game-id g001

.venv/bin/python tools/build/build_decrypto_style_diagnostics.py \
  --annotation-root annotations \
  --limit 240

.venv/bin/python tools/package/export_gooseomni_benchmark.py \
  --annotation-root annotations \
  --benchmark-root benchmark/gooseomni_v1

.venv/bin/python tools/validate/validate_decrypto_style_outputs.py \
  --annotation-root annotations \
  --benchmark-root benchmark/gooseomni_v1 \
  --output benchmark/gooseomni_v1/reports/validation.json
```

## Gold source 约束

自动生成结果只能标为 `qwen_weak` 或 `qwen_checked`。没有人工复核时，不允许标为 `human_verified`。
