# GooseOmni v2 Pilot：严格同步与轨迹驱动 ToM

`gooseomni_v2_pilot` 只用于当前六 POV 会话，不是正式跨会话榜单。旧 855 条结果在
`benchmark/registry.json` 中冻结为 `weak_epistemic_diagnostic`，不覆盖、不删除。

## 阶段

```text
ingest
→ align
→ episode
→ perceive_visual
→ perceive_audio
→ trajectory_fusion
→ information_state
→ tom_trial_build
→ codex_review
→ validate
→ package
```

## Phase-aware 双遍感知

严格 Pilot 的视觉和音频输入先按审核后的 phase 边界切分，再在 phase 内限制为最长
30 秒。分块参数必须写入数据集 YAML，并进入 config hash：

```yaml
perception:
  dual_pass: true
  speaker_confidence_min: 0.85
  max_visual_chunk_sec: 30
  max_audio_chunk_sec: 30
  split_on_phase_boundaries: true
```

感知 manifest 必须携带 `episode_id`、`phase_index`、`phase_type`、全局时间、原视频
时间和媒体 SHA-256。gameplay 中禁止 `meeting_public=true`，任何事件不得跨 phase。
视觉私有事件不会仅因时间接近而跨 POV 合并；音频只有同 phase、同 speaker、同
utterance 且至少两个 POV 一致时才可传播。

正式 Pilot 配置为 `configs/datasets/gooseomni_v2_pilot.yaml`；版本化 smoke 配置
只能写入独立 run。所有严格配置均不得引用旧
`sync_offsets.json` 或任何硬编码 correction。唯一同步文件为 run 内的
`artifacts/alignment.json`，映射公式是：

```text
raw_sec = scale * global_abs_sec + offset
```

每个 POV 至少需要三个公共事件锚点。RANSAC 拟合后的残差中位数必须不超过 1 秒，
P95 必须不超过 2 秒；失败时整个 episode 进入 quarantine，不能继续发布。

## 执行顺序

1. 在 `com300` 产生跨 POV 音频、UI/OCR、公共事件锚点，写成审计后的
   `alignment_anchors.json`。每条锚点必须包含 `global_abs_sec`、`raw_sec`、
   `event_type`、`evidence_id` 和 `confidence`。
2. 将该文件配置为 `cache.alignment_anchors`，在计算节点运行 align；align 按唯一
   仿射映射生成审计窗口，episode 审核通过后再生成 phase-aware 模型输入。
3. 在 gpu7/gpu8 提交 `configs/slurm/gooseomni_v2_pilot_qwen.slurm`，执行 episode
   共识和 Qwen3-Omni Pass A/Pass B。
4. `codex_review` 不调用 AutoML/OpenAI API。远端生成受约束 review queue，本机
   `codex exec` 使用现有 Codex provider 审核结构化结果和关键帧，远端再执行 exact
   coverage、evidence allow-list 与 repair schema 校验。无效响应失败关闭。
5. validator 通过后才能 package。发布物全部标为 `model_verified`。

常用命令：

```bash
.venv/bin/python -m gooseomni data validate-input \
  --config configs/datasets/gooseomni_v2_pilot.yaml

.venv/bin/python -m gooseomni data build \
  --config configs/datasets/gooseomni_v2_pilot.yaml --from-stage align --resume

.venv/bin/python -m gooseomni data status --run runs/gooseomni_v2_pilot
```

真实 Qwen 冒烟可设 `GOOSEOMNI_PILOT_LIMIT_CLIPS=6`，但限量结果不能通过 Pilot
发布门禁。完整 Pilot 至少需要五个有效 episode、五层各 30 条高置信 trial，并覆盖
六个 target player。正式 v2 还必须增加两个独立六 POV 会话，按会话隔离 split。

v14 工程冒烟使用 `configs/datasets/gooseomni_v2_pilot_smoke_v14.yaml`。该窗口按
phase 产生每个 POV 四段、共 24 段，必须完整运行，不设置全局 `LIMIT=6`。smoke
可以生成验证报告，但只有 `release_eligible=true` 时 package 才会执行。

v15 使用同一审核过的 alignment 与 episode，但在独立 run 中重新执行全部感知，
用于验证无人值守总控、Qwen 服务重启和两轮本机 Codex 审核。它不复用 v14 感知
结果，也不会自动 package。

v15 真实闭环结果：24 个视觉输入得到 6 个有效结果和 18 个质量隔离，24 个音频输入
得到 12 个有效结果和 12 个质量隔离，未发生 OOM。102 个视觉轨迹候选经本机 Codex
审核后保留 9 个节点，音频融合另隔离 21 个不可传播节点；最终形成 36 个信息状态，
但没有产生高置信 ToM group。validator 返回 `ok=true`、`release_eligible=false`，且未
创建 benchmark package。该结果证明自动控制闭环可运行，不代表数据门槛已经满足。

## 无人值守执行

从 Mac 挂载项目根运行：

```bash
bash tools/automation/run_strict_tom_pipeline.sh \
  --config configs/datasets/gooseomni_v2_pilot_smoke_v14.yaml \
  --run runs/gooseomni_v2_pilot_smoke_v14 \
  --nodelist gpu8
```

脚本会自动完成：提交并等待 Qwen Slurm 作业、OOM 后最多重启服务三次、轨迹 Codex
审核、ToM group Codex 审核、恢复流水线和 validate。它不会自动 package；正式门槛
必须由 validator 满足。已有成功 stage 和 decisions 会被复用，不会由失败重试覆盖。

单独重跑某一轮本机审核：

```bash
bash tools/automation/run_local_codex_review.sh \
  --run runs/gooseomni_v2_pilot_smoke_v14 \
  --kind trajectory
```
