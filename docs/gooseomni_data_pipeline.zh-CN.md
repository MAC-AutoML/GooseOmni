# GooseOmni 数据制作闭环

权威入口是声明式配置 `configs/datasets/gooseomni_v2_raw.yaml`。玩家数量、名称和原始视频路径均由配置决定，不在代码中写死。

## 完整命令

```bash
gooseomni data validate-input --config configs/datasets/gooseomni_v2_raw.yaml
gooseomni data build --config configs/datasets/gooseomni_v2_raw.yaml
gooseomni data build --config configs/datasets/gooseomni_v2_raw.yaml --resume
gooseomni data status --run runs/gooseomni_v2_raw_release
gooseomni data validate --run runs/gooseomni_v2_raw_release
```

固定阶段为：

```text
ingest -> sync -> segment -> qwen_perception -> event_fusion
       -> oracle_and_belief -> codex_review -> trial_build -> validate -> package
```

每阶段在 `run_manifest.json`、`events.jsonl` 和 `errors.jsonl` 中记录状态、输入/输出哈希和耗时。`--resume` 只复用配置哈希一致且已经完成的阶段；配置变化必须创建新 run，避免混入旧结果。

## 缓存与真实模型

### 数据契约和产物归属

- `annotation.schemas` 定义 clip 级初标和融合记录，使用 `start_sec/end_sec`、
  `confidence` 等字段；raw 流水线仍依赖这些模型。
- `benchmark.schema` 定义 segment 级标注与评测记录，使用 `segment_id`、
  `cutoff_abs_sec`、`certainty` 等字段。两种记录不能混用 schema 校验。
- raw 融合结果只归属于具体 run 的 `artifacts/fusion/`；
  `ledger_seed.build_seed_ledger` 负责将融合记录转换成 oracle ledger。
- 已删除 `data/processed/` 根目录中无 run 归属的全局事件、信息状态、
  会议发言及候选题 JSON（包括 `manual_sync_v1` 和 `round2_sync` 版本）。
  原视频、同步配置、clip manifest 和当前配置引用的缓存继续保留。
- 不再提供无参数 `annotate postprocess`；单阶段工具必须显式指定路径。
  完整构建使用上面的 `data build` 命令。

已有数据可在 YAML 的 `cache` 中声明同步、切片、Qwen 感知、oracle ledger 和历史模型裁决。缓存会复制到 run-local 目录并重新计算哈希；正式 benchmark 不直接依赖旧活动目录。

删除 `cache.sync_offsets` 或 `cache.perception` 后，对应阶段会调用真实 Qwen3-Omni。真实感知必须在 Slurm 作业中运行，并提供 `QWEN3_OMNI_SERVER_URL`。模型路径和资源参数见 `docs/qwen3_omni_annotation_pipeline.zh-CN.md`。

删除 `cache.oracle_ledger` 后，流水线会从 run-local Qwen 融合事件生成 seed ledger，包括 world event、claim、visibility、belief snapshot 和 claim link。seed 中无法可靠确定的全局事实保持 `unverified`，交给后续 Codex 裁决，不会伪造确定 gold。

删除 `cache.review_benchmark` 后，候选会调用 `GOOSEOMNI_CODEX_MODEL`（默认 `gpt-5.6-sol`）裁决。API key 只从 `OPENAI_API_KEY` 和 `OPENAI_API_BASE` 读取。裁决仅允许 `accept`、`repair`、`reject`；错误、无效 JSON 或未知 evidence ID 均失败关闭。

不使用 API 时，使用本地 Codex 直接审核：

```bash
uv run python tools/annotation/run_codex_local_review.py prepare \
  --candidates runs/raw_seed_ledger_probe_v4/candidates \
  --review-root runs/codex_direct_review_full_v4 \
  --shard-size 3

uv run python tools/annotation/run_codex_local_review.py merge \
  --review-root runs/codex_direct_review_full_v4 \
  --output runs/codex_direct_review_full_v4/decisions.jsonl
```

每个 shard 的输出受统一 JSON Schema 约束。merge 会拒绝重复/未知 trial ID、未知 evidence ID 和非法 repair；缺少输出的 shard 只记为 pending，可继续审核，不会覆盖已完成裁决。

自动发布物只允许 `gold_source=model_verified`。validator 会拒绝 JSON/JSONL 中残留的 `human_verified`，并检查 manifest SHA-256、公开泄漏、行数以及 raw-video 音视频流。

## 评测

发布后可直接进入统一 14 模型、三轨评测：

```bash
gooseomni eval plan \
  --benchmark gooseomni_v2_raw_release \
  --models all \
  --tracks leaderboard_core raw_video_smoke agentic_midgame_prediction

gooseomni eval run \
  --benchmark gooseomni_v2_raw_release \
  --model gpt4o \
  --track leaderboard_core \
  --modalities text \
  --limit 1 \
  --run runs/eval/raw_release_gpt4o_smoke
```

hidden gold 只由 scorer 读取，不进入推理请求。不兼容模态记录为 `skipped`，不计作模型失败。
